# train_models.R
# Dijalankan manual SEKALI (bukan oleh app.R). Hasilnya disimpan ke
# model_results_dini.rds dan model_results_lanjutan.rds, lalu app.R tinggal
# load file itu -- supaya app tidak training ulang setiap dibuka.
#
# Jalankan ulang script ini kapan pun data.csv berubah atau parameter model
# diubah, lalu commit ulang kedua file .rds itu.

library(dplyr)
library(rsample)
library(recipes)
library(caret)
library(e1071)
library(xgboost)
library(pROC)
library(randomForest)  # dipakai oleh rfFuncs di dalam rfeControl

set.seed(123)

# ---- 1. Load data ----
df <- read.csv("data.csv", sep = ";", header = TRUE)

categorical_vars <- c("course", "timeof_attendance", "prev_quali", "gender", "scholarship")
df <- df %>% mutate(across(all_of(categorical_vars), as.factor))

# Label risiko dibuat eksplisit: "Dropout" konsisten jadi positive class di
# confusionMatrix, glm (level kedua = "success"), dan label 1 di xgboost.
# Ini benerin bug lama: tiap model punya arah "positive" yang beda-beda.
df$target <- factor(ifelse(df$target == 0, "Dropout", "Lulus"), levels = c("Lulus", "Dropout"))

# ---- 2. Split SEBELUM feature selection (fix data leakage) ----
split    <- initial_split(df, prop = 0.8, strata = "target")
df_train <- training(split)
df_test  <- testing(split)

# ---- 3. Dua pool fitur ----
# Dini: hanya variabel yang diketahui saat mahasiswa MENDAFTAR.
# Lanjutan: semua variabel, termasuk performa semester 1 & 2.
pool_dini     <- c("course", "timeof_attendance", "prev_quali", "prev_quali_grade",
                   "admiss_grade", "gender", "scholarship", "age_enroll")
pool_lanjutan <- setdiff(colnames(df), "target")

# ---- 4. RFE per pool, HANYA di df_train ----
pilih_fitur <- function(pool) {
  control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
  rfe_result <- rfe(
    x = df_train[, pool],
    y = df_train$target,
    sizes = seq_len(length(pool)),
    rfeControl = control
  )
  list(rfe = rfe_result, fitur = predictors(rfe_result))
}

cat("Menjalankan RFE untuk pool Dini...\n")
rfe_dini <- pilih_fitur(pool_dini)
cat("Fitur terpilih (Dini):", paste(rfe_dini$fitur, collapse = ", "), "\n\n")

cat("Menjalankan RFE untuk pool Lanjutan...\n")
rfe_lanjutan <- pilih_fitur(pool_lanjutan)
cat("Fitur terpilih (Lanjutan):", paste(rfe_lanjutan$fitur, collapse = ", "), "\n\n")

# ---- 5. Helper metrics: Accuracy, Sensitivity, Specificity, AUC ----
hitung_metrics <- function(pred_class, pred_prob, actual) {
  cm      <- confusionMatrix(pred_class, actual, positive = "Dropout")
  roc_obj <- roc(actual, pred_prob, levels = c("Lulus", "Dropout"), direction = "<", quiet = TRUE)
  list(
    accuracy    = unname(cm$overall["Accuracy"]),
    sensitivity = unname(cm$byClass["Sensitivity"]),
    specificity = unname(cm$byClass["Specificity"]),
    auc         = as.numeric(auc(roc_obj)),
    conf_matrix = cm,
    roc         = roc_obj
  )
}

# Ambil probabilitas kelas "Dropout" dari model SVM (butuh probability=TRUE
# saat training) -- biar kurva ROC-nya proper, bukan cuma 2 titik dari label keras.
svm_prob_dropout <- function(model, newdata) {
  pred <- predict(model, newdata = newdata, probability = TRUE)
  list(class = pred, prob = attr(pred, "probabilities")[, "Dropout"])
}

# ---- 6. Fungsi generik: preprocessing + latih 5 model untuk satu pool ----
latih_model_suite <- function(fitur) {
  recipedata <- recipe(target ~ ., data = df_train[, c(fitur, "target")]) %>%
    step_normalize(all_numeric_predictors()) %>%
    step_dummy(all_nominal_predictors(), one_hot = TRUE)
  prepdata <- prep(recipedata)

  train_preproc <- bake(prepdata, new_data = NULL)
  test_preproc  <- bake(prepdata, new_data = df_test[, c(fitur, "target")])

  hasil <- list()

  ## --- XGBoost ---
  train_matrix <- model.matrix(target ~ . - 1, data = train_preproc)
  test_matrix  <- model.matrix(target ~ . - 1, data = test_preproc)
  # label 1 = Dropout (level kedua) -- selaras sama arah "positive" di confusionMatrix
  train_label <- as.numeric(train_preproc$target) - 1
  test_label  <- as.numeric(test_preproc$target) - 1

  dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
  dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)

  # rasio mayoritas:minoritas -- Dropout cuma ~28% data, jadi di-upweight
  # supaya model nggak males mendeteksinya.
  scale_pos_weight <- sum(train_label == 0) / sum(train_label == 1)

  params <- list(
    booster = "gbtree", objective = "binary:logistic", eval_metric = "auc",
    eta = 0.1, max_depth = 6, min_child_weight = 1,
    subsample = 0.8, colsample_bytree = 0.8,
    scale_pos_weight = scale_pos_weight
  )

  cv <- xgb.cv(
    params = params, data = dtrain, nrounds = 100, nfold = 5,
    stratified = TRUE, verbose = 0, early_stopping_rounds = 10, maximize = TRUE
  )
  xgb_model <- xgboost(params = params, data = dtrain, nrounds = cv$best_iteration, verbose = 0)

  train_prob_xgb  <- predict(xgb_model, dtrain)
  test_prob_xgb   <- predict(xgb_model, dtest)
  train_class_xgb <- factor(ifelse(train_prob_xgb > 0.5, "Dropout", "Lulus"), levels = c("Lulus", "Dropout"))
  test_class_xgb  <- factor(ifelse(test_prob_xgb  > 0.5, "Dropout", "Lulus"), levels = c("Lulus", "Dropout"))

  hasil$xgboost <- list(
    model = xgb_model,
    train = hitung_metrics(train_class_xgb, train_prob_xgb, train_preproc$target),
    test  = hitung_metrics(test_class_xgb,  test_prob_xgb,  test_preproc$target)
  )

  # bobot kelas yang sama kayak scale_pos_weight XGBoost, dipakai di semua SVM & GLM
  # di bawah -- biar nggak cuma XGBoost yang "diistimewakan" nangani imbalance.
  class_weights <- c(Lulus = 1, Dropout = scale_pos_weight)

  ## --- SVM linear ---
  tuned_linear <- tune.svm(
    x = train_preproc[, setdiff(colnames(train_preproc), "target")], y = train_preproc$target,
    type = "C-classification", kernel = "linear", cost = c(0.1, 1, 10),
    class.weights = class_weights
  )
  svm_linear <- svm(
    target ~ ., data = train_preproc, type = "C-classification", kernel = "linear",
    cost = tuned_linear$best.parameters$cost, probability = TRUE,
    class.weights = class_weights
  )
  pred_train_lin <- svm_prob_dropout(svm_linear, train_preproc)
  pred_test_lin  <- svm_prob_dropout(svm_linear, test_preproc)
  hasil$svm_linear <- list(
    model = svm_linear,
    train = hitung_metrics(pred_train_lin$class, pred_train_lin$prob, train_preproc$target),
    test  = hitung_metrics(pred_test_lin$class,  pred_test_lin$prob,  test_preproc$target)
  )

  ## --- SVM polynomial ---
  tuned_poly <- tune.svm(
    x = train_preproc[, setdiff(colnames(train_preproc), "target")], y = train_preproc$target,
    type = "C-classification", kernel = "polynomial",
    degree = c(2, 3, 4), cost = c(0.1, 1, 10),
    class.weights = class_weights
  )
  svm_poly <- svm(
    target ~ ., data = train_preproc, type = "C-classification", kernel = "polynomial",
    degree = tuned_poly$best.parameters$degree, cost = tuned_poly$best.parameters$cost,
    probability = TRUE, class.weights = class_weights
  )
  pred_train_poly <- svm_prob_dropout(svm_poly, train_preproc)
  pred_test_poly  <- svm_prob_dropout(svm_poly, test_preproc)
  hasil$svm_poly <- list(
    model = svm_poly,
    train = hitung_metrics(pred_train_poly$class, pred_train_poly$prob, train_preproc$target),
    test  = hitung_metrics(pred_test_poly$class,  pred_test_poly$prob,  test_preproc$target)
  )

  ## --- SVM radial ---
  tuned_radial <- tune.svm(
    x = train_preproc[, setdiff(colnames(train_preproc), "target")], y = train_preproc$target,
    type = "C-classification", kernel = "radial",
    cost = c(0.1, 1, 10), gamma = c(0.1, 0.5, 1),
    class.weights = class_weights
  )
  svm_radial <- svm(
    target ~ ., data = train_preproc, type = "C-classification", kernel = "radial",
    cost = tuned_radial$best.parameters$cost, gamma = tuned_radial$best.parameters$gamma,
    probability = TRUE, class.weights = class_weights
  )
  pred_train_rad <- svm_prob_dropout(svm_radial, train_preproc)
  pred_test_rad  <- svm_prob_dropout(svm_radial, test_preproc)
  hasil$svm_radial <- list(
    model = svm_radial,
    train = hitung_metrics(pred_train_rad$class, pred_train_rad$prob, train_preproc$target),
    test  = hitung_metrics(pred_test_rad$class,  pred_test_rad$prob,  test_preproc$target)
  )

  ## --- Logistic Regression ---
  # glm() tidak punya argumen class.weights seperti svm(); setara-nya pakai
  # bobot per observasi (prior weights) dengan rasio yang sama.
  obs_weights <- ifelse(train_preproc$target == "Dropout", scale_pos_weight, 1)
  glm_model <- glm(target ~ ., data = train_preproc, family = binomial, weights = obs_weights)
  train_prob_glm  <- predict(glm_model, newdata = train_preproc, type = "response")
  test_prob_glm   <- predict(glm_model, newdata = test_preproc, type = "response")
  train_class_glm <- factor(ifelse(train_prob_glm > 0.5, "Dropout", "Lulus"), levels = c("Lulus", "Dropout"))
  test_class_glm  <- factor(ifelse(test_prob_glm  > 0.5, "Dropout", "Lulus"), levels = c("Lulus", "Dropout"))

  hasil$glm <- list(
    model = glm_model,
    train = hitung_metrics(train_class_glm, train_prob_glm, train_preproc$target),
    test  = hitung_metrics(test_class_glm,  test_prob_glm,  test_preproc$target)
  )

  list(recipe = prepdata, models = hasil)
}

# ---- 7. Latih kedua pool ----
cat("Melatih model suite untuk pool Dini...\n")
suite_dini <- latih_model_suite(rfe_dini$fitur)

cat("Melatih model suite untuk pool Lanjutan...\n")
suite_lanjutan <- latih_model_suite(rfe_lanjutan$fitur)

# ---- 8. Simpan ke .rds ----
saveRDS(
  list(rfe = rfe_dini$rfe, fitur = rfe_dini$fitur,
       recipe = suite_dini$recipe, models = suite_dini$models),
  "model_results_dini.rds"
)
saveRDS(
  list(rfe = rfe_lanjutan$rfe, fitur = rfe_lanjutan$fitur,
       recipe = suite_lanjutan$recipe, models = suite_lanjutan$models),
  "model_results_lanjutan.rds"
)

# ---- 9. Ringkasan cepat buat sanity check ----
ringkas <- function(suite, nama_pool) {
  do.call(rbind, lapply(names(suite$models), function(nama_model) {
    m <- suite$models[[nama_model]]
    data.frame(
      Pool = nama_pool, Model = nama_model,
      Train_Accuracy = round(m$train$accuracy, 3), Test_Accuracy = round(m$test$accuracy, 3),
      Test_Sensitivity = round(m$test$sensitivity, 3), Test_Specificity = round(m$test$specificity, 3),
      Test_AUC = round(m$test$auc, 3)
    )
  }))
}

cat("\n=== Ringkasan performa ===\n")
print(rbind(ringkas(suite_dini, "Dini"), ringkas(suite_lanjutan, "Lanjutan")))
cat("\nTersimpan: model_results_dini.rds, model_results_lanjutan.rds\n")
