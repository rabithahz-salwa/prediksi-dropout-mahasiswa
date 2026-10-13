#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(shinydashboard)
library(shinydashboardPlus)
library(dashboardthemes)
library(caret)
library(fontawesome)
library(rpart)
library(rpart.plot)
library(leaflet)
library(e1071)
library(neuralnet)
library(pROC)
library(rsample)
library(recipes)
library(dplyr)
library(xgboost)
library(DT)
library(ggplot2)
library(reshape2)
library(DataExplorer)
library(randomForest)

# Load Data
df <- read.csv("data.csv", sep = ";", header = TRUE)

categorical_vars <- c("course", "timeof_attendance", "prev_quali", "gender", "scholarship", "target")
numeric_vars <- setdiff(colnames(df), categorical_vars)

# Data Preparation and Feature Selection
df <- df %>%
  mutate(across(all_of(categorical_vars), as.factor))

# Recursive Feature Elimination
control <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
results <- rfe(df[,1:12], df$target, sizes = c(1:12), rfeControl = control)
results
data <- df[, c("second_approve", "first_approve", "second_enroll", "age_enroll", "first_enroll", "target")]
selected_columns = df[,c("prev_quali_grade", "admiss_grade", "age_enroll", "first_enroll", "first_approve", "second_enroll", "second_approve")]
selected_columns

# Data Splitting
set.seed(123)
split <- initial_split(data, prop = 0.8, strata = "target")
df_train <- training(split)
df_test <- testing(split)
folds <- vfold_cv(df_train, v = 5)

# Data Preprocessing
recipedata <- recipe(target ~ ., data = df_train) %>%
  step_normalize(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors(), one_hot = TRUE)
prepdata <- prep(recipedata)
df_train_preproc <- bake(prepdata, new_data = NULL)
df_test_preproc <- bake(prepdata, new_data = df_test)

# Train models and store results
train_models <- function() {
  results <- list()
  
  # XGBoost
  train_matrix <- model.matrix(target ~ . - 1, data = df_train_preproc)
  test_matrix <- model.matrix(target ~ . - 1, data = df_test_preproc)
  train_label <- as.numeric(df_train_preproc$target)-1
  test_label <- as.numeric(df_test_preproc$target)-1
  
  dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
  dtest <- xgb.DMatrix(data = test_matrix, label = test_label)
  
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = 0.1,
    max_depth = 6,
    min_child_weight = 1,
    subsample = 0.8,
    colsample_bytree = 0.8
  )
  
  cv <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = 100,
    nfold = 5,
    stratified = TRUE,
    verbose = 0,
    early_stopping_rounds = 10,
    maximize = TRUE
  )
  
  best_nrounds <- cv$best_iteration
  xgb_model <- xgboost(
    params = params,
    data = dtrain,
    nrounds = best_nrounds,
    verbose = 0
  )
  
  train_pred_prob <- predict(xgb_model, dtrain)
  test_pred_prob <- predict(xgb_model, dtest)
  train_pred <- ifelse(train_pred_prob > 0.5, 1, 0)
  test_pred <- ifelse(test_pred_prob > 0.5, 1, 0)
  
  train_conf_matrix <- confusionMatrix(factor(train_pred), factor(train_label))
  test_conf_matrix <- confusionMatrix(factor(test_pred), factor(test_label))
  
  roc_train <- roc(train_label, train_pred_prob)
  roc_test <- roc(test_label, test_pred_prob)
  
  results$xgboost <- list(
    model = xgb_model,
    train_conf_matrix = train_conf_matrix,
    test_conf_matrix = test_conf_matrix,
    roc_train = roc_train,
    roc_test = roc_test
  )
  
  # SVM with linear kernel
  tune_result_linear <- tune.svm(x = df_train_preproc[, -ncol(df_train_preproc)], 
                                 y = df_train_preproc$target,
                                 type = "C-classification",
                                 kernel = "linear",
                                 cost = c(0.1, 1, 10))
  
  best_cost_linear <- tune_result_linear$best.parameters$cost
  svm_model_linear <- svm(target ~ ., data = df_train_preproc, 
                          type = "C-classification",
                          kernel = "linear",
                          cost = best_cost_linear)
  
  pred_train_linear <- predict(svm_model_linear, newdata = df_train_preproc)
  pred_test_linear <- predict(svm_model_linear, newdata = df_test_preproc)
  
  cm_train_linear <- confusionMatrix(pred_train_linear, df_train_preproc$target)
  cm_test_linear <- confusionMatrix(pred_test_linear, df_test_preproc$target)
  
  roc_train_linear <- roc(df_train_preproc$target, as.numeric(pred_train_linear))
  roc_test_linear <- roc(df_test_preproc$target, as.numeric(pred_test_linear))
  
  results$svm_linear <- list(
    model = svm_model_linear,
    train_conf_matrix = cm_train_linear,
    test_conf_matrix = cm_test_linear,
    roc_train = roc_train_linear,
    roc_test = roc_test_linear
  )
  
  # SVM with polynomial kernel
  tune_result_poly <- tune.svm(x = df_train_preproc[, -ncol(df_train_preproc)], 
                               y = df_train_preproc$target,
                               type = "C-classification",
                               kernel = "polynomial",
                               degree = c(2, 3, 4),
                               cost = c(0.1, 1, 10))
  
  best_cost_poly <- tune_result_poly$best.parameters$cost
  best_degree_poly <- tune_result_poly$best.parameters$degree
  svm_model_poly <- svm(target ~ ., data = df_train_preproc, 
                        type = "C-classification",
                        kernel = "polynomial",
                        degree = best_degree_poly,
                        cost = best_cost_poly)
  
  pred_train_poly <- predict(svm_model_poly, newdata = df_train_preproc)
  pred_test_poly <- predict(svm_model_poly, newdata = df_test_preproc)
  
  cm_train_poly <- confusionMatrix(pred_train_poly, df_train_preproc$target)
  cm_test_poly <- confusionMatrix(pred_test_poly, df_test_preproc$target)
  
  roc_train_poly <- roc(df_train_preproc$target, as.numeric(pred_train_poly))
  roc_test_poly <- roc(df_test_preproc$target, as.numeric(pred_test_poly))
  
  results$svm_poly <- list(
    model = svm_model_poly,
    train_conf_matrix = cm_train_poly,
    test_conf_matrix = cm_test_poly,
    roc_train = roc_train_poly,
    roc_test = roc_test_poly
  )
  
  # SVM with radial kernel
  tune_result_radial <- tune.svm(x = df_train_preproc[, -ncol(df_train_preproc)], 
                                 y = df_train_preproc$target,
                                 type = "C-classification",
                                 kernel = "radial",
                                 cost = c(0.1, 1, 10),
                                 gamma = c(0.1, 0.5, 1))
  
  best_cost_radial <- tune_result_radial$best.parameters$cost
  best_gamma_radial <- tune_result_radial$best.parameters$gamma
  svm_model_radial <- svm(target ~ ., data = df_train_preproc, 
                          type = "C-classification",
                          kernel = "radial",
                          cost = best_cost_radial,
                          gamma = best_gamma_radial)
  
  pred_train_radial <- predict(svm_model_radial, newdata = df_train_preproc)
  pred_test_radial <- predict(svm_model_radial, newdata = df_test_preproc)
  
  cm_train_radial <- confusionMatrix(pred_train_radial, df_train_preproc$target)
  cm_test_radial <- confusionMatrix(pred_test_radial, df_test_preproc$target)
  
  roc_train_radial <- roc(df_train_preproc$target, as.numeric(pred_train_radial))
  roc_test_radial <- roc(df_test_preproc$target, as.numeric(pred_test_radial))
  
  results$svm_radial <- list(
    model = svm_model_radial,
    train_conf_matrix = cm_train_radial,
    test_conf_matrix = cm_test_radial,
    roc_train = roc_train_radial,
    roc_test = roc_test_radial
  )
  
  # Logistic Regression
  glm_model <- glm(target ~ ., data = df_train_preproc, family = binomial)
  
  pred_train_glm <- predict(glm_model, newdata = df_train_preproc, type = "response")
  predicted_train_glm <- ifelse(pred_train_glm > 0.5, 1, 0)
  cm_train_glm <- confusionMatrix(as.factor(predicted_train_glm), df_train_preproc$target)
  roc_train_glm <- roc(df_train_preproc$target, pred_train_glm)
  
  pred_test_glm <- predict(glm_model, newdata = df_test_preproc, type = "response")
  predicted_test_glm <- ifelse(pred_test_glm > 0.5, 1, 0)
  cm_test_glm <- confusionMatrix(as.factor(predicted_test_glm), df_test_preproc$target)
  roc_test_glm <- roc(df_test_preproc$target, pred_test_glm)
  
  results$glm <- list(
    model = glm_model,
    train_conf_matrix = cm_train_glm,
    test_conf_matrix = cm_test_glm,
    roc_train = roc_train_glm,
    roc_test = roc_test_glm
  )
  
  return(results)
}

model_results <- train_models()
model_results

# Header dashboard
header <- dashboardHeader(title = "Dashboard of Dropout and Academic Success", titleWidth = 200)

ui <- dashboardPage(
  dashboardHeader(title = "Dropout Prediction App"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Beranda", tabName = "beranda", icon = icon("info-circle")),
      menuItem("Home", tabName = "home", icon = icon("home")),
      menuItem("Database", tabName = "table", icon = icon("database")),
      menuItem("Visualization", tabName = "visualization", icon = icon("chart-bar")),
      menuItem("Analyze", tabName = "analyze_tab", icon = icon("chart-line")),
      menuItem("Performance", tabName = "performance", icon = icon("chart-bar"))
    )
  ),
  dashboardBody(
    shinyDashboardThemes("blue_gradient"),
    tabItems(
      tabItem(tabName = "beranda",
              titlePanel(
                carousel(width = 12,
                         id = "mycarousel",
                         carouselItem(
                           tags$img(src = "beranda.png", width = "100%", height = "1080px")
                         ),
                         carouselItem(
                           tags$img(src = "infografis.jpg", width = "50%", height = "1419px")
                         ),
                         carouselItem(
                           tags$img(src = "foto.png", width = "100%", height = "1080px")
                         )
                )
                
              )
      ),
      tabItem(tabName = "home",
              div(h1(strong("ACADEMIC SUCCESS")),style="text-align: center;"),
              div(("Data Prediksi Drop Out Perguruan Tinggi"),style="text-align: center;"),br(),
              tags$img(src = "do.jpg", width = "100%", height = "720px"),
              p(" Image courtesy: Adobe Stock", style = "color: #808080; font-size: 12px"),
              br(),
              tabsetPanel(
                tabPanel(strong("Data"),
                         box(title = strong("Deskripsi Data"), width = 12, solidHeader = TRUE,
                             p("Dataset yang dibuat dari institusi pendidikan tinggi yang terkait dengan mahasiswa yang terdaftar di berbagai program studi sarjana. Dataset ini mencakup informasi yang diketahui pada saat pendaftaran mahasiswa dan kinerja akademik mahasiswa pada akhir semester pertama dan kedua. Data tersebut digunakan untuk membangun model klasifikasi untuk memprediksi tingkat putus sekolah dan keberhasilan akademik siswa.")
                         )
                ),
                tabPanel(strong("Data Variables"),
                         box(title = strong("Attribute Information"), width = 12, solidHeader = TRUE,
                             p("Dataset ini memiliki 13 kolom (variabel) dengan penjelasan sebagai berikut:"),
                             tags$ol(
                               tags$li(
                                 strong("course: "), "Program studi yang diambil:",
                                 tags$ul(
                                   tags$li("33: Teknologi Produksi Biofuel"),
                                   tags$li("171: Animasi dan Desain Multimedia"),
                                   tags$li("8014: Layanan Sosial (kehadiran malam)"),
                                   tags$li("9003: Agronomi"),
                                   tags$li("9070: Desain Komunikasi"),
                                   tags$li("9085: Keperawatan Hewan"),
                                   tags$li("9119: Teknik Informatika"),
                                   tags$li("9130: Persamaan budaya"),
                                   tags$li("9147: Manajemen"),
                                   tags$li("9238: Layanan Sosial"),
                                   tags$li("9254: Pariwisata"),
                                   tags$li("9500: Keperawatan"),
                                   tags$li("9556: Kebersihan Mulut"),
                                   tags$li("9670: Manajemen Periklanan dan Pemasaran"),
                                   tags$li("9773: Jurnalisme dan Komunikasi"),
                                   tags$li("9853: Pendidikan Dasar"),
                                   tags$li("9991: Manajemen (kehadiran malam)")
                                 )
                               ),
                               tags$li(
                                 strong("timeof_attendance: "), "Jenis kedatangan (0 = Siang, 1 = Malam)."
                               ),
                               tags$li(
                                 strong("prev_quali: "), "Pendidikan sebelumnya."
                               ),
                               tags$li(
                                 strong("prev_quali_grade: "), "Nilai dari pendidikan sebelumnya (antara 0 dan 200)."
                               ),
                               tags$li(
                                 strong("admiss_grade: "), "Nilai penerimaan (antara 0 dan 200)."
                               ),
                               tags$li(
                                 strong("gender: "), "Jenis Kelamin (0 = Laki-Laki, 1 = Perempuan)."
                               ),
                               tags$li(
                                 strong("scholarship: "), "Apakah seorang pemegang beasiswa? (0 = Tidak, 1 = Ya)."
                               ),
                               tags$li(
                                 strong("age_enroll: "), "Usia mahasiswa saat pendaftaran."
                               ),
                               tags$li(
                                 strong("first_enroll: "), "Jumlah unit kurikuler yang terdaftar pada semester pertama."
                               ),
                               tags$li(
                                 strong("first_approve: "), "Jumlah unit kurikuler yang lulus pada semester pertama."
                               ),
                               tags$li(
                                 strong("second_enroll: "), "Jumlah unit kurikuler yang terdaftar pada semester kedua."
                               ),
                               tags$li(
                                 strong("second_approve: "), "Jumlah unit kurikuler yang lulus pada semester kedua."
                               ),
                               tags$li(
                                 strong("target: "), "Variabel target yang menunjukkan apakah mahasiswa akan dropout (0 = Ya, 1 = Tidak)."
                               )
                             )
                         )
                )
              )
      ),
      tabItem(tabName = "table",  
              h2(strong("Database")),
              tabsetPanel(
                tabPanel(strong("Data Table"), 
                         DTOutput("data_table")),
                tabPanel(strong("Summary Statistics"), 
                         verbatimTextOutput("summary_stats"))
              )
      ),
      tabItem(tabName = "visualization",
              fluidRow(
                column(width = 12, h2("Visualization"))
              ),
              fluidRow(
                column(width = 3,
                       selectInput("plot_type", "Pilih Jenis Plot:", 
                                   choices = c("Scatter Plot", "Bar Chart", "Heatmap", "Box Plot", "Histogram", "Violin Plot")),
                       conditionalPanel(
                         condition = "input.plot_type == 'Scatter Plot'",
                         selectInput("x_var", "Pilih Variabel X:", choices = colnames(df)),
                         selectInput("y_var", "Pilih Variabel Y:", choices = colnames(df))
                       ),
                       conditionalPanel(
                         condition = "input.plot_type == 'Box Plot' || input.plot_type == 'Violin Plot'",
                         selectInput("x_var", "Pilih Variabel X:", choices = colnames(df)),
                         selectInput("y_var", "Pilih Variabel Y:", choices = numeric_vars)
                       ),
                       conditionalPanel(
                         condition = "input.plot_type == 'Bar Chart' || input.plot_type == 'Histogram'",
                         selectInput("x_var", "Pilih Variabel:", choices = colnames(df))
                       ),
                       conditionalPanel(
                         condition = "input.plot_type == 'Heatmap'",
                         checkboxGroupInput("heatmap_vars", "Pilih Variabel:", choices = colnames(selected_columns), selected = colnames(selected_columns))
                       )
                ),
                column(width = 9,
                       plotOutput("plot", height = "600px", width = "800px")
                )
              )
      ),
      tabItem(tabName = "analyze_tab",  # Ganti tabName di sini
              h2(strong("Analyze")),
              tabsetPanel(
                tabPanel(strong("Confusion Matrix"),
                         selectInput("model_conf", "Select Model:", 
                                     choices = c("XGBoost" = "xgboost", 
                                                 "SVM" = "svm", 
                                                 "Logistic Regression" = "glm")),
                         uiOutput("svm_conf_ui"),
                         fluidRow(
                           column(6,
                                  h3("Training Data"),
                                  verbatimTextOutput("train_conf_matrix")),
                           column(6,
                                  h3("Testing Data"),
                                  verbatimTextOutput("test_conf_matrix"))
                         )
                ),
                tabPanel(strong("ROC Curve/AUC"),
                         selectInput("model_roc", "Select Model:", 
                                     choices = c("XGBoost" = "xgboost", 
                                                 "SVM" = "svm", 
                                                 "Logistic Regression" = "glm")),
                         uiOutput("svm_roc_ui"),
                         fluidRow(
                           column(6,
                                  h3("Training Data"),
                                  plotOutput("train_roc_plot")),
                           column(6,
                                  h3("Testing Data"),
                                  plotOutput("test_roc_plot"))
                         )
                )
              )
      ),
      tabItem(
        tabName = "performance",
        fluidRow(
          box(
            title = "Performance",
            width = 12,
            status = "primary",
            solidHeader = TRUE,
            collapsible = TRUE,
            tableOutput("performance_table")
          )
        )
      )
    )
  )
)

server <- function(input, output) {
  output$data_table <- renderDT({
    datatable(df)
  })
  
  output$summary_stats <- renderPrint({
    summary(df)
  })
  
  output$plot <- renderPlot({
    plot <- NULL
    
    if (input$plot_type == "Scatter Plot") {
      plot <- ggplot(df, aes_string(x = input$x_var, y = input$y_var)) +
        geom_point() +
        theme_minimal()
      
    } else if (input$plot_type == "Bar Chart") {
      counts <- df %>%
        count(!!sym(input$x_var)) %>%
        arrange(desc(n))
      df[[input$x_var]] <- factor(df[[input$x_var]], levels = counts[[input$x_var]])
      plot <- ggplot(df, aes_string(x = input$x_var)) +
        geom_bar() +
        theme_minimal()
      
    } else if (input$plot_type == "Heatmap") {
      selected_vars <- input$heatmap_vars
      corr_df <- cor(df[selected_vars], use = "complete.obs")
      melted_corr_df <- melt(corr_df)
      
      plot <- ggplot(melted_corr_df, aes(Var1, Var2, fill = value)) +
        geom_tile() +
        geom_text(aes(label = round(value, 2)), color = "black", size = 4) +
        scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                             midpoint = 0, limit = c(-1, 1), space = "Lab", 
                             name="Correlation") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1),
              axis.text.y = element_text(size = 12),
              axis.title = element_text(size = 14),
              plot.title = element_text(size = 16, hjust = 0.5)) +
        coord_fixed()
      
    } else if (input$plot_type == "Box Plot") {
      if (input$x_var %in% categorical_vars) {
        plot <- ggplot(df, aes_string(x = paste0("as.factor(", input$x_var, ")"), y = input$y_var)) +
          geom_boxplot() +
          theme_minimal()
      } else {
        plot <- ggplot(df, aes_string(x = input$x_var, y = input$y_var)) +
          geom_boxplot() +
          theme_minimal()
      }
      
    } else if (input$plot_type == "Histogram") {
      plot <- ggplot(df, aes_string(x = input$x_var)) +
        geom_histogram(binwidth = 1, color="black", fill="#96C9F4") +
        theme_minimal()
      
    } else if (input$plot_type == "Violin Plot") {
      if (input$x_var %in% categorical_vars) {
        plot <- ggplot(df, aes_string(x = paste0("as.factor(", input$x_var, ")"), y = input$y_var)) +
          geom_violin() +
          theme_minimal()
      } else {
        plot <- ggplot(df, aes_string(x = input$x_var, y = input$y_var)) +
          geom_violin() +
          theme_minimal()
      }
    }
    
    print(plot)
  })
  
  selected_model_conf <- reactive({
    if (input$model_conf == "svm") {
      return(model_results[[paste("svm", input$svm_conf_kernel, sep = "_")]])
    } else {
      return(model_results[[input$model_conf]])
    }
  })
  
  selected_model_roc <- reactive({
    if (input$model_roc == "svm") {
      return(model_results[[paste("svm", input$svm_roc_kernel, sep = "_")]])
    } else {
      return(model_results[[input$model_roc]])
    }
  })
  
  output$svm_conf_ui <- renderUI({
    if (input$model_conf == "svm") {
      selectInput("svm_conf_kernel", "Select SVM Kernel:", 
                  choices = c("Linear Kernel" = "linear", 
                              "Polynomial Kernel" = "poly", 
                              "RBF Kernel" = "radial"))
    }
  })
  
  output$svm_roc_ui <- renderUI({
    if (input$model_roc == "svm") {
      selectInput("svm_roc_kernel", "Select SVM Kernel:", 
                  choices = c("Linear Kernel" = "linear", 
                              "Polynomial Kernel" = "poly", 
                              "RBF Kernel" = "radial"))
    }
  })
  
  output$train_conf_matrix <- renderPrint({
    selected_model_conf()$train_conf_matrix
  })
  
  output$test_conf_matrix <- renderPrint({
    selected_model_conf()$test_conf_matrix
  })
  
  output$train_roc_plot <- renderPlot({
    roc_train <- selected_model_roc()$roc_train
    plot(roc_train, main = paste("Training ROC Curve -", input$model_roc), col = "blue", lwd = 2)
    text(0.6, 0.2, paste("AUC =", round(roc_train$auc, 2)), col = "blue")
  })
  
  output$test_roc_plot <- renderPlot({
    roc_test <- selected_model_roc()$roc_test
    plot(roc_test, main = paste("Testing ROC Curve -", input$model_roc), col = "red", lwd = 2)
    text(0.6, 0.2, paste("AUC =", round(roc_test$auc, 2)), col = "red")
  })
  
  output$performance_table <- renderTable({
    data.frame(
      Model = c("XGBoost", "SVM Linear", "SVM Polynomial", "SVM RBF", "Logistic Regression"),
      Train_Accuracy = c(model_results$xgboost$train_conf_matrix$overall["Accuracy"], 
                         model_results$svm_linear$train_conf_matrix$overall["Accuracy"],
                         model_results$svm_poly$train_conf_matrix$overall["Accuracy"],
                         model_results$svm_radial$train_conf_matrix$overall["Accuracy"],
                         model_results$glm$train_conf_matrix$overall["Accuracy"]),
      Test_Accuracy = c(model_results$xgboost$test_conf_matrix$overall["Accuracy"], 
                        model_results$svm_linear$test_conf_matrix$overall["Accuracy"],
                        model_results$svm_poly$test_conf_matrix$overall["Accuracy"],
                        model_results$svm_radial$test_conf_matrix$overall["Accuracy"],
                        model_results$glm$test_conf_matrix$overall["Accuracy"])
    )
  })
  
  observeEvent(input$predict, {
    new_data <- data.frame(
      course = factor(input$sex, levels = c(33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991)),
      timeof_attendance = factor(input$timeof_attendance, levels = c(0, 1)),
      prev_quali = factor(input$prev_quali, levels = c(1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43)),
      prev_quali_grade = input$prev_quali_grade,
      admiss_grade = input$admiss_grade,
      gender = factor(input$gender, levels = c(0, 1)),
      scholarship = factor(input$scholarship, levels = c(0, 1)),
      age_enroll = input$age_enroll,
      first_enroll = input$first_enroll,
      first_approve = input$first_approve,
      second_enroll = input$second_enroll,
      second_approve = input$second_approve
    )
    
    new_data_preproc <- bake(prepdata, new_data = new_data)
    
    pred <- predict(results$xgboost$model, as.matrix(new_data_preproc))
    pred_class <- ifelse(pred > 0.5, 1, 0)
    
    output$prediction_result <- renderText({
      paste("Predicted class:", pred_class)
    })
  })
}

shinyApp(ui, server)