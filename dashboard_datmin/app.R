#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Model dilatih di train_models.R (dijalankan manual, hasilnya di
# model_results_dini.rds / model_results_lanjutan.rds) -- app ini cuma load,
# tidak training ulang setiap dibuka.
#

library(shiny)
library(shinydashboard)
library(shinydashboardPlus)
library(dashboardthemes)
library(caret)
library(e1071)
library(pROC)
library(recipes)
library(dplyr)
library(xgboost)
library(DT)
library(ggplot2)
library(reshape2)

# ---- Load data (untuk tab Database & Visualization) ----
df <- read.csv("data.csv", sep = ";", header = TRUE)

categorical_vars <- c("course", "timeof_attendance", "prev_quali", "gender", "scholarship", "target")
numeric_vars <- setdiff(colnames(df), categorical_vars)

df <- df %>%
  mutate(across(all_of(setdiff(categorical_vars, "target")), as.factor))

# Label dibuat terbaca & konsisten dengan train_models.R ("Dropout" = positive class)
df$target <- factor(ifelse(df$target == 0, "Dropout", "Lulus"), levels = c("Lulus", "Dropout"))

selected_columns <- df[, c("prev_quali_grade", "admiss_grade", "age_enroll", "first_enroll",
                           "first_approve", "second_enroll", "second_approve")]

# ---- Load model hasil training (lihat train_models.R) ----
model_dini     <- readRDS("model_results_dini.rds")
model_lanjutan <- readRDS("model_results_lanjutan.rds")
model_pools    <- list(Dini = model_dini, Lanjutan = model_lanjutan)

# ---- Pilihan & builder untuk form Prediksi (dinamis ikut fitur tiap pool) ----
course_choices <- c(
  "33: Teknologi Produksi Biofuel" = "33", "171: Animasi dan Desain Multimedia" = "171",
  "8014: Layanan Sosial (kehadiran malam)" = "8014", "9003: Agronomi" = "9003",
  "9070: Desain Komunikasi" = "9070", "9085: Keperawatan Hewan" = "9085",
  "9119: Teknik Informatika" = "9119", "9130: Persamaan Budaya" = "9130",
  "9147: Manajemen" = "9147", "9238: Layanan Sosial" = "9238",
  "9254: Pariwisata" = "9254", "9500: Keperawatan" = "9500",
  "9556: Kebersihan Mulut" = "9556", "9670: Manajemen Periklanan dan Pemasaran" = "9670",
  "9773: Jurnalisme dan Komunikasi" = "9773", "9853: Pendidikan Dasar" = "9853",
  "9991: Manajemen (kehadiran malam)" = "9991"
)

buat_input_prediksi <- function(fitur) {
  switch(fitur,
    course = selectInput("pred_course", "Program Studi:", choices = course_choices),
    timeof_attendance = selectInput("pred_timeof_attendance", "Waktu Kuliah:",
                                     choices = c("Siang" = "0", "Malam" = "1")),
    prev_quali = selectInput("pred_prev_quali", "Kode Pendidikan Sebelumnya:",
                              choices = setNames(levels(df$prev_quali), levels(df$prev_quali))),
    prev_quali_grade = numericInput("pred_prev_quali_grade", "Nilai Pendidikan Sebelumnya (0-200):",
                                     value = 130, min = 0, max = 200),
    admiss_grade = numericInput("pred_admiss_grade", "Nilai Penerimaan (0-200):",
                                 value = 130, min = 0, max = 200),
    gender = selectInput("pred_gender", "Jenis Kelamin:",
                          choices = c("Laki-laki" = "0", "Perempuan" = "1")),
    scholarship = selectInput("pred_scholarship", "Penerima Beasiswa:",
                               choices = c("Tidak" = "0", "Ya" = "1")),
    age_enroll = numericInput("pred_age_enroll", "Usia saat Mendaftar:", value = 20, min = 16, max = 70),
    first_enroll = numericInput("pred_first_enroll", "Unit Kurikuler Terdaftar - Semester 1:",
                                 value = 6, min = 0, max = 30),
    first_approve = numericInput("pred_first_approve", "Unit Kurikuler Lulus - Semester 1:",
                                  value = 6, min = 0, max = 30),
    second_enroll = numericInput("pred_second_enroll", "Unit Kurikuler Terdaftar - Semester 2:",
                                  value = 6, min = 0, max = 30),
    second_approve = numericInput("pred_second_approve", "Unit Kurikuler Lulus - Semester 2:",
                                   value = 6, min = 0, max = 30)
  )
}

ui <- dashboardPage(
  dashboardHeader(title = "Dropout Prediction App"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Beranda", tabName = "beranda", icon = icon("info-circle")),
      menuItem("Home", tabName = "home", icon = icon("home")),
      menuItem("Prediksi", tabName = "prediksi", icon = icon("magnifying-glass-chart")),
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
                                 strong("target: "), "Variabel target yang menunjukkan apakah mahasiswa akan dropout."
                               )
                             )
                         )
                )
              )
      ),
      tabItem(tabName = "prediksi",
              h2(strong("Prediksi Risiko Drop Out")),
              p("Pilih jenis model sesuai informasi yang kamu punya tentang mahasiswa, isi data di bawah, lalu klik Prediksi."),
              fluidRow(
                column(width = 4,
                       box(width = 12, solidHeader = TRUE, title = "Input Data Mahasiswa",
                           selectInput("pred_pool", "Jenis Model:",
                                       choices = c("Deteksi Dini (saat pendaftaran)" = "Dini",
                                                   "Deteksi Lanjutan (setelah semester 1-2)" = "Lanjutan")),
                           uiOutput("pred_form"),
                           actionButton("predict", "Prediksi", icon = icon("magnifying-glass"), class = "btn-primary")
                       )
                ),
                column(width = 8,
                       box(width = 12, solidHeader = TRUE, title = "Hasil Prediksi",
                           uiOutput("prediction_result")
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
      tabItem(tabName = "analyze_tab",
              h2(strong("Analyze")),
              tabsetPanel(
                tabPanel(strong("Confusion Matrix"),
                         selectInput("pool_conf", "Jenis Model:",
                                     choices = c("Deteksi Dini" = "Dini", "Deteksi Lanjutan" = "Lanjutan")),
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
                         selectInput("pool_roc", "Jenis Model:",
                                     choices = c("Deteksi Dini" = "Dini", "Deteksi Lanjutan" = "Lanjutan")),
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
            title = "Performance - Deteksi Dini vs Deteksi Lanjutan",
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

  # ---- Analyze ----
  selected_model_conf <- reactive({
    suite <- model_pools[[input$pool_conf]]$models
    if (input$model_conf == "svm") {
      suite[[paste("svm", input$svm_conf_kernel, sep = "_")]]
    } else {
      suite[[input$model_conf]]
    }
  })

  selected_model_roc <- reactive({
    suite <- model_pools[[input$pool_roc]]$models
    if (input$model_roc == "svm") {
      suite[[paste("svm", input$svm_roc_kernel, sep = "_")]]
    } else {
      suite[[input$model_roc]]
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
    selected_model_conf()$train$conf_matrix
  })

  output$test_conf_matrix <- renderPrint({
    selected_model_conf()$test$conf_matrix
  })

  output$train_roc_plot <- renderPlot({
    roc_train <- selected_model_roc()$train$roc
    plot(roc_train, main = paste("Training ROC Curve -", input$model_roc), col = "blue", lwd = 2)
    text(0.6, 0.2, paste("AUC =", round(as.numeric(auc(roc_train)), 2)), col = "blue")
  })

  output$test_roc_plot <- renderPlot({
    roc_test <- selected_model_roc()$test$roc
    plot(roc_test, main = paste("Testing ROC Curve -", input$model_roc), col = "red", lwd = 2)
    text(0.6, 0.2, paste("AUC =", round(as.numeric(auc(roc_test)), 2)), col = "red")
  })

  # ---- Performance ----
  output$performance_table <- renderTable({
    nama_model <- c(xgboost = "XGBoost", svm_linear = "SVM Linear", svm_poly = "SVM Polynomial",
                     svm_radial = "SVM Radial", glm = "Logistic Regression")

    baris <- function(label_pool, suite) {
      do.call(rbind, lapply(names(suite), function(nama) {
        m <- suite[[nama]]
        data.frame(
          Pool = label_pool,
          Model = nama_model[[nama]],
          Train_Accuracy = round(m$train$accuracy, 3),
          Test_Accuracy = round(m$test$accuracy, 3),
          Test_Sensitivity = round(m$test$sensitivity, 3),
          Test_Specificity = round(m$test$specificity, 3),
          Test_AUC = round(m$test$auc, 3)
        )
      }))
    }

    rbind(
      baris("Deteksi Dini", model_dini$models),
      baris("Deteksi Lanjutan", model_lanjutan$models)
    )
  })

  # ---- Prediksi ----
  output$pred_form <- renderUI({
    fitur <- model_pools[[input$pred_pool]]$fitur
    tagList(lapply(fitur, buat_input_prediksi))
  })

  hasil_prediksi <- reactiveVal(NULL)

  observeEvent(input$predict, {
    pool  <- input$pred_pool
    fitur <- model_pools[[pool]]$fitur

    nilai <- lapply(fitur, function(f) {
      v <- input[[paste0("pred_", f)]]
      if (f %in% c("course", "timeof_attendance", "prev_quali", "gender", "scholarship")) {
        factor(v, levels = levels(df[[f]]))
      } else {
        as.numeric(v)
      }
    })
    names(nilai) <- fitur
    new_data <- as.data.frame(nilai, stringsAsFactors = FALSE)

    new_data_preproc <- bake(model_pools[[pool]]$recipe, new_data = new_data)
    prob_dropout <- predict(model_pools[[pool]]$models$xgboost$model, as.matrix(new_data_preproc))

    kategori <- if (prob_dropout >= 0.7) "Tinggi" else if (prob_dropout >= 0.4) "Sedang" else "Rendah"
    warna <- if (prob_dropout >= 0.7) "#d9534f" else if (prob_dropout >= 0.4) "#f0ad4e" else "#5cb85c"

    hasil_prediksi(list(prob = prob_dropout, kategori = kategori, warna = warna, pool = pool))
  })

  output$prediction_result <- renderUI({
    hp <- hasil_prediksi()
    if (is.null(hp)) {
      return(p("Isi data di sebelah kiri, lalu klik Prediksi.", style = "color: #888;"))
    }
    tagList(
      h3(paste0("Risiko Drop Out: ", round(hp$prob * 100, 1), "%")),
      div(strong(paste("Kategori Risiko:", hp$kategori)),
          style = paste0("color: white; background-color: ", hp$warna,
                         "; padding: 10px 16px; border-radius: 6px; display: inline-block;")),
      p(style = "margin-top:15px; color:#555;",
        paste0("Model yang digunakan: ",
               if (hp$pool == "Dini") "Deteksi Dini (variabel saat pendaftaran)."
               else "Deteksi Lanjutan (performa semester 1-2)."))
    )
  })
}

shinyApp(ui, server)
