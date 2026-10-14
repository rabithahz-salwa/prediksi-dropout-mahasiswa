#
# RETENSI - Dashboard Prediksi Risiko Drop Out Mahasiswa
#
# Dataset: Realinho, V., Machado, J., Baptista, L., & Martins, M.V. (2021).
# Predict students' dropout and academic success (Version 1.0) [Dataset].
# Zenodo. https://doi.org/10.5281/zenodo.5777340
#
# Model dilatih di train_models.R (dijalankan manual), hasilnya di
# model_results_dini.rds / model_results_lanjutan.rds -- app ini cuma load.
#

library(shiny)
library(bslib)
library(recipes)
library(xgboost)
library(dplyr)

source("data_prep.R")

# ---- Data & model ----
df <- muat_data_mentah()
df <- df %>% mutate(across(all_of(kolom_kategorikal), as.factor))
df <- siapkan_target(df)

model_dini     <- readRDS("model_results_dini.rds")
model_lanjutan <- readRDS("model_results_lanjutan.rds")
model_pools    <- list(Dini = model_dini, Lanjutan = model_lanjutan)

total_mahasiswa       <- nrow(df)
historis_dropout      <- mean(df$target == "Dropout")
akurasi_model_dipakai <- model_lanjutan$models$xgboost$test$accuracy

# Breakdown per prodi -- program studi dengan n < 20 dikecualikan (sampel
# terlalu kecil buat jadi sorotan).
per_prodi <- df %>%
  mutate(nama_prodi = unname(course_labels[as.character(course)])) %>%
  group_by(nama_prodi) %>%
  summarise(n = n(), dropout_rate = mean(target == "Dropout"), .groups = "drop") %>%
  filter(n >= 20) %>%
  arrange(desc(dropout_rate))
per_prodi_overview <- head(per_prodi, 8)

nama_model <- c(glm = "Regresi Logistik", svm_linear = "SVM Linear", svm_poly = "SVM Polynomial",
                 svm_radial = "SVM Radial", xgboost = "XGBoost")

# ---- Fitur mana yang ditonjolkan di form Prediksi vs disembunyikan di
#      "Variabel Lainnya" (dengan nilai default) -- dipakai utk pool yang
#      fitur-nya banyak (Lanjutan, 34 fitur). Diurutkan dari kontribusi
#      (XGBoost gain) tertinggi, dihitung langsung dari model, bukan
#      di-hardcode -- supaya otomatis ikut kalau model dilatih ulang.
urutkan_fitur_by_importance <- function(model_suite) {
  imp <- xgb.importance(model = model_suite$models$xgboost$model)
  fitur <- model_suite$fitur
  imp$fitur_asli <- sapply(imp$Feature, function(nm) {
    cocok <- fitur[sapply(fitur, function(f) startsWith(nm, f))]
    if (length(cocok) == 0) return(NA_character_)
    cocok[which.max(nchar(cocok))]
  })
  ringkas <- imp %>% filter(!is.na(fitur_asli)) %>%
    group_by(fitur_asli) %>% summarise(gain = sum(Gain), .groups = "drop") %>%
    arrange(desc(gain))
  c(ringkas$fitur_asli, setdiff(fitur, ringkas$fitur_asli))
}

urutan_fitur_pool <- lapply(model_pools, urutkan_fitur_by_importance)
N_UTAMA <- 12

nilai_default <- function(kolom) {
  x <- df[[kolom]]
  if (is.numeric(x)) {
    round(median(x, na.rm = TRUE), 1)
  } else {
    names(sort(table(x), decreasing = TRUE))[1]
  }
}

# ---- Helper UI kecil ----
bar_html <- function(pct, rata_rata = NULL, warna = "var(--abu-teks)") {
  tick <- if (!is.null(rata_rata)) {
    sprintf('<div class="bar-tick" style="left:%.1f%%"></div>', rata_rata * 100)
  } else ""
  sprintf(
    '<div class="bar-track">%s<div class="bar-fill" style="width:%.1f%%;background:%s"></div></div>',
    tick, pct * 100, warna
  )
}

risk_kategori <- function(prob) {
  if (prob >= 0.6) list(label = "Tinggi", kelas = "tinggi", glyph = "▲")
  else if (prob >= 0.3) list(label = "Sedang", kelas = "sedang", glyph = "■")
  else list(label = "Rendah", kelas = "rendah", glyph = "●")
}

risk_badge <- function(prob) {
  k <- risk_kategori(prob)
  tags$span(class = paste("risk-badge", k$kelas), HTML(paste0(k$glyph, " ", toupper(k$label))))
}

saran_tindak_lanjut <- function(prob) {
  if (prob >= 0.6) "Disarankan pendampingan akademik tambahan dalam waktu dekat."
  else if (prob >= 0.3) "Disarankan pemantauan berkala dan komunikasi dengan dosen wali."
  else "Risiko rendah berdasarkan data saat ini; tidak ada tindakan khusus yang disarankan."
}

# Field kategorikal -> label map; field numerik nggak perlu disebut di sini.
field_label_map <- list(
  course = course_labels, application_mode = application_mode_labels,
  prev_quali = prev_quali_labels, mother_quali = parent_quali_labels,
  father_quali = parent_quali_labels, mother_occup = occup_labels,
  father_occup = occup_labels, nationality = nationality_labels,
  tuition_ok = ya_tidak_labels, scholarship = ya_tidak_labels, debtor = ya_tidak_labels,
  displaced = ya_tidak_labels, international = ya_tidak_labels,
  gender = gender_labels, timeof_attendance = waktu_labels
)

field_judul <- c(
  course = "Program studi", application_mode = "Mode pendaftaran",
  application_order = "Urutan pilihan (0 = pilihan pertama)",
  prev_quali = "Pendidikan sebelumnya", prev_quali_grade = "Nilai pendidikan sebelumnya (0-200)",
  admiss_grade = "Nilai penerimaan (0-200)", mother_quali = "Pendidikan Ibu",
  father_quali = "Pendidikan Ayah", mother_occup = "Pekerjaan Ibu", father_occup = "Pekerjaan Ayah",
  nationality = "Kewarganegaraan", tuition_ok = "UKT lunas tepat waktu", debtor = "Status utang/tunggakan",
  scholarship = "Penerima beasiswa", displaced = "Status perantau (displaced)",
  international = "Mahasiswa internasional", gender = "Jenis kelamin",
  timeof_attendance = "Waktu kuliah", age_enroll = "Usia saat mendaftar",
  first_enroll = "Semester 1 - Mata kuliah terdaftar", first_approve = "Semester 1 - Mata kuliah lulus",
  first_credited = "Semester 1 - Mata kuliah diakui (credit transfer)",
  first_eval = "Semester 1 - Jumlah evaluasi", first_grade = "Semester 1 - Rata-rata nilai (0-20)",
  first_no_eval = "Semester 1 - Mata kuliah tanpa evaluasi",
  second_enroll = "Semester 2 - Mata kuliah terdaftar", second_approve = "Semester 2 - Mata kuliah lulus",
  second_credited = "Semester 2 - Mata kuliah diakui (credit transfer)",
  second_eval = "Semester 2 - Jumlah evaluasi", second_grade = "Semester 2 - Rata-rata nilai (0-20)",
  second_no_eval = "Semester 2 - Mata kuliah tanpa evaluasi",
  unemployment_rate = "Tingkat pengangguran (%) saat masuk", inflation_rate = "Tingkat inflasi (%) saat masuk",
  gdp = "Pertumbuhan GDP (%) saat masuk"
)

buat_input_prediksi <- function(fitur) {
  id <- paste0("pred_", fitur)
  default <- nilai_default(fitur)
  judul <- field_judul[[fitur]]
  if (fitur %in% names(field_label_map)) {
    label_map <- field_label_map[[fitur]]
    choices <- if (fitur %in% c("tuition_ok", "scholarship", "debtor", "displaced", "international", "gender", "timeof_attendance")) {
      setNames(names(label_map), label_map)
    } else {
      setNames(names(label_map), paste0(names(label_map), ": ", label_map))
    }
    if (fitur %in% c("tuition_ok", "scholarship", "debtor", "displaced", "international", "gender", "timeof_attendance")) {
      radioButtons(id, judul, choices = choices, selected = as.character(default), inline = TRUE)
    } else {
      selectInput(id, judul, choices = choices, selected = as.character(default))
    }
  } else {
    numericInput(id, judul, value = as.numeric(default))
  }
}

# ---- CSS ----
css_tambahan <- tags$head(
  tags$link(rel = "stylesheet", href = paste0(
    "https://fonts.googleapis.com/css2?",
    "family=Archivo:ital,wght@0,400;0,600;0,700;0,800&",
    "family=Source+Serif+4:ital@0;1&display=swap"
  )),
  tags$style(HTML("
    :root {
      --kapur: #FAFAF9; --abu-garis: #E4E5E7; --abu-teks: #6B6E75;
      --tinta: #17181B; --biru-tautan: #2B5FAD;
      --risiko-tinggi: #C0392B; --risiko-sedang: #B7791F; --risiko-rendah: #2E7D46;
    }
    body { font-family: 'Archivo', -apple-system, sans-serif; background: var(--kapur); color: var(--tinta); }
    .narasi { font-family: 'Source Serif 4', Georgia, serif; }
    .navbar { background: var(--kapur) !important; border-bottom: 1px solid var(--abu-garis); }
    .navbar-brand { display: flex !important; align-items: center !important; padding-top: 14px; padding-bottom: 14px; margin-right: 28px !important; padding-right: 28px; border-right: 1px solid var(--abu-garis); }
    .brand-block { display: flex; flex-direction: column; justify-content: center; }
    .brand-main { font-size: 1.2rem; font-weight: 800; letter-spacing: 0.01em; line-height: 1.15; }
    .brand-sub { font-size: 0.76rem; color: var(--abu-teks); font-weight: 600; line-height: 1.2; margin-top: 2px; }
    .navbar .navbar-nav { align-items: center; }
    .eyebrow { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--abu-teks); margin-bottom: 6px; }
    h1.headline { font-size: 2rem; font-weight: 800; letter-spacing: -0.01em; max-width: 26ch; }
    .stat-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 24px; margin: 28px 0; }
    .stat-block { border-top: 2px solid var(--tinta); padding-top: 10px; }
    .stat-label { font-size: 0.78rem; color: var(--abu-teks); font-weight: 600; margin-bottom: 4px; }
    .stat-value { font-size: 2.3rem; font-weight: 800; font-variant-numeric: tabular-nums; letter-spacing: -0.02em; }
    .stat-caption { font-size: 0.8rem; color: var(--abu-teks); }
    .stat-value.accent-risk { color: var(--risiko-tinggi); }
    .section-title { font-size: 1.3rem; font-weight: 800; margin-bottom: 4px; }
    .section-sub { color: var(--abu-teks); font-size: 0.85rem; margin-bottom: 14px; }
    table.listing { width: 100%; border-collapse: collapse; }
    table.listing th { text-align: left; font-size: 0.72rem; color: var(--abu-teks); font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; border-bottom: 1px solid var(--abu-garis); padding: 6px 10px 6px 0; }
    table.listing td { padding: 10px 10px 10px 0; border-bottom: 1px solid var(--abu-garis); vertical-align: middle; }
    .prodi-row { cursor: pointer; }
    .prodi-row:hover td { background: #F1F1EF; }
    .bar-track { position: relative; background: var(--abu-garis); height: 8px; border-radius: 2px; width: 100%; min-width: 120px; }
    .bar-fill { position: absolute; left: 0; top: 0; height: 100%; border-radius: 2px; }
    .bar-tick { position: absolute; top: -3px; width: 1px; height: 14px; background: var(--tinta); z-index: 2; }
    .risk-badge { font-size: 0.74rem; font-weight: 700; letter-spacing: 0.02em; }
    .risk-badge.tinggi { color: var(--risiko-tinggi); }
    .risk-badge.sedang { color: var(--risiko-sedang); }
    .risk-badge.rendah { color: var(--risiko-rendah); }
    .model-option-group { display: flex; flex-direction: column; gap: 12px; margin-bottom: 4px; }
    .model-card { border: 1px solid var(--abu-garis); border-radius: 4px; padding: 14px 16px; cursor: pointer; }
    .model-card.aktif { border: 2px solid var(--tinta); padding: 13px 15px; }
    .model-card .head { display: flex; align-items: center; gap: 10px; }
    .model-card .dot { width: 16px; height: 16px; min-width: 16px; border-radius: 50%; border: 1.5px solid var(--abu-teks); position: relative; }
    .model-card.aktif .dot { border-color: var(--tinta); }
    .model-card.aktif .dot::after { content: ''; position: absolute; inset: 3px; border-radius: 50%; background: var(--tinta); }
    .model-card .judul { font-weight: 700; }
    .model-card .ket { font-size: 0.82rem; color: var(--abu-teks); margin-top: 6px; margin-left: 26px; }
    .form-group-label { font-size: 0.74rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--tinta); border-top: 1px solid var(--abu-garis); padding-top: 8px; margin: 18px 0 10px; }
    .form-group-label:first-of-type { border-top: none; margin-top: 0; }
    .result-card { border: 1px solid var(--abu-garis); border-radius: 4px; padding: 20px; }
    .result-eyebrow { font-size: 0.74rem; color: var(--abu-teks); font-weight: 700; text-transform: uppercase; }
    .result-number { font-size: 3.2rem; font-weight: 800; color: var(--risiko-tinggi); line-height: 1.1; font-variant-numeric: tabular-nums; }
    .result-number.sedang { color: var(--risiko-sedang); }
    .result-number.rendah { color: var(--risiko-rendah); }
    .risk-scale { display: flex; height: 8px; border-radius: 2px; overflow: hidden; margin: 10px 0 4px; position: relative; }
    .risk-scale .seg { height: 100%; }
    .risk-scale .tick { position: absolute; top: -3px; width: 2px; height: 14px; background: var(--tinta); }
    .scale-labels { display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--abu-teks); margin-bottom: 14px; }
    .disclaimer { font-size: 0.78rem; color: var(--abu-teks); margin-top: 14px; }
    .concept-wrap { border: 2px dashed var(--abu-garis); border-radius: 6px; padding: 28px 30px; }
    .concept-badge { display: inline-block; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; border: 1px dashed var(--abu-teks); color: var(--abu-teks); padding: 4px 10px; border-radius: 3px; margin-bottom: 14px; }
    .perf-table th, .perf-table td { padding: 8px 14px 8px 0; }
    .perf-table .baris-terbaik { font-weight: 700; }
    .confusion-grid { display: grid; grid-template-columns: 140px 1fr 1fr; gap: 1px; background: var(--abu-garis); border: 1px solid var(--abu-garis); }
    .confusion-grid > div { background: var(--kapur); padding: 12px; }
    .confusion-grid .head { font-size: 0.72rem; color: var(--abu-teks); font-weight: 700; text-transform: uppercase; }
    .confusion-grid .sel-benar { background: var(--tinta); color: #fff; }
    .confusion-grid .angka { font-size: 1.6rem; font-weight: 800; }
    .confusion-grid .ket { font-size: 0.74rem; opacity: 0.8; }
    details.lainnya { border: 1px solid var(--abu-garis); border-radius: 4px; padding: 12px 16px; margin-top: 16px; }
    details.lainnya summary { cursor: pointer; font-weight: 700; font-size: 0.86rem; }
    details.lainnya .isi { margin-top: 14px; }
  "))
)

# ---- UI ----
ui <- page_navbar(
  id = "main_nav",
  title = tags$div(class = "brand-block",
    tags$div("RETENSI", class = "brand-main"),
    tags$div("Prediksi risiko drop-out mahasiswa", class = "brand-sub")
  ),
  theme = bs_theme(bg = "#FAFAF9", fg = "#17181B", base_font = font_google("Archivo")),
  header = css_tambahan,
  fillable = FALSE,

  nav_panel("Overview", value = "overview",
    div(class = "container-fluid", style = "max-width:1200px; padding:28px 20px;",
      div(class = "eyebrow", "Ringkasan data historis"),
      h1(class = "headline", textOutput("headline_overview", inline = TRUE)),
      p(class = "section-sub", style = "max-width:70ch; margin-top:10px;",
        "Dashboard ini memperkirakan risiko drop-out dari data pendaftaran, latar belakang sosial-ekonomi, dan performa akademik awal, sehingga pendampingan dapat dipertimbangkan sebelum masalah akademik berlanjut."),
      div(class = "stat-strip",
        div(class = "stat-block",
            div(class = "stat-label", "Total Mahasiswa"),
            div(class = "stat-value", format(total_mahasiswa, big.mark = ".", decimal.mark = ",")),
            div(class = "stat-caption", "rekam data pelatihan model")),
        div(class = "stat-block",
            div(class = "stat-label", "Historis Drop-out"),
            div(class = "stat-value accent-risk", paste0(round(historis_dropout * 100, 1), "%")),
            div(class = "stat-caption", "dari seluruh mahasiswa dalam data")),
        div(class = "stat-block",
            div(class = "stat-label", "Akurasi Model Terpilih"),
            div(class = "stat-value", paste0(round(akurasi_model_dipakai * 100, 1), "%")),
            div(class = "stat-caption", "XGBoost · data performa semester 1-2"))
      ),
      fluidRow(
        column(8,
          div(class = "section-title", "Proporsi drop-out per program studi"),
          div(class = "section-sub", "Diurutkan dari proporsi tertinggi. Klik baris untuk membuka Eksplorasi Data dengan filter prodi tersebut."),
          uiOutput("tabel_prodi"),
          p(class = "disclaimer", "Program studi dengan sampel di bawah 20 mahasiswa dikecualikan.")
        ),
        column(4,
          div(class = "section-title", "Bagian lain"),
          tags$ul(style = "list-style:none; padding:0;",
            tags$li(style="margin-bottom:16px;", actionLink("ke_prediksi", strong("Prediksi Risiko ›")), br(),
                    span(class="section-sub", "Perkirakan risiko satu mahasiswa, baik saat pendaftaran maupun setelah semester 1-2.")),
            tags$li(style="margin-bottom:16px;", actionLink("ke_eksplorasi", strong("Eksplorasi Data ›")), br(),
                    span(class="section-sub", "Bandingkan proporsi drop-out menurut karakteristik mahasiswa.")),
            tags$li(actionLink("ke_performa", strong("Performa Model ›")), br(),
                    span(class="section-sub", "Seberapa tepat model, dan di mana ia masih keliru."))
          ),
          p(class = "disclaimer", style="margin-top:20px;",
            "Sumber: Realinho, V., Machado, J., Baptista, L., & Martins, M.V. (2021). Predict students' dropout and academic success [Dataset]. Zenodo. doi.org/10.5281/zenodo.5777340")
        )
      )
    )
  ),

  nav_panel("Prediksi Risiko", value = "prediksi",
    div(class = "container-fluid", style = "max-width:1200px; padding:28px 20px;",
      div(class = "eyebrow", "Prediksi per mahasiswa"),
      h1(class = "headline", "Perkiraan risiko drop-out"),
      p(class = "section-sub", "Isi data satu mahasiswa untuk melihat perkiraan risikonya. Hasil ini alat bantu pertimbangan, bukan keputusan."),
      fluidRow(
        column(6,
          div(class = "form-group-label", "Jenis model"),
          uiOutput("pilihan_pool"),
          uiOutput("pred_form"),
          br(),
          actionButton("predict", "Hitung risiko", class = "btn", style = "background:var(--tinta); color:#fff; border-radius:4px; font-weight:700;"),
          actionLink("reset_pred", "Atur ulang", style = "margin-left:14px;")
        ),
        column(6, uiOutput("prediction_result"))
      )
    )
  ),

  nav_panel("Eksplorasi Data", value = "eksplorasi",
    div(class = "container-fluid", style = "max-width:1200px; padding:28px 20px;",
      div(class = "eyebrow", "Eksplorasi data"),
      h1(class = "headline", "Proporsi drop-out menurut karakteristik mahasiswa"),
      fluidRow(
        column(6, selectInput("filter_prodi", "Filter program studi",
                               choices = c("Semua program studi", per_prodi$nama_prodi))),
        column(6, div(style = "text-align:right; padding-top:28px;",
                       span(style = "font-size:1.6rem; font-weight:800;", textOutput("n_terfilter", inline = TRUE)),
                       span(class = "disclaimer", " mahasiswa"))
        )
      ),
      br(),
      uiOutput("breakdown_eksplorasi"),
      p(class = "disclaimer", "Garis vertikal menunjukkan rata-rata drop-out pada filter aktif.")
    )
  ),

  nav_panel("Performa Model", value = "performa",
    div(class = "container-fluid", style = "max-width:1200px; padding:28px 20px;",
      fluidRow(
        column(8,
          div(class = "eyebrow", "Performa model"),
          h1(class = "headline", "Perbandingan model pada data uji")
        ),
        column(4, div(style = "text-align:right; padding-top:20px;",
          radioButtons("pool_performa", NULL, choices = c("Deteksi Dini" = "Dini", "Deteksi Lanjutan" = "Lanjutan"),
                       selected = "Lanjutan", inline = TRUE)
        ))
      ),
      uiOutput("tabel_performa"),
      br(), br(),
      fluidRow(
        column(6,
          div(class = "section-title", "Matriks kesalahan — XGBoost"),
          uiOutput("confusion_display")
        ),
        column(6,
          div(class = "section-title", "Cara membaca"),
          uiOutput("narasi_confusion")
        )
      )
    )
  ),

  nav_panel("Visi Pengembangan", value = "visi",
    div(class = "container-fluid", style = "max-width:1200px; padding:28px 20px;",
      div(class = "concept-wrap",
        div(class = "concept-badge", "Konsep · Belum Diimplementasikan"),
        h1(class = "headline", "Visi pengembangan: dari prediksi ke pendampingan"),
        p(class = "section-sub", style = "max-width: 70ch; font-size:0.92rem;",
          "Versi saat ini hanya memperkirakan risiko dari data akademik dan sosial-ekonomi anonim. Bila kelak terhubung ke sistem akademik kampus, prediksi dapat menjadi antrian pendampingan yang dikelola dosen wali dan konselor. Halaman ini menggambarkan arah tersebut — belum ada fungsi atau data nyata di baliknya."),
        br(),
        div(class = "section-title", "Contoh antrian pendampingan"),
        tags$table(class = "listing", style = "width:100%; margin-top:10px;",
          tags$thead(tags$tr(tags$th("Mahasiswa"), tags$th("Skor risiko"), tags$th("Status kontak"), tags$th("Penanggung jawab"))),
          tags$tbody(
            tags$tr(tags$td(strong("Mahasiswa A"), br(), span(class="disclaimer","[NIM] · Teknik Informatika")), tags$td("82%", risk_badge(0.82)), tags$td("Belum dihubungi"), tags$td("Dosen wali")),
            tags$tr(tags$td(strong("Mahasiswa B"), br(), span(class="disclaimer","[NIM] · Agronomi")), tags$td("76%", risk_badge(0.76)), tags$td("Terjadwal"), tags$td("Konselor")),
            tags$tr(tags$td(strong("Mahasiswa C"), br(), span(class="disclaimer","[NIM] · Manajemen")), tags$td("64%", risk_badge(0.64)), tags$td("Sudah didampingi"), tags$td("Dosen wali")),
            tags$tr(tags$td(strong("Mahasiswa D"), br(), span(class="disclaimer","[NIM] · Jurnalistik")), tags$td("51%", risk_badge(0.51)), tags$td("Belum dihubungi"), tags$td("Staf beasiswa"))
          )
        ),
        br(),
        fluidRow(
          column(6,
            div(class = "section-title", "Data tambahan yang dibutuhkan"),
            tags$ul(
              tags$li("Identitas mahasiswa (NIM, nama) dari sistem akademik, dengan akses terbatas"),
              tags$li("Status kontak dan jadwal pendampingan"),
              tags$li("Catatan pendampingan dari dosen wali atau konselor"),
              tags$li("Penanggung jawab tiap mahasiswa")
            )
          ),
          column(6,
            div(class = "section-title", "Pertimbangan sebelum diterapkan"),
            tags$ul(
              tags$li("Akses berbasis peran: hanya pendamping yang ditugaskan dapat melihat identitas."),
              tags$li("Skor risiko tidak ditampilkan ke mahasiswa sebagai label."),
              tags$li("Prediksi memicu percakapan, bukan sanksi atau keputusan akademik."),
              tags$li("Model dievaluasi ulang tiap angkatan untuk memeriksa bias antar kelompok, termasuk variabel sosial-ekonomi.")
            )
          )
        )
      )
    )
  )
)

# ---- Server ----
server <- function(input, output, session) {

  # ===== Overview =====
  output$headline_overview <- renderText({
    sprintf("Hampir %s dari 10 mahasiswa dalam data historis tidak bertahan sampai selesai.",
            round(historis_dropout * 10))
  })

  output$tabel_prodi <- renderUI({
    rata2 <- historis_dropout
    baris <- lapply(seq_len(nrow(per_prodi_overview)), function(i) {
      p <- per_prodi_overview[i, ]
      tags$tr(class = "prodi-row",
        onclick = sprintf("Shiny.setInputValue('prodi_clicked', '%s', {priority: 'event'})", p$nama_prodi),
        tags$td(p$nama_prodi),
        tags$td(style = "width:45%;", HTML(bar_html(p$dropout_rate, rata2, "var(--tinta)"))),
        tags$td(style = "text-align:right; font-weight:700;", paste0(round(p$dropout_rate * 100, 1), "%")),
        tags$td(style = "width:16px; color:var(--abu-teks);", "›")
      )
    })
    tags$table(class = "listing",
      tags$thead(tags$tr(tags$th("Program Studi"), tags$th(""), tags$th(style="text-align:right;","Drop-out"), tags$th(""))),
      tags$tbody(baris)
    )
  })

  observeEvent(input$prodi_clicked, {
    updateSelectInput(session, "filter_prodi",
                       choices = c("Semua program studi", per_prodi$nama_prodi),
                       selected = input$prodi_clicked)
    nav_select("main_nav", selected = "eksplorasi", session = session)
  })
  observeEvent(input$ke_prediksi, nav_select("main_nav", selected = "prediksi", session = session))
  observeEvent(input$ke_eksplorasi, nav_select("main_nav", selected = "eksplorasi", session = session))
  observeEvent(input$ke_performa, nav_select("main_nav", selected = "performa", session = session))

  # ===== Prediksi Risiko =====
  pool_dipilih <- reactiveVal("Lanjutan")
  observeEvent(input$pool_pick, pool_dipilih(input$pool_pick))

  output$pilihan_pool <- renderUI({
    pilihan <- list(
      list(nilai = "Dini", judul = "Deteksi Dini",
           ket = "Dipakai saat pendaftaran, ketika belum ada data akademik di perguruan tinggi."),
      list(nilai = "Lanjutan", judul = "Deteksi Lanjutan",
           ket = "Dipakai setelah mahasiswa melewati semester 1-2, dengan data performa akademik.")
    )
    div(class = "model-option-group",
      lapply(pilihan, function(p) {
        aktif <- identical(pool_dipilih(), p$nilai)
        div(class = paste("model-card", if (aktif) "aktif"),
            onclick = sprintf("Shiny.setInputValue('pool_pick', '%s', {priority: 'event'})", p$nilai),
            div(class = "head", div(class = "dot"), div(class = "judul", p$judul)),
            div(class = "ket", p$ket)
        )
      })
    )
  })

  output$pred_form <- renderUI({
    fitur_urut <- urutan_fitur_pool[[pool_dipilih()]]
    utama   <- head(fitur_urut, N_UTAMA)
    lainnya <- setdiff(fitur_urut, utama)

    tagList(
      div(class = "form-group-label", "Variabel utama (paling berpengaruh)"),
      lapply(utama, buat_input_prediksi),
      if (length(lainnya) > 0) {
        tags$details(class = "lainnya",
          tags$summary(sprintf("Variabel lainnya (%d) — opsional, sudah diisi nilai tipikal", length(lainnya))),
          div(class = "isi", lapply(lainnya, buat_input_prediksi))
        )
      }
    )
  })

  hasil_prediksi <- reactiveVal(NULL)

  observeEvent(input$predict, {
    pool  <- pool_dipilih()
    fitur <- model_pools[[pool]]$fitur

    nilai <- lapply(fitur, function(f) {
      v <- input[[paste0("pred_", f)]]
      if (is.null(v)) v <- nilai_default(f)
      if (f %in% kolom_kategorikal) {
        factor(as.character(v), levels = levels(df[[f]]))
      } else {
        as.numeric(v)
      }
    })
    names(nilai) <- fitur
    new_data <- as.data.frame(nilai, stringsAsFactors = FALSE)

    new_data_preproc <- bake(model_pools[[pool]]$recipe, new_data = new_data)
    prob_dropout <- as.numeric(predict(model_pools[[pool]]$models$xgboost$model, as.matrix(new_data_preproc)))

    hasil_prediksi(list(prob = prob_dropout, pool = pool))
  })

  observeEvent(input$reset_pred, hasil_prediksi(NULL))

  output$prediction_result <- renderUI({
    hp <- hasil_prediksi()
    if (is.null(hp)) {
      return(div(class = "result-card",
                  div(class = "result-eyebrow", "Hasil prediksi"),
                  p(style = "color:var(--abu-teks); margin-top:10px;", "Isi data di sebelah kiri, lalu klik “Hitung risiko”.")))
    }
    k <- risk_kategori(hp$prob)
    pos <- hp$prob * 100
    tagList(
      div(class = "result-card",
        div(class = "result-eyebrow", sprintf("Hasil prediksi · Deteksi %s · XGBoost", hp$pool)),
        div(class = paste("result-number", k$kelas), paste0(round(pos, 1), "%")),
        div(risk_badge(hp$prob)),
        div(class = "risk-scale",
          div(class = "seg", style = "width:30%; background:#DCEFE1;"),
          div(class = "seg", style = "width:30%; background:#F3E3C4;"),
          div(class = "seg", style = "width:40%; background:#F3D3CD;"),
          div(class = "tick", style = sprintf("left:%.1f%%;", pos))
        ),
        div(class = "scale-labels", span("Rendah <30%"), span("Sedang 30-60%"), span("Tinggi ≥60%")),
        div(class = "form-group-label", "Tindak lanjut yang disarankan"),
        p(class = "narasi", style = "font-size:1.05rem;", saran_tindak_lanjut(hp$prob)),
        p(class = "disclaimer", "Saran disusun dari aturan per kategori risiko, bukan keputusan otomatis. Gunakan bersama pertimbangan dosen wali atau konselor.")
      )
    )
  })

  # ===== Eksplorasi Data =====
  df_terfilter <- reactive({
    d <- df %>% mutate(nama_prodi = unname(course_labels[as.character(course)]))
    if (input$filter_prodi != "Semua program studi") d <- d %>% filter(nama_prodi == input$filter_prodi)
    d
  })

  output$n_terfilter <- renderText(format(nrow(df_terfilter()), big.mark = ".", decimal.mark = ","))

  buat_breakdown <- function(d, var, label_map = NULL, judul) {
    rata2 <- mean(d$target == "Dropout")
    ringkas <- d %>% group_by(.data[[var]]) %>%
      summarise(n = n(), dropout = mean(target == "Dropout"), .groups = "drop")
    if (!is.null(label_map)) ringkas[[var]] <- unname(label_map[as.character(ringkas[[var]])])
    baris <- lapply(seq_len(nrow(ringkas)), function(i) {
      r <- ringkas[i, ]
      tags$tr(tags$td(as.character(r[[var]])), tags$td(style="text-align:right;", format(r$n, big.mark=".", decimal.mark=",")),
              tags$td(style = "width:40%;", HTML(bar_html(r$dropout, rata2, "var(--tinta)"))),
              tags$td(style = "text-align:right; font-weight:700;", paste0(round(r$dropout * 100, 1), "%")))
    })
    div(style = "margin-bottom:26px;",
      div(class = "section-title", style = "font-size:1.02rem;", judul),
      tags$table(class = "listing",
        tags$thead(tags$tr(tags$th("Kategori"), tags$th(style="text-align:right;","N"), tags$th(""), tags$th(style="text-align:right;","Drop-out"))),
        tags$tbody(baris)
      )
    )
  }

  output$breakdown_eksplorasi <- renderUI({
    d <- df_terfilter()
    tagList(
      fluidRow(
        column(4, buat_breakdown(d, "scholarship", ya_tidak_labels, "Status Beasiswa")),
        column(4, buat_breakdown(d, "tuition_ok", ya_tidak_labels, "UKT Lunas Tepat Waktu")),
        column(4, buat_breakdown(d, "debtor", ya_tidak_labels, "Status Utang/Tunggakan"))
      ),
      fluidRow(
        column(4, buat_breakdown(d, "gender", gender_labels, "Jenis Kelamin")),
        column(4, {
          d2 <- d %>% mutate(bin = cut(age_enroll, c(0, 20, 25, 150), labels = c("≤20 tahun", "21-25 tahun", ">25 tahun")))
          buat_breakdown(d2, "bin", NULL, "Usia saat Mendaftar")
        }),
        column(4, {
          d2 <- d %>% mutate(bin = cut(second_approve, c(-1, 0, 3, 6, 100), labels = c("0", "1-3", "4-6", "7+")))
          buat_breakdown(d2, "bin", NULL, "Mata Kuliah Lulus Semester 2")
        })
      )
    )
  })

  # ===== Performa Model =====
  output$tabel_performa <- renderUI({
    suite <- model_pools[[input$pool_performa]]$models
    urutan <- c("glm", "svm_linear", "svm_poly", "svm_radial", "xgboost")
    dipakai <- "xgboost"
    baris <- lapply(urutan, function(n) {
      m <- suite[[n]]
      tags$tr(class = if (n == dipakai) "baris-terbaik" else "",
        tags$td(nama_model[[n]], if (n == dipakai) span(class="disclaimer", " DIPAKAI DI PREDIKSI RISIKO") else NULL),
        tags$td(style="text-align:right;", paste0(round(m$test$accuracy * 100, 1), "%")),
        tags$td(style="text-align:right;", paste0(round(m$test$sensitivity * 100, 1), "%")),
        tags$td(style="text-align:right;", paste0(round(m$test$specificity * 100, 1), "%")),
        tags$td(style="text-align:right;", sprintf("%.2f", m$test$auc))
      )
    })
    tags$table(class = "listing perf-table",
      tags$thead(tags$tr(tags$th("Algoritma"), tags$th(style="text-align:right;","Akurasi"),
                          tags$th(style="text-align:right;","Sensitivity"), tags$th(style="text-align:right;","Specificity"),
                          tags$th(style="text-align:right;","AUC"))),
      tags$tbody(baris)
    )
  })

  cm_aktif <- reactive({
    model_pools[[input$pool_performa]]$models$xgboost$test$conf_matrix$table
  })

  output$confusion_display <- renderUI({
    cm <- cm_aktif()
    n_total <- sum(cm)
    tp <- cm["Dropout", "Dropout"]; fn <- cm["Bertahan", "Dropout"]
    fp <- cm["Dropout", "Bertahan"]; tn <- cm["Bertahan", "Bertahan"]
    tagList(
      p(class = "disclaimer", sprintf("%s mahasiswa pada data uji", format(n_total, big.mark=".", decimal.mark=","))),
      div(class = "confusion-grid",
        div(), div(class="head", "Diprediksi drop-out"), div(class="head", "Diprediksi bertahan"),
        div(class="head", "Aktual drop-out"),
        div(class = "sel-benar", div(class="angka", tp), div(class="ket", "terdeteksi")),
        div(div(class="angka", fn), div(class="ket", "terlewat")),
        div(class="head", "Aktual bertahan"),
        div(div(class="angka", fp), div(class="ket", "peringatan berlebih")),
        div(div(class="angka", tn), div(class="ket", "tepat bertahan"))
      )
    )
  })

  output$narasi_confusion <- renderUI({
    cm <- cm_aktif()
    tp <- cm["Dropout", "Dropout"]; fn <- cm["Bertahan", "Dropout"]
    aktual_dropout <- tp + fn
    tagList(
      p(class = "narasi", sprintf(
        "Dari %d mahasiswa yang benar-benar drop-out, model menemukan %d di antaranya. Inilah yang diukur oleh sensitivity (%s).",
        aktual_dropout, tp, paste0(round(tp/aktual_dropout*100,1), "%")
      )),
      p(class = "narasi", sprintf(
        "%d mahasiswa terlewat, yaitu diprediksi bertahan padahal drop-out. Kesalahan jenis ini paling perlu diperhatikan, karena mahasiswa tersebut tidak akan muncul di daftar prioritas pendampingan.", fn
      )),
      p(class = "disclaimer", sprintf(
        "Angka akurasi %s%% (XGBoost, Deteksi %s) dan matriks ini seluruhnya dihitung dari data uji asli.",
        round(model_pools[[input$pool_performa]]$models$xgboost$test$accuracy * 100, 1), input$pool_performa
      ))
    )
  })
}

shinyApp(ui, server)
