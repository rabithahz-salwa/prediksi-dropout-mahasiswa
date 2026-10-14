# data_prep.R
# Loading data + semua label kategori, di-source() bareng oleh train_models.R
# dan app.R supaya definisinya satu tempat, nggak dobel/nggak sinkron.
#
# Sumber data: Realinho, V., Machado, J., Baptista, L., & Martins, M.V. (2021).
# Predict students' dropout and academic success (Version 1.0) [Dataset].
# Zenodo. https://doi.org/10.5281/zenodo.5777340
# (diakses via mirror UCI Machine Learning Repository, Zenodo-nya restricted-access)

library(dplyr)

muat_data_mentah <- function(path = "data_uci_full.csv") {
  df <- read.csv(path, sep = ";", header = TRUE, check.names = FALSE)
  colnames(df) <- trimws(colnames(df))

  nama_baru <- c(
    "Marital status" = "marital_status", "Application mode" = "application_mode",
    "Application order" = "application_order", "Course" = "course",
    "Daytime/evening attendance" = "timeof_attendance", "Previous qualification" = "prev_quali",
    "Previous qualification (grade)" = "prev_quali_grade", "Nacionality" = "nationality",
    "Mother's qualification" = "mother_quali", "Father's qualification" = "father_quali",
    "Mother's occupation" = "mother_occup", "Father's occupation" = "father_occup",
    "Admission grade" = "admiss_grade", "Displaced" = "displaced",
    "Educational special needs" = "special_needs", "Debtor" = "debtor",
    "Tuition fees up to date" = "tuition_ok", "Gender" = "gender",
    "Scholarship holder" = "scholarship", "Age at enrollment" = "age_enroll",
    "International" = "international",
    "Curricular units 1st sem (credited)" = "first_credited",
    "Curricular units 1st sem (enrolled)" = "first_enroll",
    "Curricular units 1st sem (evaluations)" = "first_eval",
    "Curricular units 1st sem (approved)" = "first_approve",
    "Curricular units 1st sem (grade)" = "first_grade",
    "Curricular units 1st sem (without evaluations)" = "first_no_eval",
    "Curricular units 2nd sem (credited)" = "second_credited",
    "Curricular units 2nd sem (enrolled)" = "second_enroll",
    "Curricular units 2nd sem (evaluations)" = "second_eval",
    "Curricular units 2nd sem (approved)" = "second_approve",
    "Curricular units 2nd sem (grade)" = "second_grade",
    "Curricular units 2nd sem (without evaluations)" = "second_no_eval",
    "Unemployment rate" = "unemployment_rate", "Inflation rate" = "inflation_rate",
    "GDP" = "gdp", "Target" = "target"
  )
  colnames(df) <- unname(nama_baru[colnames(df)])
  df
}

# ---- Daftar variabel kategorikal (dikonversi ke factor) ----
kolom_kategorikal <- c(
  "marital_status", "application_mode", "course", "timeof_attendance", "prev_quali",
  "nationality", "mother_quali", "father_quali", "mother_occup", "father_occup",
  "displaced", "special_needs", "debtor", "tuition_ok", "gender", "scholarship",
  "international"
)

siapkan_target <- function(df) {
  df$target <- factor(ifelse(df$target == "Dropout", "Dropout", "Bertahan"),
                       levels = c("Bertahan", "Dropout"))
  df
}

# ---- Label kategori, dari codebook resmi UCI (archive.ics.uci.edu/dataset/697) ----
# Course -- catatan: "9130" itu "Equinculture" (peternakan/studi kuda), BUKAN
# "Persamaan Budaya" seperti yang tertulis di dashboard versi sebelumnya --
# itu salah terjemahan lama yang baru ketahuan & dibenarkan di sini.
course_labels <- c(
  "33" = "Teknologi Produksi Biofuel", "171" = "Animasi dan Desain Multimedia",
  "8014" = "Pelayanan Sosial (kelas malam)", "9003" = "Agronomi",
  "9070" = "Desain Komunikasi", "9085" = "Keperawatan Hewan",
  "9119" = "Teknik Informatika", "9130" = "Peternakan Kuda (Equinculture)",
  "9147" = "Manajemen", "9238" = "Pelayanan Sosial",
  "9254" = "Pariwisata", "9500" = "Keperawatan",
  "9556" = "Kesehatan Gigi dan Mulut", "9670" = "Manajemen Periklanan dan Pemasaran",
  "9773" = "Jurnalistik dan Komunikasi", "9853" = "Pendidikan Dasar",
  "9991" = "Manajemen (kelas malam)"
)

prev_quali_labels <- c(
  "1" = "Pendidikan Menengah (SMA/sederajat)", "2" = "Pendidikan Tinggi - Sarjana (Bachelor)",
  "3" = "Pendidikan Tinggi - Degree", "4" = "Pendidikan Tinggi - Magister",
  "5" = "Pendidikan Tinggi - Doktor", "6" = "Sedang Menempuh Pendidikan Tinggi",
  "9" = "Kelas 12 - Tidak Tamat", "10" = "Kelas 11 - Tidak Tamat",
  "12" = "Kelas 11 (Lainnya)", "14" = "Kelas 10",
  "15" = "Kelas 10 - Tidak Tamat", "19" = "Pendidikan Dasar Siklus 3 (setara Kelas 9-11)",
  "38" = "Pendidikan Dasar Siklus 2 (setara Kelas 6-8)", "39" = "Kursus Spesialisasi Teknologi",
  "40" = "Pendidikan Tinggi - Degree Siklus 1", "42" = "Kursus Teknik Tinggi Profesional",
  "43" = "Pendidikan Tinggi - Magister Siklus 2"
)

# Dipakai bareng utk mother_quali & father_quali (skema kode sama, kodenya
# superset dari prev_quali_labels + beberapa kategori tambahan).
parent_quali_labels <- c(prev_quali_labels,
  "11" = "Kelas 7 (sistem lama)", "13" = "Sekolah Menengah Komplementer Tahun ke-2",
  "18" = "Kursus Niaga Umum", "20" = "Sekolah Menengah Komplementer",
  "22" = "Kursus Teknis-Profesional", "25" = "Sekolah Menengah Komplementer - Tidak Tamat",
  "26" = "Kelas 7", "27" = "Siklus 2 Sekolah Menengah Umum",
  "29" = "Kelas 9 - Tidak Tamat", "30" = "Kelas 8",
  "31" = "Kursus Umum Administrasi dan Niaga", "33" = "Akuntansi dan Administrasi Tambahan",
  "34" = "Tidak Diketahui", "35" = "Tidak Bisa Membaca/Menulis",
  "36" = "Bisa Membaca, Belum Tamat Kelas 4", "37" = "Pendidikan Dasar Siklus 1 (setara Kelas 4-5)",
  "41" = "Kursus Studi Tinggi Khusus", "44" = "Pendidikan Tinggi - Doktor Siklus 3"
)

marital_status_labels <- c(
  "1" = "Belum Menikah", "2" = "Menikah", "3" = "Janda/Duda",
  "4" = "Cerai", "5" = "Hidup Bersama (tanpa menikah)", "6" = "Berpisah Secara Hukum"
)

application_mode_labels <- c(
  "1" = "Fase 1 - Jalur Umum", "2" = "Ordonansi No. 612/93",
  "5" = "Fase 1 - Jalur Khusus (Azores)", "7" = "Pemegang Kursus Tinggi Lain",
  "10" = "Ordonansi No. 854-B/99", "15" = "Mahasiswa Internasional (S1)",
  "16" = "Fase 1 - Jalur Khusus (Madeira)", "17" = "Fase 2 - Jalur Umum",
  "18" = "Fase 3 - Jalur Umum", "26" = "Ordonansi No. 533-A/99 (Rencana Berbeda)",
  "27" = "Ordonansi No. 533-A/99 (Institusi Lain)", "39" = "Usia di atas 23 Tahun",
  "42" = "Transfer", "43" = "Pindah Jurusan",
  "44" = "Pemegang Diploma Spesialisasi Teknologi", "51" = "Pindah Institusi/Jurusan",
  "53" = "Pemegang Diploma Siklus Pendek", "57" = "Pindah Institusi/Jurusan (Internasional)"
)

nationality_labels <- c(
  "1" = "Portugis", "2" = "Jerman", "6" = "Spanyol", "11" = "Italia",
  "13" = "Belanda", "14" = "Inggris", "17" = "Lituania", "21" = "Angola",
  "22" = "Tanjung Verde", "24" = "Guinea", "25" = "Mozambik", "26" = "Sao Tome",
  "32" = "Turki", "41" = "Brasil", "62" = "Rumania", "100" = "Moldova",
  "101" = "Meksiko", "103" = "Ukraina", "105" = "Rusia", "108" = "Kuba", "109" = "Kolombia"
)

ya_tidak_labels <- c("0" = "Tidak", "1" = "Ya")
gender_labels <- c("0" = "Perempuan", "1" = "Laki-laki")
waktu_labels <- c("0" = "Malam", "1" = "Siang")

# Dipakai bareng utk mother_occup & father_occup (skema klasifikasi
# pekerjaan sama, kode father_occup supersetnya mother_occup).
occup_labels <- c(
  "0" = "Pelajar/Mahasiswa",
  "1" = "Pejabat Legislatif/Eksekutif, Direktur, Manajer Eksekutif",
  "2" = "Spesialis Aktivitas Intelektual dan Ilmiah",
  "3" = "Teknisi dan Profesi Tingkat Menengah",
  "4" = "Staf Administratif",
  "5" = "Pekerja Jasa Personal, Keamanan, dan Penjual",
  "6" = "Petani dan Pekerja Terampil Pertanian/Perikanan/Kehutanan",
  "7" = "Pekerja Terampil Industri, Konstruksi, dan Pengrajin",
  "8" = "Operator Mesin dan Perakitan",
  "9" = "Pekerja Tidak Terampil",
  "10" = "Profesi Angkatan Bersenjata",
  "90" = "Situasi Lainnya",
  "99" = "(Kosong/Tidak Diisi)",
  "101" = "Perwira Angkatan Bersenjata", "102" = "Sersan Angkatan Bersenjata",
  "103" = "Personel Angkatan Bersenjata Lainnya",
  "112" = "Direktur Jasa Administratif dan Komersial",
  "114" = "Direktur Hotel, Katering, Perdagangan, dan Jasa Lain",
  "121" = "Spesialis Sains Fisika, Matematika, dan Teknik",
  "122" = "Profesional Kesehatan", "123" = "Guru/Pengajar",
  "124" = "Spesialis Keuangan, Akuntansi, Administrasi, Hubungan Publik/Komersial",
  "125" = "Spesialis Teknologi Informasi dan Komunikasi (TIK)",
  "131" = "Teknisi Sains dan Teknik Tingkat Menengah",
  "132" = "Teknisi dan Profesional Kesehatan Tingkat Menengah",
  "134" = "Teknisi Tingkat Menengah Bidang Hukum, Sosial, Olahraga, dan Budaya",
  "135" = "Teknisi Teknologi Informasi dan Komunikasi",
  "141" = "Staf Kantor, Sekretaris, dan Operator Pengolah Data",
  "143" = "Operator Data, Akuntansi, Statistik, Keuangan, dan Pencatatan",
  "144" = "Staf Pendukung Administratif Lainnya",
  "151" = "Pekerja Jasa Personal", "152" = "Penjual",
  "153" = "Pekerja Perawatan Personal",
  "154" = "Personel Jasa Perlindungan dan Keamanan",
  "161" = "Petani Berorientasi Pasar dan Pekerja Terampil Peternakan",
  "163" = "Petani/Peternak/Nelayan Subsisten",
  "171" = "Pekerja Konstruksi Terampil (non-elektrisi)",
  "172" = "Pekerja Terampil Metalurgi dan Pengerjaan Logam",
  "173" = "Pekerja Terampil Percetakan, Instrumen Presisi, Perhiasan, Kerajinan",
  "174" = "Pekerja Terampil Listrik dan Elektronik",
  "175" = "Pekerja Pengolahan Makanan, Kayu, Pakaian, dan Industri/Kerajinan Lain",
  "181" = "Operator Mesin dan Pabrik Tetap", "182" = "Pekerja Perakitan",
  "183" = "Pengemudi Kendaraan dan Operator Alat Berat",
  "191" = "Pekerja Kebersihan",
  "192" = "Pekerja Tidak Terampil Pertanian/Peternakan/Perikanan/Kehutanan",
  "193" = "Pekerja Tidak Terampil Industri Ekstraktif, Konstruksi, Manufaktur, Transportasi",
  "194" = "Asisten Persiapan Makanan",
  "195" = "Pedagang Kaki Lima (non-makanan) dan Penyedia Jasa Jalanan"
)
