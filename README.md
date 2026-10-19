# RETENSI — Student Dropout Risk Prediction

**[🔗 Try the live dashboard](https://salwaefendii.shinyapps.io/prediksi-dropout-mahasiswa/)**

A dashboard that estimates the risk of a student dropping out mid-program, so support can be considered early — not after the student has already left.

> **Note:** the dashboard interface itself is in Indonesian (built for Indonesian higher-education staff — academic advisors, counselors, scholarship offices). This README is in English; the table in the [Live Demo](#live-demo) section below explains what each Indonesian-labeled page does.

---

## Problem

Nearly **1 in 3 students** in the dataset used here did not stay enrolled through completion. Dropout is costly — for students who lose time and money, and for institutions that lose students who could have been helped if the warning signs had been caught earlier.

The problem is that those signs often show up too late: a collapsing GPA, falling attendance, or months of unpaid tuition. The question this project asks: **can those signs be predicted earlier**, from data the institution already has from the day a student enrolls?

## Approach

The dashboard is built on the **"Predict Students' Dropout and Academic Success"** dataset (Realinho et al., 2021) — 4,424 students, 37 variables covering enrollment data, socioeconomic background (parents' education and occupation), and first- and second-semester academic performance.

Two models are trained for two different points in a student's journey, because the data available differs depending on when the question is asked:

- **Early Detection** (*Deteksi Dini*) — used at enrollment, before any academic record exists yet.
- **Follow-up Detection** (*Deteksi Lanjutan*) — used once a student has completed semesters 1–2, when early academic performance is available.

Five algorithms were compared for each model (Logistic Regression, SVM with 3 kernels, and XGBoost), validated with 5-fold cross-validation, with class-imbalance handling (class weighting, not accuracy alone) and feature selection (RFE) run **after** the train/test split — so the evaluation isn't biased by "peeking" at the test set. **XGBoost** was chosen as the production prediction engine, not because it had the single highest accuracy, but because it was the most balanced at catching genuinely at-risk students (its recall/sensitivity was the most consistent across both models).

Built with **R Shiny** and **bslib** (Bootstrap 5).

## Key Insight

**Financial/administrative status turned out to be a stronger predictor than family background** — at least at the enrollment stage. When the Early Detection model was left to choose its own most important variables (via *Recursive Feature Elimination*), it picked tuition payment status, debt status, scholarship status, program of study, and admission track — **not** parents' education or occupation, even though those variables were available.

The numbers are fairly stark: students who are **late on tuition payments** have an **86.6%** dropout rate, compared to **24.7%** for those who pay on time — more than a 3x difference, and this is data the institution already has from the first semester, long before academic grades become a usable signal.

Once semester 1–2 data becomes available, a single variable — **the number of courses passed in semester 2** — accounts for roughly **40% of the model's entire predictive power**. That tracks: early academic performance is the most direct signal of whether a student is struggling.

## Live Demo

🔗 **https://salwaefendii.shinyapps.io/prediksi-dropout-mahasiswa/**

The dashboard has 5 sections (page names are in Indonesian, as shown in the live app):

| Page (as labeled in-app) | What it does |
|---|---|
| **Overview** | Data summary: total students, historical dropout rate, model accuracy, and a breakdown by program of study |
| **Prediksi Risiko** *(Risk Prediction)* | Enter one (hypothetical) student's data, get a dropout risk estimate + category + suggested follow-up |
| **Eksplorasi Data** *(Data Exploration)* | Compare dropout rates across student characteristics (scholarship, tuition status, age, etc.) |
| **Performa Model** *(Model Performance)* | Comparison of all 5 algorithms, confusion matrix, with a plain-language explanation of how to read it |
| **Visi Pengembangan** *(Development Vision)* | A concept for future development (not implemented) — from prediction to an actual support/outreach system |

## Screenshots

**Overview — data summary and model performance**
![Overview](screenshots/01_overview.jpg)

**Risk Prediction — result for one student**
![Prediksi Risiko](screenshots/02_prediksi_risiko.jpg)

**Model Performance — comparison of 5 algorithms**
![Performa Model](screenshots/03_performa_model.jpg)

---

## For Developers

Project structure, under `dashboard_datmin/`:

```
dashboard_datmin/
├── app.R                      # Shiny dashboard (bslib)
├── train_models.R             # Training pipeline (run manually)
├── data_prep.R                # Data loading + category labels, shared by app.R & train_models.R
├── data_uci_full.csv          # Dataset (see citation below)
├── model_results_dini.rds     # Trained models — Early Detection pool
└── model_results_lanjutan.rds # Trained models — Follow-up Detection pool
```

To run locally: open `dashboard_datmin/app.R` in RStudio and click **Run App**. To retrain the models (e.g. after the data changes), run `train_models.R` manually first.

### Deliberate scope limitations
- The model predicts **Dropout vs. Retained** (*Bertahan* — Enrolled + Graduate combined), not the dataset's original 3-class target — because the goal here is dropout-risk detection, not graduation prediction.
- The dashboard does not store real student records, names, or contact status — see the **Visi Pengembangan** (Development Vision) page for the direction a real support-tracking system would need to take.

## Dataset Citation

> Realinho, V., Machado, J., Baptista, L., & Martins, M.V. (2021). *Predict students' dropout and academic success* (Version 1.0) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.5777340
