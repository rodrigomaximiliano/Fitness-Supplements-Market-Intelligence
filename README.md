# 📊 Fitness Supplements Market Analysis – United States (2021–2026)

## 🎯 Executive Summary

This project analyzes the U.S. fitness supplements market using:

- **Google Trends** (5-year historical data)
- **Google Trends** (last 12 months deep dive)
- **Meta Ads** competitive analysis
- **Keyword research**
- **Social sentiment analysis** (X/Twitter)

### 🎯 Objectives

The objective is to identify:

- **Structural growth categories**
- **Seasonal demand patterns**
- **Investment risks**
- **Underserved segments**
- **Strategic opportunities for 2026**

The analysis combines quantitative time-series evaluation with qualitative advertising and positioning insights to deliver actionable business recommendations.

---

## 📁 Project Structure

```
Suplementos-fitness-analisis/
│
├── data/
│   ├── multiTimeline-lastfiveyears.csv
│   └── multiTimelinelast12monts.csv
│
├── google-sheets/
│   ├── Supplement short analysis - Recommendations.csv
│   ├── Supplement short analysis - Matrix.csv
│   ├── Supplement short analysis - Keyword Research.csv
│   ├── Supplement short analysis - Ads Analysis .csv
│   ├── Supplement short analysis - Sentiment Analysis.csv
│   └── Supplement short analysis - Trends.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── analysis/
│   ├── __init__.py
│   ├── data_loader.py
│   └── metrics.py
│
├── figures/
│   ├── 01_five_year_trend.png
│   ├── 02_cagr.png
│   └── ...
│
├── presentation/
│   └── fitness_market_research.pptx
│
├── requirements.txt
└── README.md
```

---

## 🔎 Data Sources

### 1️⃣ Google Trends
- **Weekly search interest**
- **February 2021 – February 2026 (5-year view)** plus a **last‑12‑months export** for recent momentum
- **U.S. market only**
- **Categories analyzed:**
  - Creatine
  - Protein powder
  - Pre workout
  - Weight loss supplements
  - Omega 3

### 2️⃣ Meta Ads Library
- **Sample of long-running and high-performing supplement ads**
- **Creative format analysis**
- **Duration & positioning review**
- **Hook structure breakdown**

### 3️⃣ Keyword Research
- **Search volume**
- **Competition level**
- **Commercial intent**
- **Trend direction**

### 4️⃣ Social Sentiment (X/Twitter)
- **Conversation tone**
- **Common objections**
- **Emotional triggers**
- **Identity mapping**

---

## 📈 Key Findings (Last 12 Months)

### 🚀 Creatine – Structural Growth Signal
- **+28.7% YoY growth**
- **Strong Q1 demand peak**
- **Expanding beyond bodybuilders into mainstream fitness**

**Strategic Insight:** Creatine appears to be transitioning from niche supplement to performance mainstream product.

### 📉 Weight Loss – Declining Interest
- **−18.2% YoY decline**
- **Reduced advertising traction**
- **High competition, lower engagement**

**Strategic Insight:** Traditional "fat loss" positioning is losing momentum compared to performance and strength narratives.

### 👩 Women Segment – Underserved Market
- **Minimal ad targeting observed**
- **Growing female strength training participation**
- **Limited identity-driven messaging**

**Strategic Insight:** Clear opportunity for brand differentiation via female-focused positioning.

### 🎥 Creative Format Insight
- **UGC / creator-style ads outperform branded product ads**
- **Winning pattern:**
  - Identity-based hooks ("For women who lift")
  - Relatable storytelling
  - Authentic framing over polished branding

---

## 📊 5-Year Structural Analysis (2021–2026)

### Objectives
The long-term analysis aims to:
- Validate whether creatine growth is structural or temporary
- Identify consistent seasonal patterns
- Measure category volatility
- Detect long-term decline or maturity signals

### Quantitative Framework

#### 1️⃣ CAGR (Compound Annual Growth Rate)
- Measures structural growth vs short-term hype

#### 2️⃣ Seasonality Analysis
- Monthly average index
- Identification of recurring January peaks
- Quarterly pattern stability

#### 3️⃣ Volatility Analysis
- Standard deviation
- Coefficient of variation
- Demand stability scoring

#### 4️⃣ Momentum Analysis
- Comparison:
  - Last 12 months
  - vs
  - Previous 12 months

---

## 📊 Strategic Category Assessment

| Category     | Structural Trend   | Seasonality | Volatility | Strategic Outlook     |
|-------------|--------------------|-------------|------------|-----------------------|
| Creatine    | Strong Uptrend     | High (Jan)  | Medium     | Scale investment      |
| Protein     | Stable Growth      | Moderate    | Low        | Maintain & optimize   |
| Pre-workout | Cyclical           | Moderate    | Medium     | Tactical campaigns    |
| Weight Loss | Structural Decline | High (Jan)  | High       | Reduce exposure       |
| Mass Gainer | Flat / Niche       | Low         | Low        | Selective targeting   |

---

## 🧠 Market Interpretation

### 🟢 Performance > Aesthetics Shift

The data suggests a macro shift from:
- **Aesthetic-driven motivation** (fat loss)
- **to:**
- **Performance-driven motivation** (strength, muscle, energy)

This indicates:
- Cultural shift in fitness behavior
- Strength training normalization
- Social identity-driven consumption

---

## 🎯 2026 Strategic Investment Recommendation

### Primary Focus: Creatine
- Increase media allocation
- Expand creative testing
- Target female strength segment
- Prioritize UGC creators

### Secondary Focus: Protein
- Bundle with creatine
- Optimize messaging around recovery
- Position as foundational supplement

### Reduce Exposure: Weight Loss
- Avoid heavy scaling
- Test repositioning toward performance
- Monitor structural decline

---

## 💰 Budget Allocation Framework (Example)

| Category    | Investment Level | Risk  | Priority  |
|------------|------------------|-------|-----------|
| Creatine   | High             | Medium| 🔥 Primary |
| Protein    | Medium           | Low   | Stable    |
| Pre-workout| Medium           | Medium| Tactical  |
| Weight Loss| Low              | High  | Defensive |
| Mass Gainer| Low              | Low   | Niche     |

---

## 🧩 Business Value Delivered

This project enables:
- **Evidence-based marketing allocation**
- **Reduced investment risk**
- **Seasonal campaign optimization**
- **Category diversification strategy**
- **Identity-driven positioning insights**

---

## 🛠 Tools & Skills Demonstrated

- **Time-series analysis**
- **Market trend evaluation**
- **Business intelligence thinking**
- **Competitive ad breakdown**
- **Strategic recommendation modeling**
- **Data storytelling**

---

## 📦 What we have so far (technical status)

- **Data layer**
  - Raw Google Trends exports in `data/` (5-year weekly series plus 12‑month deep dive).
  - Enriched qualitative/strategic CSVs in `google-sheets/`:
    - `... Recommendations.csv` → executive summary & testing framework.
    - `... Matrix.csv` → creative strategy matrix by persona.
    - `... Keyword Research.csv` → SEO/SEM keyword evaluation.
    - `... Ads Analysis .csv` → long‑running ads and creative benchmarks.
    - `... Sentiment Analysis.csv` → X/Twitter sentiment & objection mapping.
    - `... Trends.csv` → multi‑year trend, CAGR and priority per category.
  - `google-sheets/README.md` documents each CSV in English.

- **Analysis & logic**
  - Core analysis implemented in `notebooks/analysis.ipynb` (Python 3.11):
    - Data loading and cleaning for 5‑year Google Trends series.
    - Metrics: CAGR, YoY momentum, seasonality, volatility (CV), regression.
    - Strategic “dashboard” combining trends, seasonality, momentum and scorecard.
  - Reusable helpers in the `analysis/` package:
    - `load_trends_5y`, `load_trends_12m` for data ingestion.
    - `yearly_averages`, `cagr_table`, `yoy_momentum`, `seasonality_table`, `volatility_table`, `quarterly_averages`, `linear_trend` for metrics.

- **Outputs**
  - Figures saved in `figures/` (trend, CAGR, YoY, seasonality, volatility, quarterly view, regression, dashboard).
  - `presentation/fitness_market_research.pptx` for stakeholder‑friendly delivery.

- **Data hygiene improvements**
  - Normalized `google-sheets` CSVs for easier use in code:
    - Fixed mixed numeric/percentage columns in `... Trends.csv` (all numeric now).
    - Replaced `#ERROR!` with `N/A` in `... Sentiment Analysis.csv`.
    - Treated large `Ad ID` in `... Ads Analysis .csv` as string to avoid scientific notation issues.

---

## 🔄 How this repository works (end‑to‑end)

- **1. Data ingestion**
  - Raw search interest comes from Google Trends exports in `data/` (5‑year + last‑12‑months windows).
  - Qualitative and strategic context comes from Google Sheets exports in `google-sheets/` (recommendations, personas, keyword research, ads, sentiment, trends).

- **2. Analysis**
  - Python notebook (`notebooks/analysis.ipynb`) and the `analysis/` package:
    - Load and clean the time series (`load_trends_5y`, `load_trends_12m`).
    - Compute yearly / quarterly / monthly aggregates.
    - Derive metrics such as CAGR, YoY momentum, volatility and linear trend.

- **3. Synthesis**
  - Visual outputs are saved under `figures/` and plugged into the PowerPoint in `presentation/`.
  - Strategic insights are consolidated and documented in:
    - The main `README.md` (this file).
    - `google-sheets/Recommendations.csv` and the persona/keyword/sentiment matrices.

### ✅ Final outcome (what this analysis shows)

- **Creatine** is a **structural growth category**: strong 5‑year CAGR, positive regression slope, and accelerating YoY momentum — recommendation: **scale investment** (especially Q1 and Q3) and prioritize women + creatine positioning.
- **Protein powder** shows **stable, predictable growth**: maintain and optimize, often bundled with creatine to increase AOV.
- **Weight loss supplements** are in **structural decline**: shrinking YoY and weaker intent — avoid scaling classic “fat loss” narratives; reframe towards performance/energy if tested at all.
- **Women + performance** is the **largest underserved segment**: very little ad targeting despite strong search/sentiment signals.
- **UGC / creator formats** with identity‑ or problem‑first hooks **outperform polished brand ads** and remain profitable for hundreds of days, making them the preferred creative starting point.

---

## 📅 Project Context

- **Date:** February 2026
- **Market:** United States
- **Use Case:** Marketing investment decision support
- **Scope:** Supplement category growth analysis

---

## 🚀 Next Steps

- **Predictive demand modeling**
- **Scenario simulation** (optimistic vs conservative growth)
- **Market share proxy modeling**
- **Competitor concentration index**
- **Female segment demand deep dive**

---

## 👤 Author

**Rodrigo Maximiliano Portillo**  
Backend Developer → Data & Market Analytics Focus  
Argentina

