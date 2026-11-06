# Exploratory Data Analysis (EDA) on the Titanic Dataset

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)](https://pandas.pydata.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-0.13+-lightblue.svg)](https://seaborn.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.15+-orange.svg)](https://plotly.com/)

An end-to-end, reproducible Exploratory Data Analysis (EDA) and anomaly profiling project on the RMS Titanic passenger dataset. This study investigates demographic, socioeconomic, and spatial factors governing passenger survival using descriptive statistics, univariate and bivariate distributions, Tukey outlier boxplots, correlation matrices, multicollinearity diagnostics (VIF), and interactive Plotly visualizations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Objective](#2-project-objective)
3. [Dataset Description](#3-dataset-description)
4. [Technologies Used](#4-technologies-used)
5. [Project Structure](#5-project-structure)
6. [EDA Analytical Workflow](#6-eda-analytical-workflow)
7. [Descriptive Statistical Analysis](#7-descriptive-statistical-analysis)
8. [Univariate Analysis](#8-univariate-analysis)
9. [Bivariate Analysis](#9-bivariate-analysis)
10. [Multivariate Analysis & Collinearity](#10-multivariate-analysis--collinearity)
11. [Correlation Matrix](#11-correlation-matrix)
12. [Interactive Visualizations (Plotly)](#12-interactive-visualizations-plotly)
13. [Key Findings & Anomaly Detection](#13-key-findings--anomaly-detection)
14. [Feature-Level Inferences for Machine Learning](#14-feature-level-inferences-for-machine-learning)
15. [Installation Instructions](#15-installation-instructions)
16. [Execution Instructions](#16-execution-instructions)
17. [Output Files Directory](#17-output-files-directory)
18. [Conclusion](#18-conclusion)

---

## 1. Project Overview

The sinking of the RMS Titanic in April 1912 remains one of the most studied maritime tragedies in modern history. Out of an estimated 2,224 passengers and crew, over 1,500 perished due to an inadequate number of lifeboats and systemic operational decisions during evacuation. 

This project performs an extensive exploratory analysis of the **891 passenger records** in the curated Titanic manifest to decipher the primary determinants of survival. The project strictly preserves raw historical records, identifies distribution extremes without arbitrary row deletion, and provides a structured foundation for downstream machine learning pipelines.

---

## 2. Project Objective

The core objective is to **understand data through rigorous descriptive statistics and visual analytics**:
1. **Summary Statistics**: Compute measures of central tendency (mean, median, mode), dispersion (standard deviation, variance, IQR), and distribution shape (skewness, kurtosis).
2. **Distribution & Outlier Characterization**: Generate histograms, kernel density curves, and Tukey boxplots to profile continuous features and isolate physical outliers.
3. **Relationship Modeling**: Assess inter-feature dependencies via pairplots, Pearson correlation matrices, and Variance Inflation Factor (VIF) collinearity diagnostics.
4. **Pattern & Anomaly Identification**: Identify empirical trends (e.g., survival by gender, passenger class, family size thresholds, embarkation port, zero-fare tickets).
5. **Machine Learning Preprocessing Hypotheses**: Formulate feature engineering and imputation strategies grounded in data evidence.

---

## 3. Dataset Description

The dataset resides at `dataset/Titanic-Dataset.csv` containing **891 rows** and **12 columns**:

| Feature Name | Analytical Type | Description | Values / Range | Missing Values |
| :--- | :--- | :--- | :--- | :--- |
| **`PassengerId`** | Numerical (Discrete) | Unique passenger sequence identifier | $1 \dots 891$ | 0 (0.0%) |
| **`Survived`** | Categorical (Binary) | Survival outcome indicator | `0 = Perished`, `1 = Survived` | 0 (0.0%) |
| **`Pclass`** | Categorical (Ordinal) | Socioeconomic ticket tier | `1 = 1st`, `2 = 2nd`, `3 = 3rd` | 0 (0.0%) |
| **`Name`** | Categorical (Nominal) | Passenger full name with honorific title | Text string | 0 (0.0%) |
| **`Sex`** | Categorical (Binary) | Biological sex | `male`, `female` | 0 (0.0%) |
| **`Age`** | Numerical (Continuous) | Passenger age in years | $0.42 \dots 80.00$ | 177 (19.87%) |
| **`SibSp`** | Numerical (Discrete) | Number of siblings/spouses aboard | $0 \dots 8$ | 0 (0.0%) |
| **`Parch`** | Numerical (Discrete) | Number of parents/children aboard | $0 \dots 6$ | 0 (0.0%) |
| **`Ticket`** | Categorical (Alphanumeric) | Ticket reference number | 681 unique codes | 0 (0.0%) |
| **`Fare`** | Numerical (Continuous) | Ticket fare in British pounds (£) | £0.00 to £512.33 | 0 (0.0%) |
| **`Cabin`** | Categorical (Alphanumeric) | Cabin room and deck allocation | 147 unique strings | 687 (77.10%) |
| **`Embarked`** | Categorical (Nominal) | Port of boarding | `C = Cherbourg`, `Q = Queenstown`, `S = Southampton` | 2 (0.22%) |

---

## 4. Technologies Used

- **Programming Language**: Python 3.12+
- **Data Manipulation**: `pandas`, `numpy`
- **Statistical Computing**: `scipy.stats`, `statsmodels`
- **Static Visualizations**: `matplotlib`, `seaborn`
- **Interactive Visualizations**: `plotly.express`, `plotly.graph_objects`
- **Notebook Development**: `jupyter`, `notebook`, `nbconvert`

---

## 5. Project Structure

```
exploratory-data-analysis/
│
├── dataset/
│   └── Titanic-Dataset.csv              # Original raw Titanic passenger dataset
│
├── notebooks/
│   └── exploratory_data_analysis.ipynb  # Comprehensive, executed 14-section Jupyter Notebook
│
├── outputs/
│   ├── visualizations/                  # High-resolution static PNGs & standalone Plotly HTMLs
│   │   ├── univariate_numerical_distributions.png
│   │   ├── univariate_categorical_distributions.png
│   │   ├── boxplot_analysis.png
│   │   ├── fare_age_boxplots.png
│   │   ├── bivariate_survival_by_gender.png
│   │   ├── bivariate_survival_by_class.png
│   │   ├── bivariate_survival_by_age.png
│   │   ├── bivariate_survival_by_fare.png
│   │   ├── bivariate_family_size.png
│   │   ├── multivariate_pairplot.png
│   │   ├── correlation_heatmap.png
│   │   ├── multivariate_bubble_chart.png
│   │   ├── survival_distribution.html
│   │   ├── survival_by_gender.html
│   │   ├── survival_by_class.html
│   │   ├── fare_distribution.html
│   │   ├── age_distribution.html
│   │   └── feature_relationships.html
│   │
│   └── reports/                         # Quantitative summaries and statistical tables
│       ├── descriptive_statistics.csv
│       ├── missing_values_summary.csv
│       ├── categorical_frequency_summary.csv
│       ├── correlation_matrix.csv
│       └── eda_summary_report.md
│
├── src/
│   ├── __init__.py                      # Package initializer
│   └── generate_notebook.py             # Reproducible notebook generator utility
│
├── .gitignore                           # Git ignore rules for virtual environments and caches
├── LICENSE                              # MIT License
├── README.md                            # Complete analytical documentation
└── requirements.txt                     # Explicit project dependency list
```

---

## 6. EDA Analytical Workflow

```
┌────────────────────────────────────────────────────────┐
│             1. Data Ingestion & Schema Audit           │
│   (Load dataset, verify dimensions, nulls, duplicates) │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│          2. Descriptive Statistical Profiling          │
│   (Mean, median, IQR, variance, skewness, kurtosis)    │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│      3. Univariate & Tukey Outlier Boxplot Analysis     │
│  (Histograms, KDEs, countplots, IQR anomaly detection) │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│         4. Bivariate Demographic & Survival Analysis   │
│ (Survival by Gender, Pclass, Age, Fare, Port, Family)  │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│    5. Multivariate, Correlation & VIF Diagnostics      │
│  (Seaborn pairplot, Pearson heatmap, VIF collinearity) │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│      6. Interactive Visualizations & Standalone HTML   │
│   (Plotly charts for 3D exploration and hover details) │
└───────────────────────────┬────────────────────────────┘
                            │
┌───────────────────────────▼────────────────────────────┐
│      7. Feature Inferences & Quantitative Reporting    │
│  (Export CSV summaries, markdown tables, findings doc) │
└────────────────────────────────────────────────────────┘
```

---

## 7. Descriptive Statistical Analysis

Extended statistical properties computed for numerical features across central tendency, dispersion, range, and higher moments:

| Feature | Count | Mean | Median | Mode | Std Dev | Min | 25% | 50% | 75% | Max | IQR | Skewness | Kurtosis |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Age** | 714 | 29.6991 | 28.0000 | 24.00 | 14.5265 | 0.42 | 20.125 | 28.0000 | 38.00 | 80.00 | 17.88 | +0.3891 | +0.1783 |
| **Fare** | 891 | 32.2042 | 14.4542 | 8.05 | 49.6934 | 0.00 | 7.910 | 14.4542 | 31.00 | 512.33 | 23.09 | +4.7873 | +33.3981 |
| **SibSp** | 891 | 0.5230 | 0.0000 | 0.00 | 1.1027 | 0.00 | 0.000 | 0.0000 | 1.00 | 8.00 | 1.00 | +3.6954 | +17.8804 |
| **Parch** | 891 | 0.3816 | 0.0000 | 0.00 | 0.8061 | 0.00 | 0.000 | 0.0000 | 0.00 | 6.00 | 0.00 | +2.7491 | +9.7781 |
| **Pclass** | 891 | 2.3086 | 3.0000 | 3.00 | 0.8361 | 1.00 | 2.000 | 3.0000 | 3.00 | 3.00 | 1.00 | -0.6305 | -1.2800 |
| **Survived** | 891 | 0.3838 | 0.0000 | 0.00 | 0.4866 | 0.00 | 0.000 | 0.0000 | 1.00 | 1.00 | 1.00 | +0.4785 | -1.7750 |

### Categorical Frequency Overview

- **`Sex`**: 577 males (**64.76%**), 314 females (**35.24%**).
- **`Pclass`**: 3rd Class: 491 (**55.11%**), 1st Class: 216 (**24.24%**), 2nd Class: 184 (**20.65%**).
- **`Embarked`**: Southampton (S): 644 (**72.28%**), Cherbourg (C): 168 (**18.86%**), Queenstown (Q): 77 (**8.64%**), Missing: 2 (**0.22%**).
- **`Survived`**: Perished (0): 549 (**61.62%**), Survived (1): 342 (**38.38%**).

---

## 8. Univariate Analysis

Univariate analysis characterizes the isolated marginal distribution of each feature:

- **Numerical Distributions**:
  - `Age`: Moderately right-skewed (+0.39) bell curve with a distinctive density cluster below 5 years (infants/toddlers).
  - `Fare`: Extremely right-skewed (+4.79) and leptokurtic (+33.40); 75% of fares were under £31.00, while the maximum reached £512.33.
  - `SibSp` & `Parch`: Highly sparse discrete distributions where 68.2% and 76.1% of observations are zero.
  - *Saved Visualization*: [`outputs/visualizations/univariate_numerical_distributions.png`](outputs/visualizations/univariate_numerical_distributions.png)

- **Categorical Distributions**:
  - Highlights the predominant presence of working-class men travelling third class and embarking from Southampton.
  - *Saved Visualization*: [`outputs/visualizations/univariate_categorical_distributions.png`](outputs/visualizations/univariate_categorical_distributions.png)

---

## 9. Bivariate Analysis

Bivariate analysis establishes empirical conditional relationships between predictor variables and survival status:

### Survival by Gender
- **Female Survival**: **74.20%** (233 / 314 survived)
- **Male Survival**: **18.89%** (109 / 577 survived)
- *Odds Ratio*: $\approx 12.3$. Female travelers had more than 12 times greater odds of surviving than male travelers, demonstrating adherence to the "women and children first" evacuation protocol.
- *Saved Chart*: [`outputs/visualizations/bivariate_survival_by_gender.png`](outputs/visualizations/bivariate_survival_by_gender.png)

### Survival by Passenger Class
- **1st Class**: **62.96%** (136 / 216 survived)
- **2nd Class**: **47.28%** (87 / 184 survived)
- **3rd Class**: **24.24%** (119 / 491 survived)
- *Gradient*: A severe monotonic decline in survival probability from Upper to Lower class.
- *Saved Chart*: [`outputs/visualizations/bivariate_survival_by_class.png`](outputs/visualizations/bivariate_survival_by_class.png)

### Survival by Age & Fare
- **Age**: Children under 10 achieved **61.29%** survival (38 / 62), compared to 35.8% for adults aged 20–40. The oldest recorded passenger (80 years old) survived.
- **Fare**: Non-survivors had a median fare of £10.50 (mean £22.12), whereas survivors had a median fare of £26.00 (mean £48.40).
- *Saved Charts*: 
  - [`outputs/visualizations/bivariate_survival_by_age.png`](outputs/visualizations/bivariate_survival_by_age.png)
  - [`outputs/visualizations/bivariate_survival_by_fare.png`](outputs/visualizations/bivariate_survival_by_fare.png)

### Survival by Total Family Size (`FamilySize = SibSp + Parch + 1`)
- **Solo Travelers** (`FamilySize = 1`): **30.35%** survival rate.
- **Small Families** (`FamilySize = 2 to 4`): Elevated survival rates of **55.28%**, **57.84%**, and **72.41%**.
- **Large Families** (`FamilySize >= 5`): Survival plummeted below **20.0%** (0% for families of 8 and 11).
- *Saved Chart*: [`outputs/visualizations/bivariate_family_size.png`](outputs/visualizations/bivariate_family_size.png)

---

## 10. Multivariate Analysis & Collinearity

### Multivariate Pairplot
A 4x4 pairwise grid (`Age`, `Fare`, `SibSp`, `Parch`) stratified by `Survived` demonstrates that survival density concentrates in regions of higher fare and lower age cohorts.
- *Saved Visualization*: [`outputs/visualizations/multivariate_pairplot.png`](outputs/visualizations/multivariate_pairplot.png)

### Collinearity and Variance Inflation Factor (VIF)
Multicollinearity was quantitatively examined using the Variance Inflation Factor ($VIF = \frac{1}{1 - R_j^2}$):

| Feature | VIF Value | Collinearity Classification |
| :--- | :--- | :--- |
| **`Pclass`** | 1.6618 | Low (Safe) |
| **`Age`** | 1.2013 | Low (Safe) |
| **`SibSp`** | 1.2756 | Low (Safe) |
| **`Parch`** | 1.2636 | Low (Safe) |
| **`Fare`** | 1.5867 | Low (Safe) |

**Conclusion**: All VIF values remain well below 5.0. No problematic multicollinearity exists among candidate continuous predictors.

---

## 11. Correlation Matrix

Pearson correlation coefficients computed across numerical attributes:

| Feature | `Survived` | `Pclass` | `Age` | `SibSp` | `Parch` | `Fare` | `FamilySize` |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Survived`** | **1.000** | -0.338 | -0.077 | -0.035 | +0.082 | +0.257 | +0.017 |
| **`Pclass`** | -0.338 | **1.000** | -0.369 | +0.083 | +0.018 | **-0.550** | +0.066 |
| **`Age`** | -0.077 | -0.369 | **1.000** | -0.308 | -0.189 | +0.096 | -0.302 |
| **`SibSp`** | -0.035 | +0.083 | -0.308 | **1.000** | +0.415 | +0.160 | **+0.891** |
| **`Parch`** | +0.082 | +0.018 | -0.189 | +0.415 | **1.000** | +0.216 | **+0.783** |
| **`Fare`** | +0.257 | -0.550 | +0.096 | +0.160 | +0.216 | **1.000** | +0.217 |
| **`FamilySize`** | +0.017 | +0.066 | -0.302 | +0.891 | +0.783 | +0.217 | **1.000** |

- *Saved Heatmap*: [`outputs/visualizations/correlation_heatmap.png`](outputs/visualizations/correlation_heatmap.png)

---

## 12. Interactive Visualizations (Plotly)

Six standalone interactive HTML applications were generated using Plotly:

| File Name | Chart Type | Interactive Capabilities |
| :--- | :--- | :--- |
| [`survival_distribution.html`](outputs/visualizations/survival_distribution.html) | Donut / Pie Chart | Hover proportions, category toggle, dynamic count callout |
| [`survival_by_gender.html`](outputs/visualizations/survival_by_gender.html) | Grouped Bar Chart | Gender-level counts, rate comparisons, zoom/pan |
| [`survival_by_class.html`](outputs/visualizations/survival_by_class.html) | Grouped Bar Chart | Class-tier survival vs death counts, hover metadata |
| [`fare_distribution.html`](outputs/visualizations/fare_distribution.html) | Histogram + Marginal Box | Bin filtering, outlier hovering, quartile markers |
| [`age_distribution.html`](outputs/visualizations/age_distribution.html) | Histogram + Violin Plot | Age cohort density, survivor vs deceased distribution |
| [`feature_relationships.html`](outputs/visualizations/feature_relationships.html) | 3D Interactive Scatter | 3D rotation, Age vs Fare vs Pclass, passenger name tooltip |

*All HTML files are fully self-contained and run in any modern web browser without server setup or API tokens.*

---

## 13. Key Findings & Anomaly Detection

1. **Female Preference**: Females survived at almost 4x the rate of males (74.20% vs 18.89%).
2. **Socioeconomic Survival Stratification**: Class 1 survival was 2.6x higher than Class 3 survival (62.96% vs 24.24%).
3. **Childhood Protection**: Children under 10 years had a 61.29% survival rate.
4. **Family Size Non-Linearity**: Solo travelers and large families faced severe survival penalties (< 31%), whereas small family units of 2 to 4 members had peak survival (up to 72.41%).
5. **Zero-Fare Anomaly**: 15 records have a fare of £0.00. Historical investigation confirms these were crew guarantee members, shipping company employees, or complimentary passages.
6. **Luxury Suite Outliers**: Three passengers paid £512.33 on ticket 35930 (First Class parlour suites). These are authentic records that should be retained.
7. **Missing Data Structure**:
   - `Cabin`: Missing in 77.10% of records; missingness is Not at Random (MNAR) and strongly correlated with lower passenger classes.
   - `Age`: Missing in 19.87% of records; missingness is Missing at Random (MAR) across classes.

---

## 14. Feature-Level Inferences for Machine Learning

| Feature | Key Observation | Statistical Meaning | Recommended Preprocessing for ML |
| :--- | :--- | :--- | :--- |
| **`Survived`** | 38.38% positive class | Moderate class imbalance | Stratified $k$-fold cross-validation |
| **`Pclass`** | Strong negative correlation with survival | Socioeconomic access to boat decks | Ordinal encoding or one-hot vectors |
| **`Sex`** | 74.2% female vs 18.9% male survival | Primary evacuation sorting criterion | Binary encoding (`female=1, male=0`) |
| **`Age`** | 177 missing entries; protective for $<10$ | Non-linear survival relationship | Impute median by `Title` and `Pclass`; bin into life stages |
| **`Fare`** | Heavily skewed (4.79); max £512.33 | Proxy for deck tier and wealth | Apply $\log(1 + x)$ transformation |
| **`SibSp`** & **`Parch`** | Peak survival for 2–4 relatives | Non-linear family size impact | Engineer `FamilySize` and `IsAlone` binary indicator |
| **`Embarked`** | C: 55.4%, Q: 39.0%, S: 33.7% | Confounded by Class 1 concentration | Impute mode ('S'); one-hot encode |
| **`Cabin`** | 77.1% missing | Missingness indicates lower deck | Extract `CabinDeck` letter; encode `'Unknown'` category |
| **`Name`** | Contains structured titles | Titles proxy age, marital status, and rank | Extract `Title` (Mr, Mrs, Miss, Master, Rare) |

---

## 15. Installation Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.12)
- Git

### Clone the Repository
```bash
git clone https://github.com/pranavm107/exploratory-data-analysis.git
cd exploratory-data-analysis
```

### Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 16. Execution Instructions

### Option A: Run the Jupyter Notebook in Interactive Mode
```bash
jupyter notebook notebooks/exploratory_data_analysis.ipynb
```
From the menu, select **Kernel -> Restart & Run All**.

### Option B: Execute Notebook Headless via CLI
```bash
python3 -m nbconvert --to notebook --execute notebooks/exploratory_data_analysis.ipynb --inplace
```

### Option C: Regenerate the Notebook from Source
```bash
python3 src/generate_notebook.py
python3 -m nbconvert --to notebook --execute notebooks/exploratory_data_analysis.ipynb --inplace
```

---

## 17. Output Files Directory

All generated files are tracked and organized in dedicated output directories:

### Analytical Reports (`outputs/reports/`)
- `descriptive_statistics.csv`: Complete parametric and non-parametric statistics table.
- `missing_values_summary.csv`: Column-by-column missingness, uniqueness, and data types.
- `categorical_frequency_summary.csv`: Frequency counts and percentage shares.
- `correlation_matrix.csv`: Pearson correlation matrix across numerical variables.
- `eda_summary_report.md`: High-level executive analytical briefing.

### Visualizations (`outputs/visualizations/`)
- **Static Charts (300 DPI PNG)**:
  - `univariate_numerical_distributions.png`
  - `univariate_categorical_distributions.png`
  - `boxplot_analysis.png`
  - `fare_age_boxplots.png`
  - `bivariate_survival_by_gender.png`
  - `bivariate_survival_by_class.png`
  - `bivariate_survival_by_age.png`
  - `bivariate_survival_by_fare.png`
  - `bivariate_family_size.png`
  - `multivariate_pairplot.png`
  - `correlation_heatmap.png`
  - `multivariate_bubble_chart.png`
- **Interactive Visualizations (Standalone HTML)**:
  - `survival_distribution.html`
  - `survival_by_gender.html`
  - `survival_by_class.html`
  - `fare_distribution.html`
  - `age_distribution.html`
  - `feature_relationships.html`

---

## 18. Conclusion

This exploratory analysis reveals that Titanic survival was heavily dictated by institutional protocol and socioeconomic positioning rather than purely random chance:
- **Gender and Class Dominance**: Female passengers in 1st Class experienced the highest survival rate (> 96%), whereas male passengers in 3rd Class faced the lowest (< 14%).
- **Family Dynamics**: Traveling with a small family unit (2 to 4 members) conferred significant survival advantages over traveling solo or in large family units.
- **Statistical Integrity**: All numerical features exhibit low multicollinearity (VIF < 1.70), and heavy skewness in `Fare` can be resolved via log-transformation without data loss.

This exploratory phase establishes the statistical baseline and feature engineering roadmap for future predictive modeling.

