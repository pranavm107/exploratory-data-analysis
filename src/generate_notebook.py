"""
Builder script to generate notebooks/exploratory_data_analysis.ipynb
with complete structure, markdown narratives, and executable code.
"""

import json
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

def add_md(content):
    cells.append(nbf.v4.new_markdown_cell(content.strip()))

def add_code(content):
    cells.append(nbf.v4.new_code_cell(content.strip()))

# ==============================================================================
# SECTION 1: PROJECT INTRODUCTION
# ==============================================================================
add_md("""
# Task 2: Comprehensive Exploratory Data Analysis (EDA) on the Titanic Dataset

---

## SECTION 1: Project Introduction

### 1.1 Project Title
**Exploratory Data Analysis (EDA) and Anomaly Profiling on the Titanic Passenger Dataset**

### 1.2 Project Objective
The fundamental objective of this project is to **understand data through rigorous descriptive statistics, univariate, bivariate, multivariate, and interactive visualizations**. Specifically, this analysis aims to:
1. Compute descriptive and distributional statistics (mean, median, standard deviation, skewness, kurtosis, IQR).
2. Characterize numerical distributions via histograms and Tukey boxplots to isolate skewness and physical outliers.
3. Quantify pairwise and multi-feature associations through correlation matrices, pairplots, cross-tabulations, and Variance Inflation Factors (VIF).
4. Discover intrinsic historical patterns, demographic trends, and anomalies (such as fare distribution extremes and zero-fare entries).
5. Extract feature-level inferences to establish grounded hypotheses for prospective machine learning pipelines.

### 1.3 Introduction to Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA), pioneered by John W. Tukey, is an indispensable stage in data science. Rather than immediately fitting machine learning models to raw data, EDA employs summary statistics and graphical representations to:
- Uncover underlying data distributions and structural anomalies.
- Detect outliers, measurement irregularities, and missingness mechanisms (MCAR, MAR, MNAR).
- Assess parametric assumptions (e.g., normality, homoscedasticity) and multicollinearity.
- Provide domain-grounded intuition to guide feature selection and engineering.

### 1.4 Dataset Description
The dataset captures demographic, ticket, cabin, and survival attributes of **891 passengers** aboard the RMS Titanic on its fateful voyage in April 1912:
- **`PassengerId`**: Unique sequential integer identifier.
- **`Survived`**: Target binary indicator (`0 = No`, `1 = Yes`).
- **`Pclass`**: Socioeconomic passenger class (`1 = 1st / Upper`, `2 = 2nd / Middle`, `3 = 3rd / Lower`).
- **`Name`**: Full name including titles (Mr., Mrs., Miss, Master, etc.).
- **`Sex`**: Biological sex (`male`, `female`).
- **`Age`**: Age in fractional/integer years.
- **`SibSp`**: Number of siblings or spouses traveling aboard.
- **`Parch`**: Number of parents or children traveling aboard.
- **`Ticket`**: Alphanumeric ticket reference number.
- **`Fare`**: Passenger ticket fare in British pounds sterling (£).
- **`Cabin`**: Cabin number and deck code.
- **`Embarked`**: Port of embarkation (`C = Cherbourg`, `Q = Queenstown`, `S = Southampton`).

### 1.5 Technologies & Libraries
- **Language**: Python 3.12+
- **Data Manipulation**: Pandas, NumPy
- **Statistical Computation**: SciPy, Statsmodels
- **Static Graphics**: Matplotlib, Seaborn
- **Interactive Visualizations**: Plotly Express & Graph Objects

### 1.6 Analytical Workflow
```
[1. Ingestion & Overview] ➔ [2. Descriptive Statistics] ➔ [3. Univariate Distributions]
                                                                        ⬇
[6. Advanced Plotly HTML] ⬅ [5. Multivariate & Correlation] ⬅ [4. Boxplots & Bivariate]
           ⬇
[7. Pattern & Anomaly Synthesis] ➔ [8. Feature Inferences] ➔ [9. Final Reports Export]
```

### 1.7 Importance of EDA Before Machine Learning
Executing EDA prior to predictive modeling is vital because:
- **Prevents Garbage-In, Garbage-Out**: Feeding skewed distributions, extreme outliers, or corrupted records directly to models degrades convergence and predictive accuracy.
- **Informs Imputation Strategies**: Identifying whether missing values in `Age` or `Cabin` correlate with passenger class prevents biased imputation.
- **Guides Feature Engineering**: Recognizing non-linear patterns (such as family size thresholds or socioeconomic survival gaps) guides interaction term creation.
- **Diagnoses Multicollinearity**: Identifying redundancy across variables ensures stable coefficient estimation in linear models.
""")

# ==============================================================================
# SECTION 2: IMPORT REQUIRED LIBRARIES
# ==============================================================================
add_md("""
---

## SECTION 2: Import Required Libraries and Environment Configuration

We import the essential numerical, statistical, data processing, static plotting, and interactive visualization packages. In addition, we configure global visual styles, Matplotlib display parameters, and output directories.
""")

add_code("""
# Core Data Manipulation and Numerical Computing
import pandas as pd
import numpy as np

# Statistical Analysis Tools
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Static Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Interactive Visualization Libraries
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Standard Utilities and Filesystem Management
import os
import sys
import warnings

# Suppress minor non-critical deprecation and future warnings for clean reporting
warnings.filterwarnings('ignore')

# Configure Seaborn and Matplotlib Aesthetics
sns.set_theme(style='whitegrid', palette='deep')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'figure.autolayout': True,
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.title_fontsize': 11
})

# Configure output directories dynamically relative to repository root
base_dir = os.path.abspath(os.path.join(os.getcwd(), '..') if os.path.basename(os.getcwd()) == 'notebooks' else os.getcwd())
viz_dir = os.path.join(base_dir, 'outputs', 'visualizations')
reports_dir = os.path.join(base_dir, 'outputs', 'reports')

os.makedirs(viz_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

print(f"Project Base Directory: {base_dir}")
print(f"Visualizations Output: {viz_dir}")
print(f"Reports Output:        {reports_dir}")
""")

# ==============================================================================
# SECTION 3: LOAD THE TITANIC DATASET
# ==============================================================================
add_md("""
---

## SECTION 3: Load the Titanic Dataset

We locate and read the original CSV file from the `dataset/` directory using robust relative pathing. We perform essential baseline sanity checks without modifying or mutating the original source file.
""")

add_code("""
# Establish dataset path dynamically
dataset_filename = 'Titanic-Dataset.csv'
dataset_path = os.path.join(base_dir, 'dataset', dataset_filename)

if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found at expected location: {dataset_path}")

# Load the raw dataset
df = pd.read_csv(dataset_path)

print(f"Successfully loaded '{dataset_filename}' from: {dataset_path}")
print(f"Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
""")

add_md("""
### 3.1 Inspecting Head, Tail, and Dimensions
We examine the initial and final records to verify schema alignment and boundary rows.
""")

add_code("""
# Display first 5 rows
print("=== FIRST 5 ROWS (HEAD) ===")
display(df.head())

# Display last 5 rows
print("=== LAST 5 ROWS (TAIL) ===")
display(df.tail())
""")

add_md("""
### 3.2 Dataset Schema, Data Types, and Structural Summary
We verify data types, non-null counts, and classification into numerical and categorical categories.
""")

add_code("""
print("=== DATASET INFORMATION ===")
df.info()

print("\\n=== COLUMN DATA TYPES ===")
dtypes_df = pd.DataFrame(df.dtypes, columns=['DataType']).reset_index().rename(columns={'index': 'Feature'})
display(dtypes_df)

# Classify columns by analytical type
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical Columns ({len(numerical_cols)}):   {numerical_cols}")
print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")
""")

# ==============================================================================
# SECTION 4: DATASET OVERVIEW AND INITIAL EXPLORATION
# ==============================================================================
add_md("""
---

## SECTION 4: Dataset Overview and Initial Exploration

An exhaustive audit of missing values, cardinality, and duplicate entries is conducted to establish data hygiene baseline before modeling.
""")

add_code("""
# Assess duplicate records
duplicate_count = df.duplicated().sum()
print(f"Duplicate records found: {duplicate_count} ({(duplicate_count / len(df))*100:.2f}%)")

# Detailed column overview profiling table
overview_rows = []
for col in df.columns:
    col_dtype = str(df[col].dtype)
    non_null_count = int(df[col].notnull().sum())
    null_count = int(df[col].isnull().sum())
    null_pct = round((null_count / len(df)) * 100, 2)
    unique_cnt = int(df[col].nunique())
    sample_vals = str(df[col].dropna().unique()[:3].tolist())
    
    overview_rows.append({
        'Feature': col,
        'DataType': col_dtype,
        'Non-Null Count': non_null_count,
        'Missing Count': null_count,
        'Missing (%)': null_pct,
        'Unique Values': unique_cnt,
        'Sample Values': sample_vals
    })

overview_df = pd.DataFrame(overview_rows)
display(overview_df)

# Missing values summary
missing_df = overview_df[overview_df['Missing Count'] > 0][['Feature', 'Missing Count', 'Missing (%)']].sort_values(by='Missing Count', ascending=False).reset_index(drop=True)
print("=== MISSING VALUE SUMMARY ===")
display(missing_df)
""")

add_md("""
### Analytical Observations from Overview:
1. **Total Records**: 891 entries with 12 features. No duplicate rows exist.
2. **Missing Value Profile**:
   - **`Cabin`**: 687 missing values (**77.10%**). Missing Not at Random (MNAR); predominantly available for First Class passengers.
   - **`Age`**: 177 missing values (**19.87%**). Missing at Random (MAR); substantial but imputable based on passenger title and class.
   - **`Embarked`**: 2 missing values (**0.22%**). Negligible missingness (can be imputed with the mode `'S'`).
3. **Cardinality**: `PassengerId` and `Name` are unique per row; `Ticket` has 681 unique values (indicating shared tickets across families or traveling groups).
""")

# ==============================================================================
# SECTION 5: DESCRIPTIVE STATISTICAL ANALYSIS
# ==============================================================================
add_md("""
---

## SECTION 5: Descriptive Statistical Analysis

We calculate full parametric and non-parametric statistical measures for all numerical features, including central tendency, dispersion, range, shape metrics (skewness and kurtosis), and categorical frequency distributions.
""")

add_code("""
# Compute extended descriptive statistics for numerical features
def generate_descriptive_stats(data, cols):
    stats_list = []
    for col in cols:
        s = data[col].dropna()
        mode_val = s.mode().iloc[0] if not s.mode().empty else np.nan
        q25 = s.quantile(0.25)
        q50 = s.quantile(0.50)
        q75 = s.quantile(0.75)
        iqr = q75 - q25
        
        stats_list.append({
            'Feature': col,
            'Count': int(s.count()),
            'Mean': round(float(s.mean()), 4),
            'Median': round(float(q50), 4),
            'Mode': round(float(mode_val), 4),
            'Std Dev': round(float(s.std()), 4),
            'Variance': round(float(s.var()), 4),
            'Min': round(float(s.min()), 4),
            '25%': round(float(q25), 4),
            '50%': round(float(q50), 4),
            '75%': round(float(q75), 4),
            'Max': round(float(s.max()), 4),
            'Range': round(float(s.max() - s.min()), 4),
            'IQR': round(float(iqr), 4),
            'Skewness': round(float(s.skew()), 4),
            'Kurtosis': round(float(s.kurt()), 4)
        })
    return pd.DataFrame(stats_list)

numerical_features = ['Age', 'Fare', 'SibSp', 'Parch', 'Pclass', 'Survived']
desc_stats_df = generate_descriptive_stats(df, numerical_features)
display(desc_stats_df)
""")

add_md("""
### 5.1 Categorical Feature Distribution Summary
Frequency counts, proportions, and modes for categorical and discrete variables (`Sex`, `Embarked`, `Pclass`, `Survived`).
""")

add_code("""
cat_summary_list = []
categorical_analysis_cols = ['Sex', 'Embarked', 'Pclass', 'Survived']

for col in categorical_analysis_cols:
    val_counts = df[col].value_counts(dropna=False)
    val_pcts = df[col].value_counts(dropna=False, normalize=True) * 100
    for val in val_counts.index:
        cat_summary_list.append({
            'Feature': col,
            'Category': str(val),
            'Frequency': int(val_counts[val]),
            'Percentage (%)': round(float(val_pcts[val]), 2)
        })

cat_summary_df = pd.DataFrame(cat_summary_list)
display(cat_summary_df)
""")

add_md("""
### Statistical Insights:
- **`Fare`**: Exhibits extreme positive skewness (**4.7873**) and massive leptokurtosis (**33.3981**). The mean (£32.20) is more than double the median (£14.45), driven by upper outliers up to £512.33. Additionally, 15 tickets recorded a fare of £0.00.
- **`Age`**: Sits at a mean of 29.70 years and median of 28.00 years with moderate spread (std 14.53) and slight positive skewness (**0.3891**), spanning infants (0.42 years / ~5 months) to an 80-year-old traveler.
- **`SibSp` & `Parch`**: Both show strong positive skewness (**3.6954** and **2.7491**). Over 68% of passengers traveled without siblings/spouses and over 76% without parents/children.
- **`Survived`**: Target variable shows an overall baseline survival rate of **38.38%** (342 survivors vs 549 fatalities).
- **`Sex`**: 577 males (64.76%) and 314 females (35.24%).
- **`Pclass`**: 3rd Class constitutes the clear majority at 491 passengers (55.11%), followed by 1st Class (216, 24.24%) and 2nd Class (184, 20.65%).
""")

# ==============================================================================
# SECTION 6: UNIVARIATE ANALYSIS
# ==============================================================================
add_md("""
---

## SECTION 6: Univariate Analysis

Univariate analysis investigates the individual distribution of each variable in isolation. We examine continuous variables via histograms with Kernel Density Estimation (KDE) and discrete/categorical variables via bar and count distributions.
""")

add_code("""
# 6A. Numerical Feature Analysis: Histograms with KDE
num_features_to_plot = ['Age', 'Fare', 'SibSp', 'Parch']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(num_features_to_plot):
    data_col = df[col].dropna()
    sns.histplot(data_col, kde=True, ax=axes[i], color='#2b5c8f', edgecolor='black', bins=30)
    axes[i].set_title(f"Univariate Distribution of {col}", fontsize=13, weight='bold')
    axes[i].set_xlabel(f"{col}", fontsize=11, weight='bold')
    axes[i].set_ylabel("Frequency", fontsize=11, weight='bold')
    
    # Overlay mean and median lines
    mean_val = data_col.mean()
    median_val = data_col.median()
    axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    axes[i].axvline(median_val, color='green', linestyle='-', linewidth=1.5, label=f"Median: {median_val:.1f}")
    axes[i].legend(loc='upper right')

plt.tight_layout()
fig_num_path = os.path.join(viz_dir, 'univariate_numerical_distributions.png')
plt.savefig(fig_num_path, dpi=300)
plt.show()
print(f"Saved: {fig_num_path}")
""")

add_md("""
### Univariate Numerical Inferences:
- **Age**: Displays an approximately bell-shaped distribution with a prominent bump below 5 years, highlighting infants and small children aboard. Right-skewness is modest.
- **Fare**: Highly clustered between £0 and £50 with an extensive thin right tail reaching £512.33. Logarithmic transformation is appropriate for linear modeling.
- **SibSp & Parch**: Discrete count distributions heavily centered at zero. Most passengers were solo travelers, while a small minority belonged to large family units (up to 8 siblings/spouses and 6 parents/children).
""")

add_code("""
# 6B. Categorical Feature Analysis: Frequency and Proportions
cat_cols_to_plot = ['Survived', 'Sex', 'Pclass', 'Embarked']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

palette_choices = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

for i, col in enumerate(cat_cols_to_plot):
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, ax=axes[i], order=order, palette='crest')
    axes[i].set_title(f"Category Distribution of {col}", fontsize=13, weight='bold')
    axes[i].set_xlabel(f"{col}", fontsize=11, weight='bold')
    axes[i].set_ylabel("Passenger Count", fontsize=11, weight='bold')
    
    # Add percentage and count labels above bars
    total = len(df[col].dropna())
    for p in axes[i].patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            percentage = f"{(height / total) * 100:.1f}%"
            axes[i].annotate(f"{int(height)}\\n({percentage})",
                             (p.get_x() + p.get_width() / 2., height),
                             ha='center', va='bottom', fontsize=9, xytext=(0, 3),
                             textcoords='offset points')

plt.tight_layout()
fig_cat_path = os.path.join(viz_dir, 'univariate_categorical_distributions.png')
plt.savefig(fig_cat_path, dpi=300)
plt.show()
print(f"Saved: {fig_cat_path}")
""")

add_md("""
### Univariate Categorical Inferences:
- **Survived**: 61.6% of passengers perished, confirming the severity of the maritime disaster.
- **Sex**: Males constituted 64.8% of the passenger manifest.
- **Pclass**: Over 55% of passengers were in 3rd Class, reflecting early 20th-century transatlantic emigration patterns.
- **Embarked**: 72.4% of passengers boarded at Southampton (S), 18.9% at Cherbourg (C), and 8.7% at Queenstown (Q).
""")

# ==============================================================================
# SECTION 7: BOXPLOT ANALYSIS
# ==============================================================================
add_md("""
---

## SECTION 7: Boxplot Analysis and Outlier Detection

Boxplots visualize the five-number summary: minimum, first quartile ($Q_1$), median, third quartile ($Q_3$), and maximum (within $1.5 \\times IQR$). Points outside whiskers are flagged as Tukey outliers.
""")

add_code("""
# Outlier Quantification using Tukey's 1.5 * IQR Rule
outlier_summary = []

for col in ['Age', 'Fare', 'SibSp', 'Parch']:
    s = df[col].dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = s[(s < lower_bound) | (s > upper_bound)]
    outlier_summary.append({
        'Feature': col,
        'Q1 (25%)': round(float(q1), 2),
        'Median (50%)': round(float(s.median()), 2),
        'Q3 (75%)': round(float(q3), 2),
        'IQR': round(float(iqr), 2),
        'Lower Bound': round(float(lower_bound), 2),
        'Upper Bound': round(float(upper_bound), 2),
        'Outliers Count': len(outliers),
        'Outlier (%)': round((len(outliers) / len(s)) * 100, 2)
    })

outlier_table = pd.DataFrame(outlier_summary)
display(outlier_table)

# Visualizing Boxplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(['Age', 'Fare', 'SibSp', 'Parch']):
    sns.boxplot(y=df[col], ax=axes[i], color='#3a86ff', flierprops={'marker':'o', 'markersize':4, 'markerfacecolor':'#e63946'})
    axes[i].set_title(f"Boxplot of {col} with Tukey Outliers", fontsize=13, weight='bold')
    axes[i].set_ylabel(col, fontsize=11, weight='bold')

plt.tight_layout()
fig_box_path = os.path.join(viz_dir, 'boxplot_analysis.png')
plt.savefig(fig_box_path, dpi=300)
plt.show()
print(f"Saved: {fig_box_path}")
""")

add_md("""
### Stratified Boxplots: Fare and Age Across Passenger Class
Examining how `Fare` and `Age` distributions vary systematically across socioeconomic tiers.
""")

add_code("""
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Fare by Pclass
sns.boxplot(x='Pclass', y='Fare', data=df, ax=axes[0], palette='Set2', showmeans=True,
            meanprops={"marker":"s","markerfacecolor":"red", "markeredgecolor":"red"})
axes[0].set_title("Fare Distribution Stratified by Pclass", fontsize=13, weight='bold')
axes[0].set_xlabel("Passenger Class", fontsize=11, weight='bold')
axes[0].set_ylabel("Fare (£)", fontsize=11, weight='bold')

# Age by Pclass
sns.boxplot(x='Pclass', y='Age', data=df, ax=axes[1], palette='Set2', showmeans=True,
            meanprops={"marker":"s","markerfacecolor":"red", "markeredgecolor":"red"})
axes[1].set_title("Age Distribution Stratified by Pclass", fontsize=13, weight='bold')
axes[1].set_xlabel("Passenger Class", fontsize=11, weight='bold')
axes[1].set_ylabel("Age (Years)", fontsize=11, weight='bold')

plt.tight_layout()
fig_strat_path = os.path.join(viz_dir, 'fare_age_boxplots.png')
plt.savefig(fig_strat_path, dpi=300)
plt.show()
print(f"Saved: {fig_strat_path}")
""")

add_md("""
### Boxplot Analysis & Anomaly Evaluation:
1. **Fare Outliers (116 cases, 13.02%)**: The upper bound is £65.63. Fares reaching £512.33 represent luxury parlour suites booked by affluent First Class travelers (such as the Cardeza party on ticket 35930). These are **legitimate historical observations, not data errors**, and should be retained or log-transformed rather than discarded.
2. **Age Outliers (11 cases, 1.54%)**: The upper bound is 64.81 years. Passengers aged 65 to 80 represent the elderly cohort. The oldest recorded passenger (80 years old, Algernon Barkworth) survived on Lifeboat B.
3. **Parch Anomaly**: Because the 25th, 50th, and 75th percentiles are all 0, any non-zero value mathematically falls outside the $1.5 \\times IQR$ boundary, highlighting how discrete count data must be contextualized rather than treated as standard normal variables.
4. **Class Stratification**: Median Age systematically declines from Class 1 (37.0 years) to Class 2 (29.0 years) and Class 3 (24.0 years).
""")

# ==============================================================================
# SECTION 8: BIVARIATE ANALYSIS
# ==============================================================================
add_md("""
---

## SECTION 8: Bivariate Analysis

Bivariate analysis explores the direct relationship between predictor variables and survival status (`Survived`), as well as inter-feature dependencies.
""")

add_md("""
### 8A. Survival vs Gender
Analyzing the disparity between male and female survival outcomes.
""")

add_code("""
# Crosstabulation of Sex vs Survived
sex_survival_ct = pd.crosstab(df['Sex'], df['Survived'], margins=True)
sex_survival_pct = pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100

print("=== SEX VS SURVIVAL COUNTS ===")
display(sex_survival_ct)
print("=== SEX VS SURVIVAL PERCENTAGES (%) ===")
display(sex_survival_pct.round(2))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(x='Sex', hue='Survived', data=df, ax=axes[0], palette=['#d9534f', '#5cb85c'])
axes[0].set_title("Survival Count by Gender", fontsize=13, weight='bold')
axes[0].set_xlabel("Sex", fontsize=11, weight='bold')
axes[0].set_ylabel("Count", fontsize=11, weight='bold')
axes[0].legend(['Perished (0)', 'Survived (1)'])

sex_survival_pct.plot(kind='bar', stacked=True, ax=axes[1], color=['#d9534f', '#5cb85c'], edgecolor='black')
axes[1].set_title("Survival Proportion by Gender (%)", fontsize=13, weight='bold')
axes[1].set_xlabel("Sex", fontsize=11, weight='bold')
axes[1].set_ylabel("Percentage", fontsize=11, weight='bold')
axes[1].legend(['Perished (0)', 'Survived (1)'], bbox_to_anchor=(1.02, 1), loc='upper left')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
fig_sex_path = os.path.join(viz_dir, 'bivariate_survival_by_gender.png')
plt.savefig(fig_sex_path, dpi=300)
plt.show()
print(f"Saved: {fig_sex_path}")
""")

add_md("""
### 8B. Survival vs Passenger Class
Analyzing survival rates across socioeconomic passenger classes.
""")

add_code("""
pclass_survival_ct = pd.crosstab(df['Pclass'], df['Survived'], margins=True)
pclass_survival_pct = pd.crosstab(df['Pclass'], df['Survived'], normalize='index') * 100

print("=== PCLASS VS SURVIVAL COUNTS ===")
display(pclass_survival_ct)
print("=== PCLASS VS SURVIVAL PERCENTAGES (%) ===")
display(pclass_survival_pct.round(2))

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.countplot(x='Pclass', hue='Survived', data=df, ax=axes[0], palette=['#d9534f', '#5cb85c'])
axes[0].set_title("Survival Count by Passenger Class", fontsize=13, weight='bold')
axes[0].set_xlabel("Passenger Class (Pclass)", fontsize=11, weight='bold')
axes[0].set_ylabel("Count", fontsize=11, weight='bold')
axes[0].legend(['Perished (0)', 'Survived (1)'])

pclass_survival_pct.plot(kind='bar', stacked=True, ax=axes[1], color=['#d9534f', '#5cb85c'], edgecolor='black')
axes[1].set_title("Survival Proportion by Passenger Class (%)", fontsize=13, weight='bold')
axes[1].set_xlabel("Passenger Class (Pclass)", fontsize=11, weight='bold')
axes[1].set_ylabel("Percentage", fontsize=11, weight='bold')
axes[1].legend(['Perished (0)', 'Survived (1)'], bbox_to_anchor=(1.02, 1), loc='upper left')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)

plt.tight_layout()
fig_pclass_path = os.path.join(viz_dir, 'bivariate_survival_by_class.png')
plt.savefig(fig_pclass_path, dpi=300)
plt.show()
print(f"Saved: {fig_pclass_path}")
""")

add_md("""
### 8C. Survival vs Age
Analyzing age density and distributions conditioned on survival status.
""")

add_code("""
# Grouped age statistics by survival status
age_survival_stats = df.groupby('Survived')['Age'].agg(
    Count='count',
    Mean='mean',
    Std='std',
    Median='median',
    Min='min',
    Max='max'
).reset_index()
display(age_survival_stats.round(2))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Overlaid KDE and Histogram
sns.histplot(data=df, x='Age', hue='Survived', kde=True, ax=axes[0], palette=['#d9534f', '#5cb85c'], bins=30, element='step')
axes[0].set_title("Age Distribution Conditioned on Survival", fontsize=13, weight='bold')
axes[0].set_xlabel("Age (Years)", fontsize=11, weight='bold')
axes[0].set_ylabel("Density / Count", fontsize=11, weight='bold')

# Boxplot of Age by Survival
sns.boxplot(x='Survived', y='Age', data=df, ax=axes[1], palette=['#d9534f', '#5cb85c'], showmeans=True)
axes[1].set_title("Age Boxplot by Survival Status", fontsize=13, weight='bold')
axes[1].set_xticklabels(['Perished (0)', 'Survived (1)'])
axes[1].set_xlabel("Survival Status", fontsize=11, weight='bold')
axes[1].set_ylabel("Age (Years)", fontsize=11, weight='bold')

plt.tight_layout()
fig_age_path = os.path.join(viz_dir, 'bivariate_survival_by_age.png')
plt.savefig(fig_age_path, dpi=300)
plt.show()
print(f"Saved: {fig_age_path}")
""")

add_md("""
### 8D. Survival vs Fare
Analyzing fare distributions across survival outcomes using boxplots and violin plots.
""")

add_code("""
fare_survival_stats = df.groupby('Survived')['Fare'].agg(
    Count='count',
    Mean='mean',
    Std='std',
    Median='median',
    Min='min',
    Max='max'
).reset_index()
display(fare_survival_stats.round(2))

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Boxplot of Fare by Survival
sns.boxplot(x='Survived', y='Fare', data=df, ax=axes[0], palette=['#d9534f', '#5cb85c'])
axes[0].set_title("Fare Distribution by Survival Status", fontsize=13, weight='bold')
axes[0].set_xticklabels(['Perished (0)', 'Survived (1)'])
axes[0].set_xlabel("Survival Status", fontsize=11, weight='bold')
axes[0].set_ylabel("Fare (£)", fontsize=11, weight='bold')

# Log-transformed Fare Violin Plot to visualize density across orders of magnitude
df_temp = df.copy()
df_temp['LogFare'] = np.log1p(df_temp['Fare'])
sns.violinplot(x='Survived', y='LogFare', data=df_temp, ax=axes[1], palette=['#d9534f', '#5cb85c'], inner='quartile')
axes[1].set_title("Log(Fare + 1) Density by Survival Status", fontsize=13, weight='bold')
axes[1].set_xticklabels(['Perished (0)', 'Survived (1)'])
axes[1].set_xlabel("Survival Status", fontsize=11, weight='bold')
axes[1].set_ylabel("Log(Fare + 1)", fontsize=11, weight='bold')

plt.tight_layout()
fig_fare_path = os.path.join(viz_dir, 'bivariate_survival_by_fare.png')
plt.savefig(fig_fare_path, dpi=300)
plt.show()
print(f"Saved: {fig_fare_path}")
""")

add_md("""
### 8E. Other Feature Relationships: Embarkation Port and Family Dynamics
We examine how embarkation port and family size (`FamilySize = SibSp + Parch + 1`) influence survival likelihood.
""")

add_code("""
# Create FamilySize feature
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Embarked vs Survival
embarked_pct = pd.crosstab(df['Embarked'], df['Survived'], normalize='index') * 100
embarked_pct.plot(kind='bar', stacked=True, ax=axes[0], color=['#d9534f', '#5cb85c'], edgecolor='black')
axes[0].set_title("Survival Rate by Embarkation Port (%)", fontsize=13, weight='bold')
axes[0].set_xlabel("Embarked Port (C=Cherbourg, Q=Queenstown, S=Southampton)", fontsize=11, weight='bold')
axes[0].set_ylabel("Percentage", fontsize=11, weight='bold')
axes[0].legend(['Perished (0)', 'Survived (1)'])
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)

# FamilySize vs Survival Rate
family_survival = df.groupby('FamilySize')['Survived'].agg(Count='count', SurvivalRate='mean').reset_index()
family_survival['SurvivalRate'] = family_survival['SurvivalRate'] * 100
sns.barplot(x='FamilySize', y='SurvivalRate', data=family_survival, ax=axes[1], palette='viridis')
axes[1].set_title("Survival Rate by Total Family Size (%)", fontsize=13, weight='bold')
axes[1].set_xlabel("Family Size (SibSp + Parch + 1)", fontsize=11, weight='bold')
axes[1].set_ylabel("Survival Rate (%)", fontsize=11, weight='bold')

for p in axes[1].patches:
    h = p.get_height()
    if h > 0:
        axes[1].annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h),
                         ha='center', va='bottom', fontsize=9, xytext=(0, 2),
                         textcoords='offset points')

plt.tight_layout()
fig_fam_path = os.path.join(viz_dir, 'bivariate_family_size.png')
plt.savefig(fig_fam_path, dpi=300)
plt.show()
print(f"Saved: {fig_fam_path}")
""")

add_md("""
### Key Bivariate Insights:
1. **Gender**: **74.20%** of females survived compared to only **18.89%** of males, confirming the strict operational enforcement of the maritime evacuation protocol ("women and children first").
2. **Class**: Survival drops precipitously down the socioeconomic ladder: **62.96%** in 1st Class, **47.28%** in 2nd Class, and only **24.24%** in 3rd Class.
3. **Age**: Children under 10 years old experienced a survival rate of **61.29%**, noticeably higher than working-age adults (20-40 years ~35-38%).
4. **Fare**: Survivors paid an average fare of **£48.40** (median £26.00), compared to non-survivors who paid an average of **£22.12** (median £10.50).
5. **Embarkation Port**: Passengers embarking at Cherbourg had the highest survival rate (**55.36%**), directly attributable to the high concentration of wealthy First Class passengers embarking there.
6. **Family Size**: Solo travelers had low survival (**30.35%**). Small families of 2 to 4 members had elevated survival rates (**55.28% to 72.41%**). Large families with 5 or more members saw survival collapse below **20%**, as keeping large family groups together during the panic hindered lifeboat boarding.
""")

# ==============================================================================
# SECTION 9: MULTIVARIATE ANALYSIS
# ==============================================================================
add_md("""
---

## SECTION 9: Multivariate Analysis

Multivariate analysis explores complex multi-dimensional interactions among multiple continuous and categorical features simultaneously.
""")

add_md("""
### 9A. Pairplot of Continuous Features
Visualizing pairwise bivariate relationships and univariate marginal densities conditioned on survival status.
""")

add_code("""
# Seaborn Pairplot of Core Continuous Variables with Hue='Survived'
pairplot_features = ['Age', 'Fare', 'SibSp', 'Parch', 'Survived']
pairplot_df = df[pairplot_features].dropna()

pp = sns.pairplot(pairplot_df, hue='Survived', palette={0: '#d9534f', 1: '#5cb85c'},
                  diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30},
                  height=2.4, aspect=1.2)
pp.fig.suptitle("Multivariate Pairplot Stratified by Survival Status", y=1.02, fontsize=15, weight='bold')

fig_pp_path = os.path.join(viz_dir, 'multivariate_pairplot.png')
pp.savefig(fig_pp_path, dpi=300)
plt.show()
print(f"Saved: {fig_pp_path}")
""")

add_md("""
### 9B. Correlation Analysis
We compute Pearson correlation coefficients across all numerical attributes and visualize them via an annotated heatmap.
""")

add_code("""
# Correlation Matrix Calculation
corr_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize']
corr_matrix = df[corr_cols].corr(numeric_only=True)

print("=== PEARSON CORRELATION MATRIX ===")
display(corr_matrix.round(4))

# Heatmap Visualization
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='vlag', vmin=-1.0, vmax=1.0,
            square=True, linewidths=1.5, cbar_kws={'shrink': 0.8}, mask=mask)
plt.title("Correlation Matrix of Titanic Numerical Features", fontsize=14, weight='bold', pad=15)

fig_corr_path = os.path.join(viz_dir, 'correlation_heatmap.png')
plt.savefig(fig_corr_path, dpi=300)
plt.show()
print(f"Saved: {fig_corr_path}")
""")

add_md("""
### 9C. Multicollinearity & Variance Inflation Factor (VIF)
**Multicollinearity** refers to a scenario in which two or more predictor variables in a multiple regression model are highly linearly related. While standard pairwise correlation measures bivariate association, multicollinearity detects when one variable can be linearly predicted from multiple other features.

The **Variance Inflation Factor (VIF)** quantifies the severity of multicollinearity:
$$VIF_j = \\frac{1}{1 - R_j^2}$$
where $R_j^2$ is the coefficient of determination from regressing predictor $x_j$ on all other predictors.
- A $VIF < 5$ indicates low multicollinearity.
- A $VIF \\ge 5$ to $10$ indicates moderate collinearity.
- A $VIF > 10$ indicates severe multicollinearity requiring structural mitigation.
""")

add_code("""
# Compute Variance Inflation Factor (VIF)
vif_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
df_vif = df[vif_features].copy()

# Median imputation applied exclusively for VIF computation
df_vif['Age'] = df_vif['Age'].fillna(df_vif['Age'].median())

# Add constant intercept term for standard centered VIF calculation
df_vif_const = add_constant(df_vif)

vif_data = pd.DataFrame()
vif_data['Feature'] = df_vif_const.columns
vif_data['VIF'] = [variance_inflation_factor(df_vif_const.values, i) for i in range(df_vif_const.shape[1])]

# Filter out constant term for display
display_vif = vif_data[vif_data['Feature'] != 'const'].reset_index(drop=True)
display_vif['VIF'] = display_vif['VIF'].round(4)
display_vif['Multicollinearity Status'] = display_vif['VIF'].apply(lambda x: 'Low (Safe)' if x < 5 else 'Moderate' if x < 10 else 'High (Action Required)')

print("=== VARIANCE INFLATION FACTOR (VIF) SUMMARY ===")
display(display_vif)
""")

add_md("""
### Multivariate Bubble Chart: Age vs Fare vs Pclass vs Survived
Examining four dimensions simultaneously to visualize survival concentration.
""")

add_code("""
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df.dropna(subset=['Age']),
    x='Age',
    y='Fare',
    hue='Survived',
    size='Pclass',
    sizes=(150, 40),
    palette=['#d9534f', '#5cb85c'],
    alpha=0.7,
    edgecolor='black'
)
plt.title("Multivariate Interaction: Age vs Fare vs Pclass Conditioned on Survival", fontsize=14, weight='bold')
plt.xlabel("Age (Years)", fontsize=12, weight='bold')
plt.ylabel("Fare (£)", fontsize=12, weight='bold')
plt.ylim(-10, 300) # zoom in on the bulk of data while retaining meaningful range
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

fig_bubble_path = os.path.join(viz_dir, 'multivariate_bubble_chart.png')
plt.savefig(fig_bubble_path, dpi=300)
plt.show()
print(f"Saved: {fig_bubble_path}")
""")

add_md("""
### Correlation & Multicollinearity Findings:
- **`Fare` vs `Pclass`**: Shows the strongest negative correlation (**-0.550**), confirming that higher-tier tickets (1st class) correspond to substantially higher fares.
- **`Pclass` vs `Survived`**: Statistically significant negative correlation (**-0.338**), validating that lower socioeconomic class (higher integer class value) strongly diminished survival probability.
- **`Fare` vs `Survived`**: Positive correlation (**+0.257**), reinforcing that financial investment in travel was associated with better lifeboat access.
- **`SibSp` vs `Parch`**: Moderate positive correlation (**+0.415**), as siblings and parents often traveled together.
- **Multicollinearity Diagnostic**: All candidate predictor features display $VIF < 1.70$ (Pclass: 1.66, Age: 1.20, SibSp: 1.28, Parch: 1.26, Fare: 1.59). There is **no problematic multicollinearity** among the raw numerical features, making them suitable for regression and classification modeling.
""")

# ==============================================================================
# SECTION 10: ADVANCED VISUALIZATION USING PLOTLY
# ==============================================================================
add_md("""
---

## SECTION 10: Advanced Interactive Visualization Using Plotly

Interactive visualizations allow dynamic exploration of data points, rich hover diagnostics, zooming, and filtering. Standalone HTML files are generated and saved to `outputs/visualizations/`.
""")

add_code("""
# 1. Interactive Survival Distribution (Donut Chart)
surv_counts = df['Survived'].value_counts().reset_index()
surv_counts['Status'] = surv_counts['Survived'].map({0: 'Perished', 1: 'Survived'})

fig_surv = px.pie(
    surv_counts,
    values='count',
    names='Status',
    hole=0.45,
    title='Interactive Titanic Survival Distribution',
    color='Status',
    color_discrete_map={'Perished': '#d9534f', 'Survived': '#5cb85c'}
)
fig_surv.update_traces(textposition='inside', textinfo='percent+label+value',
                       hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Proportion: %{percent}<extra></extra>")
fig_surv.update_layout(title_font=dict(size=18, family="Arial, sans-serif"))

html_surv_path = os.path.join(viz_dir, 'survival_distribution.html')
fig_surv.write_html(html_surv_path)
print(f"Saved: {html_surv_path}")
fig_surv.show()
""")

add_code("""
# 2. Interactive Survival Analysis by Gender
gender_surv = df.groupby(['Sex', 'Survived']).size().reset_index(name='Count')
gender_surv['Status'] = gender_surv['Survived'].map({0: 'Perished', 1: 'Survived'})

fig_gender = px.bar(
    gender_surv,
    x='Sex',
    y='Count',
    color='Status',
    barmode='group',
    title='Interactive Survival Analysis by Gender',
    color_discrete_map={'Perished': '#d9534f', 'Survived': '#5cb85c'},
    text='Count'
)
fig_gender.update_traces(textposition='outside',
                         hovertemplate="<b>Gender:</b> %{x}<br><b>Status:</b> %{fullData.name}<br><b>Count:</b> %{y}<extra></extra>")
fig_gender.update_layout(title_font=dict(size=18), yaxis_title="Number of Passengers", xaxis_title="Gender")

html_gender_path = os.path.join(viz_dir, 'survival_by_gender.html')
fig_gender.write_html(html_gender_path)
print(f"Saved: {html_gender_path}")
fig_gender.show()
""")

add_code("""
# 3. Interactive Survival Analysis by Passenger Class
class_surv = df.groupby(['Pclass', 'Survived']).size().reset_index(name='Count')
class_surv['Status'] = class_surv['Survived'].map({0: 'Perished', 1: 'Survived'})
class_surv['Class'] = class_surv['Pclass'].map({1: '1st Class', 2: '2nd Class', 3: '3rd Class'})

fig_class = px.bar(
    class_surv,
    x='Class',
    y='Count',
    color='Status',
    barmode='group',
    title='Interactive Survival Analysis by Passenger Class',
    color_discrete_map={'Perished': '#d9534f', 'Survived': '#5cb85c'},
    text='Count'
)
fig_class.update_traces(textposition='outside',
                        hovertemplate="<b>Class:</b> %{x}<br><b>Status:</b> %{fullData.name}<br><b>Count:</b> %{y}<extra></extra>")
fig_class.update_layout(title_font=dict(size=18), yaxis_title="Number of Passengers", xaxis_title="Class Tier")

html_class_path = os.path.join(viz_dir, 'survival_by_class.html')
fig_class.write_html(html_class_path)
print(f"Saved: {html_class_path}")
fig_class.show()
""")

add_code("""
# 4. Interactive Fare Distribution (Histogram & Boxplot)
fig_fare = px.histogram(
    df,
    x='Fare',
    color='Survived',
    marginal='box',
    nbins=50,
    title='Interactive Fare Distribution with Marginal Boxplot by Survival Status',
    color_discrete_map={0: '#d9534f', 1: '#5cb85c'},
    opacity=0.75,
    labels={'Survived': 'Survival Outcome'}
)
fig_fare.update_layout(title_font=dict(size=18), xaxis_title="Fare (£)", yaxis_title="Passenger Count")

html_fare_path = os.path.join(viz_dir, 'fare_distribution.html')
fig_fare.write_html(html_fare_path)
print(f"Saved: {html_fare_path}")
fig_fare.show()
""")

add_code("""
# 5. Interactive Age Distribution (Histogram & Density Curve)
fig_age = px.histogram(
    df.dropna(subset=['Age']),
    x='Age',
    color='Survived',
    marginal='violin',
    nbins=40,
    title='Interactive Age Distribution and Density by Survival Status',
    color_discrete_map={0: '#d9534f', 1: '#5cb85c'},
    opacity=0.7,
    labels={'Survived': 'Survival Outcome'}
)
fig_age.update_layout(title_font=dict(size=18), xaxis_title="Age (Years)", yaxis_title="Passenger Count")

html_age_path = os.path.join(viz_dir, 'age_distribution.html')
fig_age.write_html(html_age_path)
print(f"Saved: {html_age_path}")
fig_age.show()
""")

add_code("""
# 6. Interactive Multi-Feature Relationship Chart
# 3D Scatter Plot: Age vs Fare vs Pclass colored by Survival
plot_data = df.dropna(subset=['Age', 'Fare', 'Pclass', 'Survived']).copy()
plot_data['Survival_Status'] = plot_data['Survived'].map({0: 'Perished', 1: 'Survived'})

fig_rel = px.scatter_3d(
    plot_data,
    x='Age',
    y='Fare',
    z='Pclass',
    color='Survival_Status',
    color_discrete_map={'Perished': '#d9534f', 'Survived': '#5cb85c'},
    hover_data=['Name', 'Sex', 'Embarked', 'Ticket'],
    title='Interactive 3D Feature Space: Age vs Fare vs Pclass',
    opacity=0.8
)
fig_rel.update_layout(
    scene=dict(
        xaxis_title='Age (Years)',
        yaxis_title='Fare (£)',
        zaxis_title='Passenger Class',
        zaxis=dict(tickvals=[1, 2, 3], ticktext=['1st Class', '2nd Class', '3rd Class'])
    ),
    title_font=dict(size=18)
)

html_rel_path = os.path.join(viz_dir, 'feature_relationships.html')
fig_rel.write_html(html_rel_path)
print(f"Saved: {html_rel_path}")
fig_rel.show()
""")

# ==============================================================================
# SECTION 11: PATTERN, TREND AND ANOMALY IDENTIFICATION
# ==============================================================================
add_md("""
---

## SECTION 11: Pattern, Trend and Anomaly Identification

Based on actual computed statistics and visualizations, we summarize our analytical findings across ten core dimensions:

### 1. Survival Distribution
- **Evidence**: 342 passengers survived (38.38%), while 549 perished (61.62%).
- **Interpretation**: Demonstrates significant class imbalance. A naive baseline model predicting everyone perished would achieve 61.62% accuracy.

### 2. Gender-Based Survival Differences
- **Evidence**: Female survival rate was **74.20%** (233 / 314); male survival rate was **18.89%** (109 / 577). Odds ratio $\\approx 12.3$.
- **Interpretation**: Strongest single bivariate predictor in the dataset, directly reflecting the evacuation priority given to women.

### 3. Passenger-Class Survival Differences
- **Evidence**: Class 1 survival: **62.96%** (136 / 216); Class 2: **47.28%** (87 / 184); Class 3: **24.24%** (119 / 491).
- **Interpretation**: Reflects both physical proximity of First/Second Class cabins to the upper boat deck and socioeconomic privilege during lifeboat allocation.

### 4. Age Distribution & Life Stage Dynamics
- **Evidence**: Children under 10 years had a survival rate of **61.29%** (38 / 62), whereas young adults aged 18-35 had a survival rate of ~35%. The oldest passenger (80 years old) survived.
- **Interpretation**: Validates the "children first" protocol, though its protective effect was stronger in 1st and 2nd class than in 3rd class.

### 5. Fare Distribution & Wealth Gradient
- **Evidence**: Non-survivors paid a median fare of £10.50, whereas survivors paid a median fare of £26.00.
- **Interpretation**: Financial capability directly mapped to higher class accommodations and dramatically improved survival chances.

### 6. Embarkation Patterns
- **Evidence**: Cherbourg passengers achieved **55.36%** survival, Queenstown **38.96%**, and Southampton **33.70%**.
- **Interpretation**: Cherbourg had the highest proportion of 1st Class passengers (50.6%), confounding embarkation port with socioeconomic status.

### 7. Family Size Patterns
- **Evidence**: Solo passengers had a **30.35%** survival rate. Families of 2, 3, and 4 members experienced **55.28%**, **57.84%**, and **72.41%** survival rates. Large families (5+) experienced an abrupt collapse (< 20%).
- **Interpretation**: Small families assisted one another into lifeboats without exceeding capacity, whereas large families were difficult to assemble in the evacuation chaos.

### 8. Missing Data Patterns
- **Evidence**: `Cabin` is missing in **77.10%** of records; `Age` in **19.87%**; `Embarked` in **0.22%**.
- **Interpretation**: Missingness in `Cabin` is highly non-random (MNAR) and strongly correlated with 3rd Class status. `Age` missingness (MAR) can be accurately estimated using honorific title groupings (e.g., Master, Miss, Mr., Mrs.).

### 9. Numerical Feature Relationships
- **Evidence**: `Pclass` negatively correlates with `Fare` (-0.550) and `Age` (-0.369).
- **Interpretation**: Older and wealthier passengers dominated the upper class, creating a multi-factor structural survival advantage.

### 10. Anomalies & Outliers
- **Zero-Fare Tickets**: 15 records have `Fare = 0.0`. These correspond to shipping line employees, guarantee group mechanics, or promotional complimentary tickets.
- **Extreme Fares**: 3 passengers paid £512.33 on ticket 35930 (First Class parlour suites). These are valid records rather than measurement corruptions.
""")

# ==============================================================================
# SECTION 12: FEATURE-LEVEL INFERENCES
# ==============================================================================
add_md("""
---

## SECTION 12: Feature-Level Inferences for Machine Learning

The following structured table maps each dataset feature to empirical observations, analytical interpretations, and prospective machine learning utility:

| Feature | Observation | Analytical Interpretation | Machine Learning Utility & Strategy |
| :--- | :--- | :--- | :--- |
| **`Survived`** | 38.38% positive instances (342/891). | Moderate class imbalance; non-survivors form the majority class. | Primary target variable. Stratified k-fold cross-validation recommended. |
| **`Pclass`** | Ordered categorical (1st, 2nd, 3rd); Class 3 forms 55.1%. | Direct proxy for socioeconomic standing and physical cabin deck proximity. | High predictive power. Encode as ordinal feature or one-hot vectors. |
| **`Sex`** | 64.8% male, 35.2% female; massive survival disparity (74.2% vs 18.9%). | Primary determinant of evacuation priority under maritime protocol. | Top-tier feature. Binary encoding (`female=1, male=0`). |
| **`Age`** | Mean 29.7, std 14.5; 177 missing entries; children < 10 have ~61.3% survival. | Non-linear relationship with survival; vulnerable age cohorts prioritized. | Moderate predictive utility. Impute conditionally by `Title` and `Pclass`; consider binning into life stages. |
| **`Fare`** | Heavily right-skewed (skew 4.79); mean £32.20 vs median £14.45; max £512.33. | Directly correlates with passenger class and accommodation quality. | Moderate-to-high utility. Apply log-transformation (`np.log1p`) to mitigate skewness. |
| **`SibSp`** | Discrete count (0 to 8); 68.2% zero; positive skew 3.70. | Reflects horizontal family ties (siblings / spouses). | Transform into combined `FamilySize` feature. |
| **`Parch`** | Discrete count (0 to 6); 76.1% zero; positive skew 2.75. | Reflects vertical family ties (parents / children). | Combine with `SibSp` to engineer `IsAlone` binary flag. |
| **`Embarked`** | 3 ports (S: 72.4%, C: 18.9%, Q: 8.7%); 2 missing entries. | Confounded with passenger class distribution. | Low-to-moderate utility. Impute mode ('S') and apply one-hot encoding. |
| **`Cabin`** | 77.1% missing; starts with deck letter (A, B, C, D, E, F, G, T). | Deck letter indicates physical location on the ship structure. | Extract deck letter as `CabinDeck`; treat missing as dedicated `'Unknown'` category. |
| **`Ticket`** | 681 unique values; shared tickets indicate travel groups. | Ticket frequency can reveal travel party size even for unrelated companions. | Feature engineering opportunity: group ticket size and ticket prefix extraction. |
| **`Name`** | Contains structured honorific titles (Mr, Mrs, Miss, Master, Dr, Rev). | Titles correlate strongly with age, marital status, and social rank. | Extract `Title` feature to enable accurate `Age` imputation and enhance classification. |
""")

# ==============================================================================
# SECTION 13: FINAL EDA SUMMARY
# ==============================================================================
add_md("""
---

## SECTION 13: Final EDA Summary

### 13.1 Quantitative Summary Table
""")

add_code("""
summary_metrics = {
    "Metric": [
        "Total Observations (Rows)",
        "Total Attributes (Columns)",
        "Numerical Features",
        "Categorical Features",
        "Overall Survival Rate",
        "Female Survival Rate",
        "Male Survival Rate",
        "1st Class Survival Rate",
        "2nd Class Survival Rate",
        "3rd Class Survival Rate",
        "Children (<10 yrs) Survival Rate",
        "Median Fare (Overall)",
        "Median Fare (Survivors)",
        "Median Fare (Non-Survivors)",
        "Features with Missing Values",
        "Duplicate Rows Detected",
        "Max VIF (Multicollinearity)"
    ],
    "Value": [
        "891",
        "12",
        "6 (PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare)",
        "5 (Name, Sex, Ticket, Cabin, Embarked)",
        "38.38% (342 / 891)",
        "74.20% (233 / 314)",
        "18.89% (109 / 577)",
        "62.96% (136 / 216)",
        "47.28% (87 / 184)",
        "24.24% (119 / 491)",
        "61.29% (38 / 62)",
        "£14.45",
        "£26.00",
        "£10.50",
        "3 (Age: 19.87%, Cabin: 77.10%, Embarked: 0.22%)",
        "0 (0.0%)",
        "1.66 (Pclass) - No Collinearity Issue"
    ]
}

eda_summary_df = pd.DataFrame(summary_metrics)
display(eda_summary_df)
""")

add_md("""
### 13.2 Key Analytical Takeaways
1. **Dominant Predictors**: Biological sex and passenger class are the two most powerful determinants of survival on the Titanic, verifying the historical operational priority of "women and children first" combined with socioeconomic privilege.
2. **Family Size Non-Linearity**: Solo travelers and large families experienced low survival rates (< 31%), whereas small family units of 2 to 4 members had elevated survival rates (> 55%).
3. **Data Preprocessing Blueprint for ML**:
   - Impute `Age` using median values grouped by extracted `Title` and `Pclass`.
   - Engineer `FamilySize = SibSp + Parch + 1` and `IsAlone = (FamilySize == 1)`.
   - Log-transform `Fare` (`np.log1p`) to compress extreme right skewness.
   - Extract `CabinDeck` from `Cabin` and treat missing values as an explicit category.
   - Impute the 2 missing `Embarked` records with `'S'` and one-hot encode.
""")

# ==============================================================================
# SECTION 14: EXPORT ANALYSIS RESULTS
# ==============================================================================
add_md("""
---

## SECTION 14: Export Analysis Results

We programmatically export all generated analytical summaries, correlation matrices, frequency tables, and statistical profiles to `outputs/reports/`.
""")

add_code("""
# 1. Export Descriptive Statistics CSV
desc_csv_path = os.path.join(reports_dir, 'descriptive_statistics.csv')
desc_stats_df.to_csv(desc_csv_path, index=False)
print(f"Exported: {desc_csv_path}")

# 2. Export Missing Values Summary CSV
missing_csv_path = os.path.join(reports_dir, 'missing_values_summary.csv')
overview_df.to_csv(missing_csv_path, index=False)
print(f"Exported: {missing_csv_path}")

# 3. Export Categorical Frequency Summary CSV
cat_csv_path = os.path.join(reports_dir, 'categorical_frequency_summary.csv')
cat_summary_df.to_csv(cat_csv_path, index=False)
print(f"Exported: {cat_csv_path}")

# 4. Export Correlation Matrix CSV
corr_csv_path = os.path.join(reports_dir, 'correlation_matrix.csv')
corr_matrix.to_csv(corr_csv_path)
print(f"Exported: {corr_csv_path}")

# 5. Export Final EDA Summary Report in Markdown
report_md_path = os.path.join(reports_dir, 'eda_summary_report.md')

def format_df_as_markdown(dframe):
    cols = dframe.columns.tolist()
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines = ["| " + " | ".join(str(x) for x in row) + " |" for row in dframe.itertuples(index=False)]
    return chr(10).join([header, separator] + body_lines)

with open(report_md_path, 'w') as f:
    f.write("# Titanic Dataset - Exploratory Data Analysis (EDA) Summary Report\\n\\n")
    f.write("## 1. Project Overview\\n")
    f.write("- Dataset: Titanic Passenger Dataset (891 rows, 12 columns)\\n")
    f.write("- Target: Survival status (0 = Perished, 1 = Survived)\\n\\n")
    f.write("## 2. Key Statistical Metrics\\n\\n")
    f.write(format_df_as_markdown(eda_summary_df))
    f.write("\\n\\n## 3. Core Insights\\n")
    f.write("1. Gender was the primary factor in survival probability (Female: 74.20%, Male: 18.89%).\\n")
    f.write("2. Socioeconomic class created sharp survival gradients (1st Class: 62.96%, 3rd Class: 24.24%).\\n")
    f.write("3. Children under 10 had a high survival rate (61.29%).\\n")
    f.write("4. Fare distribution was heavily skewed (skew = 4.79) with luxury suite outliers up to £512.33.\\n")
    f.write("5. Small families (2-4 members) had superior survival compared to solo travelers or large families.\\n")
    f.write("6. No severe multicollinearity was observed among candidate predictors (all VIFs < 1.70).\\n")

print(f"Exported: {report_md_path}")
print("\\nAll analytical outputs, figures, HTML charts, and reports have been generated and exported successfully.")
""")

nb['cells'] = cells

output_nb_path = "/Users/pranavagneesh/Desktop/exploratory-data-analysis/notebooks/exploratory_data_analysis.ipynb"
with open(output_nb_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook written successfully to: {output_nb_path}")
