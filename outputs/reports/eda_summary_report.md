# Titanic Dataset - Exploratory Data Analysis (EDA) Summary Report

## 1. Project Overview
- Dataset: Titanic Passenger Dataset (891 rows, 12 columns)
- Target: Survival status (0 = Perished, 1 = Survived)

## 2. Key Statistical Metrics

| Metric | Value |
| --- | --- |
| Total Observations (Rows) | 891 |
| Total Attributes (Columns) | 12 |
| Numerical Features | 6 (PassengerId, Survived, Pclass, Age, SibSp, Parch, Fare) |
| Categorical Features | 5 (Name, Sex, Ticket, Cabin, Embarked) |
| Overall Survival Rate | 38.38% (342 / 891) |
| Female Survival Rate | 74.20% (233 / 314) |
| Male Survival Rate | 18.89% (109 / 577) |
| 1st Class Survival Rate | 62.96% (136 / 216) |
| 2nd Class Survival Rate | 47.28% (87 / 184) |
| 3rd Class Survival Rate | 24.24% (119 / 491) |
| Children (<10 yrs) Survival Rate | 61.29% (38 / 62) |
| Median Fare (Overall) | £14.45 |
| Median Fare (Survivors) | £26.00 |
| Median Fare (Non-Survivors) | £10.50 |
| Features with Missing Values | 3 (Age: 19.87%, Cabin: 77.10%, Embarked: 0.22%) |
| Duplicate Rows Detected | 0 (0.0%) |
| Max VIF (Multicollinearity) | 1.66 (Pclass) - No Collinearity Issue |

## 3. Core Insights
1. Gender was the primary factor in survival probability (Female: 74.20%, Male: 18.89%).
2. Socioeconomic class created sharp survival gradients (1st Class: 62.96%, 3rd Class: 24.24%).
3. Children under 10 had a high survival rate (61.29%).
4. Fare distribution was heavily skewed (skew = 4.79) with luxury suite outliers up to £512.33.
5. Small families (2-4 members) had superior survival compared to solo travelers or large families.
6. No severe multicollinearity was observed among candidate predictors (all VIFs < 1.70).
