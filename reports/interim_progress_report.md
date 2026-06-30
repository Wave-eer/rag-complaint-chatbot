# Interim Progress Report: AlphaCare Insurance Solutions (ACIS) Analytics Project

## 1. Business Objective and Context
AlphaCare Insurance Solutions (ACIS), a South African auto-insurance company, is preparing for an aggressive growth phase. To succeed in a highly competitive market, ACIS must transition from traditional, intuition-based pricing models to an analytics-driven pricing strategy. By leveraging the 18-month historical claim dataset (February 2014 – August 2015), our primary objective is to discover 'low-risk' customer segments where premiums can be competitively reduced to attract new clients without sacrificing profitability. 

Two anchor metrics are central to evaluating this objective:
* **Loss Ratio (TotalClaims / TotalPremium):** Measures the proportion of premium income paid out in claims. A lower ratio indicates higher profitability and lower risk.
* **Margin (TotalPremium − TotalClaims):** Represents the raw financial return per policy. Optimization of this metric drives overall corporate growth.

## 2. Completed Work and Initial Analysis (Tasks 1 & 2)

### Exploratory Data Analysis (Task 1)
Our initial analysis focused on understanding the underlying patterns within the dataset to inform downstream modeling.
* **Data Summarization & Quality:** We computed descriptive statistics for key financial variables (e.g., TotalPremium, TotalClaims). Dtypes across categorical (e.g., VehicleType, Province), date, and numerical columns were verified. Missing values were identified, and an appropriate imputation strategy was applied to prevent downstream bias.
* **Univariate, Bivariate & Multivariate Analysis:** We examined the distributions of key features. Bivariate analysis revealed strong relationships between `TotalPremium` and `TotalClaims` across different `ZipCodes`.
* **Geographic Trends & Outlier Detection:** We compared cover types, premiums, and vehicle makes geographically across provinces. Furthermore, we identified and managed outliers in financial variables to ensure robust predictive modeling.
* **Guiding Questions Addressed:** Our analysis directly answered critical business questions, highlighting variations in the overall Loss Ratio across Provinces, Vehicle Types, and Genders, while also profiling the risk associated with different vehicle makes and models. *(Note: Please reference Notebook `01_EDA.ipynb` for detailed, properly labeled visualizations supporting these findings).*

### Data Version Control Pipeline (Task 2)
To ensure reproducibility and auditability—crucial elements in a highly regulated insurance environment—we implemented Data Version Control (DVC). We configured a remote storage pipeline and successfully tracked both the raw and cleaned versions of the dataset (`insurance_data.csv` and `insurance_data_cleaned.csv`). This pipeline guarantees that every statistical model and test can be traced back to the exact snapshot of data it was trained on, minimizing data drift risks.

## 3. Next Steps and Key Areas of Focus (Tasks 3 & 4)

Moving forward, our efforts will shift towards statistical validation and machine learning implementation.

### A/B Hypothesis Testing (Task 3)
We will conduct rigorous statistical testing using Chi-squared tests for categorical KPIs (Claim Frequency) and t-tests or z-tests for numerical KPIs (Claim Severity, Margin). We will evaluate four specific null hypotheses:
1. There are no risk differences across provinces.
2. There are no risk differences between zip codes.
3. There are no significant margin differences between zip codes.
4. There is no significant risk difference between women and men.

### Statistical Modeling & Risk-Based Pricing (Task 4)
Our modeling efforts will focus on two primary goals: 
1. Predicting claim severity accurately.
2. Optimizing premiums using the data-driven formula: 
   `Premium = P(claim) × Predicted Severity + Expense Loading + Profit Margin`

To achieve this, we will implement and evaluate three algorithms:
1. **Linear Regression** (Baseline)
2. **Random Forest** 
3. **XGBoost** 

Finally, to ensure our pricing models remain transparent and interpretable for business stakeholders and regulatory compliance, we will apply **SHAP** (or LIME) to identify and clearly explain the top 5–10 most influential risk features driving our premium pricing.
