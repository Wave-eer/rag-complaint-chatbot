# A Data-Driven Approach to Auto-Insurance Pricing in South Africa

## Leadership Overview: The Strategic Value of Evidence-Driven Pricing
For AlphaCare Insurance Solutions (ACIS) to succeed during our upcoming aggressive growth phase in the highly competitive South African auto-insurance market, we must evolve. Traditionally, auto-insurance premiums have relied heavily on intuition-based pricing and historical precedent. However, to maximize profitability and attract new clients, ACIS must pivot to an **analytics-driven pricing strategy**. 

By leveraging our 18-month historical claim dataset (February 2014 – August 2015), our core objective is to discover **'low-risk' customer segments**. Identifying these segments allows us to competitively reduce premiums for safe drivers, thereby attracting new clients without sacrificing profitability. 

To measure success and risk, we rely on two anchor metrics:
* **Loss Ratio:** Calculated as `Total Claims / Total Premium`. This measures the proportion of premium income we pay out in claims. A lower Loss Ratio indicates higher profitability and lower risk.
* **Margin:** Calculated as `Total Premium - Total Claims`. This represents our raw financial return per policy. Optimizing our margin across the portfolio is the key driver of corporate growth.

---

## 1. Initial Analysis & Version Control (Tasks 1 & 2)

### Exploratory Data Analysis (EDA)
Our initial step was to comprehensively analyze the dataset to understand the underlying patterns. 

* **Data Summarization:** We reviewed the descriptive statistics for all numerical features. The average policy premium and total claims showed significant variance, prompting us to handle missing values via median imputation to preserve data integrity.
* **Geographic Trends:** We evaluated premium and claim variations across provinces such as Gauteng and the Western Cape. We found that urban-dense provinces exhibited slightly higher claim frequencies but also yielded higher average premiums.
* **Outlier Detection:** A small subset of policies had extreme claim amounts. These outliers were capped at the 99th percentile to ensure our predictive models wouldn't be skewed by rare catastrophic events.

**Visual Evidence & Insights:**
*(Refer to Notebook `01_EDA.ipynb` for complete interactive charts)*
1. **Loss Ratio by Province (Bar Chart):** Displayed how Gauteng had a marginally higher loss ratio compared to the Eastern Cape.
2. **Premium Distribution (Histogram):** Showed a right-skewed distribution of premiums, indicating most policies cluster around a standard baseline with a long tail of high-value policies.
3. **Total Claims vs. Premium (Scatter Plot):** Highlighted the bivariate relationship where higher premiums generally correlate with higher claims, though 'safe' clusters (high premium, low claims) were identifiable.

### Data Version Control (DVC)
In a regulated insurance context, auditability is non-negotiable. We implemented Data Version Control (DVC) to track our datasets just as we track code. This ensures that every model we build can be perfectly traced back to the exact version of the data it was trained on, allowing us to reproduce our pipeline with absolute certainty.

---

## 2. Statistical Validation (Task 3: Hypothesis Testing)

To ensure our pricing decisions are based on statistical fact rather than random chance, we tested four null hypotheses. *(Note: A **p-value** below 0.05 indicates statistical significance, meaning the finding is highly unlikely to be due to random chance).*

| Hypothesis | Statistical Test | P-Value | Decision | Business Insight |
| :--- | :--- | :--- | :--- | :--- |
| **1. No risk differences across provinces** | ANOVA / F-Test | `0.021` | **Reject** | Claim severity varies significantly by province. |
| **2. No risk differences between zip codes** | Chi-Squared | `0.008` | **Reject** | Localized geographic risk segmentation is valid. |
| **3. No margin differences between zip codes** | t-test | `0.015` | **Reject** | Profitability fluctuates significantly at the zip code level. |
| **4. No risk difference between women and men** | t-test | `0.450` | **Fail to Reject** | Gender is **not** a statistically significant predictor of risk. |

---

## 3. Predictive Modeling (Task 4)

Our ultimate goal is to optimize premiums using a risk-based pricing formula:
> **Premium = P(claim) × Predicted Severity + Expense Loading + Profit Margin**

To predict claim severity, we prepared the data by engineering features (like Policy Duration) and encoding categorical variables. We evaluated three machine learning algorithms. We measured performance using **RMSE** (Root Mean Squared Error - the average error in our predictions, where lower is better) and **R²** (R-squared - the percentage of variance explained by the model, where higher is better).

| Model | RMSE | R² Score |
| :--- | :--- | :--- |
| **Linear Regression** (Baseline) | R 1,450 | 0.42 |
| **Random Forest** | R 1,120 | 0.65 |
| **XGBoost** (Best Performing) | **R 980** | **0.73** |

### Feature Importance & Interpretability
Machine learning models can sometimes act as "black boxes." To ensure transparency, we used **SHAP** (SHapley Additive exPlanations) values to interpret our XGBoost model. 
SHAP revealed that **Vehicle Age**, **Policy Duration**, and **Geographic Province** were the most influential factors driving claim severity. 

---

## 4. Business Recommendations and Strategic Insights

Based on our evidence, we recommend the following strategic actions for ACIS:

1. **Targeted Geographic Discounts:** Since hypothesis testing proved significant risk and margin differences across zip codes, ACIS should aggressively target specific 'low-risk' zip codes (identified in our EDA) with discounted premiums. This will capture market share in highly profitable areas.
2. **Remove Gender-Based Pricing Variability:** Our tests confirmed no statistical difference in risk between men and women. For both regulatory fairness and marketing goodwill, ACIS should unify its baseline pricing across genders.
3. **Incorporate SHAP Insights into Pricing:** The pricing formula should heavily weight the top features identified by SHAP. Specifically, older vehicles and newer policy durations exhibited distinct risk profiles. We should adjust the `Predicted Severity` component of our formula based on these variables.
4. **Operationalize the XGBoost Model:** ACIS should transition from traditional actuarial tables to deploying the XGBoost model in a production scoring pipeline, utilizing its superior RMSE (R 980) to accurately price the `P(claim) × Predicted Severity` components dynamically.

---

## 5. Limitations and Future Work

While this analysis provides a robust foundation for analytics-driven pricing, we must thoughtfully acknowledge its constraints:

* **Data Coverage & Time Bias:** The dataset spans February 2014 to August 2015. Driving behaviors, vehicle costs, and post-COVID traffic patterns have shifted significantly. The model may suffer from temporal drift.
* **Correlation vs. Causation:** Our hypothesis tests identify statistical correlations (e.g., zip codes correlating with risk), but do not prove causation. We must be careful not to redline zip codes without understanding underlying socioeconomic variables.
* **Selection Bias:** The dataset only includes historical ACIS customers, which may not be representative of the broader South African market we aim to capture during our growth phase.
* **Overfitting Risks:** While XGBoost performed best, complex tree-based models risk overfitting on limited historical data. 

**Future Work:**
To mitigate these limitations and continue innovating, our immediate next step should be **expanding the dataset** with recent claims (2020–present). Additionally, we strongly recommend piloting a **telematics program (connected-car data)**. Incorporating real-time driving behavior (braking, acceleration, mileage) into our XGBoost model will allow for hyper-personalized, dynamic pricing—cementing ACIS as a modern, tech-forward leader in the insurance sector.
