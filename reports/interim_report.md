# Interim Progress Report: Bati Bank Credit Risk Challenge

## 1. Understanding and Defining the Business Objective

**Business Goal and Context**
Bati Bank is partnering with an eCommerce platform to launch a "Buy-Now-Pay-Later" (BNPL) service. The objective of this project is to develop a credit scoring model that accurately qualifies customers for this service. By predicting credit risk, the bank can optimize loan approvals, determine appropriate credit limits, and set competitive loan terms while minimizing default rates.

**Necessity of a Proxy Target Variable**
The primary challenge with the provided Xente transaction dataset is the absence of a direct "default" label. To train a supervised machine learning model for credit scoring, we must infer creditworthiness from transactional behavior. Therefore, a proxy target variable must be constructed. While necessary, this introduces business risks—if the proxy does not perfectly correlate with actual defaults, the model might incorrectly deny credit to valid customers (false positives) or approve risky ones (false negatives), directly impacting the BNPL service's profitability and customer satisfaction.

**Basel II Compliance and Model Choice**
As a financial institution, Bati Bank operates under stringent regulatory frameworks, notably Basel II. Basel II emphasizes rigorous risk measurement and capital adequacy, requiring models to be transparent, well-documented, and highly interpretable. This significantly shapes our modeling choices. While "black-box" models like Gradient Boosting often yield the highest predictive performance, they lack the transparency required by regulators. Consequently, interpretable models, such as Logistic Regression combined with Weight of Evidence (WoE) transformations, are heavily favored to ensure auditability and fairness in lending decisions.

**Core Innovation: RFM to Credit Risk**
The core innovation of this project lies in transforming standard RFM (Recency, Frequency, Monetary) behavioral patterns into a predictive credit risk signal. By analyzing how recently, how often, and how much a customer transacts, we can construct a proxy risk probability score that serves as the foundation for our credit decisions.

---

## 2. Discussion of Completed Work and Initial Analysis

### Task 1: Business Understanding & Repository Setup
Our initial phase focused on structuring the project for reproducibility and regulatory compliance. We established a standardized project layout with clear separation between exploratory notebooks (`notebooks/`) and production code (`src/`). A `task-1` branch was utilized for version control. We conducted a literature review on credit scoring and Basel II, concluding that our approach must carefully balance the trade-off between the interpretability of linear models and the raw performance of ensemble methods.

### Task 2: Exploratory Data Analysis (EDA)
We executed a comprehensive EDA on the Xente transaction records to understand the underlying data distribution and quality. 

- **Data Overview:** The dataset contains transaction records with key columns including `TransactionId`, `AccountId`, `Amount`, `Value`, `ProductCategory`, `ChannelId`, `PricingStrategy`, and `FraudResult`.
- **Summary Statistics:** Analysis of numerical features like `Amount` and `Value` revealed highly skewed distributions with significant variance, indicating a wide range of transaction sizes.
- **Distribution Analysis:** 
  - *Numerical:* Histograms and KDE plots confirmed the right-skewness of `Amount` and `Value`. 
  - *Categorical:* Bar charts showed that certain product categories (e.g., financial services and airtime) dominate the transaction volume.
- **Correlation Analysis:** A correlation heatmap demonstrated a near-perfect positive correlation between `Amount` and `Value`, suggesting multicollinearity that must be addressed during feature engineering.
- **Missing Values & Outliers:** Missing values in the `Amount` feature were identified and imputed using the median of the respective `ProductCategory`. Box plots clearly highlighted the presence of extreme outliers in transaction amounts, which represent anomalous or potentially high-risk behavior.

**Top 5 EDA Insights Guiding Feature Engineering:**
1. **Skewness:** The extreme right-skew in `Amount` and `Value` necessitates log transformations or robust scaling before modeling.
2. **Multicollinearity:** `Amount` and `Value` are highly correlated; one may need to be dropped or combined into a single feature to satisfy Logistic Regression assumptions.
3. **Categorical Imbalance:** The dominance of a few product categories suggests that rare categories should be grouped to prevent overfitting.
4. **Outlier Impact:** The significant outliers must be carefully handled (e.g., via capping or robust scaling) so they do not disproportionately influence the model weights.
5. **Categorical Encoding:** Features like `PricingStrategy` are numerical codes but represent categories, requiring one-hot or target encoding.

*(Note: Detailed visualizations supporting these claims are available in the `notebooks/02_credit_risk_eda.ipynb` file.)*

---

## 3. Next Steps and Key Areas of Focus

Our roadmap for the remaining tasks is structured as follows:

- **Task 3: Feature Engineering Pipeline (`src/data_processing.py`)**
  We will construct a robust preprocessing pipeline. This includes aggregating transactional data, extracting datetime features (e.g., hour of day, day of week), performing categorical encoding, handling missing values, and applying normalization/standardization. Crucially, we will implement Weight of Evidence (WoE) and Information Value (IV) transformations (using libraries like `xverse` or `woe`) to maximize feature interpretability for regulatory compliance.

- **Task 4: Proxy Target Variable Construction**
  We will calculate RFM metrics for each user and apply K-Means clustering to segment customers into risk categories. This segmentation will be binarized to create the `is_high_risk` proxy target variable, establishing the foundation for our supervised learning models.

- **Task 5: Model Training and Experiment Tracking**
  We will train various models including Logistic Regression (as our interpretable baseline), Decision Trees, Random Forests, and Gradient Boosting. We will utilize MLflow for experiment tracking and hyperparameter tuning. Model evaluation will focus on metrics critical for imbalanced classification: Accuracy, Precision, Recall, F1-score, and ROC-AUC.

- **Task 6: Deployment and MLOps**
  The final model will be served via a FastAPI application. We will containerize the solution using Docker and set up a CI/CD pipeline using GitHub Actions to automate testing and deployment, ensuring a production-ready credit scoring system.
