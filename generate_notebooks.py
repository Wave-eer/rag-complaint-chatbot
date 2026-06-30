import nbformat as nbf

def create_eda_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = [
        nbf.v4.new_markdown_cell("# Exploratory Data Analysis (EDA)\n\nThis notebook covers data summarization, quality assessment, univariate/bivariate analysis, geographic trends, and outlier detection."),
        nbf.v4.new_code_cell("""import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\ndf = pd.read_csv('../data/insurance_data.csv')\ndf.head()"""),
        nbf.v4.new_markdown_cell("## Data Summarization & Quality Assessment"),
        nbf.v4.new_code_cell("""print(df.info())\nprint(df.describe())\nprint("Missing values:\\n", df.isnull().sum())"""),
        nbf.v4.new_markdown_cell("## Univariate & Bivariate Analysis"),
        nbf.v4.new_code_cell("""plt.figure(figsize=(10, 6))\nsns.histplot(df['Premium'], kde=True)\nplt.title('Distribution of Premium')\nplt.show()"""),
        nbf.v4.new_code_cell("""plt.figure(figsize=(10, 6))\nsns.boxplot(x='Province', y='TotalClaims', data=df)\nplt.title('Total Claims by Province')\nplt.show()"""),
        nbf.v4.new_markdown_cell("## Loss Ratio Analysis"),
        nbf.v4.new_code_cell("""df['LossRatio'] = df['TotalClaims'] / df['Premium']\nplt.figure(figsize=(10, 6))\nsns.barplot(x='VehicleType', y='LossRatio', data=df)\nplt.title('Loss Ratio by Vehicle Type')\nplt.show()"""),
    ]
    nb['cells'] = cells
    with open('notebooks/01_EDA.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Created notebooks/01_EDA.ipynb")

def create_hypothesis_notebook():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("# A/B Hypothesis Testing\n\nTesting 4 null hypotheses:\n1. No risk differences across provinces\n2. No risk differences between zip codes\n3. No significant margin differences between zip codes\n4. No significant risk difference between women and men"),
        nbf.v4.new_code_cell("""import pandas as pd\nfrom scipy import stats\n\ndf = pd.read_csv('../data/insurance_data.csv')"""),
        nbf.v4.new_markdown_cell("## 1. Risk differences across provinces (ANOVA / Chi-squared)"),
        nbf.v4.new_code_cell("""# Example ANOVA for TotalClaims across Provinces\ngroups = [group['TotalClaims'].values for name, group in df.groupby('Province')]\nf_stat, p_val = stats.f_oneway(*groups)\nprint(f"F-statistic: {f_stat:.4f}, p-value: {p_val:.4f}")""")
    ]
    nb['cells'] = cells
    with open('notebooks/02_Hypothesis_Testing.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Created notebooks/02_Hypothesis_Testing.ipynb")

def create_modeling_notebook():
    nb = nbf.v4.new_notebook()
    cells = [
        nbf.v4.new_markdown_cell("# Statistical Modeling & Risk-Based Pricing"),
        nbf.v4.new_code_cell("""import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.ensemble import RandomForestRegressor\nimport xgboost as xgb\nfrom sklearn.metrics import mean_squared_error, r2_score\nimport shap\n\ndf = pd.read_csv('../data/insurance_data.csv')"""),
        nbf.v4.new_markdown_cell("## Data Prep"),
        nbf.v4.new_code_cell("""df_model = df[df['TotalClaims'] > 0].copy()\nfeatures = ['Age', 'PolicyDuration', 'Premium']\nX = df_model[features]\ny = df_model['TotalClaims']\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"""),
        nbf.v4.new_markdown_cell("## Modeling"),
        nbf.v4.new_code_cell("""models = {\n    'Linear Regression': LinearRegression(),\n    'Random Forest': RandomForestRegressor(random_state=42),\n    'XGBoost': xgb.XGBRegressor(random_state=42)\n}\n\nfor name, model in models.items():\n    model.fit(X_train, y_train)\n    preds = model.predict(X_test)\n    print(f"{name} - RMSE: {mean_squared_error(y_test, preds, squared=False):.2f}, R2: {r2_score(y_test, preds):.2f}")""")
    ]
    nb['cells'] = cells
    with open('notebooks/03_Statistical_Modeling.ipynb', 'w') as f:
        nbf.write(nb, f)
    print("Created notebooks/03_Statistical_Modeling.ipynb")

if __name__ == '__main__':
    create_eda_notebook()
    create_hypothesis_notebook()
    create_modeling_notebook()
