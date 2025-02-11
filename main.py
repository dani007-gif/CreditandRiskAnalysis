import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve )
import matplotlib.pyplot as plt
import seaborn as sbn

#Convert dataset to csv
xlsx_file = r"E:\Projects\git_projects\CreditandRiskAnalysis\Credit data.xlsx"
csv_file = r"E:\Projects\git_projects\CreditandRiskAnalysis\Credit data2.csv"

# Read Excel file and convert to CSV
df = pd.read_excel(xlsx_file, engine="openpyxl")  # Ensure openpyxl is installed
df.to_csv(csv_file, index=False)

print(f"File converted and saved as {csv_file}")

#Loading the dataset

ds = pd.read_csv("E:\Projects\git_projects\CreditandRiskAnalysis\Credit data2.csv")
# Dataset overview
print("Dataset Overview:")
print(ds.info())
print("\nMissing Values Summary:")
print(ds.isnull().sum())
# Summary Statistics for Numerical Features
print("\nNumerical Feature Statistics:")
print(ds.describe())


# Target Class Distribution
plt.figure(figsize=(6, 4))
sbn.countplot(x='SeriousDlqin2yrs', data=ds)
plt.title('Target Class Distribution')
plt.show()

# Correlation Matrix
plt.figure(figsize=(12, 8))
sbn.heatmap(ds.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

#Data Preprocessing

# Handle Missing Values
ds['MonthlyIncome'] = ds['MonthlyIncome'].fillna(ds['MonthlyIncome'].median())
ds['NumberOfDependents'] = ds['NumberOfDependents'].fillna(ds['NumberOfDependents'].mode()[0])

# Visualize distributions
key_features = ['RevolvingUtilizationOfUnsecuredLines', 'DebtRatio', 'MonthlyIncome', 'age']
i = 0

while i < len(key_features):
    column = key_features[i]
    print(column)
    plt.figure(figsize=(8, 5))
    sbn.histplot(ds[column], bins=20, kde=True, color='blue')
    plt.title(f'Distribution of {column} (Before Outlier Treatment)')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
    i = i+1

# Outlier Treatment
def mad_based_outlier_treatment(df, column):
    median = df[column].median()
    mad = np.median(np.abs(df[column] - median))
    upper_limit = median + 3 * mad
    lower_limit = median - 3 * mad
    df[column] = np.clip(df[column], lower_limit, upper_limit)


for col in key_features:
    mad_based_outlier_treatment(ds, col)

# Visualize distributions after outlier treatment
j =0
while j <  len(key_features):
    column = key_features[j]
    plt.figure(figsize=(8, 5))
    sbn.histplot(ds[column], bins=20, kde=True, color='green')
    plt.title(f'Distribution of {column} (After Outlier Treatment)')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()
    j = j+1

#WoE Transformation
# WoE and IV Calculation Function
def calculate_woe_iv(data, feature, target):
    temp_dataframe = data[[feature, target]].copy()
    temp_dataframe['total'] = 1

    grouped = temp_dataframe.groupby(feature, observed=False).agg(
        good=(target, lambda x: x.sum()),
        total=('total', 'sum')
    )
    grouped['bad'] = grouped['total'] - grouped['good']

    grouped['good_dist'] = grouped['good'] / grouped['good'].sum()
    grouped['bad_dist'] = grouped['bad'] / grouped['bad'].sum()
    grouped['woe'] = np.log(grouped['good_dist'] / grouped['bad_dist'].replace(0, 1e-9))
    grouped['iv'] = (grouped['good_dist'] - grouped['bad_dist']) * grouped['woe']

    iv = grouped['iv'].sum()
    return grouped[['woe']], iv

# Apply WoE to Features
woe_features = ['NumberOfDependents', 'DebtRatio', 'MonthlyIncome', 'RevolvingUtilizationOfUnsecuredLines']
for feature in woe_features:
    if ds[feature].nunique() > 10:
        ds[f'{feature}_bin'] = pd.qcut(ds[feature], q=10, duplicates='drop')
        feature_to_use = f'{feature}_bin'
    else:
        feature_to_use = feature

    woe, iv = calculate_woe_iv(ds, feature_to_use, 'SeriousDlqin2yrs')
    ds[f'{feature}_woe'] = ds[feature_to_use].map(woe['woe']).replace([np.inf, -np.inf], np.nan)
    ds[f'{feature}_woe'] = ds[f'{feature}_woe'].astype(float).fillna(0)
    print(f"IV for {feature}: {iv:.4f}")

#Train-Test Split

# Prepare Data
X = ds[[f'{feature}_woe' for feature in woe_features]]
y = ds['SeriousDlqin2yrs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)   

#Model Training and Evaluation
# Logistic Regression
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train, y_train)

# Evaluate Logistic Regression
y_pred_log = logistic_model.predict(X_test)
y_pred_prob_log = logistic_model.predict_proba(X_test)[:, 1]
accuracy_log = accuracy_score(y_test, y_pred_log)
roc_auc_log = roc_auc_score(y_test, y_pred_prob_log)

# Random Forest
random_forest_model = RandomForestClassifier(random_state=42, n_estimators=100)
random_forest_model.fit(X_train, y_train)

# Evaluate Random Forest
y_pred_rf = random_forest_model.predict(X_test)
y_pred_prob_rf = random_forest_model.predict_proba(X_test)[:, 1]
accuracy_rf = accuracy_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_prob_rf)

# Logistic Regression Scorecard

def generate_scorecard(log_model, features, pdo=20, base_score=600, base_odds=50):
    coefficients = log_model.coef_[0]
    intercept = log_model.intercept_[0]
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    scorecard = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients,
        'Points': -coefficients * factor
    })
    scorecard['Intercept Points'] = -intercept * factor + offset
    return scorecard

# Generate the scorecard
scorecard = generate_scorecard(logistic_model, X_train.columns)
print("\nLogistic Regression Scorecard:")
print(scorecard)


# ROC Curve Comparison
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_prob_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_prob_rf)

plt.figure(figsize=(10, 6))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {roc_auc_log:.2f})', color='blue')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.title('ROC Curve-Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Print Key Metrics
results = {
    "Logistic Regression": {
        "Accuracy": accuracy_log,
        "ROC-AUC": roc_auc_log,
    },
    "Random Forest": {
        "Accuracy": accuracy_rf,
        "ROC-AUC": roc_auc_rf,
    },
}
print("\nModel Evaluation Results: \n")

for i in results:
  print(i, results[i])