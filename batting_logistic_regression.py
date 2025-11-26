import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# Load datasets
batting = pd.read_csv("Batting.csv")
hof = pd.read_csv("HallOfFame.csv")

# Create HOF label
hof['HOF'] = (hof['inducted'] == 'Y').astype(int)
hof = hof[['playerID', 'HOF']]
df = batting.merge(hof, on='playerID', how='left')
df['HOF'] = df['HOF'].fillna(0)

# Aggregate career totals safely
numeric_cols = ["G","AB","R","H","2B","3B","HR","RBI","SB","CS","BB","SO","IBB","HBP","SH","SF"]
df_agg = df.groupby('playerID').agg({**{col:'sum' for col in numeric_cols}, 'HOF':'max'}).reset_index()

# Select top features from RF and target
top_features = ["G","AB","SO","R","H","RBI"]
X = df_agg[top_features]
y = df_agg['HOF']

# Train/test split (no stratify to avoid small-class errors)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression with balanced class weights
lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# Predictions and evaluation
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:,1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (HOF):", precision_score(y_test, y_pred))
print("Recall (HOF):", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature coefficients
coefficients = pd.Series(lr.coef_[0], index=top_features).sort_values(key=abs, ascending=False)
print("Feature coefficients:")
print(coefficients)
