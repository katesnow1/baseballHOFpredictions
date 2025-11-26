import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
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

# Select features and target
X = df_agg[numeric_cols]
y = df_agg['HOF']

# Train/test split (no stratify to avoid small-class errors)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance with random oversampling
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)

# Train Random Forest
rf = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=1, class_weight="balanced", random_state=42)
rf.fit(X_res, y_res)

# Predictions and evaluation
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:,1]

print("\n Random Forest Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision (HOF):", precision_score(y_test, y_pred))
print("Recall (HOF):", recall_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n Feature Importance:")
print(importances.head(20))
