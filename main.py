import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# Load Dataset
# =========================

df = pd.read_csv("startup_success_dataset.csv")

# =========================
# Basic Information
# =========================

print("Dataset Shape:", df.shape)

print("\nOutcome Distribution:")
print(df["outcome"].value_counts())

# =========================
# EDA Visualizations
# =========================

plt.figure(figsize=(8, 5))
sns.countplot(x="outcome", data=df)
plt.title("Startup Outcome Distribution")
plt.savefig("outcome_distribution.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x="outcome", y="revenue_million", data=df)
plt.title("Revenue vs Outcome")
plt.savefig("revenue_vs_outcome.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x="outcome", y="founder_experience_years", data=df)
plt.title("Founder Experience vs Outcome")
plt.savefig("experience_vs_outcome.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(x="outcome", y="funding_rounds", data=df)
plt.title("Funding Rounds vs Outcome")
plt.savefig("funding_vs_outcome.png")
plt.close()

print("\nEDA graphs saved successfully!")

# =========================
# Label Encoding
# =========================

le = LabelEncoder()

df["investor_type"] = le.fit_transform(df["investor_type"])
df["sector"] = le.fit_transform(df["sector"])
df["founder_background"] = le.fit_transform(df["founder_background"])
df["outcome"] = le.fit_transform(df["outcome"])

# =========================
# Features and Target
# =========================

X = df.drop("outcome", axis=1)
y = df["outcome"]

# =========================
# Train-Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# =========================
# Logistic Regression
# =========================

lr = LogisticRegression(max_iter=10000)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)

# =========================
# Decision Tree
# =========================

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)

# =========================
# Random Forest
# =========================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

# =========================
# Results
# =========================

print("\n========== MODEL ACCURACY ==========")

print(f"Logistic Regression : {lr_acc:.4f}")
print(f"Decision Tree       : {dt_acc:.4f}")
print(f"Random Forest       : {rf_acc:.4f}")

# =========================
# Best Model
# =========================

best_accuracy = max(lr_acc, dt_acc, rf_acc)

if best_accuracy == rf_acc:
    print("\nBest Model: Random Forest")
elif best_accuracy == lr_acc:
    print("\nBest Model: Logistic Regression")
else:
    print("\nBest Model: Decision Tree")

print("\nProject Completed Successfully!")

