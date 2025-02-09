import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_curve,auc
# Initialize Random Forest Classifier
model = LogisticRegression()

# Train the model
model.fit(x_train, y_train)

# Make predictions on the test set
y_pred = model.predict(x_test)

# Evaluate Model Performance
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Compute FPR, TPR, and Thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Compute AUC (Area Under Curve)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()

# Random Forest Algoritham
#-------------------xxxx----------
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Initialize Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred_rf = rf_model.predict(x_test)

# Evaluate the Random Forest Model
print(f"Random Forest Model Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

#Boosting Algorithm
#-----------xxxxxxxxxxx--------------
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Initialize XGBoost Classifier
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)

# Train the model
xgb_model.fit(x_train, y_train)

# Make predictions on the test set
y_pred_xgb = xgb_model.predict(x_test)

# Evaluate the XGBoost Model
print(f"XGBoost Model Accuracy: {accuracy_score(y_test, y_pred_xgb):.2f}")
print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("\nXGBoost Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_xgb))
