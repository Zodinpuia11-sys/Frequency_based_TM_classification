import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

data = pd.read_excel(r"\For_training_models.xlsx")

X = data[['Freq_non_trivial', 'Freq_trivial', 'Difference']]
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, 'scaler_rf.joblib')

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tprs, aucs = [], []
mean_fpr = np.linspace(0, 1, 100)

fold_styles = [
    {'linestyle': 'none', 'marker': '*', 'markersize': 6, 'color': 'red'},
    {'linestyle': '--', 'linewidth': 2.5, 'color': 'blue'},
    {'linestyle': 'none', 'marker': 'x', 'markersize': 6, 'color': 'green'},
    {'linestyle': '-.', 'linewidth': 2.5, 'color': 'purple'},
    {'linestyle': 'none', 'marker': '.', 'markersize': 5, 'color': 'orange'}
]

plt.figure(figsize=(10, 10))
for i, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    rf_cv_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_cv_model.fit(X_cv_train, y_cv_train)
    y_cv_prob = rf_cv_model.predict_proba(X_cv_val)[:, 1]

    fpr, tpr, _ = roc_curve(y_cv_val, y_cv_prob)
    roc_auc = auc(fpr, tpr)

    roc_auc_floor = np.floor(roc_auc * 100) / 100
    aucs.append(roc_auc)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    plt.plot(fpr, tpr, label=f'Fold {i+1} (AUC = {roc_auc_floor:.2f})', **fold_styles[i])

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

mean_auc_floor = np.floor(mean_auc * 100) / 100
plt.plot(mean_fpr, mean_tpr, color='black', lw=3, linestyle='-',
         label=f'Mean ROC (AUC = {mean_auc_floor:.2f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)

plt.title('Cross-Validation ROC Curves (RF)', fontsize=22)
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

legend = plt.legend(loc='lower right', fontsize=14)
for text in legend.get_texts():
    text.set_fontweight('bold')

plt.grid(alpha=0.3)
plt.show()

rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'rf_model.joblib')

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

report = classification_report(y_test, y_pred_rf, output_dict=True)
precision = report['1']['precision']
recall = report['1']['recall']
f1_score = report['1']['f1-score']
roc_auc = roc_auc_score(y_test, y_prob_rf)

print(f"Precision (Class 1): {precision:.4f}")
print(f"Recall (Class 1): {recall:.4f}")
print(f"F1 Score (Class 1): {f1_score:.4f}")
print(f"AUC: {roc_auc:.4f}")

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure()
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2,
         label='Random Forest ROC curve (area = %.2f)' % roc_auc_rf)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Random Forest (Test Set)')
plt.legend(loc="lower right")
plt.show()

print("AUC Score (Test Set):", round(roc_auc_score(y_test, y_prob_rf), 2))
print("Model Score (Test Set):", round(rf_model.score(X_test, y_test), 2))

precision_rf, recall_rf, thresholds_rf = precision_recall_curve(y_test, y_prob_rf)
plt.figure(figsize=(8, 6))
plt.plot(recall_rf, precision_rf, color='blue', lw=2,
         label='Random Forest Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Test Set)')
plt.legend(loc="lower left")
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(y_prob_rf[y_test == 0], color='blue', alpha=0.7, label='Class 0')
plt.hist(y_prob_rf[y_test == 1], color='red', alpha=0.7, label='Class 1')
plt.xlabel("Predicted Probability")
plt.ylabel("Number of Data Points")
plt.title("Distribution of Predicted Probabilities")
plt.legend()
plt.show()

importances = rf_model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importances:")
print(feature_importance_df)

plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
plt.title("Feature Importance - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()
