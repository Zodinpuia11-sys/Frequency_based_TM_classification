import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

data = pd.read_excel(r"C:\Users\HP\OneDrive\Desktop\ML in TM\For_training_models.xlsx")

X = data[['Freq_non_trivial', 'Freq_trivial', 'Difference']]
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, 'scaler_svm.joblib')

svm_model = SVC(kernel='linear', probability=True, random_state=42)

cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='roc_auc')
print("Cross-validation Scores:", cv_scores)
print("Mean Cross-validation Score:", cv_scores.mean())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

plt.figure(figsize=(10, 10))

for i, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    svm_model_cv = SVC(kernel='linear', probability=True, random_state=42)
    svm_model_cv.fit(X_cv_train, y_cv_train)
    y_cv_prob = svm_model_cv.predict_proba(X_cv_val)[:, 1]

    fpr, tpr, _ = roc_curve(y_cv_val, y_cv_prob)
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    plt.plot(fpr, tpr, lw=2, alpha=0.7, label=f'Fold {i+1} (AUC = {roc_auc:.3f})')

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='black', lw=3, linestyle='--',
         label=f'Mean ROC (AUC = {mean_auc:.3f})')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)

plt.title('Cross-Validation ROC Curves (SVM)', fontsize=20)
plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(alpha=0.3)
plt.show()

print("Fold-wise AUC Scores:", aucs)
print("Mean AUC Score:", np.mean(aucs))

plt.figure(figsize=(8, 6))
bars = plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='skyblue', edgecolor='black')

mean_score = np.mean(cv_scores)
plt.axhline(y=mean_score, color='red', linestyle='--', linewidth=2,
            label=f'Mean = {mean_score:.4f}')

plt.title('Cross-Validation ROC AUC Scores (SVM)', fontsize=18)
plt.xlabel('Fold Number', fontsize=14)
plt.ylabel('ROC AUC Score', fontsize=14)
plt.xticks(range(1, len(cv_scores) + 1),
           [f'Fold {i}' for i in range(1, len(cv_scores)+1)], fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 1.05)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01,
             f'{height:.3f}', ha='center', fontsize=11)

plt.tight_layout()
plt.show()

svm_model.fit(X_train, y_train)

joblib.dump(svm_model, 'svm_model.joblib')

y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

print("\nSVM Classification Report (Test Set):")
print(classification_report(y_test, y_pred_svm))

plt.figure(figsize=(10, 10))
cm = confusion_matrix(y_test, y_pred_svm)
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", linewidths=2, linecolor='black',
                 annot_kws={"size": 20})
plt.title("Confusion Matrix - SVM", fontsize=30)
plt.xlabel("Predicted Labels", fontsize=20)
plt.ylabel("True Labels", fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=20)
plt.show()

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

plt.figure(figsize=(10, 10))
plt.plot(fpr_svm, tpr_svm, color='blue', lw=3,
         label=f'SVM ROC curve (area = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('Receiver Operating Characteristic - Test Set (SVM)', fontsize=30)
plt.legend(loc="lower right", fontsize=20)
plt.show()

precision_svm, recall_svm, thresholds_svm = precision_recall_curve(y_test, y_prob_svm)
plt.figure(figsize=(8, 6))
plt.plot(recall_svm, precision_svm, color='blue', lw=2,
         label='SVM Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Test Set)')
plt.legend(loc="lower left")
plt.show()

plt.figure(figsize=(8, 6))
plt.hist(y_prob_svm[y_test == 0], color='blue', alpha=0.7, label='Class 0')
plt.hist(y_prob_svm[y_test == 1], color='red', alpha=0.7, label='Class 1')
plt.xlabel("Predicted Probability")
plt.ylabel("Number of Data Points")
plt.title("Distribution of Predicted Probabilities")
plt.legend()
plt.show()

print("\n--- SVM Model Parameters ---")
print("Weight vector (w):", svm_model.coef_)
print("Bias term (b):", svm_model.intercept_)
print("Number of support vectors for each class:", svm_model.n_support_)
print("Total number of support vectors:", len(svm_model.support_))

w_norm = np.linalg.norm(svm_model.coef_)
geometric_margin = 1 / w_norm
print(f"Geometric Margin: {geometric_margin:.4f}")

decision_values = svm_model.decision_function(X_test)

y_test_signed = y_test.replace({0: -1, 1: 1})
margin_violations = np.sum(y_test_signed * decision_values < 1)
print(f"Number of margin violations (slack variables > 0): {margin_violations}")
