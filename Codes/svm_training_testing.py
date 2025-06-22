import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
# import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

data = pd.read_excel(r"\mnt\For_training_models.xlsx")

# Separate features and labels
X = data[['Freq_non_trivial', 'Freq_trivial', 'Difference']]
y = data['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler_svm.joblib')

# Train SVM model
svm_model = SVC(kernel='linear',  probability=True, random_state=42)

# K-Fold Cross Validation
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='roc_auc')
print("Cross-validation Scores:", cv_scores)
print("Mean Cross-validation Score:", cv_scores.mean())

svm_model.fit(X_train, y_train)

# Bar plot
plt.figure(figsize=(8, 6))
bars = plt.bar(range(1, len(cv_scores) + 1), cv_scores, color='skyblue', edgecolor='black')

# Mean line
mean_score = np.mean(cv_scores)
plt.axhline(y=mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_score:.4f}')

# Labels & titles
plt.title('Cross-Validation ROC AUC Scores (SVM)', fontsize=18)
plt.xlabel('Fold Number', fontsize=14)
plt.ylabel('ROC AUC Score', fontsize=14)
plt.xticks(range(1, len(cv_scores) + 1), [f'Fold {i}' for i in range(1, len(cv_scores)+1)], fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, 1.05)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Annotate scores on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01, f'{height:.3f}', ha='center', fontsize=11)

plt.tight_layout()
plt.show()

svm_model.fit(X_train, y_train)

# Save the model
joblib.dump(svm_model, 'svm_model.joblib')

# Predict and evaluate
y_pred_svm = svm_model.predict(X_test)
y_prob_svm = svm_model.predict_proba(X_test)[:, 1]

print("SVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Precision, Recall, F1, and AUC values
report = classification_report(y_test, y_pred_svm, output_dict=True)
precision = report['1']['precision']
recall = report['1']['recall']
f1_score = report['1']['f1-score']
roc_auc = roc_auc_score(y_test, y_prob_svm)

print(f"Precision (Class 1): {precision:.4f}")
print(f"Recall (Class 1): {recall:.4f}")
print(f"F1 Score (Class 1): {f1_score:.4f}")
print(f"AUC: {roc_auc:.4f}")

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# ROC-AUC for SVM
fpr_svm, tpr_svm, _ = roc_curve(y_test, y_prob_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)
plt.figure()
plt.plot(fpr_svm, tpr_svm, color='blue', lw=2, label='SVM ROC curve (area = %0.2f)' % roc_auc_svm)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - SVM')
plt.legend(loc="lower right")
plt.show()

# AUC Score
print("AUC Score:", roc_auc_score(y_test, y_prob_svm))

# Model Score
print("Model Score:", svm_model.score(X_test, y_test))

# Precision-Recall Curve
precision_svm, recall_svm, thresholds_svm = precision_recall_curve(y_test, y_prob_svm)

plt.figure(figsize=(8, 6))
plt.plot(recall_svm, precision_svm, color='blue', lw=2, label='SVM Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()

# Distribution of Predicted Probabilities
plt.figure(figsize=(8, 6))
plt.hist(y_prob_svm[y_test == 0], color='blue', alpha=0.7, label='Class 0')
plt.hist(y_prob_svm[y_test == 1], color='red', alpha=0.7, label='Class 1')
plt.xlabel("Predicted Probability")
plt.ylabel("Number of Data Points")
plt.title("Distribution of Predicted Probabilities")
plt.legend()
plt.show()

# Model Parameters
print("\n--- SVM Model Parameters ---")
print("Weight vector (w):", svm_model.coef_)
print("Bias term (b):", svm_model.intercept_)
print("Number of support vectors for each class:", svm_model.n_support_)
print("Total number of support vectors:", len(svm_model.support_))

# Geometric Margin
w_norm = np.linalg.norm(svm_model.coef_)
geometric_margin = 1 / w_norm
print(f"Geometric Margin: {geometric_margin:.4f}")

# Margin Violations (slack variables > 0)
decision_values = svm_model.decision_function(X_test)
# Convert y_test to +/-1
y_test_signed = y_test.replace({0: -1, 1: 1})
margin_violations = np.sum(y_test_signed * decision_values < 1)
print(f"Number of margin violations (slack variables > 0): {margin_violations}")
