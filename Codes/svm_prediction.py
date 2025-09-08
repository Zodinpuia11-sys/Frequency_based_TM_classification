import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
svm_model = joblib.load('svm_model.joblib')
scaler = joblib.load('scaler_svm.joblib')

# Load the unseen data to classify
unseen_data_path = r"mnt\Data_to_classify.xlsx"
unseen_data = pd.read_excel(unseen_data_path)

# Separate the features and the electronegativity difference
X_unseen = unseen_data[['Freq_non_trivial', 'Freq_trivial', 'Difference']]
X_unseen = scaler.transform(X_unseen)

# Predict probabilities on the unseen data
y_prob_unseen = svm_model.predict_proba(X_unseen)[:, 1]

# Filter the rows where the probability is above 70% and the electronegativity difference is below 1
filtered_data = unseen_data[(y_prob_unseen >= 0.7) & (unseen_data['Electronegativity_difference'] < 1)]

# Add the predicted probabilities to the filtered data
filtered_data.loc[:, 'Prediction_Probability'] = y_prob_unseen[(y_prob_unseen >= 0.7 ) & (unseen_data['Electronegativity_difference'] < 1)]

# Save the filtered data to a new file
output_file_path = r"\mnt\svm_classified.xlsx"
filtered_data.to_excel(output_file_path, index=False)
