import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
rf_model = joblib.load('rf_model.joblib')
scaler = joblib.load('scaler_rf.joblib')

# Load the unseen data
unseen_data_path = r"\mnt\Data_to_classify.xlsx"
unseen_data = pd.read_excel(unseen_data_path)

# Separate the features and the electronegativity difference
X_unseen = unseen_data[['Freq_non_trivial', 'Freq_trivial', 'Difference']]
X_unseen = scaler.transform(X_unseen)

# Predict probabilities on the unseen data
y_prob_rf = rf_model.predict_proba(X_unseen)[:, 1]

# Filter the rows
filtered_data_rf = unseen_data[(y_prob_rf >= 0.9) & (unseen_data['Electronegativity_difference'] < 0.4)]

# Add the predicted probabilities to the filtered data
filtered_data_rf.loc[:, 'Prediction_Probability'] = y_prob_rf[(y_prob_rf >= 0.9) & (unseen_data['Electronegativity_difference'] < 0.4)]

# Ensure the 'Spacegroup' column is included
filtered_data_rf = filtered_data_rf[['Spacegroup', 'Material', 'Freq_non_trivial', 'Freq_trivial', 'Difference', 'Electronegativity_difference', 'Prediction_Probability']]

# Save the filtered data to a new file
output_file_path_rf = r"\mnt\rf_classified.xlsx"
filtered_data_rf.to_excel(output_file_path_rf, index=False)
