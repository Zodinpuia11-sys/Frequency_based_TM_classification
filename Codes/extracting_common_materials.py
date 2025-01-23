import pandas as pd

file1 = r"\mnt\svm_classified.xlsx"
file2 = r"\mnt\rf_classified.xlsx"

df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# Find common rows based on the 'Material' and 'Space Group' columns
common_df = pd.merge(df1, df2, on=['Material', 'Spacegroup'])

# Drop duplicate rows resulting from the merge
common_df = common_df.drop_duplicates(subset=['Material', 'Spacegroup'])

# Save the common data to a new Excel file
output_file = r"\mnt\common_predicted.xlsx"
common_df.to_excel(output_file, index=False)
