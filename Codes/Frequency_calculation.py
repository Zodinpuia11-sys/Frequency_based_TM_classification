import re
import pandas as pd
from collections import Counter

# Function to extract elements and their frequencies from a string
def extract_elements(line):
    pattern = r'([A-Z][a-z]*)(\d*)'
    elements = re.findall(pattern, line)

    element_dict = {}
    for element, count in elements:
        count = int(count) if count else 1
        if element in element_dict:
            element_dict[element] += count
        else:
            element_dict[element] = count
    return element_dict

# Path to your input files (use raw strings to avoid path escape issues)
input_filename_1 = r"\mnt\Non-trivial.txt"
input_filename_2 = r"\mnt\Trivial.txt"  # Second file to include
output_filename_sorted = r"\mnt\Frequencies.xlsx"  # Save as Excel

# Function to process a file and return a dictionary of element frequencies
def process_file(input_filename):
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    element_counter = Counter()

    for line in lines:
        line = line.strip()
        element_dict = extract_elements(line)
        element_counter.update(element_dict)

    return dict(sorted(element_counter.items()))

element_counter_1 = process_file(input_filename_1)
df1 = pd.DataFrame(list(element_counter_1.items()), columns=['Element', 'Frequency_1'])

element_counter_2 = process_file(input_filename_2)
df2 = pd.DataFrame(list(element_counter_2.items()), columns=['Element', 'Frequency_2'])

# Merge the two dataframes on 'Element' column to preserve both frequencies
df = pd.merge(df1, df2, on='Element', how='outer').fillna(0)

# Calculate the sums of Frequency_2 and Frequency_1
sum_f_2 = df['Frequency_2'].sum()
sum_f_1 = df['Frequency_1'].sum()

# Calculate the rescaled values
df['Rescaled_frequency_2'] = (df['Frequency_2'] / sum_f_2) * sum_f_1

# Calculate the difference between Rescaled_frequency_2 and Frequency_1
df['Difference'] = df['Rescaled_frequency_2'] - df['Frequency_1']

df = df.drop(columns=['Frequency_2'])
df.to_excel(output_filename_sorted, index=False)
