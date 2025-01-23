import pandas as pd

frequency_file = r"\mnt\Frequencies.xlsx"
material_file = r"\mnt\Trivial.txt" # Non-trivial.txt for Non-trivial material file

frequency_data = pd.read_excel(frequency_file)

# Function to read the material .txt file and return the data
def read_material_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    material_data = []
    for line in lines:
        parts = line.strip().split(',')
        spacegroup = int(parts[0].strip())  # Spacegroup number
        material_formula = parts[1].strip()  # Material formula
        material_data.append([spacegroup, material_formula])

    return pd.DataFrame(material_data, columns=['Spacegroup', 'Material'])

material_data = read_material_file(material_file)

# Modify the parse_formula function to extract only the element symbols
def parse_formula(formula):
    elements = []
    counts = []
    i = 0
    while i < len(formula):
        element = ''
        while i < len(formula) and formula[i].isalpha():
            element += formula[i]
            i += 1

        count = ''
        while i < len(formula) and formula[i].isdigit():
            count += formula[i]
            i += 1

        count = int(count) if count else 1
        elements.append(element)
        counts.append(count)

    return list(zip(elements, counts))

# Update the calculate_weighted_sum function with the new parsing logic
def calculate_weighted_sum(material, frequency_data):
    parsed_formula = parse_formula(material)

    total_atoms = sum(count for _, count in parsed_formula)

    freq_non_trivial_sum = 0
    freq_trivial_sum = 0
    difference_sum = 0

    for element, count in parsed_formula:
        proportion = count / total_atoms
        row = frequency_data[frequency_data['Element'] == element]
        if not row.empty:
            freq_non_trivial_sum += proportion * row['Frequency_1'].values[0]
            freq_trivial_sum += proportion * row['Rescaled_frequency_2'].values[0]
            difference_sum += proportion * row['Difference'].values[0]

    return freq_non_trivial_sum, freq_trivial_sum, difference_sum

# Applying the function to the material data and adding the calculated features
material_data[['Freq_non_trivial', 'Freq_trivial', 'Difference']] = material_data['Material'].apply(
    lambda x: pd.Series(calculate_weighted_sum(x, frequency_data))
)

# Add the "Label" column with all values set to 0(trivial)/1(non-trivial)
material_data['Label'] = 0

output_file = r"\mnt\Data_for_training_and_testing_trivial.xlsx"
material_data.to_excel(output_file, index=False)
