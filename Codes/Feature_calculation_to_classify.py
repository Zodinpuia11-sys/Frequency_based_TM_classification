import pandas as pd

frequency_file = r"\mny\Frequencies.xlsx"  # Replace with the actual file path
material_file = r"\mnt\To_classify.txt"  # Replace with the actual txt file path

frequency_data = pd.read_excel(frequency_file)

# Electronegativity values
electronegativity = {
    'H': 2.20, 'He': 0.00, 'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44,
    'F': 3.98, 'Ne': 0.00, 'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58,
    'Cl': 3.16, 'Ar': 0.00, 'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66,
    'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01,
    'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 0.00, 'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33,
    'Nb': 1.60, 'Mo': 2.16, 'Tc': 1.90, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
    'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'I': 2.66, 'Xe': 0.00, 'Cs': 0.79, 'Ba': 0.89, 'La': 1.10,
    'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Pm': 1.13, 'Sm': 1.17, 'Eu': 1.20, 'Gd': 1.20, 'Tb': 1.23,
    'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.10, 'Lu': 1.27, 'Hf': 1.30, 'Ta': 1.50,
    'W': 2.36, 'Re': 1.90, 'Os': 2.16, 'Ir': 2.20, 'Pt': 2.28, 'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62,
    'Pb': 2.33, 'Bi': 2.02, 'Po': 2.00, 'At': 2.2, 'Rn': 0.00, 'Fr': 0.70, 'Ra': 0.90, 'Ac': 1.10,
    'Te': 2.01, 'Th': 1.30, 'U': 1.38, 'Pa': 1.50,
}


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


# Modify the parse_formula function to extract only the element symbols and check if it's binary
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


# Filter out binary materials
def is_binary_material(formula):
    parsed_formula = parse_formula(formula)
    return len(parsed_formula) == 2  # Binary material should have exactly two elements

# Filter the materials that are binary
binary_material_data = material_data[material_data['Material'].apply(is_binary_material)]


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


# Calculate electronegativity difference for binary materials
def calculate_electronegativity_difference(formula):
    parsed_formula = parse_formula(formula)

    if len(parsed_formula) == 2:
        element_1, element_2 = parsed_formula[0][0], parsed_formula[1][0]
        if element_1 in electronegativity and element_2 in electronegativity:
            electronegativity_diff = abs(electronegativity[element_1] - electronegativity[element_2])
            return electronegativity_diff
    return None

# Applying the function to the binary material data and adding the calculated features
binary_material_data[['Freq_non_trivial', 'Freq_trivial', 'Difference']] = binary_material_data['Material'].apply(
    lambda x: pd.Series(calculate_weighted_sum(x, frequency_data))
)

# Calculate electronegativity difference for each material
binary_material_data['Electronegativity_difference'] = binary_material_data['Material'].apply(
    calculate_electronegativity_difference)

output_file = r"\mnt\Data_to_classify.xlsx"
binary_material_data.to_excel(output_file, index=False)
