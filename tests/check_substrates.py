import pandas as pd
from plotting import get_substrate_category

# Read the substrats file
df = pd.read_excel('data/substrats.xlsx')

# Define our categorization lists
stone_substrates = ['c', 's', 'b', 'sh', 'san', 'vr', 'ba', 'sw', 'cw', 'cn', 'tsc', 'tsa', 'tsb', 'tss', 'sti', 'asp', 'mor', 'pav', 'cli', 'rt', 'so', 'ir']
wood_substrates = ['st', 'wp', 'wft', 'wfb', 'w', 'wb', 'wf', 'dst', 'wst', 'stb', 'stp', 'stq', 'stc', 'bft', 'bfb', 'exr', 'pwf', 'th', 'fq', 'fs', 'fp']
soil_substrates = ['t', 'eb', 'dg', 'sdh']
moss_substrates = ['m', 'mb', 'mc', 'ma']
leaf_substrates = ['le', 'leb', 'lel', 'ler', 'lea', 'lei', 'len', 'leo', 'leq', 'les', 'let', 'leu', 'lef', 'lej', 'lep', 'lem', 'lec', 'lei', 'ne', 'nep', 'nea', 'nes', 'net', 'nec']
other_substrates = ['pls', 'as', 'rf', 'ust', 'ush', 'ud', 'uft', 'uc', 'ut', 'cac', 'lia']

# Create a set of all explicitly categorized substrates
all_categorized = set(stone_substrates + wood_substrates + soil_substrates + moss_substrates + leaf_substrates + other_substrates)

# Get all substrate codes from the Excel file
all_substrates = set(df['AfkSub'].str.lower().dropna())

# Find substrates that are not explicitly categorized
uncategorized = all_substrates - all_categorized

print(f'Total substrates in Excel: {len(df)}')
print(f'Explicitly categorized substrates: {len(all_categorized)}')
print(f'Substrates defaulting to Bark category: {len(uncategorized)}')
print('\nSubstrates not explicitly categorized (defaulting to Bark):')
for sub in sorted(uncategorized):
    mask = df['AfkSub'].str.lower() == sub
    if any(mask):
        print(f"{sub}: {df.loc[mask.values, 'NaamSub'].iloc[0]}")

print('\nCategorization by type:')
print(f'Stone substrates: {len(stone_substrates)}')
print(f'Wood substrates: {len(wood_substrates)}')
print(f'Soil substrates: {len(soil_substrates)}')
print(f'Moss substrates: {len(moss_substrates)}')
print(f'Leaf substrates: {len(leaf_substrates)}')
print(f'Other substrates: {len(other_substrates)}')

# Get categories for all substrates
categories = {}
for _, row in df.iterrows():
    if not pd.isna(row['AfkSub']):
        cat = get_substrate_category(row['AfkSub'])
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((row['AfkSub'], row['NaamSub']))

# Print summary
print('\nCategorization Summary:')
print('-' * 50)
total = 0
for cat, items in sorted(categories.items()):
    print(f'\n{cat} substrates ({len(items)}):')
    for code, name in sorted(items):
        print(f'  {code}: {name}')
        
        
import pandas as pd
import numpy as np

# Read the data
data = pd.read_excel('data/data.xlsx')

# Group by species and count occurrences in different MTB/64 squares
species_freq = data.groupby('SPECIES')['KMHOK'].nunique().reset_index()
species_freq = species_freq.sort_values('KMHOK', ascending=False)

# Create frequency classes based on MTB occurrences
bins = [0, 1, 2, 4, 8, 16, float('inf')]
labels = ['very rare (vr)', 'rare (r)', 'moderately rare (mr)', 'moderately frequent (mf)', 'frequent (f)', 'very frequent (vf)']
species_freq['freq_class'] = pd.cut(species_freq['KMHOK'], bins=bins, labels=labels)

# Print examples from each frequency class
print('\nExamples of species in each frequency class:')
for freq_class in labels:
    class_species = species_freq[species_freq['freq_class'] == freq_class]
    print(f'\n{freq_class}:')
    print(f'Number of species: {len(class_species)}')
    print('Examples (with number of grid squares):')
    for _, row in class_species.head(3).iterrows():
        print(f'  - {row['SPECIES']} ({row['KMHOK']} grid squares)')"

# Examples of species in each frequency class:

# very rare (vr):
# Number of species: 103
# Examples (with number of grid squares):
#   - ochrocheila (1 grid squares)
#   - perisidiosa (1 grid squares)
#   - percrenata (1 grid squares)

# rare (r):
# Number of species: 60
# Examples (with number of grid squares):
#   - hippocastani (2 grid squares)
#   - crustulata (2 grid squares)
#   - confusa (2 grid squares)

# moderately rare (mr):
# Number of species: 38
# Examples (with number of grid squares):
#   - cladoniae (4 grid squares)
#   - pseudotsugae (4 grid squares)
#   - pyxidata (4 grid squares)

# moderately frequent (mf):
# Number of species: 53
# Examples (with number of grid squares):
#   - sulphurella (8 grid squares)
#   - abscondita (8 grid squares)
#   - angulosa (8 grid squares)

# frequent (f):
# Number of species: 52
# Examples (with number of grid squares):
#   - rupestris (16 grid squares)
#   - weillii (16 grid squares)
#   - monomorpha (16 grid squares)

# very frequent (vf):
# Number of species: 226
# Examples (with number of grid squares):
#   - tenella (560 grid squares)
#   - parietina (559 grid squares)
#   - adscendens (548 grid squares)