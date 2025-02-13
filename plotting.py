import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Load the data from the Excel files
data = pd.read_excel('data/data.xlsx')
substrats = pd.read_excel('data/substrats.xlsx')

# Ensure the 'DATE' column is in datetime format
data['DATE'] = pd.to_datetime(data['DATE'])

# The 3rd and 4th columns in the data refer to categories of lichen
unique_species = data['SPECIES'].unique()
print(len(unique_species)) # 533 unique species

unique_genus = data['GENUS'].unique()
print(len(unique_genus)) # 235 unique genus

# The 6th column Sub refers to the substrate, so where the lichen grows on. 
# They are all abbreviations of trees, shrubs, stones, etc.
def get_substrate_category(sub_code):
    """
    Categorize substrate codes into main categories based on their characteristics
    """
    if pd.isna(sub_code):
        return 'Unknown'
        
    sub_code = str(sub_code)  # Convert to string but preserve case
    
    # Stone substrates (including tiles and stone-related)
    stone_substrates = [
        'c', 's', 'b', 'sh', 'san', 'vr', 'ba', 'sw', 'cw', 'cn', 
        'tsc', 'tsa', 'tsb', 'tss', 'sti', 'asp', 'mor', 'pav', 'cli', 
        'rt', 'so', 'ir', 'tl'
    ]
    
    # Wood substrates (including processed wood, dead wood, and tree-related)
    wood_substrates = [
        'st', 'wp', 'wft', 'wfb', 'w', 'wb', 'wf', 'dst', 'wst',
        'stb', 'stp', 'stq', 'stc',  # Specific stumps
        'bft', 'bfb', 'exr',  # Exposed roots and bark
        'pwf', 'th',  # Processed wood (painted wood, thatch)
        'fq', 'fs', 'fp',  # Fallen specific trees
        'bfp', 'tf', 'dbs',  # Bark fence post, tree-fern, dead branch shrub
        'gn', 'Gn'  # Ganoderma (wood-decaying fungus)
    ]
    
    # Soil/Earth substrates (including ground cover)
    soil_substrates = [
        't', 'eb', 'dg',  # Added dead grass
        'sdh',  # Small dead herbs
        'sts'  # Stone on soil
    ]
    
    # Moss substrates
    moss_substrates = ['m', 'mb', 'mc', 'ma', 'mo', 'Ma', 'Mo']
    
    # Leaf substrates (including needles and cones)
    leaf_substrates = [
        'le', 'leb', 'lel', 'ler', 'lea', 'lei', 'len', 'leo', 'leq', 
        'les', 'let', 'leu', 'lef', 'lej', 'lep', 'lem', 'lec', 'lei',
        'ne', 'nep', 'nea', 'nes', 'net', 'nec',  # Needles
        'ce',  # Cone
        'leh', 'leR'  # Leaf Chamaecyparis and other leaf types
    ]
    
    # Other specific substrates (including non-woody plants and artificial)
    other_substrates = [
        'pls', 'as', 'rf', 'ust', 'ush', 'ud', 'uft', 'uc', 'ut',
        'cac', 'lia', 'lv', 'he', 'He', 'Lv', 'po\\', 'pn*', 'qr', 'qre', 'fbf' 
    ]
    
    # Tree/shrub genera that should be categorized as Bark
    bark_substrates = [
        'aa', 'ab', 'ac', 'acm', 'acn', 'ae', 'ah', 'ai', 'al', 'ali', 'aln', 'am', 
        'ame', 'an', 'ap', 'api', 'apl', 'aps', 'ar', 'ara', 'arb', 'arc', 'ars', 
        'ass', 'au', 'az', 'aza', 'bau', 'be', 'br', 'bu', 'bul', 'bur', 'ca', 
        'cas', 'cat', 'cc', 'cd', 'cea', 'cec', 'cf', 'cg', 'ch', 'cho', 'chr', 
        'ci', 'cis', 'cl', 'cla', 'cm', 'cn', 'co', 'coc', 'col', 'con', 'cou', 'cp', 
        'cr', 'cre', 'cry', 'cs', 'csi', 'ct', 'cu', 'cul', 'cum', 'cyt', 'da', 
        'do', 'dr', 'en', 'ep', 'er', 'erm', 'ery', 'eu', 'euo', 'fa', 'fi', 'fn', 
        'fo', 'for', 'fr', 'ft', 'fua', 'ga', 'geo', 'gi', 'gr', 'gu', 'hb', 
        'hi', 'hp', 'hu', 'hur', 'hy', 'id', 'in', 'ix', 'ja', 'jac', 'jn', 
        'ju', 'la', 'lai', 'lau', 'li', 'lib', 'lic', 'lno', 'lr', 'ly', 'man', 
        'may', 'me', 'mel', 'met', 'mg', 'mh', 'mi', 'mic', 'ml', 'mn', 
        'mol', 'mr', 'ms', 'mt', 'my', 'na', 'ner', 'no', 'oc', 'ol', 'or', 
        'os', 'pa', 'pac', 'pah', 'pal', 'par', 'pau', 'pc', 'pca', 'pdu', 'pe', 
        'ph', 'phy', 'pi', 'pic', 'pit', 'pl', 'pla', 'pm', 'pn', 'po', 'pod', 
        'pol', 'pp', 'pr', 'prd', 'prt', 'ps', 'psp', 'pt', 'pth', 'ptr', 'pu', 
        'py', 'qco', 'qf', 'qil', 'qpa', 'qpu', 'qpy', 'qro', 'qru', 'qs', 'qu', 
        'qut', 'rd', 'rh', 'ri', 'rm', 'ro', 'rs', 'ru', 'rz', 'rzm', 'sa', 'sal', 
        'sd', 'sek', 'sm', 'sn', 'sp', 'sq', 'sr', 'sy', 'sz', 'te', 'tel', 'th',
        'ti', 'tm', 'tn', 'tr', 'ts', 'tx', 'ul', 'va', 'vi', 'we', 'wi', 'yu', 'yue',
        'Ba', 'Cn', 'So', 'Th', 'Qu', 'Qro', 'Sa', 'Po', 'Pn', 'Be', 'Bu', 'Tx',
        'Sz', 'Beq', 'Euo', 'Chr'
    ]
    
    # Convert input to lowercase for comparison
    sub_code_lower = sub_code.lower()
    
    # Check for exact matches in substrate lists
    if sub_code_lower in [x.lower() for x in stone_substrates]:
        return 'Stone'
    elif sub_code_lower in [x.lower() for x in wood_substrates]:
        return 'Wood'
    elif sub_code_lower in [x.lower() for x in soil_substrates]:
        return 'Soil'
    elif sub_code_lower in [x.lower() for x in moss_substrates]:
        return 'Moss'
    elif sub_code_lower in [x.lower() for x in leaf_substrates]:
        return 'Leaf'
    elif sub_code_lower in [x.lower() for x in other_substrates]:
        return 'Other'
    elif sub_code_lower in [x.lower() for x in bark_substrates]:
        return 'Bark'
        
    # If not found in any list, mark as Other
    return 'Other'

def categorize_substrate(sub_code):
    """Categorize substrate codes into main categories"""
    category = get_substrate_category(sub_code)
    
    # Map our detailed categories to the five main categories
    main_category_map = {
        'Bark': 'Bark',
        'Stone': 'Stone',
        'Wood': 'Wood',
        'Soil': 'Soil',
        'Moss': 'Soil',  # Group moss with soil substrates
        'Leaf': 'Bark',  # Group leaves with bark substrates
        'Other': 'Other',
        'Unknown': 'Unknown'
    }
    
    return main_category_map.get(category, 'Other')

def get_display_name(latin_name):
    """Get display name for substrate, using Latin name to ensure uniqueness"""
    # Common mapping of Latin genera to English names
    english_names = {
        'Quercus robur': 'English Oak',
        'Quercus rubra': 'Red Oak',
        'Quercus petraea': 'Sessile Oak',
        'Quercus ilex': 'Holm Oak',
        'Betula': 'Birch',
        'Fagus': 'Beech',
        'Fraxinus': 'Ash',
        'Tilia': 'Linden',
        'Ulmus': 'Elm',
        'Salix': 'Willow',
        'Populus tremula': 'Aspen',
        'Populus alba': 'White Poplar',
        'Acer platanoides': 'Norway Maple',
        'Acer pseudoplatanus': 'Sycamore Maple'
    }
    
    # Wood-related terms
    wood_terms = {
        'st': 'Stump',
        'wp': 'Fence post',
        'wft': 'Fallen trunk',
        'wfb': 'Fallen branch',
        'dst': 'Dead standing tree',
        'w': 'Unidentified wood'
    }
    
    if latin_name in wood_terms:
        return wood_terms[latin_name]
    
    # For species with unique English names
    if latin_name in english_names:
        return english_names[latin_name]
    
    # For other cases, return the Latin name
    return latin_name

def plot_frequency_distribution(data):
    """Create frequency distribution plot showing both MTB/64 distribution and frequency classes
    This function analyzes the geographical distribution of lichen species using two key pieces of data:

    - KMHOK (MTB/64 Grid Squares):
        These are geographical grid coordinates that divide the study area into squares
        Each number (like 511225, 511311) represents a specific grid square on the map
        MTB/64 is a standardized grid system used for mapping species distributions
    SPECIES:
        This column contains the names of lichen species found in each grid square
"""
    # Create figure with extra space for note
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Group by species and count occurrences in different MTB/64 squares
    # Shows how widespread each species is
    # X-axis: Species (ranked from most to least widespread)
    # Y-axis: Number of different grid squares where each species was found
    # This tells us which species are found in many locations vs. few locations
    species_freq = data.groupby('SPECIES')['KMHOK'].nunique().reset_index()
    
    species_freq = species_freq.sort_values('KMHOK', ascending=False)
    
    # Create frequency classes based on MTB occurrences
    bins = [0, 1, 2, 4, 8, 16, float('inf')]
    labels = ['vr', 'r', 'mr', 'mf', 'f', 'vf']
    species_freq['freq_class'] = pd.cut(species_freq['KMHOK'], bins=bins, labels=labels)
    
    # Calculate percentages for pie chart
    freq_dist = species_freq['freq_class'].value_counts()
    freq_percentages = freq_dist / len(species_freq) * 100
    
    # Sort the frequencies to match the legend order
    freq_percentages = freq_percentages.reindex(['vr', 'r', 'mr', 'mf', 'f', 'vf'])
    
    # Bar plot showing distribution of species across MTB/64 squares
    ax1.bar(range(len(species_freq)), species_freq['KMHOK'])
    ax1.set_xlabel('Species (ranked by occurrence)')
    ax1.set_ylabel('Number of MTB/64 Grid Squares')
    ax1.set_title('Distribution of Species Across Grid Squares')
    
    # Pie chart with consistent colors
    colors = ['lightgray', 'darkgray', 'gray', 'dimgray', 'black', 'darkslategray']
    wedges, texts, autotexts = ax2.pie(freq_percentages, 
                                      labels=[f'{l} {v:.1f}%' for l, v in zip(freq_percentages.index, freq_percentages)],
                                      colors=colors,
                                      autopct='%1.1f%%')
    ax2.set_title('Frequency Class Distribution')
    
    # Add legend with matching colors and clearer MTB/64 references
    legend_labels = [
        'very rare (vr): 1 MTB/64 grid square',
        'rare (r): 2-3 MTB/64 grid squares',
        'moderately rare (mr): 4-7 MTB/64 grid squares',
        'moderately frequent (mf): 8-15 MTB/64 grid squares',
        'frequent (f): 16-31 MTB/64 grid squares',
        'very frequent (vf): >31 MTB/64 grid squares'
    ]
    ax2.legend(wedges, legend_labels,
               loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Add explanatory note
    note = ("Note: These plots show the geographical distribution of lichen species:\n"
           "Left: Number of MTB/64 grid squares where each species was found, ranked from most to least widespread.\n"
           "Right: Distribution of species across frequency classes based on their occurrence in MTB/64 grid squares.\n"
           "This helps understand which species are widespread vs. geographically restricted.")
    plt.figtext(0.1, 0.02, note, wrap=True, horizontalalignment='left', fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    return fig

def plot_substrate_distribution(data):
    """Create vertical bar plot of substrate distribution"""
    
    # Get substrate categories for each observation
    data['SubstrateCategory'] = data['SUB'].apply(get_substrate_category)
    
    # Count observations per substrate category
    substrate_counts = data.groupby('SubstrateCategory')['AMOUNT'].sum().sort_values(ascending=False)
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create vertical bar plot
    ax = plt.gca()
    bars = ax.bar(range(len(substrate_counts)), substrate_counts.values, color='#1f77b4')
    
    # Customize the plot
    ax.set_xticks(range(len(substrate_counts)))
    ax.set_xticklabels(substrate_counts.index, rotation=0)
    ax.set_ylabel('Number of Observations')
    ax.set_title('Distribution of Lichen Observations Across Substrate Types')
    
    # Add value labels on top of each bar
    for i, v in enumerate(substrate_counts.values):
        ax.text(i, v + (max(substrate_counts.values) * 0.02), str(int(v)), ha='center')
    
    # Add legend with substrate categories and their contents
    legend_text = {
        'Bark': 'Tree species: Aa, Ab, Ac, Acm, Acn, Ae, Ah, Ai, Al, Ali, Aln, Am, Ame, An, Ap, Api, Apl, Aps, Ar, Ara, Arb, Arc, Ars, Ass, Au, Az, Aza, Bau, Be, Br, Bu, Bul, Bur, Ca, Cas, Cat, Cc, Cd, Cea, Cee, Cf, Cg, Ch, Cho, Chr, Ci, Cis, Cl, Cla, Cm, Co, Coc, Col, Cou, Cp, Cr, Cre, Cry, Cs, Csi, Ct, Cu, Cul, Cum, Cyt, Da, Do, Dr, En, Ep, Er, Erm, Ery, Eu, Euo, Fa, Fi, Fn, Fo, For, Fr, Fua, Ga, Geo, Gi, Gr, Gu, Hb, Hi, Hp, Hu, Hur, Hy, Id, In, Ix, Ja, Jac, Jn, Ju, La, Lal, Lau, Li, Lib, Lic, Lno, Lr, Ly, Man, May, Me, Mel, Met, Mg, Mh, Mi, Mic, Ml, Mn, Mo, Mol, Mr, Ms, Mt, My, Na, Ner, No, Oc, Ol, Or, Os, Pa, Pac, Pah, Pal, Par, Pau, Pc, Pca, Pdu, Pe, Ph, Phy, Pi, Pic, Pit, Pl, Pla, Pln, Pn, Po, Pod, Pol, Pp, Pr, Prd, Prt, Ps, Psp, Pt, Pth, Ptr, Pu, Py, Qco, Qf, Qil, Qpa, Qpu, Qpy, Qro, Qru, Qs, Qu, Qut, Rd, Rh, Ri, Rm, Ro, Rs, Ru, Rz, Rzm, Sa, Sal, Sd, Sek, Sm, Sn, Sp, Sq, Sr, Sy, Sz, Te, Tel, Ti, Tm, Tn, Tr, Ts, Tx, Ul, Va, Vi, We, Wi, Yu, Yue, ac, be, con, ft, pc, qro, sa, sm',
        'Stone': 'Rock and stone substrates: Ba, C, Cn, So, asp, b, ba, c, cli, cn, cw, ir, mor, pav, rt, s, san, sh, so, sti, sw, ti, tsa, tsb, tsc, tss, vr',
        'Wood': 'Dead wood and processed wood: Gn, Th, bfb, bfp, bft, dbs, dst, exr, fp, fq, fs, pwf, st, stb, stc, stp, stq, tf, th, w, wb, wf, wfb, wft, wp, wst',
        'Soil': 'Ground and soil substrates: dg, eb, sdh, sts, t',
        'Leaf': 'Living and dead leaves: le, leb, lel, ler, lea, lei, len, leo, leq, les, let, leu, lef, lej, lep, lem, lec, lei, ne, nep, nea, nes, net, nec, ce, leh, leR',
        'Moss': 'Moss and moss-covered substrates: m, mb, mc, ma, mo, Ma, Mo',
        'Other': 'Miscellaneous substrates: \'t, *, 2c, ?, Beg, E, H, He, Lv, Po?, Rn, Se, Y, as, cac, cro, cru, f, fbf, h, lia, p, pls, pn*, po, q, qr, qre, rf, rft, rw, sab, sas, tt, uc, ud, uft, ush, ust, ut, wfp, wqo',
        'Unknown': 'Substrates with unknown or missing information: nan'
    }
    
    # Format legend text with line breaks every few substrates
    formatted_legend = []
    for category in substrate_counts.index:  # Use the order from the plot
        if category in legend_text:            
            substrates = legend_text[category].split(': ')[1].split(', ')
            # Create groups of 8 substrates per line
            grouped_substrates = [', '.join(substrates[i:i+8]) for i in range(0, len(substrates), 8)]
            formatted_text = f"{category}: {legend_text[category].split(': ')[0]}\n" + '\n'.join(grouped_substrates)
            formatted_legend.append(formatted_text)
    
    # Add legend to the right of the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    plt.figtext(0.65, 0.1, '\n\n'.join(formatted_legend), fontsize=8, va='bottom')
    
    # Set y-axis to start at 0
    ax.set_ylim(0, max(substrate_counts.values) * 1.1)  # Add 10% padding for labels
    
    return plt.gcf()

def plot_species_per_substrate(data):
    """Create vertical bar plot of unique species per substrate"""
    
    # Get substrate categories for each observation
    data['SubstrateCategory'] = data['SUB'].apply(get_substrate_category)
    
    # Count unique species per substrate category
    species_counts = data.groupby('SubstrateCategory')['SPECIES'].nunique().sort_values(ascending=False)
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Create vertical bar plot
    ax = plt.gca()
    bars = ax.bar(range(len(species_counts)), species_counts.values, color='#1f77b4')
    
    # Customize the plot
    ax.set_xticks(range(len(species_counts)))
    ax.set_xticklabels(species_counts.index, rotation=0)
    ax.set_ylabel('Number of Species')
    ax.set_title('Number of Unique Species on Different Substrate Types')
    
    # Add value labels on top of each bar
    for i, v in enumerate(species_counts.values):
        ax.text(i, v + 5, str(int(v)), ha='center')
    
    # Add legend with substrate categories and their contents
    legend_text = {
        'Bark': 'Tree species: Aa, Ab, Ac, Acm, Acn, Ae, Ah, Ai, Al, Ali, Aln, Am, Ame, An, Ap, Api, Apl, Aps, Ar, Ara, Arb, Arc, Ars, Ass, Au, Az, Aza, Bau, Be, Br, Bu, Bul, Bur, Ca, Cas, Cat, Cc, Cd, Cea, Cee, Cf, Cg, Ch, Cho, Chr, Ci, Cis, Cl, Cla, Cm, Co, Coc, Col, Cou, Cp, Cr, Cre, Cry, Cs, Csi, Ct, Cu, Cul, Cum, Cyt, Da, Do, Dr, En, Ep, Er, Erm, Ery, Eu, Euo, Fa, Fi, Fn, Fo, For, Fr, Fua, Ga, Geo, Gi, Gr, Gu, Hb, Hi, Hp, Hu, Hur, Hy, Id, In, Ix, Ja, Jac, Jn, Ju, La, Lal, Lau, Li, Lib, Lic, Lno, Lr, Ly, Man, May, Me, Mel, Met, Mg, Mh, Mi, Mic, Ml, Mn, Mol, Mr, Ms, Mt, My, Na, Ner, No, Oc, Ol, Or, Os, Pa, Pac, Pah, Pal, Par, Pau, Pc, Pca, Pdu, Pe, Ph, Phy, Pi, Pic, Pit, Pl, Pla, Pln, Pn, Po, Pod, Pol, Pp, Pr, Prd, Prt, Ps, Psp, Pt, Pth, Ptr, Pu, Py, Qco, Qf, Qil, Qpa, Qpu, Qpy, Qro, Qru, Qs, Qu, Qut, Rd, Rh, Ri, Rm, Ro, Rs, Ru, Rz, Rzm, Sa, Sal, Sd, Sek, Sm, Sn, Sp, Sq, Sr, Sy, Sz, Te, Tel, Ti, Tm, Tn, Tr, Ts, Tx, Ul, Va, Vi, We, Wi, Yu, Yue, ac, be, con, ft, pc, qro, sa, sm',
        'Stone': 'Rock and stone substrates: Ba, C, Cn, So, asp, b, ba, c, cli, cn, cw, ir, mor, pav, rt, s, san, sh, so, sti, sw, ti, tsa, tsb, tsc, tss, vr',
        'Wood': 'Dead wood and processed wood: Gn, Th, bfb, bfp, bft, dbs, dst, exr, fp, fq, fs, pwf, st, stb, stc, stp, stq, tf, th, w, wb, wf, wfb, wft, wp, wst',
        'Soil': 'Ground and soil substrates: dg, eb, sdh, sts, t',
        'Leaf': 'Living and dead leaves: le, leb, lel, ler, lea, lei, len, leo, leq, les, let, leu, lef, lej, lep, lem, lec, lei, ne, nep, nea, nes, net, nec, ce, leh, leR',
        'Moss': 'Moss and moss-covered substrates: m, mb, mc, ma, mo, Ma, Mo',
        'Other': 'Miscellaneous substrates: \'t, *, 2c, ?, Beg, E, H, He, Lv, Po?, Rn, Se, Y, as, cac, cro, cru, f, fbf, h, lia, p, pls, pn*, po, q, qr, qre, rf, rft, rw, sab, sas, tt, uc, ud, uft, ush, ust, ut, wfp, wqo',
        'Unknown': 'Substrates with unknown or missing information: nan'
    }
    
    # Format legend text with line breaks every few substrates
    formatted_legend = []
    for category in species_counts.index:  # Use the order from the plot
        if category in legend_text:            
            substrates = legend_text[category].split(': ')[1].split(', ')
            # Create groups of 8 substrates per line
            grouped_substrates = [', '.join(substrates[i:i+8]) for i in range(0, len(substrates), 8)]
            formatted_text = f"{category}: {legend_text[category].split(': ')[0]}\n" + '\n'.join(grouped_substrates)
            formatted_legend.append(formatted_text)
    
    # Add legend to the right of the plot
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])
    plt.figtext(0.65, 0.1, '\n\n'.join(formatted_legend), fontsize=8, va='bottom')
    
    # Set y-axis to start at 0
    ax.set_ylim(0, max(species_counts.values) * 1.1)  # Add 10% padding for labels
    
    return plt.gcf()

def plot_top_substrates(data):
    """Create horizontal bar plot of top 25 substrates by observations"""
    
    # Calculate total observations and unique species per substrate
    substrate_stats = data.groupby('SUB').agg({
        'AMOUNT': 'sum',
        'SPECIES': 'nunique'
    }).sort_values('AMOUNT', ascending=True)
    
    # Get top 20 substrates
    top_substrates = substrate_stats.tail(25)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create horizontal bar plot
    ax = plt.gca()
    y_pos = range(len(top_substrates))
    
    # Plot bars for observations
    obs_bars = ax.barh(y_pos, top_substrates['AMOUNT'], color='skyblue', label='Number of Observations')
    
    # Add species count as text
    for i, (obs, species) in enumerate(zip(top_substrates['AMOUNT'], top_substrates['SPECIES'])):
        ax.text(obs + obs * 0.02, i, f'  {int(obs):,}  ({species} species)', va='center')
    
    # Customize the plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_substrates.index)
    ax.set_xlabel('Number of Observations')
    ax.set_title('Top 25 Substrates by Number of Observations')
    
    plt.tight_layout()
    return plt.gcf()

def plot_bark_distribution(data, substrats):
    """Create plot for bark substrates"""
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Process bark substrates
    bark_substrates = [
        code for code in substrats['AfkSub']
        if get_substrate_category(code) == 'Bark'
    ]
    bark_data = data[data['SUB'].isin(bark_substrates)]
    bark_counts = bark_data.groupby('SUB')['AMOUNT'].sum().sort_values(ascending=True)
    
    # Function to create labels with English names (falling back to Latin)
    def create_label(code):
        name = substrats[substrats['AfkSub'] == code]['NaamSub'].iloc[0]
        return f"{code} ({name})" if pd.notna(name) else code
    
    # Plot bark substrates
    bars = ax.barh(range(len(bark_counts)), bark_counts.values)
    ax.set_yticks(range(len(bark_counts)))
    ax.set_yticklabels([create_label(code) for code in bark_counts.index], fontsize=8)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, i, f' {int(width)}', va='center')
    ax.set_xlabel('Number of Observations')
    ax.set_title('Distribution of Observations on Bark Substrates')
    
    # Add explanatory note
    note = ("Note: This plot shows the distribution of lichen observations on bark substrates from living trees and shrubs.\n"
           "The numbers indicate the total count of individual lichen occurrences recorded for each substrate.")
    plt.figtext(0.1, 0.02, note, wrap=True, horizontalalignment='left', fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the plot
    plt.savefig('plots/bark_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_wood_distribution(data, substrats):
    """Create plot for wood substrates"""
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    ax = plt.gca()
    
    # Process wood substrates
    wood_substrates = [
        code for code in substrats['AfkSub']
        if get_substrate_category(code) == 'Wood'
    ]
    wood_data = data[data['SUB'].isin(wood_substrates)]
    wood_counts = wood_data.groupby('SUB')['AMOUNT'].sum().sort_values(ascending=True)
    
    # Function to create labels with English names (falling back to Latin)
    def create_label(code):
        name = substrats[substrats['AfkSub'] == code]['NaamSub'].iloc[0]
        return f"{code} ({name})" if pd.notna(name) else code
    
    # Plot wood substrates
    bars = ax.barh(range(len(wood_counts)), wood_counts.values)
    ax.set_yticks(range(len(wood_counts)))
    ax.set_yticklabels([create_label(code) for code in wood_counts.index], fontsize=8)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax.text(width, i, f' {int(width)}', va='center')
    ax.set_xlabel('Number of Observations')
    ax.set_title('Distribution of Observations on Wood Substrates')
    
    # Add explanatory note
    note = ("Note: This plot shows the distribution of lichen observations on wood substrates including dead wood,\n"
           "processed wood, and wood products. The numbers indicate the total count of individual lichen occurrences\n"
           "recorded for each substrate.")
    plt.figtext(0.1, 0.02, note, wrap=True, horizontalalignment='left', fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    # Save the plot
    plt.savefig('plots/wood_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_woody_plants_distribution(data, substrats):
    """Create separate plots for bark and wood substrates"""
    plot_bark_distribution(data, substrats)
    plot_wood_distribution(data, substrats)

def main():
    """Main function to create all plots"""
    # Read data
    data = pd.read_excel('data/data.xlsx')
    substrats = pd.read_excel('data/substrats.xlsx')
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Create and save frequency distribution plot
    fig1 = plot_frequency_distribution(data)
    fig1.savefig('plots/frequency_distribution.png', bbox_inches='tight')
    plt.close(fig1)
    
    # Create and save substrate distribution plot
    fig2 = plot_substrate_distribution(data)
    fig2.savefig('plots/substrate_distribution.png', bbox_inches='tight')
    plt.close(fig2)
    
    # Create and save species per substrate plot
    fig3 = plot_species_per_substrate(data)
    fig3.savefig('plots/species_per_substrate.png', bbox_inches='tight')
    plt.close(fig3)
    
    # Create and save top substrates plot
    fig4 = plot_top_substrates(data)
    fig4.savefig('plots/top_substrates.png', bbox_inches='tight')
    plt.close(fig4)
    
    # Create and save woody plants distribution plot
    plot_woody_plants_distribution(data, substrats)

if __name__ == '__main__':
    main()
