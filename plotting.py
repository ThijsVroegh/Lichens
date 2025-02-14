import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from functools import lru_cache
from typing import Dict, List, Tuple

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
@lru_cache(maxsize=None)
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

def get_display_name(latin_name: str) -> str:
    """Get English display name for Latin species name"""
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
    """Create frequency distribution plot showing MTB/64 distribution and classes.
    It effectively counts the number of unique grid squares per species and visualizes
    this distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Count number of MTB/64 squares per species
    species_freq = data.groupby('SPECIES')['KMHOK'].nunique().reset_index()
    
    # Create histogram of number of grid squares
    # This shows how many species are found in X number of grid squares
    hist_values, hist_bins, _ = ax1.hist(
        species_freq['KMHOK'], 
        bins=20,  # Adjust number of bins as needed
        color='skyblue',
        edgecolor='black'
    )
    
    # Add value labels on top of bars
    for i in range(len(hist_values)):
        ax1.text(
            (hist_bins[i] + hist_bins[i+1])/2,  # Center of bar
            hist_values[i],  # Height of bar
            f'{int(hist_values[i])}',  # Number of species
            ha='center',
            va='bottom'
        )
    
    # Make axes labels clearer
    ax1.set_xlabel('Number of MTB/64 Grid Squares (Species Range)')
    ax1.set_ylabel('Number of Species')
    ax1.set_title('Distribution of Species by Geographic Range\n(How many species occur in X grid squares)')
    
    # Rest of the pie chart code remains the same
    # Create frequency classes based on MTB occurrences
    bins = [0, 1, 2, 4, 8, 16, float('inf')]
    labels = ['vr', 'r', 'mr', 'mf', 'f', 'vf']
    species_freq['freq_class'] = pd.cut(species_freq['KMHOK'], bins=bins, labels=labels)
    
    # Calculate percentages for pie chart
    freq_dist = species_freq['freq_class'].value_counts()
    freq_percentages = freq_dist / len(species_freq) * 100
    freq_percentages = freq_percentages.reindex(['vr', 'r', 'mr', 'mf', 'f', 'vf'])
    
    # Pie chart
    colors = ['lightgray', 'darkgray', 'gray', 'dimgray', 'black', 'darkslategray']
    wedges, texts, autotexts = ax2.pie(freq_percentages, 
                                      labels=[f'{l} {v:.1f}%' for l, v in zip(freq_percentages.index, freq_percentages)],
                                      colors=colors,
                                      autopct='%1.1f%%')
    ax2.set_title('Frequency Class Distribution')
    
    # Update legend
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
    
    # Update note
    note = ("**Note:** These plots show the geographical distribution of lichen species:\n\n"
           "_Left:_ Histogram showing how many species (y-axis) are found in a given number of grid squares (x-axis).\n"
           "_Right:_ Distribution of species across frequency classes based on their occurrence in MTB/64 grid squares.\n\n"
           "This helps understand the rarity patterns of species in the dataset.")
    
    fig.text(0.05, 0.02, note, 
             wrap=True,
             horizontalalignment='left',
             verticalalignment='bottom',
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8),
             transform=fig.transFigure)
    
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

def plot_temporal_trends(data):
    """Create plot showing relative occurrences of species over time
    
    This function analyzes temporal trends while accounting for sampling effort by:
    1. Calculating total observations per year
    2. Converting species counts to relative frequencies
    3. Showing both absolute and relative trends
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Get yearly totals and species counts
    yearly_data = data.groupby([data['DATE'].dt.year]).agg({
        'AMOUNT': 'sum',  # Total observations
        'SPECIES': 'nunique'  # Unique species
    }).reset_index()
    
    # Plot 1: Absolute numbers
    ax1.plot(yearly_data['DATE'], yearly_data['AMOUNT'], 
             marker='o', label='Total observations', color='blue')
    ax1.set_ylabel('Number of observations', color='blue')
    
    # Add second y-axis for species counts
    ax1_twin = ax1.twinx()
    ax1_twin.plot(yearly_data['DATE'], yearly_data['SPECIES'],
                  marker='s', label='Unique species', color='red')
    ax1_twin.set_ylabel('Number of unique species', color='red')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    ax1.set_title('Absolute Numbers: Observations and Species Richness')
    
    # Plot 2: Relative frequencies
    # Calculate species relative frequencies
    yearly_species = data.groupby([data['DATE'].dt.year, 'SPECIES'])['AMOUNT'].sum().reset_index()
    pivot_species = yearly_species.pivot(index='DATE', columns='SPECIES', values='AMOUNT').fillna(0)
    
    # Convert to relative frequencies
    relative_freq = pivot_species.div(pivot_species.sum(axis=1), axis=0)
    
    # Select top 10 most common species for visualization
    top_species = pivot_species.sum().nlargest(10).index
    
    # Plot relative frequencies for top species
    for species in top_species:
        ax2.plot(relative_freq.index, relative_freq[species], 
                 label=species, marker='o', alpha=0.7)
    
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Relative Frequency')
    ax2.set_title('Relative Frequencies of Top 10 Species Over Time')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add explanatory note
    note = ("Note: These plots show temporal trends in lichen observations:\n"
           "Top: Absolute numbers of total observations and unique species per year\n"
           "Bottom: Relative frequencies of the 10 most common species,\n"
           "normalized by total yearly observations to account for sampling effort")
    plt.figtext(0.1, 0.02, note, wrap=True, horizontalalignment='left', fontsize=8)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    return fig

def plot_trend_analysis(data):
    """Create plot showing species with strongest positive and negative trends
    
    This function:
    1. Calculates relative frequencies for each species per year
    2. Fits a linear regression to identify trends
    3. Shows top 5 increasing and top 5 decreasing species
    """
    # Calculate yearly relative frequencies
    yearly_species = data.groupby([data['DATE'].dt.year, 'SPECIES'])['AMOUNT'].sum().reset_index()
    yearly_totals = yearly_species.groupby('DATE')['AMOUNT'].sum()
    yearly_species['relative_freq'] = yearly_species.apply(
        lambda x: x['AMOUNT'] / yearly_totals[x['DATE']], axis=1
    )
    
    # Pivot data for trend analysis
    species_trends = yearly_species.pivot(
        index='DATE', columns='SPECIES', values='relative_freq'
    ).fillna(0)
    
    # Calculate trends using linear regression
    trends = {}
    for species in species_trends.columns:
        x = np.arange(len(species_trends.index))
        y = species_trends[species].values
        slope, _ = np.polyfit(x, y, 1)
        trends[species] = slope
    
    # Get top 5 increasing and decreasing species
    trend_series = pd.Series(trends)
    top_increasing = trend_series.nlargest(5)
    top_decreasing = trend_series.nsmallest(5)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot increasing trends
    for species in top_increasing.index:
        data = species_trends[species]
        ax1.plot(species_trends.index, data, marker='o', label=f"{species} ({top_increasing[species]:.2e})")
        
        # Add trend line
        x = np.arange(len(species_trends.index))
        z = np.polyfit(x, data, 1)
        p = np.poly1d(z)
        ax1.plot(species_trends.index, p(x), '--', alpha=0.5)
    
    ax1.set_title('Top 5 Species with Strongest Increasing Trends')
    ax1.set_ylabel('Relative Frequency')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot decreasing trends
    for species in top_decreasing.index:
        data = species_trends[species]
        ax2.plot(species_trends.index, data, marker='o', label=f"{species} ({top_decreasing[species]:.2e})")
        
        # Add trend line
        x = np.arange(len(species_trends.index))
        z = np.polyfit(x, data, 1)
        p = np.poly1d(z)
        ax2.plot(species_trends.index, p(x), '--', alpha=0.5)
    
    ax2.set_title('Top 5 Species with Strongest Decreasing Trends')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Relative Frequency')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add explanatory note with improved formatting
    note = ("**Note:** These plots show species with the strongest trends over time:\n\n"
           "_Top panel:_ Species with the strongest increasing relative frequency\n"
           "_Bottom panel:_ Species with the strongest decreasing relative frequency\n\n"
           "Numbers in parentheses show the slope of the trend line (change in relative frequency per year). "
           "Dashed lines indicate the fitted linear trends.")
    
    # Use figure-width text box for note
    fig.text(0.05, 0.02, note,
             wrap=True,
             horizontalalignment='left',
             verticalalignment='bottom',
             fontsize=10,
             bbox=dict(facecolor='white', alpha=0.8),
             transform=fig.transFigure)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    return fig

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
    
    # Create and save temporal trends plot
    fig6 = plot_temporal_trends(data)
    fig6.savefig('plots/temporal_trends.png', bbox_inches='tight')
    plt.close(fig6)
    
    # Create and save trend analysis plot
    fig7 = plot_trend_analysis(data)
    fig7.savefig('plots/trend_analysis.png', bbox_inches='tight')
    plt.close(fig7)

if __name__ == '__main__':
    main()
