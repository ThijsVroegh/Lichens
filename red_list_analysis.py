import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, List, Tuple

def load_data():
    """Load the data from the Excel file"""
    data = pd.read_excel('data/data.xlsx')
    return data

def process_red_list_data(data):
    """Process red list data to prepare for visualization"""
    # Make a copy of the data to avoid modifying the original
    df = data.copy()
    
    # Create a unique species dataset to avoid duplicate counts
    unique_species = df.drop_duplicates(subset=['SPECIES'])
    
    # Map RED_LIST 'v' to a boolean - only handle 'v' (no 'vge' as per updated data)
    unique_species['is_red_list'] = unique_species['RED_LIST'] == 'v'
    
    # Create a category for species not on any red list (Not threatened)
    unique_species['CATEGORY'] = unique_species['CATEGORY'].fillna('nt')  # nt = not threatened
    
    # Create a dictionary to map abbreviations to full names
    category_names = {
        'ge': 'Sensitive',
        'kw': 'Vulnerable',
        'be': 'Threatened',
        'eb': 'Seriously threatened',
        've': 'Disappeared',
        'nt': 'Not threatened'
    }
    
    # Count unique species in each category
    category_counts = unique_species.groupby('CATEGORY').size()
    
    # Calculate percentages
    total_species = category_counts.sum()
    category_percentages = (category_counts / total_species * 100).round(1)
    
    # Create a DataFrame with counts and percentages
    result_df = pd.DataFrame({
        'Category': [category_names.get(cat, cat) for cat in category_counts.index],
        'Original': list(category_counts.index),
        'Count': category_counts.values,
        'Percentage': category_percentages.values
    })
    
    # Also calculate total red list species vs non-red list
    red_list_count = unique_species['is_red_list'].sum()
    non_red_list_count = len(unique_species) - red_list_count
    
    red_list_df = pd.DataFrame({
        'Status': ['Red List', 'Not on Red List'],
        'Count': [red_list_count, non_red_list_count],
        'Percentage': [(red_list_count / len(unique_species) * 100).round(1), 
                       (non_red_list_count / len(unique_species) * 100).round(1)]
    })
    
    return result_df, red_list_df, unique_species

def plot_red_list_pie_chart(result_df, red_list_df):
    """Create a pie chart of red list categories in English only"""
    # Set up the figure with two parts: pie chart and table
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), 
                                   gridspec_kw={'width_ratios': [1.2, 0.8]})
    
    # Define colors for each category (using a harmonious color palette)
    # Using a colorblind-friendly palette for better accessibility
    colors = {
        'Not threatened': '#abd9e9',  # Light blue
        'Sensitive': '#fdae61',       # Orange
        'Vulnerable': '#f46d43',      # Dark orange
        'Threatened': '#d73027',      # Red
        'Seriously threatened': '#a50026',  # Dark red
        'Disappeared': '#313695'      # Dark blue
    }
    
    # Sort data for better visualization - put "Not threatened" first, then sort by conservation concern
    category_order = ['Not threatened', 'Sensitive', 'Vulnerable', 'Threatened', 'Seriously threatened', 'Disappeared']
    
    # Filter to only include categories that exist in our data
    display_order = [cat for cat in category_order if cat in result_df['Category'].values]
    
    # Add any categories that might be in our data but not in the predefined order
    for cat in result_df['Category']:
        if cat not in display_order:
            display_order.append(cat)
    
    # Reindex the dataframe according to our display order
    result_df_sorted = pd.DataFrame([result_df[result_df['Category'] == cat].iloc[0] 
                                     for cat in display_order if cat in result_df['Category'].values])
    
    # Create the pie chart with percentages outside the pie
    wedges, texts = ax1.pie(
        result_df_sorted['Percentage'], 
        labels=None,  # No direct labels
        colors=[colors.get(cat, 'gray') for cat in result_df_sorted['Category']],
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1},
        radius=0.8  # Slightly smaller pie to make room for external labels
    )
    
    # Add percentage labels outside the pie chart
    for i, wedge in enumerate(wedges):
        # Calculate angle for text placement
        ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
        x = 0.9 * np.cos(np.deg2rad(ang))  # 0.9 = radius + margin
        y = 0.9 * np.sin(np.deg2rad(ang))
        
        # Determine horizontal alignment based on angle
        ha = 'center'
        if ang < 180:
            ha = 'left'
        elif ang > 180:
            ha = 'right'
            
        # Add percentage text
        percent = result_df_sorted['Percentage'].iloc[i]
        if percent > 2:  # Only show percentages > 2% to avoid clutter
            ax1.text(x, y, f"{percent}%", ha=ha, va='center', fontsize=12, 
                    fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                                                fc='white', ec="none", alpha=0.7))
    
    # Add a legend explaining the categories
    ax1.legend(
        wedges,
        [f"{cat} ({int(count)} species)" for cat, count in zip(result_df_sorted['Category'], result_df_sorted['Count'])],
        title="Red List Categories",
        loc="center left",
        bbox_to_anchor=(0, 0.5),
        frameon=False,
        fontsize=10
    )
    
    ax1.set_title('Distribution of Red List Categories', fontsize=16, pad=20)
    
    # Create a table with category counts and percentages
    ax2.axis('off')  # Turn off the axis
    ax2.set_title('Red List Categories', fontsize=16, pad=20)
    
    # Prepare table data
    table_data = [
        ['Category', 'Species', '%'],
    ]
    
    # Add rows for each category
    for _, row in result_df_sorted.iterrows():
        table_data.append([
            f"{row['Category']}",
            f"{int(row['Count'])}",
            f"{row['Percentage']}%"
        ])
    
    # Add row for total
    table_data.append([
        'Total',
        f"{int(result_df['Count'].sum())}",
        '100.0%'
    ])
    
    # Create the table
    table = ax2.table(
        cellText=table_data,
        loc='center',
        cellLoc='left',
        colWidths=[0.5, 0.25, 0.25]
    )
    
    # Customize table appearance
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)  # Adjust table size
    
    # Style the header row
    for j in range(len(table_data[0])):
        cell = table[(0, j)]
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#e6e6e6')
    
    # Style the total row
    last_row = len(table_data) - 1
    for j in range(len(table_data[0])):
        cell = table[(last_row, j)]
        cell.set_text_props(fontweight='bold')
    
    # Add descriptions of categories beneath the table
    category_descriptions = [
        "Red List Categories Explained:",
        "• Not threatened: Species with stable populations",
        "• Sensitive: Species beginning to show signs of decline",
        "• Vulnerable: Species facing a high risk of extinction",
        "• Threatened: Species facing a very high risk of extinction",
        "• Seriously threatened: Species in danger of extinction",
        "• Disappeared: Species no longer found in the region"
    ]
    
    description_text = "\n".join(category_descriptions)
    ax2.text(0.5, 0.05, description_text, 
            ha='center', va='top', 
            fontsize=10,
            bbox=dict(facecolor='#f5f5f5', alpha=0.5, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # Create plots directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Save the plot
    plt.savefig('plots/red_list_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Red list distribution plot created and saved in plots/red_list_distribution.png")
    print("\nRed List Category statistics:")
    print(result_df.to_string(index=False))
    print("\nRed List vs Non-Red List statistics:")
    print(red_list_df.to_string(index=False))

def plot_red_list_status_pie_chart(red_list_df):
    """Create a separate pie chart showing Red List vs. Non-Red List species"""
    plt.figure(figsize=(10, 8))
    
    # Create the pie chart with better colors
    colors = ['#d73027', '#abd9e9']  # Red for red list, blue for non-red list
    wedges, texts, autotexts = plt.pie(
        red_list_df['Percentage'], 
        labels=None,  # Moving to legend for better clarity
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 12, 'weight': 'bold', 'color': 'white'}
    )
    
    # Add a clear legend with counts
    plt.legend(
        wedges, 
        [f"{status} ({int(count)} species)" for status, count in zip(red_list_df['Status'], red_list_df['Count'])],
        title="Conservation Status",
        loc="center left",
        bbox_to_anchor=(0, 0.5),
        frameon=False,
        fontsize=12
    )
    
    plt.title('Red List vs. Non-Red List Species', fontsize=18, pad=20)
    
    # Add explanatory annotation
    plt.annotate(
        "Red List species are those that have been \n" +
        "officially listed as threatened or endangered \n" +
        "according to conservation criteria.",
        xy=(0.5, 0.1),
        xycoords='figure fraction',
        ha='center',
        va='center',
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="gray", alpha=0.8)
    )
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/red_list_status.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Red list status plot created and saved in plots/red_list_status.png")

def analyze_red_list_species(unique_species):
    """Analyze red list species in more detail"""
    # Filter for red list species only
    red_list_species = unique_species[unique_species['is_red_list']]
    
    # Count by category
    red_list_by_category = red_list_species.groupby('CATEGORY').size()
    
    # Calculate percentages
    total_red_list = red_list_by_category.sum()
    red_list_percentages = (red_list_by_category / total_red_list * 100).round(1)
    
    # Create a DataFrame with counts and percentages
    category_names = {
        'ge': 'Sensitive',
        'kw': 'Vulnerable',
        'be': 'Threatened',
        'eb': 'Seriously threatened',
        've': 'Disappeared',
        'nt': 'Not threatened'
    }
    
    red_list_analysis = pd.DataFrame({
        'Category': [category_names.get(cat, cat) for cat in red_list_by_category.index],
        'Count': red_list_by_category.values,
        'Percentage': red_list_percentages.values
    })
    
    print("\nAnalysis of Red List Species by Category:")
    print(red_list_analysis.to_string(index=False))
    
    return red_list_analysis

def plot_red_list_distribution_horizontal(result_df):
    """Create a horizontal bar chart showing the distribution of species across categories"""
    # Sort categories by conservation concern level
    category_order = ['Not threatened', 'Sensitive', 'Vulnerable', 'Threatened', 'Seriously threatened', 'Disappeared']
    
    # Create a new DataFrame with categories in the desired order
    plot_df = pd.DataFrame([
        {'Category': cat, 
         'Count': result_df[result_df['Category'] == cat]['Count'].values[0] if cat in result_df['Category'].values else 0,
         'Percentage': result_df[result_df['Category'] == cat]['Percentage'].values[0] if cat in result_df['Category'].values else 0}
        for cat in category_order if cat in result_df['Category'].values
    ])
    
    # Reverse order for bottom-to-top display
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    
    # Define consistent colors
    colors = {
        'Not threatened': '#abd9e9',  # Light blue
        'Sensitive': '#fdae61',       # Orange
        'Vulnerable': '#f46d43',      # Dark orange
        'Threatened': '#d73027',      # Red
        'Seriously threatened': '#a50026',  # Dark red
        'Disappeared': '#313695'      # Dark blue
    }
    
    # Create the horizontal bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.barh(
        plot_df['Category'],
        plot_df['Count'],
        color=[colors.get(cat, 'gray') for cat in plot_df['Category']],
        edgecolor='white',
        linewidth=1
    )
    
    # Add value labels on the bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 5,
            i,
            f"{int(bar.get_width())} ({plot_df['Percentage'].iloc[i]}%)",
            va='center',
            fontsize=10
        )
    
    # Add titles and labels
    plt.title('Distribution of Species by Conservation Status', fontsize=16, pad=20)
    plt.xlabel('Number of Species', fontsize=12)
    plt.ylabel('Conservation Status', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/red_list_horizontal.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Horizontal red list distribution plot created and saved in plots/red_list_horizontal.png")

def main():
    """Main function to process data and create the pie chart"""
    # Load the data
    data = load_data()
    
    # Process the red list data
    result_df, red_list_df, unique_species = process_red_list_data(data)
    
    # Create the pie chart for categories
    plot_red_list_pie_chart(result_df, red_list_df)
    
    # Create the pie chart for red list status
    plot_red_list_status_pie_chart(red_list_df)
    
    # Analyze red list species in more detail
    analyze_red_list_species(unique_species)
    
    # Create a horizontal bar chart for better category comparison
    plot_red_list_distribution_horizontal(result_df)

if __name__ == '__main__':
    main() 