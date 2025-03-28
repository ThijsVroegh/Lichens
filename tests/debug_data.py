import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Load the data
    print("Loading data...")
    data = pd.read_excel('data/data.xlsx')
    
    # Print info about the columns
    print("\nColumn names:")
    print(data.columns.tolist())
    
    # Print the first few rows
    print("\nFirst 3 rows:")
    print(data.head(3))
    
    # Check for RED_LIST and CATEGORY columns
    if 'RED_LIST' in data.columns:
        print("\nRED_LIST values (first 10):")
        print(data['RED_LIST'].value_counts().head(10))
    else:
        print("\nRED_LIST column not found!")
    
    if 'CATEGORY' in data.columns:
        print("\nCATEGORY values (first 10):")
        print(data['CATEGORY'].value_counts().head(10))
    else:
        print("\nCATEGORY column not found!")
    
    # Count unique species
    print(f"\nTotal unique species: {data['SPECIES'].nunique()}")
    
    # Create a very simple pie chart
    if 'CATEGORY' in data.columns:
        print("\nCreating simple pie chart for categories...")
        # Fill NaN values
        data['CATEGORY'] = data['CATEGORY'].fillna('nt')
        
        # Count unique species per category
        species_by_category = data.drop_duplicates(subset=['SPECIES']).groupby('CATEGORY').size()
        
        # Create a pie chart
        plt.figure(figsize=(10, 8))
        plt.pie(species_by_category, labels=species_by_category.index, autopct='%1.1f%%')
        plt.title('Distribution of Species by Category')
        
        # Create plots directory if it doesn't exist
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        # Save the plot
        plt.savefig('plots/debug_category_pie.png')
        print("Pie chart saved to plots/debug_category_pie.png")

if __name__ == '__main__':
    main() 