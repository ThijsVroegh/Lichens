# Lichen Distribution Interactive Map

This Streamlit application visualizes the geographical distribution of lichens in the Netherlands, allowing users to explore the data with interactive filters.

## Features

- **Interactive Map**: View lichen observations across the Netherlands
- **Multiple Filters**: Filter by genus, species, substrate, conservation status, and date range
- **Dynamic Visualizations**: See temporal trends and distribution statistics
- **Conservation Status Insights**: Analyze red list species distributions
- **Substrate Analysis**: Explore different substrate types and their associated species

## Installation

1. Make sure you have Python 3.7+ installed
2. Install required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

1. Navigate to the project directory
2. Run the Streamlit application:

```bash
streamlit run app.py
```

3. The application should open in your default web browser at `http://localhost:8501`

## Data Sources

The application uses two main data files:
- `data/data.xlsx`: Contains the lichen observation records
- `data/substrats.xlsx`: Contains substrate reference information

## Notes on Location Data

The KMHOK grid references in the dataset are converted to approximate geographical coordinates for visualization purposes. In a production version, a proper conversion from the Dutch grid system to latitude/longitude would be implemented.

## Customizing the Application

- To modify the color schemes, edit the `CATEGORY_COLORS` and `SUBSTRATE_COLORS` dictionaries
- To change the default map view, modify the `location` parameter in the `folium.Map()` call
- To add additional visualizations, create new functions and add them to the main layout 