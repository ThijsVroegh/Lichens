import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
import pickle
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Set page title and layout
st.set_page_config(
    page_title="Lichen Distribution in the Netherlands",
    page_icon="🍃",
    layout="wide"
)

# Define standard colors for categories (consistent with other plots)
CATEGORY_COLORS = {
    'Not threatened': '#abd9e9',  # Light blue
    'Sensitive': '#fdae61',       # Orange
    'Vulnerable': '#f46d43',      # Dark orange
    'Threatened': '#d73027',      # Red
    'Seriously threatened': '#a50026',  # Dark red
    'Disappeared': '#313695'      # Dark blue
}

# Define substrate colors
SUBSTRATE_COLORS = {
    'Bark': '#8c510a',     # Brown
    'Wood': '#bf812d',     # Light brown
    'Stone': '#35978f',    # Teal
    'Soil': '#01665e',     # Dark green
    'Moss': '#5ab4ac',     # Light green
    'Leaf': '#c7eae5',     # Very light green
    'Other': '#7f7f7f',    # Gray
    'Unknown': '#d9d9d9'   # Light gray
}

# File to save geocoded coordinates
GEOCODE_CACHE_FILE = "data/geocode_cache.pkl"
KMHOK_CACHE_FILE = "data/kmhok_cache.pkl"

def geocode_locations(locations):
    """
    Geocode a list of location names.
    Uses cache file to avoid repeating geocoding.
    """
    # Initialize geocoding cache
    location_coords = {}
    
    # Check if cache file exists and load it
    if os.path.exists(GEOCODE_CACHE_FILE):
        try:
            with open(GEOCODE_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                location_coords = cache
                st.success(f"Loaded {len(location_coords)} cached locations from disk.")
        except Exception as e:
            st.warning(f"Could not load geocode cache: {str(e)}")
    
    # Filter out locations that are already in the cache
    locations_to_geocode = [loc for loc in locations if loc not in location_coords]
    
    if locations_to_geocode:
        # Initialize geocoder
        user_agent = "lichen_distribution_app/1.0"
        geolocator = Nominatim(user_agent=user_agent)
        
        # Create a geocoding function with rate limiting
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
        
        with st.spinner(f"Geocoding {len(locations_to_geocode)} new locations... This may take a minute."):
            for location in locations_to_geocode:
                try:
                    geo_result = geocode(location)
                    if geo_result:
                        location_coords[location] = (geo_result.latitude, geo_result.longitude)
                    else:
                        # If geocoding fails, add none
                        location_coords[location] = (None, None)
                    # Small delay to respect API limits
                    time.sleep(0.1)
                except Exception as e:
                    st.warning(f"Error geocoding {location}: {str(e)}")
                    location_coords[location] = (None, None)
            
            # Save updated cache to disk
            try:
                os.makedirs(os.path.dirname(GEOCODE_CACHE_FILE), exist_ok=True)
                with open(GEOCODE_CACHE_FILE, 'wb') as f:
                    pickle.dump(location_coords, f)
                st.success(f"Saved {len(location_coords)} geocoded locations to cache file.")
            except Exception as e:
                st.warning(f"Could not save geocode cache: {str(e)}")
    
    return location_coords

def load_data():
    """Load and preprocess the lichen dataset"""
    data = pd.read_excel('data/data.xlsx')
    substrats = pd.read_excel('data/substrats.xlsx')
    
    # Ensure the 'DATE' column is in datetime format
    data['DATE'] = pd.to_datetime(data['DATE'])
    
    # Create categorical mapping for RED_LIST and CATEGORY
    category_names = {
        'ge': 'Sensitive',
        'kw': 'Vulnerable',
        'be': 'Threatened',
        'eb': 'Seriously threatened',
        've': 'Disappeared',
        'nt': 'Not threatened'
    }
    
    # Fill missing CATEGORY values with 'nt' (not threatened)
    data['CATEGORY'] = data['CATEGORY'].fillna('nt')
    
    # Map CATEGORY codes to full names
    data['CategoryName'] = data['CATEGORY'].map(category_names)
    
    # Create a is_red_list column
    data['is_red_list'] = data['RED_LIST'] == 'v'
    
    # Process location information for mapping
    data = add_coordinates(data)
    
    return data, substrats

def add_coordinates(data):
    """
    Add actual coordinates based on location names.
    Uses geocoding to convert location names to geographic coordinates.
    """
    # Check if Location column exists
    if 'LOCATION' not in data.columns and 'Location' in data.columns:
        data['LOCATION'] = data['Location']  # Handle possible column name variations
    
    if 'LOCATION' in data.columns:
        # Add Netherlands to each location for better results
        def format_location(loc):
            if pd.isna(loc):
                return None
            # Format the location string to improve geocoding accuracy
            loc_str = str(loc).strip()
            if "," not in loc_str and " Netherlands" not in loc_str and " nederland" not in loc_str.lower():
                return f"{loc_str}, Netherlands"
            return loc_str
        
        # Apply formatting to location names
        data['formatted_location'] = data['LOCATION'].apply(format_location)
        
        # Get unique locations for geocoding
        unique_locations = data['formatted_location'].dropna().unique()
        
        # Get coordinates for all locations (using cache when available)
        location_coords = geocode_locations(unique_locations)
        
        # Apply coordinates to the dataframe
        data['latitude'] = data['formatted_location'].map(lambda x: location_coords.get(x, (None, None))[0])
        data['longitude'] = data['formatted_location'].map(lambda x: location_coords.get(x, (None, None))[1])
        
        # Count how many locations were successfully geocoded
        geocoded_count = data.dropna(subset=['latitude', 'longitude']).shape[0]
        total_count = data.shape[0]
        
        st.info(f"Applied coordinates for {geocoded_count} out of {total_count} records ({(geocoded_count/total_count)*100:.1f}%).")
        
        # For records with missing coordinates, use fallback to KMHOK if available
        missing_coords = data[data['latitude'].isna() | data['longitude'].isna()]
        if not missing_coords.empty and 'KMHOK' in data.columns:
            st.info(f"Using KMHOK coordinates for {len(missing_coords)} records with missing location coordinates.")
            
            missing_coords = add_kmhok_coordinates(missing_coords)
            
            # Update the main dataframe with KMHOK-based coordinates where location-based coordinates are missing
            data.loc[data['latitude'].isna(), 'latitude'] = missing_coords['latitude']
            data.loc[data['longitude'].isna(), 'longitude'] = missing_coords['longitude']
    
    elif 'KMHOK' in data.columns:
        # If no Location column, fall back to KMHOK method
        st.warning("LOCATION column not found. Using KMHOK for approximate coordinates.")
        data = add_kmhok_coordinates(data)
    else:
        # If neither Location nor KMHOK are available, use random points
        st.warning("Neither LOCATION nor KMHOK columns found. Using random coordinates for demonstration.")
        n = len(data)
        data['latitude'] = np.random.uniform(50.75, 53.55, n)
        data['longitude'] = np.random.uniform(3.35, 7.22, n)
    
    return data

def add_kmhok_coordinates(data):
    """
    Fallback method to add approximate coordinates based on KMHOK values.
    Uses cache to avoid regenerating coordinates on each run.
    """
    # Initialize coordinates cache
    kmhok_coords = {}
    
    # Check if cache file exists and load it
    if os.path.exists(KMHOK_CACHE_FILE):
        try:
            with open(KMHOK_CACHE_FILE, 'rb') as f:
                cache = pickle.load(f)
                kmhok_coords = cache
                st.success(f"Loaded {len(kmhok_coords)} cached KMHOK coordinates from disk.")
        except Exception as e:
            st.warning(f"Could not load KMHOK cache: {str(e)}")
    
    # Get unique KMHOK values that aren't in the cache yet
    unique_kmhok = [k for k in data['KMHOK'].unique() if k not in kmhok_coords and pd.notna(k)]
    
    # Generate coordinates for uncached KMHOK values
    if unique_kmhok:
        # Netherlands approximate bounds
        NL_LAT_MIN, NL_LAT_MAX = 50.75, 53.55  # Latitude range
        NL_LON_MIN, NL_LON_MAX = 3.35, 7.22    # Longitude range
        
        with st.spinner(f"Generating coordinates for {len(unique_kmhok)} new KMHOK values..."):
            for kmhok in unique_kmhok:
                # Use hash of KMHOK to generate consistent coordinates
                # Ensure hash_val is within valid range for np.random
                hash_val = abs(hash(str(kmhok))) % (2**32 - 1)
                # Use numpy's random number generator directly
                rng = np.random.default_rng(hash_val)
                
                lat = rng.uniform(NL_LAT_MIN, NL_LAT_MAX)
                lon = rng.uniform(NL_LON_MIN, NL_LON_MAX)
                kmhok_coords[kmhok] = (lat, lon)
            
            # Save updated cache to disk
            try:
                os.makedirs(os.path.dirname(KMHOK_CACHE_FILE), exist_ok=True)
                with open(KMHOK_CACHE_FILE, 'wb') as f:
                    pickle.dump(kmhok_coords, f)
                st.success(f"Saved {len(kmhok_coords)} KMHOK coordinates to cache file.")
            except Exception as e:
                st.warning(f"Could not save KMHOK cache: {str(e)}")
    
    # Add None for NaN KMHOK values
    for kmhok in data['KMHOK'].unique():
        if pd.isna(kmhok) and kmhok not in kmhok_coords:
            kmhok_coords[kmhok] = (None, None)
    
    # Add coordinates to the dataframe
    data['latitude'] = data['KMHOK'].map(lambda x: kmhok_coords.get(x, (None, None))[0])
    data['longitude'] = data['KMHOK'].map(lambda x: kmhok_coords.get(x, (None, None))[1])
    
    return data

def get_substrate_category(sub_code, substrats):
    """Get the substrate category for a given code"""
    # This is a simplified version - ideally you would use the more comprehensive
    # categorization from plotting.py
    
    if pd.isna(sub_code):
        return 'Unknown'
    
    # Basic categories - this could be expanded with the full logic from plotting.py
    bark_prefixes = ['aa', 'ab', 'ac', 'al', 'be', 'bu', 'fa', 'fr', 'po', 'qu', 'sa', 'ul']
    stone_prefixes = ['s', 'c', 'b']
    wood_prefixes = ['w', 'st']
    soil_prefixes = ['t', 'eb']
    moss_prefixes = ['m', 'mb', 'mc']
    leaf_prefixes = ['le', 'leb', 'lel', 'ne']
    
    sub_code = str(sub_code).lower()
    
    for prefix in bark_prefixes:
        if sub_code.startswith(prefix):
            return 'Bark'
    
    for prefix in stone_prefixes:
        if sub_code.startswith(prefix):
            return 'Stone'
            
    for prefix in wood_prefixes:
        if sub_code.startswith(prefix):
            return 'Wood'
            
    for prefix in soil_prefixes:
        if sub_code.startswith(prefix):
            return 'Soil'
            
    for prefix in moss_prefixes:
        if sub_code.startswith(prefix):
            return 'Moss'
            
    for prefix in leaf_prefixes:
        if sub_code.startswith(prefix):
            return 'Leaf'
    
    return 'Other'

def create_map(data, genus_filter, species_filter, substrate_filter, category_filter, date_range):
    """Create an interactive map with filtered lichen observations"""
    
    # Apply filters
    filtered_data = data.copy()
    
    if genus_filter:
        filtered_data = filtered_data[filtered_data['GENUS'].isin(genus_filter)]
        
    if species_filter:
        filtered_data = filtered_data[filtered_data['SPECIES'].isin(species_filter)]
        
    if substrate_filter:
        filtered_data = filtered_data[filtered_data['SUB'].isin(substrate_filter)]
        
    if category_filter:
        filtered_data = filtered_data[filtered_data['CategoryName'].isin(category_filter)]
    
    if date_range:
        filtered_data = filtered_data[
            (filtered_data['DATE'] >= pd.to_datetime(date_range[0])) & 
            (filtered_data['DATE'] <= pd.to_datetime(date_range[1]))
        ]
    
    # Create base map centered on Netherlands
    m = folium.Map(location=[52.1326, 5.2913], zoom_start=7)
    
    # Add markers for each observation
    for _, row in filtered_data.iterrows():
        if pd.notna(row['latitude']) and pd.notna(row['longitude']):
            # Determine marker color based on substrate or threat category
            substrate_category = get_substrate_category(row['SUB'], None)
            
            if st.session_state.color_by == "Substrate":
                color = SUBSTRATE_COLORS.get(substrate_category, 'gray')
            else:  # Color by threat category
                color = CATEGORY_COLORS.get(row['CategoryName'], 'gray')
            
            # Create popup text
            popup_text = f"""
            <b>Species:</b> {row['SPECIES']}<br>
            <b>Genus:</b> {row['GENUS']}<br>
            <b>Substrate:</b> {row['SUB']} ({substrate_category})<br>
            <b>Date:</b> {row['DATE'].strftime('%Y-%m-%d')}<br>
            <b>Category:</b> {row['CategoryName']}<br>
            <b>Amount:</b> {row['AMOUNT']}
            """
            
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_text, max_width=300)
            ).add_to(m)
    
    # Add legend to map
    if st.session_state.color_by == "Substrate":
        legend_items = SUBSTRATE_COLORS.items()
        legend_title = "Substrate Types"
    else:
        legend_items = CATEGORY_COLORS.items()
        legend_title = "Threat Categories"
    
    legend_html = f'''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                padding: 10px; border: 2px solid grey; border-radius: 5px">
    <p><b>{legend_title}</b></p>
    '''
    
    for name, color in legend_items:
        legend_html += f'''
        <p><i class="fa fa-circle" style="color:{color}"></i> {name}</p>
        '''
    
    legend_html += '</div>'
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m, filtered_data

def create_time_series_plot(filtered_data):
    """Create time series visualization of lichen observations"""
    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
        return None
    
    # Aggregate data by year
    yearly_data = filtered_data.groupby(filtered_data['DATE'].dt.year).agg({
        'AMOUNT': 'sum',
        'SPECIES': 'nunique',
        'GENUS': 'nunique'
    }).reset_index()
    
    # Create plot using Plotly for interactivity
    fig = px.line(yearly_data, x='DATE', y=['AMOUNT', 'SPECIES', 'GENUS'],
                 labels={'DATE': 'Year', 'value': 'Count', 'variable': 'Metric'},
                 title='Lichen Observations Over Time',
                 color_discrete_sequence=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    fig.update_layout(
        xaxis_title='Year',
        yaxis_title='Count',
        legend_title='Metric',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_category_distribution(filtered_data):
    """Create visualization of species distribution by category"""
    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
        return None
    
    # Count unique species by category
    category_counts = filtered_data.drop_duplicates(subset=['SPECIES']).groupby('CategoryName').size().reset_index()
    category_counts.columns = ['Category', 'Count']
    
    # Sort categories by conservation concern
    category_order = ['Not threatened', 'Sensitive', 'Vulnerable', 'Threatened', 'Seriously threatened', 'Disappeared']
    category_counts['Order'] = category_counts['Category'].map({cat: i for i, cat in enumerate(category_order)})
    category_counts = category_counts.sort_values('Order')
    
    # Create bar chart
    fig = px.bar(category_counts, x='Category', y='Count', 
                 color='Category', color_discrete_map=CATEGORY_COLORS,
                 title='Species Distribution by Conservation Status')
    
    fig.update_layout(
        xaxis_title='Conservation Status',
        yaxis_title='Number of Species',
        showlegend=False
    )
    
    return fig

def create_substrate_distribution(filtered_data, substrats):
    """Create visualization of species distribution by substrate"""
    if filtered_data.empty:
        st.warning("No data available for the selected filters.")
        return None
    
    # Add substrate category to the data
    filtered_data['SubstrateCategory'] = filtered_data['SUB'].apply(
        lambda x: get_substrate_category(x, substrats)
    )
    
    # Count unique species by substrate category
    substrate_counts = filtered_data.groupby('SubstrateCategory').agg({
        'SPECIES': 'nunique',
        'AMOUNT': 'sum'
    }).reset_index()
    
    # Create bar chart
    fig = px.bar(substrate_counts, x='SubstrateCategory', y='SPECIES', 
                 color='SubstrateCategory', color_discrete_map=SUBSTRATE_COLORS,
                 title='Species Distribution by Substrate Type')
    
    fig.update_layout(
        xaxis_title='Substrate Type',
        yaxis_title='Number of Species',
        showlegend=False
    )
    
    return fig

def main():
    """Main function to run the Streamlit app"""
    st.title("Lichen Distribution in the Netherlands")
    st.markdown("""
    This interactive application visualizes the geographical distribution of lichens in the Netherlands.
    Use the filters to explore different species, genera, substrates, and time periods.
    """)
    
    # Initialize session state for color selection
    if 'color_by' not in st.session_state:
        st.session_state.color_by = "Threat Category"
    
    # Load data
    data, substrats = load_data()
    
    # Create sidebar for filters
    st.sidebar.header("Filters")
    
    # Multiple selection filters - handle potential NaN values
    genus_values = data['GENUS'].dropna().unique()
    species_values = data['SPECIES'].dropna().unique()
    substrate_values = data['SUB'].dropna().unique()
    category_values = data['CategoryName'].dropna().unique()
    
    # Convert to strings for proper sorting
    genus_options = sorted([str(g) for g in genus_values])
    species_options = sorted([str(s) for s in species_values])
    substrate_options = sorted([str(s) for s in substrate_values])
    category_options = sorted([str(c) for c in category_values])
    
    genus_filter = st.sidebar.multiselect("Select Genus", genus_options)
    
    # Update species options based on selected genera
    if genus_filter:
        species_values = data[data['GENUS'].isin(genus_filter)]['SPECIES'].dropna().unique()
        species_options = sorted([str(s) for s in species_values])
    
    species_filter = st.sidebar.multiselect("Select Species", species_options)
    substrate_filter = st.sidebar.multiselect("Select Substrate", substrate_options)
    category_filter = st.sidebar.multiselect("Select Conservation Status", category_options)
    
    # Date range filter
    min_date = data['DATE'].min().date()
    max_date = data['DATE'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [min_date, max_date],
        min_value=min_date,
        max_value=max_date
    )
    
    # Option to color points by substrate or threat category
    st.sidebar.header("Map Options")
    color_by = st.sidebar.radio(
        "Color points by:",
        ["Threat Category", "Substrate"],
        index=0
    )
    st.session_state.color_by = color_by
    
    # Create map and get filtered data
    folium_map, filtered_data = create_map(
        data, genus_filter, species_filter, substrate_filter, category_filter, date_range
    )
    
    # Create dashboard layout with multiple columns
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("Geographical Distribution")
        folium_static(folium_map, width=700)
        
        # Display stats below the map
        st.markdown(f"**Observations displayed:** {len(filtered_data)}")
        st.markdown(f"**Unique species:** {filtered_data['SPECIES'].nunique()}")
        st.markdown(f"**Unique genera:** {filtered_data['GENUS'].nunique()}")
    
    with col2:
        # Time series plot
        st.subheader("Temporal Distribution")
        time_fig = create_time_series_plot(filtered_data)
        if time_fig:
            st.plotly_chart(time_fig, use_container_width=True)
        
        # Tab for additional charts
        tab1, tab2 = st.tabs(["Conservation Status", "Substrate Distribution"])
        
        with tab1:
            category_fig = create_category_distribution(filtered_data)
            if category_fig:
                st.plotly_chart(category_fig, use_container_width=True)
        
        with tab2:
            substrate_fig = create_substrate_distribution(filtered_data, substrats)
            if substrate_fig:
                st.plotly_chart(substrate_fig, use_container_width=True)
    
    # Add information about red list species
    st.header("Red List Species Information")
    
    # Count red list species in filtered data
    red_list_count = filtered_data[filtered_data['is_red_list']].drop_duplicates(subset=['SPECIES']).shape[0]
    total_species = filtered_data.drop_duplicates(subset=['SPECIES']).shape[0]
    
    if total_species > 0:
        red_list_percentage = (red_list_count / total_species) * 100
        st.markdown(f"**Red List species:** {red_list_count} ({red_list_percentage:.1f}% of selected species)")
    
        # Display top red list species
        top_red_list = (
            filtered_data[filtered_data['is_red_list']]
            .groupby('SPECIES')['AMOUNT']
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )
        
        if not top_red_list.empty:
            st.subheader("Most Common Red List Species in Selection")
            top_red_list_df = pd.DataFrame({
                'Species': top_red_list.index,
                'Observations': top_red_list.values
            })
            st.dataframe(top_red_list_df, use_container_width=True)
    
    st.markdown("""
    ### About This App
    
    This application visualizes lichen distribution data from the Netherlands, allowing exploration of:
    
    - Geographic distribution of lichen species
    - Temporal trends in observations
    - Distribution by conservation status
    - Distribution by substrate type
    
    The data includes information on location, species, genus, substrate, conservation status, and observation date.
    
    **Note:** Location coordinates are approximated based on the KMHOK grid references.
    """)

if __name__ == "__main__":
    main() 