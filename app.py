import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="Bike Sharing Analysis Dashboard",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function for Plotly dark mode compatibility
def configure_plotly_for_both_modes(fig):
    """Configure Plotly figure to be visible in both light and dark mode"""
    fig.update_layout(
        # Background colors
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        
        # Text colors - using a darker color for better visibility in light mode
        font=dict(
            color="#333333" 
        ),
        
        # Grid colors that work in both modes
        xaxis=dict(
            gridcolor="rgba(80,80,80,0.2)",
            zerolinecolor="rgba(80,80,80,0.5)"
        ),
        yaxis=dict(
            gridcolor="rgba(80,80,80,0.2)",
            zerolinecolor="rgba(80,80,80,0.5)"
        )
    )
    
    return fig

# Custom CSS to improve aesthetics with both light and dark mode support
st.markdown("""
<style>
    /* Headers with colors that work in both light and dark mode */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3498db !important; /* Blue that works in both modes */
        margin-bottom: 1rem;
        text-align: center;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #3498db !important; /* Blue that works in both modes */
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: #f39c12 !important; /* Orange that works in both modes */
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .highlight {
        color: #e74c3c !important;
        font-weight: 600;
    }
    .insight-box {
        background-color: rgba(52, 152, 219, 0.1) !important;
        border-left: 4px solid #3498db !important;
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.3rem;
        color: #333333 !important; /* Dark text color for light mode visibility */
    }
    
    /* Make sure text stays visible */
    .stMarkdown {
        color: #333333 !important; /* Darker color for better light mode visibility */
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #333333 !important; /* Darker color for better light mode visibility */
    }
    
    /* Plotly-specific styles */
    .js-plotly-plot .plotly .gtitle, 
    .js-plotly-plot .plotly .xtitle,
    .js-plotly-plot .plotly .ytitle {
        fill: #333333 !important; /* Darker color for better light mode visibility */
    }
    
    .js-plotly-plot .plotly .xtick text,
    .js-plotly-plot .plotly .ytick text {
        fill: #333333 !important; /* Darker color for better light mode visibility */
    }

    /* Streamlit caption and text elements */
    .caption {
        color: #333333 !important;
    }
    
    p {
        color: #333333 !important;
    }
    
    li {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="main-header">🚲 Bike Sharing Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown("""
<p style="color: #333333;">This dashboard analyzes the bike sharing dataset from Capital Bikeshare system in Washington D.C., USA.
The data covers the years 2011 and 2012, and includes information on hourly and daily bike rentals,
along with weather and seasonal information.</p>
""", unsafe_allow_html=True)

# Data loading function
@st.cache_data
def load_data():
    day_df = pd.read_csv('day.csv')
    hour_df = pd.read_csv('hour.csv')
    
    # Convert date string to datetime
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
    
    # Create month and year columns for easier filtering
    day_df['month_name'] = day_df['dteday'].dt.strftime('%b')
    hour_df['month_name'] = hour_df['dteday'].dt.strftime('%b')
    
    # Map categorical variables for better readability
    season_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    weather_map = {1: 'Clear', 2: 'Mist/Cloudy', 3: 'Light Rain/Snow', 4: 'Heavy Rain/Snow'}
    weekday_map = {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}
    
    day_df['season_name'] = day_df['season'].map(season_map)
    day_df['weathersit_name'] = day_df['weathersit'].map(weather_map)
    day_df['weekday_name'] = day_df['weekday'].map(weekday_map)
    
    hour_df['season_name'] = hour_df['season'].map(season_map)
    hour_df['weathersit_name'] = hour_df['weathersit'].map(weather_map)
    hour_df['weekday_name'] = hour_df['weekday'].map(weekday_map)
    
    # Create workingday and holiday readable names
    day_df['workingday_name'] = day_df['workingday'].map({0: 'Weekend/Holiday', 1: 'Working Day'})
    day_df['holiday_name'] = day_df['holiday'].map({0: 'Regular Day', 1: 'Holiday'})
    
    hour_df['workingday_name'] = hour_df['workingday'].map({0: 'Weekend/Holiday', 1: 'Working Day'})
    hour_df['holiday_name'] = hour_df['holiday'].map({0: 'Regular Day', 1: 'Holiday'})
    
    # Create day category
    day_df['day_category'] = 'Working Day'
    day_df.loc[day_df['holiday'] == 1, 'day_category'] = 'Holiday'
    day_df.loc[(day_df['workingday'] == 0) & (day_df['holiday'] == 0), 'day_category'] = 'Weekend'
    
    hour_df['day_category'] = 'Working Day'
    hour_df.loc[hour_df['holiday'] == 1, 'day_category'] = 'Holiday'
    hour_df.loc[(hour_df['workingday'] == 0) & (hour_df['holiday'] == 0), 'day_category'] = 'Weekend'
    
    return day_df, hour_df

# Load the data
with st.spinner('Loading data...'):
    day_df, hour_df = load_data()

# Sidebar for filtering
st.sidebar.markdown('### Filters')

# Date range filter
date_range = st.sidebar.date_input(
    "Select date range",
    [day_df['dteday'].min().date(), day_df['dteday'].max().date()]
)

# Season filter
selected_seasons = st.sidebar.multiselect(
    "Select seasons",
    options=day_df['season_name'].unique(),
    default=day_df['season_name'].unique()
)

# Weather filter
selected_weather = st.sidebar.multiselect(
    "Select weather conditions",
    options=day_df['weathersit_name'].unique(),
    default=day_df['weathersit_name'].unique()
)

# Apply filters
filtered_day_df = day_df[
    (day_df['dteday'].dt.date >= date_range[0]) &
    (day_df['dteday'].dt.date <= date_range[1]) &
    (day_df['season_name'].isin(selected_seasons)) &
    (day_df['weathersit_name'].isin(selected_weather))
]

filtered_hour_df = hour_df[
    (hour_df['dteday'].dt.date >= date_range[0]) &
    (hour_df['dteday'].dt.date <= date_range[1]) &
    (hour_df['season_name'].isin(selected_seasons)) &
    (hour_df['weathersit_name'].isin(selected_weather))
]

# Main dashboard content
# Key metrics at the top
st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    total_rentals = filtered_day_df['cnt'].sum()
    st.metric("Total Bike Rentals", f"{total_rentals:,}")

with col2:
    avg_daily_rentals = int(filtered_day_df['cnt'].mean())
    st.metric("Avg. Daily Rentals", f"{avg_daily_rentals:,}")

with col3:
    casual_percentage = (filtered_day_df['casual'].sum() / total_rentals * 100).round(1)
    st.metric("Casual Users", f"{casual_percentage}%")

with col4:
    registered_percentage = (filtered_day_df['registered'].sum() / total_rentals * 100).round(1)
    st.metric("Registered Users", f"{registered_percentage}%")

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs(["Weather Impact", "Day Type Analysis", "Combined Analysis", "Data Explorer"])

# Tab 1: Weather Impact
with tab1:
    st.markdown('<div class="section-header">Weather Impact on Bike Rentals</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="subsection-header">Average Rentals by Weather Condition</div>', unsafe_allow_html=True)
        weather_stats = filtered_day_df.groupby('weathersit_name').agg({
            'cnt': 'mean',
            'casual': 'mean',
            'registered': 'mean'
        }).reset_index()
        
        fig = px.bar(
            weather_stats, 
            x='weathersit_name', 
            y='cnt',
            color='weathersit_name',
            labels={'weathersit_name': 'Weather Condition', 'cnt': 'Average Rentals'},
            text_auto='.0f',
            color_discrete_sequence=px.colors.qualitative.Set1  # Brighter colors
        )
        fig.update_layout(showlegend=False)
        fig = configure_plotly_for_both_modes(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="insight-box">Clear weather conditions show significantly higher rental rates compared to rainy or snowy days. Weather has a direct impact on bike rental behavior.</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="subsection-header">User Types by Weather Condition</div>', unsafe_allow_html=True)
        weather_user_type = filtered_day_df.groupby('weathersit_name').agg({
            'casual': 'mean',
            'registered': 'mean'
        }).reset_index()
        
        weather_user_type_melted = pd.melt(
            weather_user_type, 
            id_vars=['weathersit_name'], 
            value_vars=['casual', 'registered'],
            var_name='user_type', 
            value_name='avg_rentals'
        )
        
        fig = px.bar(
            weather_user_type_melted, 
            x='weathersit_name', 
            y='avg_rentals', 
            color='user_type',
            barmode='group',
            labels={
                'weathersit_name': 'Weather Condition', 
                'avg_rentals': 'Average Rentals',
                'user_type': 'User Type'
            },
            color_discrete_map={'casual': '#e74c3c', 'registered': '#3498db'}
        )
        fig = configure_plotly_for_both_modes(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="insight-box">Casual users are more affected by weather conditions, showing a steeper decline in poor weather compared to registered users who maintain more consistent usage patterns.</div>', unsafe_allow_html=True)
    
    # Temperature vs. Rentals
    st.markdown('<div class="subsection-header">Temperature and Humidity Effects</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            filtered_day_df, 
            x='temp', 
            y='cnt', 
            color='weathersit_name',
            size='cnt',
            hover_data=['dteday', 'season_name', 'hum', 'windspeed'],
            labels={
                'temp': 'Normalized Temperature', 
                'cnt': 'Total Rentals',
                'weathersit_name': 'Weather'
            },
            title='Bike Rentals by Temperature and Weather'
        )
        fig = configure_plotly_for_both_modes(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Group data by temperature and humidity ranges
        filtered_day_df['temp_bin'] = pd.cut(filtered_day_df['temp'], bins=10)
        filtered_day_df['hum_bin'] = pd.cut(filtered_day_df['hum'], bins=10)
        
        weather_heatmap = filtered_day_df.groupby(['temp_bin', 'hum_bin']).agg({
            'cnt': 'mean'
        }).reset_index()
        
        # Create temp and humidity midpoints for better labeling
        weather_heatmap['temp_mid'] = weather_heatmap['temp_bin'].apply(lambda x: x.mid)
        weather_heatmap['hum_mid'] = weather_heatmap['hum_bin'].apply(lambda x: x.mid)
        
        fig = px.density_heatmap(
            weather_heatmap, 
            x='temp_mid', 
            y='hum_mid', 
            z='cnt',
            labels={
                'temp_mid': 'Normalized Temperature', 
                'hum_mid': 'Normalized Humidity', 
                'cnt': 'Average Rentals'
            },
            title='Bike Rentals by Temperature and Humidity'
        )
        fig = configure_plotly_for_both_modes(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="insight-box">Temperature has a strong positive correlation with bike rentals up to a certain point (around 0.7 normalized temperature), after which extremely high temperatures may reduce rental rates. High humidity combined with high temperatures tends to suppress bike rental activity.</div>', unsafe_allow_html=True)

# Tab 2: Day Type Analysis
with tab2:
    st.markdown('<div class="section-header">Working Days vs. Holidays Analysis</div>', unsafe_allow_html=True)
    
    # Hourly patterns
    st.markdown('<div class="subsection-header">Hourly Rental Patterns by Day Type</div>', unsafe_allow_html=True)
    
    hourly_patterns = filtered_hour_df.groupby(['hr', 'workingday']).agg({
        'cnt': 'mean',
        'casual': 'mean',
        'registered': 'mean'
    }).reset_index()
    
    fig = px.line(
        hourly_patterns, 
        x='hr', 
        y='cnt', 
        color='workingday',
        labels={
            'hr': 'Hour of Day', 
            'cnt': 'Average Rentals',
            'workingday': 'Day Type'
        },
        title='Hourly Bike Rental Patterns',
        markers=True,
        color_discrete_map={0: '#e74c3c', 1: '#3498db'}
    )
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1),
        legend=dict(
            title='Day Type',
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=0.01,
            orientation='h'
        )
    )
    fig.update_traces(
        hovertemplate='Hour: %{x}<br>Average Rentals: %{y:.1f}'
    )
    fig = configure_plotly_for_both_modes(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="insight-box">Working days show a characteristic bimodal pattern with peaks during commuting hours (8 AM and 5-6 PM), while weekends and holidays display a more even distribution with a single broad peak in the afternoon. This clearly indicates different use cases: utilitarian commuting on working days versus recreational riding on non-working days.</div>', unsafe_allow_html=True)
    
    # User types by day category
    st.markdown('<div class="subsection-header">User Types by Day Category</div>', unsafe_allow_html=True)
    
    user_day_category = filtered_day_df.groupby('day_category').agg({
        'casual': 'mean',
        'registered': 'mean',
        'cnt': 'mean'
    }).reset_index()
    
    user_day_category_melted = pd.melt(
        user_day_category, 
        id_vars=['day_category'], 
        value_vars=['casual', 'registered'],
        var_name='user_type', 
        value_name='avg_rentals'
    )
    
    fig = px.bar(
        user_day_category_melted, 
        x='day_category', 
        y='avg_rentals', 
        color='user_type',
        barmode='group',
        labels={
            'day_category': 'Day Category', 
            'avg_rentals': 'Average Rentals',
            'user_type': 'User Type'
        },
        color_discrete_map={'casual': '#e74c3c', 'registered': '#3498db'}
    )
    fig = configure_plotly_for_both_modes(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Box plots comparing rentals by day category
        fig = px.box(
            filtered_day_df, 
            x='day_category', 
            y='cnt',
            color='day_category',
            labels={
                'day_category': 'Day Category', 
                'cnt': 'Total Rentals'
            },
            title='Distribution of Bike Rentals by Day Category'
        )
        fig = configure_plotly_for_both_modes(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Hourly distribution of user types on weekends vs working days
        st.markdown('<div class="subsection-header">User Types Throughout the Day</div>', unsafe_allow_html=True)
        
        user_hr_type = filtered_hour_df.groupby(['hr', 'workingday_name']).agg({
            'casual': 'mean',
            'registered': 'mean'
        }).reset_index()
        
        day_types = st.radio(
            "Select day type to view hourly user pattern:",
            ['Working Day', 'Weekend/Holiday']
        )
        
        hourly_user_data = filtered_hour_df[filtered_hour_df['workingday_name'] == day_types]
        hourly_user_avg = hourly_user_data.groupby('hr').agg({
            'casual': 'mean',
            'registered': 'mean'
        }).reset_index()
        
        hourly_user_melted = pd.melt(
            hourly_user_avg, 
            id_vars=['hr'], 
            value_vars=['casual', 'registered'],
            var_name='user_type', 
            value_name='avg_rentals'
        )
        
        fig = px.bar(
            hourly_user_melted, 
            x='hr', 
            y='avg_rentals', 
            color='user_type',
            barmode='stack',
            labels={
                'hr': 'Hour of Day', 
                'avg_rentals': 'Average Rentals',
                'user_type': 'User Type'
            },
            title=f'Hourly Distribution of User Types on {day_types}',
            color_discrete_map={'casual': '#e74c3c', 'registered': '#3498db'}
        )
        fig.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))
        fig = configure_plotly_for_both_modes(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('''<div class="insight-box">Registered users dominate on working days, particularly during peak commuting hours, while casual users make up a significantly larger proportion of weekend and holiday rentals. On holidays, there's a notable decrease in registered users compared to regular weekends, suggesting many registered commuters may be away or not using the service.</div>''', unsafe_allow_html=True)

# Tab 3: Combined Analysis
with tab3:
    st.markdown('<div class="section-header">Combined Analysis: Weather and Day Type</div>', unsafe_allow_html=True)
    
    # Create combo analysis
    combo_analysis = filtered_day_df.groupby(['workingday_name', 'weathersit_name']).agg({
        'cnt': 'mean',
        'casual': 'mean',
        'registered': 'mean'
    }).reset_index()
    
    # Plot interactive heatmap
    combo_pivot = combo_analysis.pivot(index='weathersit_name', columns='workingday_name', values='cnt')
    
    fig = px.imshow(
        combo_pivot,
        labels=dict(x="Day Type", y="Weather Condition", color="Average Rentals"),
        x=combo_pivot.columns,
        y=combo_pivot.index,
        text_auto='.0f',
        aspect="auto",
        title="Average Rentals by Weather and Day Type",
        color_continuous_scale='viridis'
    )
    fig = configure_plotly_for_both_modes(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    # Bar chart showing user types by weather and day type
    st.markdown('<div class="subsection-header">User Types by Weather and Day Type</div>', unsafe_allow_html=True)
    
    user_weather_day = combo_analysis.copy()
    user_weather_day['casual_pct'] = (user_weather_day['casual'] / user_weather_day['cnt'] * 100).round(1)
    user_weather_day['registered_pct'] = (user_weather_day['registered'] / user_weather_day['cnt'] * 100).round(1)
    
    fig = px.bar(
        user_weather_day,
        x='weathersit_name',
        y='cnt',
        color='workingday_name',
        barmode='group',
        facet_row='workingday_name',
        labels={
            'weathersit_name': 'Weather Condition',
            'cnt': 'Average Rentals',
            'workingday_name': 'Day Type'
        },
        title="Average Rentals by Weather Condition and Day Type",
        text_auto='.0f'
    )
    
    # Add percentage annotations with color for both light and dark mode
    for i, row in user_weather_day.iterrows():
        fig.add_annotation(
            x=row['weathersit_name'],
            y=row['cnt'] + 100,
            text=f"C: {row['casual_pct']}%<br>R: {row['registered_pct']}%",
            showarrow=False,
            xanchor='center',
            yanchor='bottom',
            font=dict(size=10, color="#333333")  # Dark color for light mode visibility
        )
    
    fig = configure_plotly_for_both_modes(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="insight-box">The combination of weather and day type reveals interesting patterns: clear weather on weekends generates the highest casual ridership, while working days maintain strong registered user activity regardless of weather (except in extreme conditions). This suggests weather has a more pronounced effect on recreational riding compared to commuting.</div>', unsafe_allow_html=True)
    
    # Season and weather interaction
    st.markdown('<div class="subsection-header">Season, Weather and Day Type Interaction</div>', unsafe_allow_html=True)
    
    season_weather_day = filtered_day_df.groupby(['season_name', 'weathersit_name', 'workingday_name']).agg({
        'cnt': 'mean'
    }).reset_index()
    
    fig = px.sunburst(
        season_weather_day,
        path=['season_name', 'weathersit_name', 'workingday_name'],
        values='cnt',
        color='cnt',
        color_continuous_scale='viridis',
        labels={
            'cnt': 'Average Rentals',
            'season_name': 'Season',
            'weathersit_name': 'Weather',
            'workingday_name': 'Day Type'
        },
        title="Hierarchical View of Factors Affecting Bike Rentals"
    )
    fig = configure_plotly_for_both_modes(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="insight-box">The hierarchical view shows the compounding effects of season, weather, and day type. Summer and fall seasons with clear weather generate the highest rental activity regardless of day type, while winter rentals are more affected by both weather conditions and day type.</div>', unsafe_allow_html=True)

# Tab 4: Data Explorer
with tab4:
    st.markdown('<div class="section-header">Data Explorer</div>', unsafe_allow_html=True)
    
    # Allow user to select variables for custom visualization
    st.markdown('<div class="subsection-header">Custom Visualization</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        chart_type = st.selectbox(
            "Select chart type",
            ["Scatter Plot", "Line Chart", "Bar Chart", "Box Plot", "Heatmap"]
        )
    
    with col2:
        dataset = st.radio(
            "Select dataset",
            ["Daily Data", "Hourly Data"]
        )
    
    # Set the dataframe based on selection
    df = filtered_day_df if dataset == "Daily Data" else filtered_hour_df
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Get categorical columns
    categorical_cols = [col for col in df.columns if col.endswith('_name') or col in ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'day_category']]
    
    # Create custom chart based on user selection
    if chart_type == "Scatter Plot":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox("Select X variable", numeric_cols, index=numeric_cols.index('temp') if 'temp' in numeric_cols else 0)
        
        with col2:
            y_var = st.selectbox("Select Y variable", numeric_cols, index=numeric_cols.index('cnt') if 'cnt' in numeric_cols else 0)
        
        with col3:
            color_var = st.selectbox("Select color variable", categorical_cols, index=categorical_cols.index('season_name') if 'season_name' in categorical_cols else 0)
        
        fig = px.scatter(
            df, 
            x=x_var, 
            y=y_var, 
            color=color_var,
            hover_data=['dteday'],
            labels={x_var: x_var, y_var: y_var, color_var: color_var},
            title=f"{y_var} vs {x_var} by {color_var}"
        )
        fig = configure_plotly_for_both_modes(fig)
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Line Chart":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox(
                "Select X variable", 
                ['dteday', 'mnth', 'hr'] if 'hr' in df.columns else ['dteday', 'mnth'],
                index=0
            )
        
        with col2:
            y_var = st.selectbox("Select Y variable", ['cnt', 'casual', 'registered'], index=0)
        
        with col3:
            color_var = st.selectbox("Select color variable", categorical_cols, index=0)
        
        # Group data for line chart
        if x_var == 'dteday':
            line_data = df.groupby([x_var, color_var])[y_var].mean().reset_index()
        else:
            line_data = df.groupby([x_var, color_var])[y_var].mean().reset_index()
        
        fig = px.line(
            line_data, 
            x=x_var, 
            y=y_var, 
            color=color_var,
            markers=True,
            labels={x_var: x_var, y_var: f"Average {y_var}", color_var: color_var},
            title=f"Average {y_var} by {x_var} and {color_var}"
        )
        fig = configure_plotly_for_both_modes(fig)
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Bar Chart":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox("Select X variable", categorical_cols, index=0)
        
        with col2:
            y_var = st.selectbox("Select Y variable", ['cnt', 'casual', 'registered'], index=0)
        
        with col3:
            color_var = st.selectbox("Select secondary grouping", [None] + categorical_cols, index=0)
        
        # Group data for bar chart
        if color_var is None:
            bar_data = df.groupby(x_var)[y_var].mean().reset_index()
            
            fig = px.bar(
                bar_data, 
                x=x_var, 
                y=y_var,
                labels={x_var: x_var, y_var: f"Average {y_var}"},
                title=f"Average {y_var} by {x_var}"
            )
        else:
            bar_data = df.groupby([x_var, color_var])[y_var].mean().reset_index()
            
            fig = px.bar(
                bar_data, 
                x=x_var, 
                y=y_var, 
                color=color_var,
                barmode='group',
                labels={x_var: x_var, y_var: f"Average {y_var}", color_var: color_var},
                title=f"Average {y_var} by {x_var} and {color_var}"
            )
        fig = configure_plotly_for_both_modes(fig)
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Box Plot":
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox("Select grouping variable", categorical_cols, index=0)
        
        with col2:
            y_var = st.selectbox("Select value variable", numeric_cols, index=numeric_cols.index('cnt') if 'cnt' in numeric_cols else 0)
        
        fig = px.box(
            df, 
            x=x_var, 
            y=y_var,
            color=x_var,
            labels={x_var: x_var, y_var: y_var},
            title=f"Distribution of {y_var} by {x_var}"
        )
        fig = configure_plotly_for_both_modes(fig)
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_type == "Heatmap":
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_var = st.selectbox("Select X variable", categorical_cols, index=0)
        
        with col2:
            y_var = st.selectbox("Select Y variable", categorical_cols, index=min(1, len(categorical_cols)-1))
        
        with col3:
            value_var = st.selectbox("Select value to aggregate", numeric_cols, index=numeric_cols.index('cnt') if 'cnt' in numeric_cols else 0)
        
        # Group data for heatmap
        heatmap_data = df.groupby([y_var, x_var])[value_var].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index=y_var, columns=x_var, values=value_var)
        
        fig = px.imshow(
            heatmap_pivot,
            labels=dict(x=x_var, y=y_var, color=f"Average {value_var}"),
            x=heatmap_pivot.columns,
            y=heatmap_pivot.index,
            aspect="auto",
            title=f"Average {value_var} by {x_var} and {y_var}",
            text_auto='.0f'
        )
        fig = configure_plotly_for_both_modes(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # Show raw data
    st.markdown('<div class="subsection-header">Raw Data</div>', unsafe_allow_html=True)
    
    show_data = st.checkbox("Show raw data")
    
    if show_data:
        st.dataframe(df)

# Footer
st.markdown("""---""")
st.markdown("""
<div class="footer">
    <p>Bike Sharing Dashboard - Created for Data Analysis Project</p>
    <p>Data source: Capital Bikeshare, Washington D.C. (2011-2012)</p>
</div>
""", unsafe_allow_html=True)