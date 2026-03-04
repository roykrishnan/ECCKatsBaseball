import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from datetime import datetime, date, timedelta
import os
import glob
import requests 

# Session configuration
SESSION_CONFIG = {
    "December 2025": {
        "bullpen_dir": "data/BullpenReports120625",
        "date_code": "120625"
    },
    "February 2026": {
        "bullpen_dir": "data/BullpenReports021526",
        "date_code": "021526"
    }
}

# Set matplotlib style for dark mode
plt.style.use('dark_background')
sns.set_style("darkgrid")

# Page configuration
st.set_page_config(
    page_title="ECC Kats Baseball Performance Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling - Kats Theme
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: white !important;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #C0C0C0 !important;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        opacity: 0.9;
    }
    .leaderboard-title {
        color: #C41E3A;
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 1rem;
        border-bottom: 2px solid #C41E3A;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #C41E3A;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(196, 30, 58, 0.2);
    }
    .rank-1 {
        background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
        color: #1a1a1a;
        font-weight: bold;
    }
    .rank-2 {
        background: linear-gradient(135deg, #C0C0C0 0%, #A9A9A9 100%);
        color: #1a1a1a;
        font-weight: bold;
    }
    .rank-3 {
        background: linear-gradient(135deg, #CD7F32 0%, #B8860B 100%);
        color: white;
        font-weight: bold;
    }
    .stSelectbox > div > div {
        background-color: #C41E3A !important;
        border: 2px solid #C41E3A !important;
    }
    .stSelectbox > div > div > div {
        color: white !important;
    }
    .stSelectbox > div > div > div > div {
        color: white !important;
    }
    .stSelectbox span {
        color: white !important;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
    }
    .stSelectbox div[data-baseweb="select"] span {
        color: white !important;
    }
            
    .stSelectbox label {
        color: white !important;
        font-weight: bold !important;
    }
            
    .stTab [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTab [data-baseweb="tab"] {
        background-color: #2d2d2d;
        color: #C0C0C0;
        border: 1px solid #C41E3A;
        border-radius: 4px 4px 0 0;
    }
    .stTab [data-baseweb="tab"][aria-selected="true"] {
        background-color: #C41E3A;
        color: #ffffff;
    }
    .stMetric > div {
        background-color: #1a1a1a !important;
        border: 1px solid #C41E3A !important;
        border-radius: 6px !important;
        padding: 0.5rem !important;
    }
    .stMetric [data-testid="metric-container"] {
        background-color: #1a1a1a !important;
        border: 1px solid #C41E3A !important;
        border-radius: 6px !important;
        padding: 0.5rem !important;
    }
    .stMetric [data-testid="metric-container"] > div {
        color: #C0C0C0 !important;
    }
    .stMetric .metric-label,
    .stMetric [data-testid="metric-container"] label {
        color: #C0C0C0 !important;
        font-weight: bold !important;
    }
    .stMetric .metric-value,
    .stMetric [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #C41E3A !important;
        font-weight: bold !important;
    }
    
    /* Force metric styling */
    .stMetric * {
        color: #C0C0C0 !important;
    }
    
    .stMetric div,
    .stMetric span,
    .stMetric p,
    .stMetric label {
        color: #C0C0C0 !important;
    }
    
    [data-testid="metric-container"] * {
        color: #C0C0C0 !important;
    }
    
    /* Accent red for important values */
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: #C41E3A !important;
    }
</style>
""", unsafe_allow_html=True)

def get_handedness_from_raw_data(data_dir="data"):
    """
    Dynamically determine pitcher handedness from fastball horizontal break.
    Negative horizontal break = LHP (ball breaks arm-side for lefty)
    Positive horizontal break = RHP (ball breaks arm-side for righty)
    """
    handedness_map = {}
    
    if not os.path.exists(data_dir):
        return handedness_map
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    for csv_file in csv_files:
        try:
            encodings_to_try = ['utf-8', 'utf-16', 'latin1', 'cp1252', 'iso-8859-1']
            lines = None
            successful_encoding = None
            
            for encoding in encodings_to_try:
                try:
                    with open(csv_file, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    successful_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if lines is None:
                continue
            
            player_name = None
            data_start_row = None
            
            for i, line in enumerate(lines):
                if 'Player Name:' in line:
                    player_name = line.split(',')[1].strip()
                elif line.startswith('No,Date'):
                    data_start_row = i
                    break
            
            if data_start_row is not None and player_name:
                pitch_data = pd.read_csv(csv_file, skiprows=data_start_row, encoding=successful_encoding)
                
                # Filter for fastballs
                fastball_data = pitch_data[pitch_data['Pitch Type'].str.contains('Fastball', case=False, na=False)]
                
                if len(fastball_data) > 0 and 'HB (trajectory)' in fastball_data.columns:
                    fastball_data = fastball_data.copy()
                    fastball_data['HB (trajectory)'] = pd.to_numeric(fastball_data['HB (trajectory)'], errors='coerce')
                    avg_hb = fastball_data['HB (trajectory)'].mean()
                    
                    if pd.notna(avg_hb):
                        # Negative HB = LHP, Positive HB = RHP
                        handedness_map[player_name] = 'LHP' if avg_hb < 0 else 'RHP'
        except Exception:
            continue
    
    return handedness_map


@st.cache_data
def get_cached_handedness_map(data_dir="data"):
    """Cache the handedness detection to avoid repeated file reads"""
    return get_handedness_from_raw_data(data_dir)

@st.cache_data
def load_rapsodo_data(data_dir):
    """Load Rapsodo pitching data from CSV files in specified directory"""
    all_player_data = []
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please ensure the data folder exists.")
    
    # Get all CSV files in the data directory
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}' directory.")
    
    for csv_file in csv_files:
        try:
            # Try different encodings to handle various file formats
            encodings_to_try = ['utf-8', 'utf-16', 'latin1', 'cp1252', 'iso-8859-1']
            lines = None
            successful_encoding = None
            
            for encoding in encodings_to_try:
                try:
                    with open(csv_file, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    successful_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if lines is None:
                st.warning(f"Could not read file {csv_file} with any supported encoding")
                continue
            
            player_id = None
            player_name = None
            data_start_row = None
            
            # Find player info and data start
            for i, line in enumerate(lines):
                if 'Player ID:' in line:
                    player_id = line.split(',')[1].strip()
                elif 'Player Name:' in line:
                    player_name = line.split(',')[1].strip()
                elif line.startswith('No,Date'):
                    data_start_row = i
                    break
            
            if data_start_row is not None and player_name and player_id:
                # Read the pitch data with the same encoding
                pitch_data = pd.read_csv(csv_file, skiprows=data_start_row, encoding=successful_encoding)
                
                # Filter out rows with missing pitch type
                pitch_data = pitch_data[pitch_data['Pitch Type'].notna()]
                pitch_data = pitch_data[pitch_data['Pitch Type'] != '-']
                pitch_data = pitch_data[pitch_data['Pitch Type'] != '']
                
                if len(pitch_data) > 0:
                    # Convert numeric columns
                    numeric_cols = ['Velocity', 'Total Spin', 'VB (trajectory)', 'HB (trajectory)', 'Release Height', 'Release Side', 'Horizontal Angle']
                    for col in numeric_cols:
                        if col in pitch_data.columns:
                            pitch_data[col] = pd.to_numeric(pitch_data[col], errors='coerce')
                    
                    # Calculate metrics for each pitch type
                    pitch_types = ['Fastball', 'ChangeUp', 'Slider']
                    player_stats = {
                        'PlayerID': player_id,
                        'PlayerName': player_name,
                        'TotalPitches': len(pitch_data)
                    }
                    
                    valid_pitch_types = []
                    
                    for pitch_type in pitch_types:
                        if pitch_type == 'Fastball':
                            pitch_data_filtered = pitch_data[pitch_data['Pitch Type'].str.contains('Fastball', case=False, na=False)]
                        elif pitch_type == 'ChangeUp':
                            pitch_data_filtered = pitch_data[
                                pitch_data['Pitch Type'].str.contains('ChangeUp|Splitter', case=False, na=False)
                            ]
                        elif pitch_type == 'Slider':
                            pitch_data_filtered = pitch_data[
                                pitch_data['Pitch Type'].str.contains('Slider|Curveball', case=False, na=False)
                            ]
                        else:
                            pitch_data_filtered = pitch_data[pitch_data['Pitch Type'].str.contains(pitch_type, case=False, na=False)]
                        
                        if len(pitch_data_filtered) > 0:
                            valid_pitch_types.append(pitch_type)
                            
                            avg_velocity = pitch_data_filtered['Velocity'].mean()
                            avg_spin_rate = pitch_data_filtered['Total Spin'].mean()
                            avg_release_height = pitch_data_filtered['Release Height'].mean() if 'Release Height' in pitch_data_filtered.columns else 5.5
                            avg_release_side = pitch_data_filtered['Release Side'].mean() if 'Release Side' in pitch_data_filtered.columns else 0
                            avg_vb = pitch_data_filtered['VB (trajectory)'].mean() if 'VB (trajectory)' in pitch_data_filtered.columns else 0
                            avg_hb = pitch_data_filtered['HB (trajectory)'].mean() if 'HB (trajectory)' in pitch_data_filtered.columns else 0
                            avg_horizontal_angle = pitch_data_filtered['Horizontal Angle'].mean() if 'Horizontal Angle' in pitch_data_filtered.columns else 0
                            
                            fastball_data = pitch_data[pitch_data['Pitch Type'].str.contains('Fastball', case=False, na=False)]
                            if len(fastball_data) > 0:
                                fastball_avg_velocity = fastball_data['Velocity'].mean()
                                speed_diff = fastball_avg_velocity - avg_velocity
                            else:
                                speed_diff = 0
                            
                            player_stats.update({
                                f'{pitch_type}_Velocity': avg_velocity,
                                f'{pitch_type}_SpinRate': avg_spin_rate,
                                f'{pitch_type}_ReleaseHeight': avg_release_height,
                                f'{pitch_type}_ReleaseSide': avg_release_side,
                                f'{pitch_type}_HorizontalAngle': avg_horizontal_angle,
                                f'{pitch_type}_SpeedDiff': speed_diff,
                                f'{pitch_type}_HorizontalBreak': abs(avg_hb) if not pd.isna(avg_hb) else 0,
                                f'{pitch_type}_VerticalBreak': avg_vb if not pd.isna(avg_vb) else 0,
                                f'{pitch_type}_Pitches': len(pitch_data_filtered)
                            })
                    
                    if valid_pitch_types:
                        all_player_data.append(player_stats)
                    else:
                        st.warning(f"No valid pitch types found for {player_name} in {csv_file}")
                else:
                    st.warning(f"No valid pitch data found for {player_name} in {csv_file}")
            else:
                st.warning(f"Could not find player info or data start in {csv_file}")
        
        except Exception as e:
            st.error(f"Error reading file {csv_file}: {str(e)}")
            continue
    
    if not all_player_data:
        raise ValueError("No valid player data found in CSV files.")
    
    df = pd.DataFrame(all_player_data)
    
    # Calculate Kats Stuff+ for each pitch type
    pitch_types = ['Fastball', 'ChangeUp', 'Slider']

    for pitch_type in pitch_types:
        velocity_col = f'{pitch_type}_Velocity'
        if velocity_col in df.columns:
            pitch_df = df[df[velocity_col].notna()].copy()
            if len(pitch_df) > 0:
                stuff_plus_col = f'{pitch_type}_Stuff+'
                stuff_plus_values = calculate_kats_stuff_plus_for_pitch_type(pitch_df, pitch_type)
                stuff_plus_mapping = dict(zip(pitch_df['PlayerName'], stuff_plus_values))
                df[stuff_plus_col] = df['PlayerName'].map(stuff_plus_mapping)
    
    # Calculate Total Stuff+ as mean of all pitch types
    stuff_plus_cols = [col for col in df.columns if col.endswith('_Stuff+')]
    if stuff_plus_cols:
        df['Total_Stuff+'] = df[stuff_plus_cols].mean(axis=1, skipna=True)
    
    return df

def calculate_kats_stuff_plus_for_pitch_type(df, pitch_type):
    """Calculate Kats Stuff+ for a specific pitch type"""
    
    def normalize_component(values, higher_is_better=True):
        if len(values) == 0 or values.std() == 0:
            return np.ones(len(values)) * 0.5
        z_scores = (values - values.mean()) / values.std()
        normalized = 1 / (1 + np.exp(-z_scores))
        if not higher_is_better:
            normalized = 1 - normalized
        return normalized
    
    def normalize_deviation_from_mean(values):
        """Reward deviation from mean - both high and low values are good"""
        if len(values) == 0 or values.std() == 0:
            return np.ones(len(values)) * 0.5
        mean_val = values.mean()
        deviations = np.abs(values - mean_val)
        if deviations.std() == 0:
            return np.ones(len(values)) * 0.5
        z_scores = (deviations - deviations.mean()) / deviations.std()
        normalized = 1 / (1 + np.exp(-z_scores))
        return normalized
    
    weights = {
        'velocity': 0.225,
        'spin_rate': 0.175,
        'release_height': 0.125,
        'release_side': 0.085,
        'horizontal_angle': 0.05,
        'speed_diff': 0.075,
        'horizontal_break': 0.10,
        'vertical_break': 0.10,
        'distinctive_shape': 0.125
    }
    
    velocity_col = f'{pitch_type}_Velocity'
    spin_col = f'{pitch_type}_SpinRate'
    height_col = f'{pitch_type}_ReleaseHeight'
    side_col = f'{pitch_type}_ReleaseSide'
    angle_col = f'{pitch_type}_HorizontalAngle'
    speed_diff_col = f'{pitch_type}_SpeedDiff'
    h_break_col = f'{pitch_type}_HorizontalBreak'
    v_break_col = f'{pitch_type}_VerticalBreak'
    
    velocity_norm = normalize_component(df[velocity_col], higher_is_better=True)
    spin_norm = normalize_component(df[spin_col], higher_is_better=True)
    speed_diff_norm = normalize_component(df[speed_diff_col], higher_is_better=True)
    horizontal_angle_norm = normalize_component(abs(df[angle_col]), higher_is_better=False)
    h_break_norm = normalize_component(df[h_break_col], higher_is_better=True)
    v_break_norm = normalize_component(df[v_break_col], higher_is_better=True)
    
    height_norm = normalize_deviation_from_mean(df[height_col])
    
    if side_col in df.columns:
        side_norm = normalize_deviation_from_mean(df[side_col])
    else:
        side_norm = np.ones(len(df)) * 0.5
    
    shape_differential = np.abs(df[h_break_col]) - np.abs(df[v_break_col])
    distinctive_shape_norm = normalize_deviation_from_mean(np.abs(shape_differential))
    
    composite_score = (
        velocity_norm * weights['velocity'] +
        spin_norm * weights['spin_rate'] +
        height_norm * weights['release_height'] +
        side_norm * weights['release_side'] +
        horizontal_angle_norm * weights['horizontal_angle'] +
        speed_diff_norm * weights['speed_diff'] +
        h_break_norm * weights['horizontal_break'] +
        v_break_norm * weights['vertical_break'] +
        distinctive_shape_norm * weights['distinctive_shape']
    )
    
    median_score = np.median(composite_score)
    std_score = np.std(composite_score)
    
    if std_score > 0:
        stuff_plus = 100 + ((composite_score - median_score) / std_score) * 20
    else:
        stuff_plus = np.ones(len(composite_score)) * 100
    
    stuff_plus = np.clip(stuff_plus, 40, 160)
    
    return stuff_plus

def create_leaderboard_chart(df, metric_col, title):
    """Create a horizontal bar chart for leaderboards using matplotlib dark mode"""
    df_sorted = df.sort_values(metric_col, ascending=False).head(10)
    df_sorted = df_sorted.sort_values(metric_col, ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    colors = []
    values = df_sorted[metric_col].values
    max_val = values.max()
    min_val = values.min()
    
    for val in values:
        normalized = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
        if normalized > 0.8:
            colors.append("#DF0E0EE1")
        elif normalized > 0.6:
            colors.append("#6D6D6D")
        elif normalized > 0.4:
            colors.append("#DEDEDE")
        else:
            colors.append("#000000")
    
    bars = ax.barh(df_sorted['PlayerName'], df_sorted[metric_col], color=colors, 
                   edgecolor='white', linewidth=1, alpha=0.8)
    
    for i, (bar, value) in enumerate(zip(bars, df_sorted[metric_col])):
        ax.text(bar.get_width() + max(df_sorted[metric_col]) * 0.01, 
                bar.get_y() + bar.get_height()/2,
                f'{value:.1f}', 
                va='center', ha='left', color='white', fontweight='bold', fontsize=10)
    
    ax.set_xlabel(metric_col.replace('_', ' '), color='white', fontsize=12, fontweight='bold')
    ax.set_title(title, color='white', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(colors='white', labelsize=10)
    ax.grid(True, alpha=0.3, color='gray')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    
    plt.tight_layout()
    return fig

def create_leaderboard_table(df, metric_col, additional_cols=None):
    """Create a formatted leaderboard table"""
    df_sorted = df.sort_values(metric_col, ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    
    display_cols = ['Rank', 'PlayerName', metric_col]
    if additional_cols:
        for col in additional_cols:
            if col in df_sorted.columns:
                display_cols.append(col)
    
    return df_sorted[display_cols]

def create_leaderboard_table_with_rank_change(current_df, baseline_df, metric_col, additional_cols=None):
    """Create a formatted leaderboard table with rank change from baseline"""
    df_sorted = current_df.sort_values(metric_col, ascending=False).reset_index(drop=True)
    df_sorted['Rank'] = range(1, len(df_sorted) + 1)
    
    if baseline_df is not None and not baseline_df.empty and metric_col in baseline_df.columns:
        baseline_sorted = baseline_df.sort_values(metric_col, ascending=False).reset_index(drop=True)
        baseline_sorted['BaselineRank'] = range(1, len(baseline_sorted) + 1)
        baseline_ranks = baseline_sorted.set_index('PlayerName')['BaselineRank'].to_dict()
        
        df_sorted['Rank Change'] = df_sorted.apply(
            lambda row: baseline_ranks.get(row['PlayerName'], None) - row['Rank'] 
            if row['PlayerName'] in baseline_ranks else None, axis=1
        )
        
        baseline_values = baseline_sorted.set_index('PlayerName')[metric_col].to_dict()
        df_sorted['Baseline'] = df_sorted['PlayerName'].map(baseline_values)
        df_sorted['Stuff+ Change'] = df_sorted[metric_col] - df_sorted['Baseline']
    else:
        df_sorted['Rank Change'] = None
        df_sorted['Baseline'] = None
        df_sorted['Stuff+ Change'] = None
    
    display_cols = ['Rank', 'PlayerName', metric_col]
    if additional_cols:
        for col in additional_cols:
            if col in df_sorted.columns:
                display_cols.append(col)
    
    if df_sorted['Baseline'].notna().any():
        display_cols.extend(['Stuff+ Change', 'Rank Change'])
    
    return df_sorted[display_cols]

# Header
st.title("ECC Kats Baseball")
st.caption("Kats Stuff+ Dashboard")

baseline_df = None  # ensure always defined

try:
    # Load February 2026 (current) data
    current_config = SESSION_CONFIG["February 2026"]
    rapsodo_df = load_rapsodo_data(current_config["bullpen_dir"])

    # Load December 2025 (baseline) data for comparison
    baseline_config = SESSION_CONFIG["December 2025"]
    try:
        baseline_df = load_rapsodo_data(baseline_config["bullpen_dir"])
    except Exception as baseline_err:
        st.sidebar.warning(f"December baseline not loaded: {baseline_err}")
        baseline_df = None

except Exception as e:
    st.error(f"Failed to load data: {str(e)}")
    st.stop()

# Show comparison indicator
if baseline_df is not None and not baseline_df.empty:
    st.info("Showing February 2026 data with rank changes vs December 2025 baseline")
else:
    st.warning("⚠️ December 2025 baseline not loaded — Biggest Movers will be empty. Check `data/BullpenReports120625/` exists and contains CSVs.")

# Sidebar with logo
try:
    st.sidebar.image("images/liquid_logo.png", width=250)
    st.sidebar.image("images/logo.png", width=250)
except FileNotFoundError:
    st.sidebar.warning("Logo not found at images/logo.png")

st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("---")

# Pitch type selector
st.sidebar.subheader("Pitch Type Selection")
pitch_type_options = ["Total", "Fastball", "ChangeUp", "Slider"]
selected_pitch_type = st.sidebar.selectbox("Select Pitch Type", pitch_type_options)

# Determine the column name for selected pitch type
if selected_pitch_type == "Total":
    stuff_plus_col = "Total_Stuff+"
    display_name = "Total Stuff+"
else:
    stuff_plus_col = f"{selected_pitch_type}_Stuff+"
    display_name = f"{selected_pitch_type} Stuff+"

st.sidebar.markdown("---")

# Team overview metrics in sidebar
st.sidebar.subheader("Team Overview")
total_players = len(rapsodo_df)

if stuff_plus_col in rapsodo_df.columns:
    avg_stuff_plus = rapsodo_df[stuff_plus_col].mean()
    median_stuff_plus = rapsodo_df[stuff_plus_col].median()
    
    st.sidebar.metric("Total Players", total_players)
    st.sidebar.metric(f"Avg {display_name}", f"{avg_stuff_plus:.1f}")
    st.sidebar.metric(f"Median {display_name}", f"{median_stuff_plus:.1f}")
else:
    st.sidebar.metric("Total Players", total_players)
    st.sidebar.warning(f"No data for {display_name}")

total_pitches = rapsodo_df['TotalPitches'].sum()
st.sidebar.metric("Total Pitches", f"{total_pitches:,}")

# Main content
st.subheader(f"Kats {display_name} Leaderboard")

if stuff_plus_col not in rapsodo_df.columns:
    st.error(f"No data available for {display_name}. Please select a different pitch type.")
    st.stop()

# Filter out players without data for this pitch type
display_df = rapsodo_df[rapsodo_df[stuff_plus_col].notna()].copy()

# Filter baseline data similarly
baseline_display_df = None
if baseline_df is not None and stuff_plus_col in baseline_df.columns:
    baseline_display_df = baseline_df[baseline_df[stuff_plus_col].notna()].copy()

if len(display_df) == 0:
    st.error(f"No players have data for {display_name}")
    st.stop()

col1, col2 = st.columns([3, 2])

with col1:
    fig_stuff = create_leaderboard_chart(
        display_df, stuff_plus_col, 
        f"Kats {display_name} Rankings"
    )
    st.pyplot(fig_stuff, use_container_width=True)

with col2:
    st.subheader("Biggest Movers vs December")
    
    if baseline_display_df is not None and not baseline_display_df.empty:
        movers_table = create_leaderboard_table_with_rank_change(
            display_df, baseline_display_df, stuff_plus_col, []
        )
        
        movers_table = movers_table[movers_table['Stuff+ Change'].notna()].copy()
        
        if not movers_table.empty:
            movers_table['Abs Change'] = movers_table['Stuff+ Change'].abs()
            movers_table = movers_table.sort_values('Abs Change', ascending=False)
            
            def format_stuff_change(val):
                if pd.isna(val) or val is None:
                    return ""
                return f"{val:+.1f}"
            
            def format_rank_change(val):
                if pd.isna(val) or val is None:
                    return ""
                val = int(round(val))
                if val > 0:
                    return f"↑{val}"
                elif val < 0:
                    return f"↓{abs(val)}"
                else:
                    return "—"
            
            display_movers = movers_table[['PlayerName', stuff_plus_col, 'Stuff+ Change', 'Rank Change']].copy()
            display_movers.columns = ['Player', 'Current', 'Stuff+ Δ', 'Rank Δ']
            
            display_movers['Stuff+ Δ'] = movers_table['Stuff+ Change'].apply(format_stuff_change)
            display_movers['Rank Δ'] = movers_table['Rank Change'].apply(format_rank_change)
            display_movers['Current'] = display_movers['Current'].round(1)
            
            def color_movers(val):
                if pd.isna(val) or val is None or val == "":
                    return ''
                if isinstance(val, str):
                    if val.startswith('+') or val.startswith('↑'):
                        return 'color: #00cc00; font-weight: bold'
                    elif val.startswith('-') or val.startswith('↓'):
                        return 'color: #ff4444; font-weight: bold'
                return ''
            
            styled_movers = display_movers.head(10).style.map(
                color_movers, subset=['Stuff+ Δ', 'Rank Δ']
            )
            
            st.dataframe(
                styled_movers,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Player": st.column_config.TextColumn("Player"),
                    "Current": st.column_config.NumberColumn("Current Stuff+", format="%.1f"),
                    "Stuff+ Δ": st.column_config.TextColumn("Stuff+ Δ"),
                    "Rank Δ": st.column_config.TextColumn("Rank Δ")
                }
            )
            
            gainers = movers_table[movers_table['Stuff+ Change'] > 0]
            losers = movers_table[movers_table['Stuff+ Change'] < 0]
            
            col_a, col_b = st.columns(2)
            with col_a:
                if not gainers.empty:
                    top_gainer = gainers.iloc[0]
                    st.metric(
                        "Biggest Gainer",
                        top_gainer['PlayerName'].split()[0],
                        f"+{top_gainer['Stuff+ Change']:.1f}"
                    )
            with col_b:
                if not losers.empty:
                    top_loser = movers_table[movers_table['Stuff+ Change'] < 0].sort_values('Stuff+ Change').iloc[0]
                    st.metric(
                        "Biggest Drop",
                        top_loser['PlayerName'].split()[0],
                        f"{top_loser['Stuff+ Change']:.1f}"
                    )
        else:
            st.warning(
                "Baseline loaded but no player names overlap between December and February. "
                "Check that player names match exactly across both sessions."
            )
    else:
        if baseline_df is None:
            st.warning("December baseline file failed to load — check sidebar for the error detail.")
        else:
            st.info(f"December baseline loaded but has no '{display_name}' data for this pitch type.")

# Full leaderboard
if selected_pitch_type == "Total":
    full_table_cols = ['TotalPitches']
    for pt in ['Fastball', 'ChangeUp', 'Slider']:
        if f'{pt}_Stuff+' in display_df.columns:
            full_table_cols.append(f'{pt}_Stuff+')
else:
    full_table_cols = [
        f'{selected_pitch_type}_Velocity',
        f'{selected_pitch_type}_SpinRate',
        f'{selected_pitch_type}_ReleaseHeight',
        f'{selected_pitch_type}_ReleaseSide',
        f'{selected_pitch_type}_HorizontalAngle',
        f'{selected_pitch_type}_SpeedDiff',
        f'{selected_pitch_type}_HorizontalBreak',
        f'{selected_pitch_type}_VerticalBreak',
        f'{selected_pitch_type}_Pitches'
    ]

full_table = create_leaderboard_table_with_rank_change(display_df, baseline_display_df, stuff_plus_col, full_table_cols)

full_column_config = {
    stuff_plus_col: st.column_config.NumberColumn(display_name, format="%.1f")
}

if selected_pitch_type == "Total":
    full_column_config["TotalPitches"] = st.column_config.NumberColumn("Total Pitches", format="%.0f")
    for pt in ['Fastball', 'ChangeUp', 'Slider']:
        if f'{pt}_Stuff+' in display_df.columns:
            full_column_config[f'{pt}_Stuff+'] = st.column_config.NumberColumn(f"{pt} Stuff+", format="%.1f")
else:
    full_column_config.update({
        f"{selected_pitch_type}_Velocity": st.column_config.NumberColumn("Velocity", format="%.1f mph"),
        f"{selected_pitch_type}_SpinRate": st.column_config.NumberColumn("Spin Rate", format="%.0f rpm"),
        f"{selected_pitch_type}_ReleaseHeight": st.column_config.NumberColumn("Release Height", format="%.2f ft"),
        f"{selected_pitch_type}_ReleaseSide": st.column_config.NumberColumn("Release Side", format="%.2f ft"),
        f"{selected_pitch_type}_HorizontalAngle": st.column_config.NumberColumn("H-Angle", format="%.1f°"),
        f"{selected_pitch_type}_SpeedDiff": st.column_config.NumberColumn("Speed Diff", format="%.1f mph"),
        f"{selected_pitch_type}_HorizontalBreak": st.column_config.NumberColumn("H-Break", format="%.1f in"),
        f"{selected_pitch_type}_VerticalBreak": st.column_config.NumberColumn("V-Break", format="%.1f in"),
        f"{selected_pitch_type}_Pitches": st.column_config.NumberColumn("Pitches", format="%.0f")
    })

if 'Stuff+ Change' in full_table.columns:
    full_column_config["Stuff+ Change"] = st.column_config.NumberColumn("Stuff+ Δ", format="%+.1f")

if 'Rank Change' in full_table.columns:
    def format_rank_change_full(val):
        if pd.isna(val) or val is None:
            return ""
        val = int(round(val))
        if val > 0:
            return f"↑{val}"
        elif val < 0:
            return f"↓{abs(val)}"
        else:
            return "→0"
    
    full_table['Rank Change'] = full_table['Rank Change'].apply(format_rank_change_full)
    full_column_config["Rank Change"] = st.column_config.TextColumn("Rank Δ")

def color_change_with_rank(val):
    if pd.isna(val) or val is None or val == "":
        return ''
    if isinstance(val, (int, float)):
        if val > 0:
            return 'color: #00cc00'
        elif val < 0:
            return 'color: #ff4444'
    elif isinstance(val, str):
        if val.startswith('↑'):
            return 'color: #00cc00'
        elif val.startswith('↓'):
            return 'color: #ff4444'
    return ''

style_columns_all = [col for col in ['Stuff+ Change', 'Rank Change'] if col in full_table.columns]

if style_columns_all:
    styled_full_table = full_table.style.map(color_change_with_rank, subset=style_columns_all)
    st.dataframe(styled_full_table, hide_index=True, use_container_width=True, column_config=full_column_config)
else:
    st.dataframe(full_table, hide_index=True, use_container_width=True, column_config=full_column_config)

# Right vs Left Handed Analysis
st.subheader(f"{display_name} - Right vs Left Handed Pitchers")

handedness_map = get_cached_handedness_map(current_config["bullpen_dir"])
display_df['Handedness'] = display_df['PlayerName'].map(handedness_map).fillna('RHP')

col1, col2 = st.columns(2)

def build_handedness_column_config(stuff_plus_col, display_name, df_table):
    """Build column config for RHP/LHP tables, handling Total view extra Stuff+ cols."""
    cfg = {stuff_plus_col: st.column_config.NumberColumn(display_name, format="%.1f")}
    for col in df_table.columns:
        if col.endswith('_Stuff+') and col != stuff_plus_col:
            cfg[col] = st.column_config.NumberColumn(col.replace('_Stuff+', ' Stuff+'), format="%.1f")
    return cfg


with col1:
    rhp_df = display_df[display_df['Handedness'] == 'RHP']
    if len(rhp_df) > 0:
        st.subheader(f"Right-Handed Pitchers ({len(rhp_df)})")
        rhp_table = create_leaderboard_table(rhp_df, stuff_plus_col, [])
        st.dataframe(
            rhp_table,
            hide_index=True,
            use_container_width=True,
            column_config=build_handedness_column_config(stuff_plus_col, display_name, rhp_table)
        )
        rhp_avg = rhp_df[stuff_plus_col].mean()
        rhp_median = rhp_df[stuff_plus_col].median()
        st.metric("RHP Average", f"{rhp_avg:.1f}")
        st.metric("RHP Median", f"{rhp_median:.1f}")
    else:
        st.info("No right-handed pitchers with data for this pitch type")

with col2:
    lhp_df = display_df[display_df['Handedness'] == 'LHP']
    if len(lhp_df) > 0:
        st.subheader(f"Left-Handed Pitchers ({len(lhp_df)})")
        lhp_table = create_leaderboard_table(lhp_df, stuff_plus_col, [])
        st.dataframe(
            lhp_table,
            hide_index=True,
            use_container_width=True,
            column_config=build_handedness_column_config(stuff_plus_col, display_name, lhp_table)
        )
        lhp_avg = lhp_df[stuff_plus_col].mean()
        lhp_median = lhp_df[stuff_plus_col].median()
        st.metric("LHP Average", f"{lhp_avg:.1f}")
        st.metric("LHP Median", f"{lhp_median:.1f}")
    else:
        st.info("No left-handed pitchers with data for this pitch type")

# Set matplotlib to dark theme
plt.style.use('dark_background')
sns.set_palette("husl")

# VALD API Configuration
VALD_CONFIG = st.secrets["VALD_CONFIG"]

@st.cache_data(ttl=300)
def get_access_token():
    """Get access token from VALD API"""
    token_data = {
        "grant_type": "client_credentials",
        "client_id": VALD_CONFIG["client_id"],
        "client_secret": VALD_CONFIG["client_secret"]
    }
    response = requests.post(VALD_CONFIG["token_url"], data=token_data)
    return response.json()["access_token"] if response.ok else None

@st.cache_data(ttl=1800)
def load_kats_players_from_csv():
    """Load players from CSV files"""
    data_dir = SESSION_CONFIG["February 2026"]["bullpen_dir"]
    kats_players = {}
    
    if not os.path.exists(data_dir):
        return {}
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    handedness_map = get_handedness_from_raw_data(data_dir)
    
    for csv_file in csv_files:
        try:
            encodings_to_try = ['utf-8', 'utf-16', 'latin1', 'cp1252', 'iso-8859-1']
            lines = None
            
            for encoding in encodings_to_try:
                try:
                    with open(csv_file, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            
            if lines is None:
                continue
            
            player_id = None
            player_name = None
            
            for line in lines:
                if 'Player ID:' in line:
                    player_id = line.split(',')[1].strip()
                elif 'Player Name:' in line:
                    player_name = line.split(',')[1].strip()
                    break
            
            if player_id and player_name:
                handedness = handedness_map.get(player_name, 'RHP')
                kats_players[player_name] = {
                    'player_id': player_id,
                    'handedness': handedness,
                    'csv_file': csv_file
                }
                
        except Exception:
            continue
    
    return kats_players

@st.cache_data(ttl=1800)
def fetch_all_profiles():
    """Fetch all profiles from External Profiles API"""
    token = get_access_token()
    if not token:
        return {}
    
    headers = {"Authorization": f"Bearer {token}"}
    profiles_url = f"{VALD_CONFIG['profiles_base_url']}/profiles?tenantId={VALD_CONFIG['tenant_id']}"
    
    try:
        response = requests.get(profiles_url, headers=headers)
        
        if response.ok:
            data = response.json()
            
            if "profiles" in data:
                profiles = data["profiles"]
                profiles_dict = {}
                
                for profile in profiles:
                    profile_id = profile.get('profileId')
                    given_name = profile.get('givenName', '').strip()
                    family_name = profile.get('familyName', '').strip()
                    full_name = f"{given_name} {family_name}".strip()
                    
                    profiles_dict[profile_id] = {
                        'profileId': profile_id,
                        'givenName': given_name,
                        'familyName': family_name,
                        'fullName': full_name,
                        'dateOfBirth': profile.get('dateOfBirth'),
                        'height': profile.get('height'),
                        'weight': profile.get('weight'),
                        'sex': profile.get('sex')
                    }
                
                return profiles_dict
        return {}
    except Exception as e:
        st.error(f"Error fetching profiles: {str(e)}")
        return {}

def match_players_to_profiles(kats_players, all_profiles):
    """Match CSV players to VALD profiles by name"""
    name_to_profile_id = {}
    
    for player_name in kats_players.keys():
        for profile_id, profile_data in all_profiles.items():
            if profile_data['fullName'] == player_name:
                name_to_profile_id[player_name] = profile_id
                break
    
    return name_to_profile_id

@st.cache_data(ttl=1800)
def get_team_id():
    """Get team ID from the v2019q3/teams endpoint"""
    token = get_access_token()
    if not token:
        return None
    
    headers = {"Authorization": f"Bearer {token}"}
    teams_url = f"{VALD_CONFIG['forcedecks_base_url']}/v2019q3/teams"
    
    try:
        response = requests.get(teams_url, headers=headers)
        
        if response.ok:
            teams = response.json()
            if teams and len(teams) > 0:
                return teams[0].get('id') or teams[0].get('teamId')
        return None
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_forcedecks_tests(profile_ids, modified_from_date):
    """Fetch ForceDecks test data using /tests endpoint"""
    if not profile_ids:
        return pd.DataFrame()
    
    token = get_access_token()
    if not token:
        return pd.DataFrame()
    
    headers = {"Authorization": f"Bearer {token}"}
    modified_date = f"{modified_from_date}T00:00:00.000Z"
    
    initial_url = f"{VALD_CONFIG['forcedecks_base_url']}/tests?tenantId={VALD_CONFIG['tenant_id']}&modifiedFromUtc={modified_date}"
    
    try:
        all_tests = []
        current_url = initial_url
        page_count = 0
        max_pages = 10
        
        while current_url and page_count < max_pages:
            page_count += 1
            response = requests.get(current_url, headers=headers)
            
            if response.status_code == 204:
                break
            
            if response.ok:
                try:
                    data = response.json()
                    tests = data if isinstance(data, list) else data.get("tests", [])
                    
                    if len(tests) > 0:
                        filtered_tests = [test for test in tests if test.get('profileId') in profile_ids]
                        all_tests.extend(filtered_tests)
                        
                        last_test = tests[-1]
                        last_modified = last_test.get('modifiedDateUtc')
                        if last_modified:
                            last_dt = datetime.fromisoformat(last_modified.replace('Z', '+00:00'))
                            next_dt = last_dt + timedelta(microseconds=1)
                            next_modified = next_dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')[:-3] + 'Z'
                            current_url = f"{VALD_CONFIG['forcedecks_base_url']}/tests?tenantId={VALD_CONFIG['tenant_id']}&modifiedFromUtc={next_modified}"
                        else:
                            current_url = None
                    else:
                        break
                        
                except Exception as e:
                    st.error(f"Error parsing response: {str(e)}")
                    break
            else:
                st.error(f"API Error {response.status_code}")
                break
        
        if all_tests:
            df = pd.DataFrame(all_tests)
            if 'modifiedDateUtc' in df.columns:
                df['modifiedDateUtc'] = pd.to_datetime(df['modifiedDateUtc'], utc=True)
                df['date'] = df['modifiedDateUtc'].dt.date
                df['time'] = df['modifiedDateUtc'].dt.time
            return df
        else:
            return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error fetching tests: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_test_trials_batch(team_id, test_ids):
    """Fetch trials (detailed rep data) for multiple tests"""
    if not team_id or not test_ids:
        return pd.DataFrame()
    
    token = get_access_token()
    if not token:
        return pd.DataFrame()
    
    headers = {"Authorization": f"Bearer {token}"}
    all_trials = []
    
    for test_id in test_ids:
        try:
            trials_url = f"{VALD_CONFIG['forcedecks_base_url']}/v2019q3/teams/{team_id}/tests/{test_id}/trials"
            response = requests.get(trials_url, headers=headers)
            
            if response.ok:
                trials_data = response.json()
                
                if isinstance(trials_data, list) and len(trials_data) > 0:
                    for trial in trials_data:
                        trial['testId'] = test_id
                    all_trials.extend(trials_data)
        
        except Exception:
            continue
    
    if all_trials:
        return pd.DataFrame(all_trials)
    else:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def fetch_dynamo_tests(profile_ids, modified_from_date):
    """Fetch Dynamo test data for specific profiles"""
    if not profile_ids:
        return pd.DataFrame()
    
    token = get_access_token()
    if not token:
        return pd.DataFrame()
    
    headers = {"Authorization": f"Bearer {token}"}
    
    now = datetime.now()
    test_from = f"{modified_from_date}T00:00:00.000Z"
    test_to = now.strftime('%Y-%m-%dT23:59:59.000Z')
    modified_from = f"{modified_from_date}T00:00:00.000Z"
    
    tenant_id = VALD_CONFIG['tenant_id']
    base = VALD_CONFIG['dynamo_base_url']
    
    url = f"{base}/v2022q2/teams/{tenant_id}/tests"
    
    all_tests = []
    page = 1
    max_pages = 20
    
    try:
        while page <= max_pages:
            params = {
                "modifiedFromUtc": modified_from,
                "testFromUtc": test_from,
                "testToUtc": test_to,
                "includeRepSummaries": "true",
                "page": page
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 204:
                break
            
            if response.ok:
                data = response.json()
                items = data.get("items", [])
                total_pages = data.get("totalPages", 1)
                
                if items:
                    filtered_tests = [test for test in items if test.get('athleteId') in profile_ids]
                    all_tests.extend(filtered_tests)
                
                if page >= total_pages:
                    break
                page += 1
            else:
                break
        
        if all_tests:
            df = pd.DataFrame(all_tests)
            df['test_date'] = pd.to_datetime(df['startTimeUTC']).dt.date
            return df
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Error fetching Dynamo tests: {str(e)}")
        return pd.DataFrame()


def extract_dynamo_metrics(dynamo_df, all_profiles):
    """Extract key metrics from Dynamo test data with player names"""
    if dynamo_df.empty:
        return pd.DataFrame()
    
    athlete_to_name = {}
    for profile_id, profile_data in all_profiles.items():
        athlete_to_name[profile_id] = profile_data['fullName']
    
    performance_data = []
    
    for _, test in dynamo_df.iterrows():
        athlete_id = test.get('athleteId')
        player_name = athlete_to_name.get(athlete_id, 'Unknown')
        
        if player_name == 'Unknown':
            continue
        
        test_type = f"{test.get('bodyRegion', '')} {test.get('movement', '')} - {test.get('position', '')}"
        rep_summaries = test.get('repetitionTypeSummaries', [])
        
        for rep in rep_summaries:
            record = {
                'testId': test.get('id'),
                'athleteId': athlete_id,
                'player_name': player_name,
                'test_date': test.get('test_date'),
                'testCategory': test.get('testCategory'),
                'bodyRegion': test.get('bodyRegion'),
                'movement': test.get('movement'),
                'position': test.get('position'),
                'test_type': test_type,
                'laterality': rep.get('laterality'),
                'repCount': rep.get('repCount'),
                'maxForceNewtons': rep.get('maxForceNewtons'),
                'avgForceNewtons': rep.get('avgForceNewtons'),
                'maxImpulseNewtonSeconds': rep.get('maxImpulseNewtonSeconds'),
                'avgImpulseNewtonSeconds': rep.get('avgImpulseNewtonSeconds'),
                'maxRateOfForceDevelopmentNewtonsPerSecond': rep.get('maxRateOfForceDevelopmentNewtonsPerSecond'),
                'avgRateOfForceDevelopmentNewtonsPerSecond': rep.get('avgRateOfForceDevelopmentNewtonsPerSecond'),
                'maxRangeOfMotionDegrees': rep.get('maxRangeOfMotionDegrees'),
                'avgRangeOfMotionDegrees': rep.get('avgRangeOfMotionDegrees'),
                'avgTimeToPeakForceSeconds': rep.get('avgTimeToPeakForceSeconds'),
            }
            performance_data.append(record)
    
    return pd.DataFrame(performance_data) if performance_data else pd.DataFrame()


def create_trunk_rotation_leaderboard(dynamo_perf_df):
    """Create leaderboard for Trunk Rotation tests"""
    
    if dynamo_perf_df.empty:
        st.warning("No Dynamo performance data available")
        return
    
    trunk_data = dynamo_perf_df[
        (dynamo_perf_df['bodyRegion'] == 'Trunk') & 
        (dynamo_perf_df['movement'].str.contains('Rotation', case=False, na=False))
    ].copy()
    
    if trunk_data.empty:
        st.warning("No Trunk Rotation data found")
        return
    
    st.subheader("Trunk Rotation Performance Analysis")
    
    total_tests = len(trunk_data)
    unique_players = trunk_data['player_name'].nunique()
    st.info(f"Found {total_tests} Trunk Rotation measurements from {unique_players} players")
    
    available_metrics = [
        ('maxForceNewtons', 'Peak Force (N)', 'Newtons'),
        ('avgForceNewtons', 'Avg Force (N)', 'Newtons'),
        ('maxImpulseNewtonSeconds', 'Peak Impulse (N·s)', 'Newton-seconds'),
        ('maxRateOfForceDevelopmentNewtonsPerSecond', 'Peak RFD (N/s)', 'Newtons/second'),
    ]
    
    metric_options = [m[1] for m in available_metrics]
    selected_metric_display = st.selectbox("Select Metric:", metric_options, key="trunk_metric")
    
    selected_metric = None
    selected_units = None
    for col, display, units in available_metrics:
        if display == selected_metric_display:
            selected_metric = col
            selected_units = units
            break
    
    if selected_metric is None or selected_metric not in trunk_data.columns:
        st.error(f"Metric {selected_metric_display} not available")
        return
    
    tab1, tab2, tab3 = st.tabs(["Overall Leaderboard", "Left vs Right Comparison", "Asymmetry Analysis"])
    
    with tab1:
        player_best = trunk_data.groupby('player_name')[selected_metric].max().reset_index()
        player_best = player_best.sort_values(selected_metric, ascending=False)
        player_best = player_best[player_best[selected_metric].notna()]
        
        if player_best.empty:
            st.warning("No valid data for selected metric")
            return
        
        group_avg = player_best[selected_metric].mean()
        group_std = player_best[selected_metric].std()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Group Average", f"{group_avg:.1f} {selected_units}")
        with col2:
            st.metric("Players Tested", len(player_best))
        with col3:
            if len(player_best) > 0:
                st.metric("Top Performer", player_best.iloc[0]['player_name'].split()[0])
        with col4:
            if len(player_best) > 0:
                st.metric("Best Value", f"{player_best.iloc[0][selected_metric]:.1f}")
        
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        colors = []
        for _, row in player_best.iterrows():
            value = row[selected_metric]
            if value >= group_avg + 0.5 * group_std:
                colors.append('#2E8B8B')
            elif value >= group_avg:
                colors.append('#4A90A4')
            elif value >= group_avg - 0.5 * group_std:
                colors.append('#FFA500')
            else:
                colors.append('#FF6B6B')
        
        bars = ax.bar(range(len(player_best)), player_best[selected_metric], color=colors, alpha=0.8)
        
        ax.axhline(y=group_avg, color='white', linestyle='--', linewidth=2, alpha=0.8,
                  label=f'Group Average: {group_avg:.1f}')
        
        ax.set_title(f'ECC Kats Baseball - Trunk Rotation\n{selected_metric_display}',
                    fontsize=16, pad=20, fontweight='bold', color='white')
        ax.set_ylabel(f'{selected_metric_display}', fontsize=12, color='white')
        ax.set_xlabel('Players (Ranked by Performance)', fontsize=12, color='white')
        
        ax.set_xticks(range(len(player_best)))
        ax.set_xticklabels([name.split()[0] for name in player_best['player_name']],
                          rotation=45, ha='right', color='white')
        
        for bar, value in zip(bars, player_best[selected_metric]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
                   f'{value:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='white')
        
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
        
        for spine in ax.spines.values():
            spine.set_color('white')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        st.subheader("Trunk Rotation Rankings")
        rankings_df = player_best.copy()
        rankings_df['Rank'] = range(1, len(rankings_df) + 1)
        group_median = player_best[selected_metric].median()
        rankings_df['vs_median'] = ((rankings_df[selected_metric] - group_median) / group_median * 100).round(1)
        rankings_df = rankings_df[['Rank', 'player_name', selected_metric, 'vs_median']]
        rankings_df.columns = ['Rank', 'Player', selected_metric_display, 'vs Median (%)']
        
        st.dataframe(rankings_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Left vs Right Rotation Comparison")
        
        left_data = trunk_data[trunk_data['laterality'] == 'LeftSide'].groupby('player_name')[selected_metric].max()
        right_data = trunk_data[trunk_data['laterality'] == 'RightSide'].groupby('player_name')[selected_metric].max()
        
        comparison_df = pd.DataFrame({
            'Left': left_data,
            'Right': right_data
        }).dropna()
        
        if comparison_df.empty:
            st.info("Need bilateral data for comparison")
        else:
            comparison_df['Asymmetry (%)'] = ((comparison_df['Left'] - comparison_df['Right']).abs() / 
                                               comparison_df[['Left', 'Right']].max(axis=1) * 100).round(1)
            comparison_df['Dominant Side'] = comparison_df.apply(
                lambda x: 'Left' if x['Left'] > x['Right'] else 'Right', axis=1
            )
            
            comparison_df = comparison_df.reset_index()
            comparison_df.columns = ['Player', 'Left (N)', 'Right (N)', 'Asymmetry (%)', 'Dominant Side']
            
            st.dataframe(comparison_df.sort_values('Asymmetry (%)', ascending=False), 
                        use_container_width=True, hide_index=True)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            fig.patch.set_facecolor('#1e1e1e')
            ax.set_facecolor('#1e1e1e')
            
            x = range(len(comparison_df))
            width = 0.35
            
            bars1 = ax.bar([i - width/2 for i in x], comparison_df['Left (N)'], width, 
                          label='Left', color='#4A90A4', alpha=0.8)
            bars2 = ax.bar([i + width/2 for i in x], comparison_df['Right (N)'], width,
                          label='Right', color='#C41E3A', alpha=0.8)
            
            ax.set_ylabel(selected_metric_display, color='white')
            ax.set_title('Left vs Right Trunk Rotation', color='white', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([name.split()[0] for name in comparison_df['Player']], 
                             rotation=45, ha='right', color='white')
            ax.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
            ax.tick_params(colors='white')
            
            for spine in ax.spines.values():
                spine.set_color('white')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    with tab3:
        st.subheader("Rotational Asymmetry Analysis")
        
        left_data_t3 = trunk_data[trunk_data['laterality'] == 'LeftSide'].groupby('player_name')[selected_metric].max()
        right_data_t3 = trunk_data[trunk_data['laterality'] == 'RightSide'].groupby('player_name')[selected_metric].max()
        
        comparison_df_t3 = pd.DataFrame({
            'Left': left_data_t3,
            'Right': right_data_t3
        }).dropna()
        
        if not comparison_df_t3.empty:
            comparison_df_t3['Asymmetry (%)'] = ((comparison_df_t3['Left'] - comparison_df_t3['Right']).abs() / 
                                                   comparison_df_t3[['Left', 'Right']].max(axis=1) * 100).round(1)
            comparison_df_t3 = comparison_df_t3.reset_index()
        
        if comparison_df_t3.empty:
            st.info("Need bilateral data for asymmetry analysis")
        else:
            high_asymmetry = comparison_df_t3[comparison_df_t3['Asymmetry (%)'] > 10]
            
            col1, col2 = st.columns(2)
            with col1:
                avg_asymmetry = comparison_df_t3['Asymmetry (%)'].mean()
                st.metric("Team Avg Asymmetry", f"{avg_asymmetry:.1f}%")
            with col2:
                st.metric("Players with >10% Asymmetry", len(high_asymmetry))
            
            if len(high_asymmetry) > 0:
                st.warning("**Players with Significant Rotational Asymmetry (>10%):**")
                for _, row in high_asymmetry.iterrows():
                    player = row.get('player_name', row.get('index', 'Unknown'))
                    asym = row['Asymmetry (%)']
                    st.write(f"- {player}: {asym:.1f}% asymmetry")
            
            st.markdown("""
            ---
            ### Training Recommendations for Rotational Asymmetry
            
            **If Asymmetry > 10%:**
            - Focus on **unilateral rotational exercises** to the weaker side
            - Implement **anti-rotation holds** (Pallof press, bird dogs)
            - Address potential **hip or thoracic mobility restrictions**
            - Consider **manual therapy** for tissue quality imbalances
            
            **For Pitchers:**
            - Some asymmetry is expected due to throwing demands
            - Monitor for **increasing asymmetry** over time
            - Ensure adequate **deceleration strength** on non-throwing side
            """)

def extract_performance_metrics_from_trials(trials_df, test_data):
    """Extract performance metrics from the results field in trial data"""
    if trials_df.empty:
        return pd.DataFrame()
    
    performance_data = []
    
    test_mapping = {}
    for _, test in test_data.iterrows():
        test_mapping[test['testId']] = {
            'profileId': test['profileId'],
            'testType': test['testType']
        }
    
    for _, trial in trials_df.iterrows():
        if 'results' not in trial or not trial['results']:
            continue
            
        test_info = test_mapping.get(trial['testId'], {})
        
        for result in trial['results']:
            if not isinstance(result, dict):
                continue
                
            metric_data = {
                'testId': trial['testId'],
                'trialId': trial['id'],
                'athleteId': trial['athleteId'],
                'profileId': test_info.get('profileId', trial['athleteId']),
                'testType': test_info.get('testType', 'Unknown'),
                'recordedUTC': trial['recordedUTC'],
                'resultId': result.get('resultId'),
                'value': result.get('value'),
                'time': result.get('time'),
                'limb': result.get('limb'),
                'repeat': result.get('repeat')
            }
            
            definition = result.get('definition', {})
            metric_data.update({
                'metric_name': definition.get('name', f"Metric_{result.get('resultId')}"),
                'metric_result': definition.get('result', ''),
                'description': definition.get('description', ''),
                'units': definition.get('unit', ''),
                'repeatable': definition.get('repeatable', False),
                'asymmetry': definition.get('asymmetry', False)
            })
            
            performance_data.append(metric_data)
    
    return pd.DataFrame(performance_data) if performance_data else pd.DataFrame()

def create_leaderboard_dashboard(perf_df, kats_players):
    """Create focused leaderboard dashboard for the four key tests"""
    
    if perf_df.empty:
        st.warning("No performance data available")
        return
    
    profile_id_to_name = {}
    for name, info in kats_players.items():
        for _, row in perf_df.iterrows():
            athlete_id = row.get('athleteId') or row.get('profileId')
            if athlete_id and athlete_id in [info.get('player_id'), name]:
                profile_id_to_name[athlete_id] = name
                break
    
    if not profile_id_to_name:
        if 'name_to_profile_id' in st.session_state:
            profile_id_to_name = {v: k for k, v in st.session_state.name_to_profile_id.items()}
    
    perf_df['player_name'] = perf_df['profileId'].map(profile_id_to_name)
    
    name_to_handedness = {name: info['handedness'] for name, info in kats_players.items()}
    perf_df['handedness'] = perf_df['player_name'].map(name_to_handedness)
    
    perf_df['testType'] = perf_df['testType'].replace('SLJ', 'SJ')
    
    test_mapping = {
        'CMJ': 'CMJ',
        'SJ': 'Squat Jump', 
        'HJ': 'Hop Jump',
        'PPU': 'Plyo Pushup'
    }
    
    target_test_codes = ['CMJ', 'SJ', 'HJ', 'PPU']
    available_tests = perf_df['testType'].unique()
    
    st.subheader("Force Plate Test Leaderboards")
    
    filtered_test_codes = [test for test in target_test_codes if test in available_tests]
    
    if not filtered_test_codes:
        st.warning(f"None of the target tests ({target_test_codes}) found in data. Available: {list(available_tests)}")
        filtered_test_codes = list(available_tests)
        test_mapping.update({code: code for code in available_tests if code not in test_mapping})
    
    tab_names = [test_mapping.get(code, code) for code in filtered_test_codes]
    tabs = st.tabs(tab_names)
    
    for i, test_code in enumerate(filtered_test_codes):
        with tabs[i]:
            display_name = test_mapping.get(test_code, test_code)
            create_test_leaderboard(perf_df, test_code, display_name)

def create_test_leaderboard(perf_df, test_code, display_name):
    """Create leaderboard for a specific test type"""
    
    test_data = perf_df[perf_df['testType'] == test_code].copy()
    
    if test_data.empty:
        st.warning(f"No data found for {display_name} ({test_code})")
        return
    
    st.subheader(f"{display_name} Performance Analysis")
    
    key_metrics = {
        'CMJ': ['Jump Height (Flight Time)', 'Peak Power', 'Takeoff Peak Force', 'RSI-modified'],
        'SJ': ['Jump Height (Flight Time)', 'Peak Power', 'Takeoff Peak Force'],
        'PPU': ['Peak Power', 'Peak Force', 'Flight Time'],
        'HJ': ['Jump Height (Flight Time)', 'Peak Force', 'Landing RFD']
    }
    
    available_metrics = test_data['metric_name'].unique()
    target_metrics = key_metrics.get(test_code, [])
    
    matched_metrics = []
    for target in target_metrics:
        for available in available_metrics:
            if target.lower() in available.lower() or available.lower() in target.lower():
                matched_metrics.append(available)
                break
    
    if not matched_metrics:
        st.warning(f"No key metrics found for {display_name}")
        st.write(f"Available metrics: {list(available_metrics)[:10]}...")
        return
    
    selected_metric = st.selectbox(f"Select {display_name} Metric:", matched_metrics)
    
    metric_data = test_data[
        (test_data['metric_name'] == selected_metric) & 
        (test_data['limb'] == 'Trial')
    ].copy()
    
    if metric_data.empty:
        st.warning(f"No data found for {selected_metric}")
        return
    
    Q1 = metric_data['value'].quantile(0.25)
    Q3 = metric_data['value'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outliers = metric_data[(metric_data['value'] < lower_bound) | (metric_data['value'] > upper_bound)]
    outlier_count = len(outliers)
    
    filtered_metric_data = metric_data[
        (metric_data['value'] >= lower_bound) & 
        (metric_data['value'] <= upper_bound)
    ].copy()
    
    if outlier_count > 0:
        st.info(f"Automatically removed {outlier_count} obvious outlier(s) for cleaner analysis")
    
    player_best = filtered_metric_data.groupby('player_name')['value'].max().reset_index()
    player_best = player_best.sort_values('value', ascending=False)
    
    if 'handedness' in filtered_metric_data.columns:
        handedness_map_local = filtered_metric_data.groupby('player_name')['handedness'].first().to_dict()
        player_best['handedness'] = player_best['player_name'].map(handedness_map_local)
    
    sample_metric = filtered_metric_data.iloc[0]
    units = sample_metric.get('units', '')
    description = sample_metric.get('description', '')
    
    st.info(f"**{selected_metric}** ({units}): {description}")
    
    group_avg = player_best['value'].mean()
    group_std = player_best['value'].std()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Group Average", f"{group_avg:.2f} {units}")
    with col2:
        above_avg = len(player_best[player_best['value'] >= group_avg])
        st.metric("Above Average", f"{above_avg}/{len(player_best)}")
    with col3:
        if len(player_best) > 0:
            best_performer = player_best.iloc[0]['player_name']
            st.metric("Top Performer", best_performer.split()[0])
        else:
            st.metric("Top Performer", "N/A")
    with col4:
        cv = (group_std / group_avg) * 100 if group_avg != 0 else 0
        st.metric("Coefficient of Variation", f"{cv:.1f}%")
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = []
    for _, row in player_best.iterrows():
        value = row['value']
        if value >= group_avg + 0.5 * group_std:
            colors.append('#2E8B8B')
        elif value >= group_avg:
            colors.append('#4A90A4')
        elif value >= group_avg - 0.5 * group_std:
            colors.append('#FFA500')
        else:
            colors.append('#FF6B6B')
    
    bars = ax.bar(range(len(player_best)), player_best['value'], color=colors, alpha=0.8)
    
    ax.axhline(y=group_avg, color='white', linestyle='--', linewidth=2, alpha=0.8, 
              label=f'Group Average: {group_avg:.2f} {units}')
    
    ax.set_title(f'ECC Kats Baseball\n{display_name} - {selected_metric}', 
                fontsize=16, pad=20, fontweight='bold')
    ax.set_ylabel(f'{selected_metric} ({units})', fontsize=12)
    ax.set_xlabel('Players (Ranked by Performance)', fontsize=12)
    
    ax.set_xticks(range(len(player_best)))
    ax.set_xticklabels([name.split()[0] for name in player_best['player_name']], 
                      rotation=45, ha='right')
    
    for bar, value in zip(bars, player_best['value']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.01,
               f'{value:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    st.subheader(f"{display_name} - {selected_metric} Rankings")
    
    rankings_df = player_best.copy()
    rankings_df['rank'] = range(1, len(rankings_df) + 1)
    rankings_df['vs_average'] = ((rankings_df['value'] - group_avg) / group_avg * 100).round(1)
    rankings_df['percentile'] = [100 - (i/len(rankings_df))*100 for i in range(len(rankings_df))]
    
    display_columns = ['rank', 'player_name', 'value', 'vs_average', 'percentile']
    column_names = ['Rank', 'Player', f'Best ({units})', 'vs Avg (%)', 'Percentile']
    
    if 'handedness' in rankings_df.columns:
        display_columns.append('handedness')
        column_names.append('Hand')
    
    display_df_local = rankings_df[display_columns].copy()
    display_df_local.columns = column_names
    
    st.dataframe(display_df_local, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("## Training Interpretation Guide")
    
    if test_code == 'CMJ':
        st.markdown("""
        ### CMJ (Countermovement Jump) Training Recommendations
        **Jump Height (Flight Time):**
        - **Below Average:** Focus on explosive power development with **explosive plyometric exercises** (box jumps, depth jumps, tuck jumps)
        
        **Peak Power:** 
        - **Low values:** Need **power-speed training** (medicine ball throws, jump squats, lift derivatives)
        
        **Peak Force:** 
        - **Below average:** **Strength deficits** - emphasize **heavy compound movements** (squats, deadlifts, hip thrusts)
        
        **RSI-modified:** 
        - **Poor reactive strength:** Requires **reactive plyometrics** (pogos, quick ground contacts, hurdle hops)
        """)
    elif test_code == 'SJ':
        st.markdown("""
        ### Squat Jump (SJ) Training Recommendations
        **Jump Height:** 
        - **Below Average:** **Concentric power deficits** - focus on **pause squats, quarter squats with heavy load**
        
        **Peak Power:** 
        - **Low values:** Need **ballistic training** (jump squats, medicine ball slams, explosive bench press)
        
        **Peak Force:** 
        - **Poor scores:** Require **maximal strength training** (1-5 rep range compound movements)
        
        **Training Parameters:**
        - **Strength focus:** Reps at 85-95% 1RM
        - **Power focus:** Reps at 30-60% 1RM with pause and explosive intent
        - **Concentric emphasis:** Remove eccentric component with pause squats and pin squats
        """)
    elif test_code == 'HJ':
        st.markdown("""
        ### Hop Jump (HJ) Training Recommendations
        **Jump Height:**
        - **Below Average:** Single-leg power deficits - emphasize **unilateral plyometrics** (single-leg bounds, lateral hops)
        
        **Peak Force:** 
        - **Low values:** **Single-leg strength needs** (Bulgarian split squats, single-leg RDLs, step-ups)
        
        **Landing RFD:** 
        - **Poor landing mechanics:** Require **eccentric control training** (landing drills, eccentric squats)
        
        **Training Parameters:**
        - **Unilateral strength:** 3-4 sets × 6-10 reps per leg with challenging load
        - **Single-leg power:** 3-5 sets × 3-6 reps per leg with explosive intent
        - **Landing mechanics:** Focus on controlled landings with 2-3 second holds
        """)
    elif test_code == 'PPU':
        st.markdown("""
        ### Plyo Pushup (PPU) Training Recommendations
        **Peak Power:**
        - **Below Average:** Upper body explosive deficit - focus on **upper body plyometrics** (plyo pushups, medicine ball chest passes, clap pushups)
        
        **Peak Force:** 
        - **Low values:** **Upper body strength needs** (bench press, weighted pushups, dips)
        
        **Flight Time:** 
        - **Poor airborne time:** Requires **explosive pushing power** (ballistic bench press, speed pushups)
        
        **Training Parameters:**
        - **Upper body strength:** Reps at 80-90% 1RM
        - **Upper body power:** Reps at 30-50% 1RM with maximal speed
        - **Plyometric progression:** Start with incline variations, progress to decline
        """)
    
    st.markdown("""
    ---
    *Note: All training recommendations should be implemented under qualified supervision with proper progression and recovery protocols.*
    """)

def main():
    st.title("ECC Kats Baseball - Force Plate Leaderboards")
    st.caption("Performance rankings for CMJ, Squat Jump, Plyo Pushup, and Hop Test")
    
    if 'kats_players' not in st.session_state:
        st.session_state.kats_players = {}
    if 'all_profiles' not in st.session_state:
        st.session_state.all_profiles = {}
    if 'name_to_profile_id' not in st.session_state:
        st.session_state.name_to_profile_id = {}
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    
    if not st.session_state.initialized:
        with st.spinner("Loading team data..."):
            kats_players = load_kats_players_from_csv()
            all_profiles = fetch_all_profiles()
            name_to_profile_id = match_players_to_profiles(kats_players, all_profiles)
            
            st.session_state.kats_players = kats_players
            st.session_state.all_profiles = all_profiles
            st.session_state.name_to_profile_id = name_to_profile_id
            st.session_state.initialized = True
    
    st.sidebar.subheader("Force Plate Team Leaderboards")
    with st.sidebar.expander("Team Info", expanded=True):
        st.write(f"**Players with Motion Capture Data:** {len(st.session_state.kats_players)}")
        st.write(f"**Matched with VALD:** {len(st.session_state.name_to_profile_id)}")
    
    selected_date = st.date_input(
        "Select Testing Date (First Test 2025/12/06)",
        value=date(2025, 12, 6),
        min_value=date(2025, 1, 1),
        max_value=date.today()
    )
    
    if st.button("Load Force Plate Data", type="primary"):
        profile_ids = list(st.session_state.name_to_profile_id.values())
        team_id = get_team_id()
        
        with st.spinner("Loading force plate data..."):
            df = fetch_forcedecks_tests(profile_ids, selected_date.strftime('%Y-%m-%d'))
            
            if not df.empty:
                df_filtered = df[df['date'] == selected_date].copy()
                
                if not df_filtered.empty:
                    test_ids = df_filtered['testId'].unique().tolist()
                    trials_df = fetch_test_trials_batch(team_id, test_ids)
                    
                    if not trials_df.empty:
                        perf_df = extract_performance_metrics_from_trials(trials_df, df_filtered)
                        
                        if not perf_df.empty:
                            st.session_state.performance_data = perf_df
                            st.success(f"Loaded {len(perf_df)} performance measurements from {len(df_filtered)} tests")
                        else:
                            st.error("No performance metrics extracted from trial data")
                    else:
                        st.error("No trial data found")
                else:
                    st.warning(f"No test data found for {selected_date}")
            else:
                st.warning("No test data found")
    
    if 'performance_data' in st.session_state and not st.session_state.performance_data.empty:
        create_leaderboard_dashboard(st.session_state.performance_data, st.session_state.kats_players)
    else:
        st.info("Click 'Load Force Plate Data' to generate leaderboards for the selected date")

    st.markdown("---")
    st.title("Rotational Power Leaderboards")
    st.caption("Trunk Rotation Testing via Isometric Pulls")

    dynamo_date = st.date_input(
        "Select Dynamo Testing Start Date",
        value=date(2025, 12, 6),
        min_value=date(2025, 1, 1),
        max_value=date.today(),
        key="dynamo_date_select"
    )

    if st.button("Load Rotational Power Data", type="primary", key="load_dynamo"):
        if 'name_to_profile_id' not in st.session_state or not st.session_state.name_to_profile_id:
            kats_players = load_kats_players_from_csv()
            all_profiles = fetch_all_profiles()
            name_to_profile_id = match_players_to_profiles(kats_players, all_profiles)
            st.session_state.kats_players = kats_players
            st.session_state.all_profiles = all_profiles
            st.session_state.name_to_profile_id = name_to_profile_id
        
        profile_ids = list(st.session_state.name_to_profile_id.values())
        
        with st.spinner("Loading Dynamo rotational data..."):
            dynamo_df = fetch_dynamo_tests(profile_ids, dynamo_date.strftime('%Y-%m-%d'))
            
            if not dynamo_df.empty:
                dynamo_perf_df = extract_dynamo_metrics(dynamo_df, st.session_state.all_profiles)
                
                if not dynamo_perf_df.empty:
                    st.session_state.dynamo_performance_data = dynamo_perf_df
                    trunk_count = len(dynamo_perf_df[dynamo_perf_df['bodyRegion'] == 'Trunk'])
                    st.success(f"Loaded {len(dynamo_perf_df)} Dynamo measurements ({trunk_count} Trunk Rotation)")
                else:
                    st.warning("No performance metrics extracted from Dynamo data")
            else:
                st.warning("No Dynamo test data found for selected date range")

    if 'dynamo_performance_data' in st.session_state and not st.session_state.dynamo_performance_data.empty:
        create_trunk_rotation_leaderboard(st.session_state.dynamo_performance_data)
    else:
        st.info("Click 'Load Rotational Power Data' to generate leaderboards for the selected date.")

if __name__ == "__main__":
    main()

# ─────────────────────────────────────────────────────────────────────────────
# Table Assessments - Merged view (Session 2 data, fallback to Session 1)
# ─────────────────────────────────────────────────────────────────────────────

def load_assessment_df(excel_file_path):
    """Read an assessment Excel file and return trimmed DataFrame, or None on failure."""
    try:
        df = pd.read_excel(excel_file_path, engine='openpyxl')
        if df.shape[1] >= 22:
            df = df.iloc[:, 0:22].copy()
        df = df.dropna(how='all')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].round(2)
        # Strip trailing/leading whitespace from all string cells (fixes name mismatches)
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        # Normalize Sleep column name to canonical lowercase form regardless of source casing
        df.columns = [
            "Sleep (avg # of hours)" if c.strip().lower() == "sleep (avg # of hours)" else c
            for c in df.columns
        ]
        return df
    except Exception:
        return None


SLEEP_COL = "Sleep (avg # of hours)"

# Players who always use Session 1 values regardless of Session 2 data
FORCE_SESSION1_PLAYERS = {"Zack", "Cole", "Caleb"}

def merge_assessment_sessions(df_s1, df_s2):
    """
    Show Session 2 as-is by default.

    For each player row: if their Sleep value is missing (NaN) in Session 2,
    replace the ENTIRE row with that player's Session 1 row instead.
    Players with no Session 1 data keep their (incomplete) Session 2 row.
    """
    if df_s1 is None and df_s2 is None:
        return None
    if df_s2 is None:
        return df_s1
    if df_s1 is None:
        return df_s2

    id_col = df_s2.columns[0]

    # Build a lookup of Session 1 rows keyed on the player identifier
    s1_lookup = df_s1.set_index(id_col)

    result_rows = []
    for _, row_s2 in df_s2.iterrows():
        player = row_s2[id_col]
        sleep_val = row_s2.get(SLEEP_COL, None)
        sleep_missing = pd.isna(sleep_val) if sleep_val is not None else True

        # If Sleep is missing OR player is hardcoded to use S1, use entire Session 1 row
        if (sleep_missing or player in FORCE_SESSION1_PLAYERS) and player in s1_lookup.index:
            s1_row = s1_lookup.loc[player].copy()
            s1_row[id_col] = player          # restore the id column value
            result_rows.append(s1_row)
        else:
            result_rows.append(row_s2)

    result = pd.DataFrame(result_rows).reset_index(drop=True)

    # Ensure column order matches Session 2 (add any extra S1 cols at the end)
    s2_cols = list(df_s2.columns)
    extra_cols = [c for c in result.columns if c not in s2_cols]
    result = result[[c for c in s2_cols if c in result.columns] + extra_cols]

    return result


def render_merged_assessment(display_df):
    """Render a single merged assessment DataFrame with outlier highlighting."""

    def highlight_outliers(val, median, std, column_name):
        if pd.isna(val) or not isinstance(val, (int, float)):
            return ''
        lower_bound = median - 2 * std
        upper_bound = median + 2 * std
        if val < lower_bound:
            return 'background-color: #cc0000; color: white; font-weight: bold'
        elif val > upper_bound:
            return 'background-color: #006400; color: white; font-weight: bold'
        return ''

    stats_info = {}
    for column in display_df.columns:
        numeric_col = pd.to_numeric(display_df[column], errors='coerce')
        if numeric_col.notna().sum() >= 3:
            median = numeric_col.median()
            std = numeric_col.std()
            if std > 0:
                stats_info[column] = {
                    'median': median,
                    'std': std,
                    'lower_bound': median - 2 * std,
                    'upper_bound': median + 2 * std
                }

    if stats_info:
        def apply_outlier_styling(row):
            styles = [''] * len(row)
            for idx, (column, val) in enumerate(row.items()):
                if column in stats_info:
                    styles[idx] = highlight_outliers(
                        val,
                        stats_info[column]['median'],
                        stats_info[column]['std'],
                        column
                    )
            return styles

        styled_df = display_df.style.apply(apply_outlier_styling, axis=1)

        col_leg1, col_leg2, col_leg3 = st.columns(3)
        with col_leg1:
            st.error("🔴 Mobility Deficiency")
        with col_leg2:
            st.success("🟢 Hyper Mobility (ensure conservation)")
        with col_leg3:
            st.caption("★ Session 2 values shown; players missing Sleep data use full Session 1 row")

        col_config = {}
        if "Sleep (avg # of hours)" in display_df.columns:
            col_config["Sleep (avg # of hours)"] = st.column_config.NumberColumn(format="%.1f")

        st.dataframe(
            styled_df,
            hide_index=True,
            use_container_width=True,
            height=600,
            column_config=col_config,
        )

        with st.expander("View Column Statistics", expanded=False):
            stats_df = pd.DataFrame(stats_info).T.round(2)
            st.dataframe(stats_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Training Translation Guide")

        st.markdown("""
        **Mobility Deficiency (red):** Focus on mobility work, dynamic stretching, and tissue quality exercises to improve movement capacity. Prioritize addressing movement restrictions before adding load or intensity.

        - Increase mobility drills and dynamic warm-ups
        - Address tissue quality (foam rolling, soft tissue work)
        - Focus on controlled articular rotations (CARs)
        - Gradually expand range of motion through progressive stretching
        """)

        st.markdown("""
        **Hyper Mobility (green):** Ensure adequate strength development to support and control this mobility, particularly at end ranges of motion.

        - Emphasize strength training throughout full range of motion
        - Focus on eccentric control and end-range strength
        - Implement tempo work and isometric holds
        - Develop motor control to utilize available mobility effectively
        """)
    else:
        st.warning("No numeric columns found for outlier detection. Displaying table without highlighting.")
        st.dataframe(display_df, hide_index=True, use_container_width=True, height=600)


st.header("Table Assessments - Team View")
st.caption("February 2026 data shown; players missing Sleep data fall back to full December 2025 row")

_df_s1 = load_assessment_df(os.path.join("data", "KatsBaseballTableAssessment.xlsx"))
_df_s2 = load_assessment_df(os.path.join("data", "KatsBaseballTableAssessment2.xlsx"))

if _df_s1 is None and _df_s2 is None:
    st.error("Neither assessment file could be loaded. Please check the data directory.")
else:
    if _df_s1 is None:
        st.warning("Session 1 (December) file not found — showing Session 2 only.")
    if _df_s2 is None:
        st.warning("Session 2 (February) file not found — showing Session 1 only.")

    _merged_df = merge_assessment_sessions(_df_s1, _df_s2)
    render_merged_assessment(_merged_df)

# Footer
st.markdown("---")
st.markdown("*ECC Kats Home Dashboard | Built by Liquid Sports Lab*")