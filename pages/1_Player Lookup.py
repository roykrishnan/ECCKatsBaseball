import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # before pyplot import
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import os
import glob

# Page configuration
st.set_page_config(
    page_title="ECC Baseball Player Lookup", 
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

# Session configuration - two data collection periods
SESSION_CONFIG = {
    "December 2025": {
        "bullpen_dir": "data/BullpenReports120626",
        "dev_reports_dir": "data/Dev Reports",
        "biomech_dir": "data/Biomech Reports",
        "assessment_file": "data/KatsBaseballTableAssessment2.xlsx",
        "date_code": "120626"
    },
    "February 2026": {
        "bullpen_dir": "data/BullpenReports021526",
        "dev_reports_dir": "data/Dev Reports",
        "biomech_dir": "data/Biomech Reports",
        "assessment_file": "data/KatsBaseballTableAssessment2.xlsx",
        "date_code": "021526"
    }
}

@st.cache_data
def load_all_player_data(data_dir):
    """Load all player data for lookup functionality"""
    all_players = {}
    
    if not os.path.exists(data_dir):
        return {}
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    if not csv_files:
        return {}
    
    # Handedness mapping - update as needed for ECC roster
    handedness_map = {}
    
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
            
            player_id = None
            player_name = None
            data_start_row = None
            
            for i, line in enumerate(lines):
                if 'Player ID:' in line:
                    player_id = line.split(',')[1].strip()
                elif 'Player Name:' in line:
                    player_name = line.split(',')[1].strip()
                elif line.startswith('No,Date'):
                    data_start_row = i
                    break
            
            if data_start_row is not None and player_name and player_id:
                pitch_data = pd.read_csv(csv_file, skiprows=data_start_row, encoding=successful_encoding)
                
                pitch_data = pitch_data[pitch_data['Pitch Type'].notna()]
                pitch_data = pitch_data[pitch_data['Pitch Type'] != '-']
                pitch_data = pitch_data[pitch_data['Pitch Type'] != '']
                
                if len(pitch_data) > 0:
                    numeric_cols = ['Velocity', 'Total Spin', 'VB (trajectory)', 'HB (trajectory)', 'Release Height', 'Release Side', 'Horizontal Angle']
                    for col in numeric_cols:
                        if col in pitch_data.columns:
                            pitch_data[col] = pd.to_numeric(pitch_data[col], errors='coerce')
                    
                    if 'Gyro Degree (deg)' in pitch_data.columns:
                        def clean_gyro_degree(value):
                            if pd.isna(value) or value == '' or value == '-':
                                return np.nan
                            try:
                                return float(value)
                            except (ValueError, TypeError):
                                str_val = str(value)
                                import re
                                numbers = re.findall(r'-?\d+\.?\d*', str_val)
                                if numbers:
                                    return float(numbers[0])
                                return np.nan
                        pitch_data['Gyro Degree (deg)'] = pitch_data['Gyro Degree (deg)'].apply(clean_gyro_degree)
                    
                    all_players[player_name] = {
                        'player_id': player_id,
                        'handedness': handedness_map.get(player_name, 'RHP'),
                        'pitch_data': pitch_data,
                        'file_path': csv_file
                    }
        
        except Exception as e:
            continue
    
    return all_players


def calculate_Kats_stuff_plus_for_pitch_type(df, pitch_type, player_fastball_velocity=None):
    """Calculate Kats Stuff+ for a specific pitch type"""
    
    weights = {
        'velocity': 0.20,
        'spin_rate': 0.15,
        'release_height': 0.10,
        'distinctive_shape': 0.15,
        'release_side': 0.08,
        'horizontal_break': 0.10,
        'vertical_break': 0.10,
        'speed_diff': 0.07,
        'horizontal_angle': 0.05,
        'movement_distinction': 0.10
    }
    
    velocity = df['Velocity'].mean()
    spin_rate = df['Total Spin'].mean()
    release_height = df['Release Height'].mean() if 'Release Height' in df.columns else 5.5
    release_side = df['Release Side'].mean() if 'Release Side' in df.columns else 0
    horizontal_angle = df['Horizontal Angle'].mean() if 'Horizontal Angle' in df.columns else 0
    h_break = abs(df['HB (trajectory)'].mean()) if 'HB (trajectory)' in df.columns else 0
    v_break = df['VB (trajectory)'].mean() if 'VB (trajectory)' in df.columns else 0
    v_break_abs = abs(v_break)
    
    if pitch_type != 'Fastball' and player_fastball_velocity is not None:
        speed_diff = max(0, player_fastball_velocity - velocity)
    else:
        speed_diff = 0

    if pitch_type == 'Fastball':
        velocity_score = min(1.0, max(0.0, (velocity - 70) / 30))
    elif pitch_type == 'ChangeUp':
        velocity_score = min(1.0, max(0.0, (velocity - 65) / 25))
    else:
        velocity_score = min(1.0, max(0.0, (velocity - 65) / 30))
    
    if pitch_type == 'Fastball':
        spin_score = min(1.0, max(0.0, (spin_rate - 1800) / 1200))
    elif pitch_type == 'ChangeUp':
        spin_score = max(0.2, min(1.0, 1.0 - (spin_rate - 1000) / 1500))
    else:
        spin_score = min(1.0, max(0.0, (spin_rate - 2000) / 1000))
    
    height_deviation = abs(release_height - 5.5)
    height_score = min(1.0, height_deviation / 1.0)
    
    side_deviation = abs(release_side - 0.0)
    side_score = min(1.0, side_deviation / 2.0)
    
    angle_score = max(0.0, 1.0 - (abs(horizontal_angle) / 20))
    speed_diff_score = min(1.0, speed_diff / 15) if pitch_type != 'Fastball' else 0.5
    h_break_score = min(1.0, h_break / 20)
    v_break_score = min(1.0, v_break_abs / 20)
    
    shape_differential = abs(h_break - v_break_abs)
    distinctive_shape_score = min(1.0, shape_differential / 15)
    
    total_movement = h_break + v_break_abs
    if total_movement > 5:
        min_break = min(h_break, v_break_abs)
        max_break = max(h_break, v_break_abs)
        if max_break > 0:
            similarity_ratio = min_break / max_break
            movement_magnitude_factor = min(1.0, total_movement / 30)
            base_distinction_score = 1.0 - similarity_ratio
            movement_distinction_score = base_distinction_score * (0.5 + 0.5 * (1 - movement_magnitude_factor * similarity_ratio))
            movement_distinction_score = max(0.0, min(1.0, movement_distinction_score))
        else:
            movement_distinction_score = 0.5
    else:
        movement_distinction_score = 0.5
    
    composite_score = (
        velocity_score * weights['velocity'] +
        spin_score * weights['spin_rate'] +
        height_score * weights['release_height'] +
        side_score * weights['release_side'] +
        angle_score * weights['horizontal_angle'] +
        speed_diff_score * weights['speed_diff'] +
        h_break_score * weights['horizontal_break'] +
        v_break_score * weights['vertical_break'] +
        distinctive_shape_score * weights['distinctive_shape'] +
        movement_distinction_score * weights['movement_distinction']
    )
    
    stuff_plus = 100 + (composite_score - 0.5) * 100
    stuff_plus = max(40, min(160, stuff_plus))
    
    return stuff_plus


def calculate_player_stuff_plus(pitch_data):
    """Calculate Stuff+ for each pitch type for a player"""
    pitch_stuff = {}
    
    fastball_data = pitch_data[
        pitch_data['Pitch Type'].str.contains('Fastball', case=False, na=False)
    ]
    player_fastball_velocity = fastball_data['Velocity'].mean() if len(fastball_data) > 0 else None
    
    pitch_type_mapping = {
        'Fastball': ['Fastball'],
        'ChangeUp': ['ChangeUp'],
        'Slider': ['Slider'],
        'Curveball': ['CurveBall'],
        'Cutter': ['Cutter'],
        'Splitter': ['Splitter']
    }
    
    for pitch_category, pitch_variants in pitch_type_mapping.items():
        pitch_type_data = pitch_data[pitch_data['Pitch Type'].isin(pitch_variants)]
        
        if len(pitch_type_data) >= 1:
            stuff_plus = calculate_Kats_stuff_plus_for_pitch_type(
                pitch_type_data, pitch_category, player_fastball_velocity
            )
            
            avg_velocity = pitch_type_data['Velocity'].mean()
            speed_diff = (player_fastball_velocity - avg_velocity) if (pitch_category != 'Fastball' and player_fastball_velocity is not None) else 0
            
            pitch_stuff[pitch_category] = {
                'stuff_plus': stuff_plus,
                'count': len(pitch_type_data),
                'avg_velocity': avg_velocity,
                'avg_spin': pitch_type_data['Total Spin'].mean(),
                'avg_h_break': abs(pitch_type_data['HB (trajectory)'].mean()) if 'HB (trajectory)' in pitch_type_data.columns else 0,
                'avg_v_break': pitch_type_data['VB (trajectory)'].mean() if 'VB (trajectory)' in pitch_type_data.columns else 0,
                'speed_diff': speed_diff
            }
    
    return pitch_stuff


def find_available_reports(player_name, reports_dir):
    """Find all available development reports for a player"""
    formatted_name = player_name.replace(" ", "")
    
    if not os.path.exists(reports_dir):
        return []
    
    pattern = os.path.join(reports_dir, f"{formatted_name}*.txt")
    report_files = glob.glob(pattern)
    
    available_reports = []
    for file_path in report_files:
        filename = os.path.basename(file_path)
        if filename.endswith('.txt') and len(filename) >= 10:
            date_part = filename[-10:-4]
            if date_part.isdigit() and len(date_part) == 6:
                display_date = f"{date_part[:2]}/{date_part[2:4]}/{date_part[4:]}"
                available_reports.append({
                    'file_path': file_path,
                    'date_code': date_part,
                    'display_date': display_date,
                    'filename': filename
                })
    
    available_reports.sort(key=lambda x: x['date_code'], reverse=True)
    return available_reports


def load_specific_pitch_development_report(file_path):
    """Load a specific pitch development report from file path"""
    try:
        encodings_to_try = ['utf-8', 'utf-16', 'latin1', 'cp1252', 'iso-8859-1']
        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    report_content = f.read().strip()
                if report_content:
                    return report_content
                break
            except UnicodeDecodeError:
                continue
        return "Development report found but could not be read properly."
    except FileNotFoundError:
        return f"Report file not found: {file_path}"
    except Exception as e:
        return f"Error loading development report: {str(e)}"


def display_pitch_development_report_section(player_name, reports_dir, fallback_reports_dir=None):
    """Display the pitch development report section with date selection"""
    st.markdown('<h3 class="section-header">Player Development Report</h3>', unsafe_allow_html=True)
    
    available_reports = find_available_reports(player_name, reports_dir)
    
    using_fallback = False
    if not available_reports and fallback_reports_dir:
        available_reports = find_available_reports(player_name, fallback_reports_dir)
        if available_reports:
            using_fallback = True
            st.info("No reports found for current session. Showing reports from previous session.")
    
    if not available_reports:
        formatted_name = player_name.replace(" ", "")
        st.warning(f"No development reports found for {player_name}")
        st.info(f"Expected filename format: `{reports_dir}/{formatted_name}MMDDYY.txt`")
        return
    
    if len(available_reports) == 1:
        selected_report = available_reports[0]
        st.info(f"Report Date: {selected_report['display_date']}")
    else:
        st.subheader("Select Report Date:")
        cols = st.columns(min(len(available_reports), 4))
        
        if 'selected_report_date' not in st.session_state:
            st.session_state.selected_report_date = available_reports[0]['date_code']
        
        selected_report = None
        for i, report in enumerate(available_reports):
            col_idx = i % 4
            with cols[col_idx]:
                is_selected = st.session_state.selected_report_date == report['date_code']
                button_label = f"🔹 {report['display_date']}" if is_selected else report['display_date']
                if st.button(button_label, key=f"report_btn_{report['date_code']}"):
                    st.session_state.selected_report_date = report['date_code']
                    st.rerun()
                if is_selected:
                    selected_report = report
        
        if selected_report is None:
            selected_report = available_reports[0]
    
    report_content = load_specific_pitch_development_report(selected_report['file_path'])
    
    if report_content.startswith("Development report found") or report_content.startswith("Error loading") or report_content.startswith("Report file not found"):
        st.error(report_content)
    else:
        st.markdown(report_content)
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"*Report Date: {selected_report['display_date']}*")
        with col2:
            if using_fallback:
                st.caption("*(From previous session)*")
        if len(available_reports) > 1:
            st.caption(f"*{len(available_reports)} reports available for {player_name}*")


def create_stuff_plus_radar_chart(pitch_stuff, player_name):
    """Create radar chart showing Stuff+ for each pitch type"""
    all_pitch_types = list(pitch_stuff.keys())
    if not all_pitch_types:
        all_pitch_types = ['Fastball', 'ChangeUp', 'Slider']
    
    stuff_values = []
    for pitch_type in all_pitch_types:
        if pitch_type in pitch_stuff:
            stuff_plus_val = pitch_stuff[pitch_type]['stuff_plus']
            percentile = (stuff_plus_val - 40) / 120
            stuff_values.append(percentile)
        else:
            stuff_values.append(0.5)
    
    N = len(all_pitch_types)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    stuff_values += stuff_values[:1]
    avg_line = [0.5] * len(angles)
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    
    ax.plot(angles, stuff_values, linewidth=3, label=f'{player_name} Stuff+', color='#C41E3A')
    ax.fill(angles, stuff_values, alpha=0.3, color="#C41E3A")
    ax.plot(angles, avg_line, linewidth=2, label='League Average (100)',
            linestyle='--', color='gray', alpha=0.8)
    
    plt.xticks(angles[:-1], all_pitch_types, color='white', size=12)
    plt.yticks(np.linspace(0, 1, 6), ['0%', '20%', '40%', '60%', '80%', '100%'],
               color='white', size=10)
    plt.title(f'{player_name} Pitch Stuff+ Radar Chart', color='white', size=16, weight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    return fig


def create_movement_chart(pitch_data):
    """Create movement chart (HB vs VB)"""
    if len(pitch_data) == 0 or 'HB (trajectory)' not in pitch_data.columns or 'VB (trajectory)' not in pitch_data.columns:
        return None
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    pitch_types = pitch_data['Pitch Type'].unique()
    import numpy
    colors = plt.cm.Set1(numpy.linspace(0, 1, len(pitch_types)))
    
    from matplotlib.patches import Ellipse
    
    for i, pitch_type in enumerate(pitch_types):
        pitch_subset = pitch_data[pitch_data['Pitch Type'] == pitch_type]
        hb_data = pitch_subset['HB (trajectory)'].dropna()
        vb_data = pitch_subset['VB (trajectory)'].dropna()
        
        if len(hb_data) > 0 and len(vb_data) > 0:
            def remove_outliers(data):
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 2.0 * IQR
                upper_bound = Q3 + 2.0 * IQR
                return data[(data >= lower_bound) & (data <= upper_bound)]
            
            if len(hb_data) >= 3:
                hb_clean = remove_outliers(hb_data)
                vb_clean = remove_outliers(vb_data)
                if len(hb_clean) >= 1 and len(vb_clean) >= 1:
                    hb_data = hb_clean
                    vb_data = vb_clean
            
            mean_hb = hb_data.mean()
            mean_vb = vb_data.mean()
            
            ax.scatter(pitch_subset['HB (trajectory)'], pitch_subset['VB (trajectory)'],
                       c=[colors[i]], label=pitch_type, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
            
            if len(hb_data) >= 2 and len(vb_data) >= 2:
                std_hb = min(hb_data.std(), 3.0)
                std_vb = min(vb_data.std(), 3.0)
                ellipse = Ellipse((mean_hb, mean_vb),
                                  width=2*std_hb, height=2*std_vb,
                                  facecolor=colors[i], alpha=0.2,
                                  edgecolor=colors[i], linewidth=2, linestyle='--')
                ax.add_patch(ellipse)
            
            ax.scatter(mean_hb, mean_vb, c=[colors[i]], s=120, marker='D',
                       edgecolors='white', linewidth=2, alpha=0.9, zorder=10)
    
    ax.axhline(y=0, color='white', linestyle='-', alpha=0.7, linewidth=1)
    ax.axvline(x=0, color='white', linestyle='-', alpha=0.7, linewidth=1)
    
    h_break_data = pitch_data['HB (trajectory)'].dropna()
    v_break_data = pitch_data['VB (trajectory)'].dropna()
    
    if len(h_break_data) > 0 and len(v_break_data) > 0:
        max_h = max(abs(h_break_data.min()), abs(h_break_data.max())) * 1.2
        max_v = max(abs(v_break_data.min()), abs(v_break_data.max())) * 1.2
        ax.set_xlim(-max_h, max_h)
        ax.set_ylim(-max_v, max_v)
    else:
        ax.set_xlim(-25, 25)
        ax.set_ylim(-15, 25)
    
    ax.set_xlabel('Horizontal Break (inches)', color='white', fontsize=12)
    ax.set_ylabel('Vertical Break (inches)', color='white', fontsize=12)
    ax.set_title('Pitch Movement Profile', color='white', size=14, weight='bold')
    ax.legend(*ax.get_legend_handles_labels(), loc='upper right', framealpha=0.8)
    ax.text(0.02, 0.02, 'Diamond = Average\nDashed ellipse = Expected range',
            transform=ax.transAxes, ha='left', va='bottom',
            color='gray', fontsize=9, alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
    ax.grid(True, alpha=0.3, linestyle=':')
    ax.text(0.02, 0.98, 'Arm Side\nRising', transform=ax.transAxes, ha='left', va='top', color='gray', fontsize=9, alpha=0.7)
    ax.text(0.98, 0.98, 'Glove Side\nRising', transform=ax.transAxes, ha='right', va='top', color='gray', fontsize=9, alpha=0.7)
    ax.text(0.02, 0.12, 'Arm Side\nSinking', transform=ax.transAxes, ha='left', va='bottom', color='gray', fontsize=9, alpha=0.7)
    ax.text(0.98, 0.12, 'Glove Side\nSinking', transform=ax.transAxes, ha='right', va='bottom', color='gray', fontsize=9, alpha=0.7)
    
    return fig


def create_stuff_plus_bar_chart(pitch_stuff, player_name):
    """Create bar chart showing Stuff+ by pitch type"""
    if not pitch_stuff:
        return None
    
    pitch_types = list(pitch_stuff.keys())
    stuff_values = [pitch_stuff[pt]['stuff_plus'] for pt in pitch_types]
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(pitch_types, stuff_values, color="#C41E3A", alpha=0.8, edgecolor='white')
    
    for bar, value in zip(bars, stuff_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}', ha='center', va='bottom', color='white', fontweight='bold')
    
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.7, label='League Average (100)')
    ax.set_ylabel('Stuff+', color='white')
    ax.set_xlabel('Pitch Type', color='white')
    ax.set_title(f'{player_name} Stuff+ by Pitch Type', color='white', size=14, weight='bold')
    ax.set_ylim(40, 160)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if len(pitch_types) > 4:
        plt.xticks(rotation=45, ha='right')
    
    return fig


def display_pitch_stuff_details(pitch_stuff, pitch_type):
    """Display detailed Stuff+ breakdown for a pitch type"""
    if pitch_type not in pitch_stuff:
        st.info(f"No {pitch_type} data available")
        return
    
    data = pitch_stuff[pitch_type]
    st.subheader(f"{pitch_type} Stuff+ Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stuff+", f"{data['stuff_plus']:.1f}")
    with col2:
        st.metric("Pitch Count", f"{data['count']}")
    with col3:
        st.metric("Avg Velocity", f"{data['avg_velocity']:.1f} mph")
    with col4:
        st.metric("Avg Spin", f"{data['avg_spin']:.0f} rpm")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg H-Break", f"{data['avg_h_break']:.1f} in")
    with col2:
        st.metric("Avg V-Break", f"{data['avg_v_break']:.1f} in")
    with col3:
        if pitch_type != 'Fastball' and data['speed_diff'] > 0:
            st.metric("Speed Diff vs FB", f"{data['speed_diff']:.1f} mph")
        else:
            st.metric("Speed Diff vs FB", "N/A")
    
    stuff_plus_val = data['stuff_plus']
    if stuff_plus_val >= 120:
        interpretation, color = "Elite", "green"
    elif stuff_plus_val >= 110:
        interpretation, color = "Above Average", "blue"
    elif stuff_plus_val >= 90:
        interpretation, color = "Average", "gray"
    elif stuff_plus_val >= 80:
        interpretation, color = "Below Average", "orange"
    else:
        interpretation, color = "Poor", "red"
    
    st.markdown(f"**Stuff+ Grade:** :{color}[{interpretation}]")


# VALD API imports and configuration
import requests

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
def fetch_all_vald_profiles():
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
                profiles_dict = {}
                for profile in data["profiles"]:
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


@st.cache_data(ttl=1800)
def get_vald_team_id():
    """Get team ID from the v2019q3/teams endpoint"""
    token = get_access_token()
    if not token:
        return None
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(f"{VALD_CONFIG['forcedecks_base_url']}/v2019q3/teams", headers=headers)
        if response.ok:
            teams = response.json()
            if teams and len(teams) > 0:
                return teams[0].get('id') or teams[0].get('teamId')
        return None
    except Exception:
        return None


@st.cache_data(ttl=600)
def fetch_player_forcedecks_tests(profile_id, modified_from_date):
    """Fetch ForceDecks test data for a specific player"""
    if not profile_id:
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
                        filtered_tests = [test for test in tests if test.get('profileId') == profile_id]
                        all_tests.extend(filtered_tests)
                        
                        last_modified = tests[-1].get('modifiedDateUtc')
                        if last_modified:
                            from datetime import datetime, timedelta
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
                break
        
        if all_tests:
            df = pd.DataFrame(all_tests)
            if 'modifiedDateUtc' in df.columns:
                df['modifiedDateUtc'] = pd.to_datetime(df['modifiedDateUtc'], utc=True)
                df['date'] = df['modifiedDateUtc'].dt.date
                df['time'] = df['modifiedDateUtc'].dt.time
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching tests: {str(e)}")
        return pd.DataFrame()


@st.cache_data(ttl=600)
def fetch_test_trials_for_player(team_id, test_ids):
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
    
    return pd.DataFrame(all_trials) if all_trials else pd.DataFrame()


def extract_player_performance_metrics(trials_df, test_data):
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


def find_player_vald_profile_id(player_name, all_profiles):
    """Find VALD profile ID for a specific player by name matching"""
    for profile_id, profile_data in all_profiles.items():
        if profile_data['fullName'] == player_name:
            return profile_id
    return None


def get_daily_values_for_metric(exercise_data, metric, exercise_code):
    """Get daily values for a metric - all repeats for HJ, max for others"""
    metric_data = exercise_data[
        (exercise_data['metric_name'] == metric) &
        (exercise_data['limb'] == 'Trial')
    ].copy()
    
    if metric_data.empty:
        return pd.DataFrame()
    
    if exercise_code == 'HJ':
        daily_values = []
        for d, group in metric_data.groupby('test_date'):
            for _, row in group.iterrows():
                daily_values.append({'test_date': d, 'value': row['value'], 'repeat': row['repeat']})
        return pd.DataFrame(daily_values) if daily_values else pd.DataFrame()
    else:
        daily_max = metric_data.groupby('test_date')['value'].max().reset_index()
        return daily_max.sort_values('test_date')


def create_cmj_quadrant_analysis(exercise_data, player_name, comparison_exercise_data=None):
    """Create a 2x2 quadrant analysis for CMJ data using E:C Ratio methodology"""
    cmj_data = exercise_data[
        (exercise_data['testType'] == 'CMJ') &
        (exercise_data['limb'] == 'Trial')
    ].copy()
    
    if cmj_data.empty:
        return None
    
    cmj_data['test_date'] = pd.to_datetime(cmj_data['recordedUTC']).dt.date
    
    comparison_cmj_data = None
    if comparison_exercise_data is not None and not comparison_exercise_data.empty:
        comparison_cmj_data = comparison_exercise_data[
            (comparison_exercise_data['testType'] == 'CMJ') &
            (comparison_exercise_data['limb'] == 'Trial')
        ].copy()
        if not comparison_cmj_data.empty:
            comparison_cmj_data['test_date'] = pd.to_datetime(comparison_cmj_data['recordedUTC']).dt.date
    
    required_metrics = {
        'concentric_impulse': ['Concentric Impulse'],
        'eccentric_deceleration': ['Eccentric Deceleration Impulse'],
        'eccentric_braking': ['Eccentric Braking Impulse']
    }
    
    def extract_metrics_from_data(data_df):
        metrics_data = {}
        for category, metric_names in required_metrics.items():
            for metric_name in metric_names:
                matching_metrics = data_df[data_df['metric_name'] == metric_name]
                if not matching_metrics.empty:
                    daily_values = matching_metrics.groupby('test_date')['value'].max().reset_index()
                    if len(daily_values) > 0:
                        metrics_data[category] = {
                            'values': daily_values['value'].values,
                            'dates': daily_values['test_date'].values,
                            'latest': daily_values['value'].iloc[-1],
                            'mean': daily_values['value'].mean(),
                            'metric_name': matching_metrics.iloc[0]['metric_name'],
                            'units': matching_metrics.iloc[0].get('units', '')
                        }
                        break
        return metrics_data
    
    metrics_data = extract_metrics_from_data(cmj_data)
    
    comparison_metrics_data = None
    if comparison_cmj_data is not None and not comparison_cmj_data.empty:
        comparison_metrics_data = extract_metrics_from_data(comparison_cmj_data)
    
    if 'concentric_impulse' not in metrics_data:
        st.warning("CMJ quadrant analysis requires Concentric Impulse metric")
        return None
    if 'eccentric_deceleration' not in metrics_data or 'eccentric_braking' not in metrics_data:
        st.warning("CMJ quadrant analysis requires both Eccentric Deceleration and Eccentric Braking Impulse metrics")
        return None
    
    concentric_data = metrics_data['concentric_impulse']
    eccentric_decel_data = metrics_data['eccentric_deceleration']
    eccentric_brake_data = metrics_data['eccentric_braking']
    
    conc_df = pd.DataFrame({'date': concentric_data['dates'], 'concentric': concentric_data['values']})
    ecc_decel_df = pd.DataFrame({'date': eccentric_decel_data['dates'], 'eccentric_decel': eccentric_decel_data['values']})
    ecc_brake_df = pd.DataFrame({'date': eccentric_brake_data['dates'], 'eccentric_brake': eccentric_brake_data['values']})
    
    combined_df = pd.merge(conc_df, ecc_decel_df, on='date', how='inner')
    combined_df = pd.merge(combined_df, ecc_brake_df, on='date', how='inner')
    
    if len(combined_df) == 0:
        st.warning("No matching dates between Concentric, Eccentric Deceleration, and Eccentric Braking Impulse data")
        return None
    
    combined_df['eccentric_total'] = combined_df['eccentric_decel'] + combined_df['eccentric_brake']
    combined_df['ec_ratio'] = combined_df['eccentric_total'] / combined_df['concentric']
    combined_df['session'] = 'Feb 2026'
    
    comparison_combined_df = None
    if comparison_metrics_data is not None:
        if all(key in comparison_metrics_data for key in ['concentric_impulse', 'eccentric_deceleration', 'eccentric_braking']):
            comp_conc_df = pd.DataFrame({'date': comparison_metrics_data['concentric_impulse']['dates'], 'concentric': comparison_metrics_data['concentric_impulse']['values']})
            comp_ecc_decel_df = pd.DataFrame({'date': comparison_metrics_data['eccentric_deceleration']['dates'], 'eccentric_decel': comparison_metrics_data['eccentric_deceleration']['values']})
            comp_ecc_brake_df = pd.DataFrame({'date': comparison_metrics_data['eccentric_braking']['dates'], 'eccentric_brake': comparison_metrics_data['eccentric_braking']['values']})
            comparison_combined_df = pd.merge(comp_conc_df, comp_ecc_decel_df, on='date', how='inner')
            comparison_combined_df = pd.merge(comparison_combined_df, comp_ecc_brake_df, on='date', how='inner')
            if len(comparison_combined_df) > 0:
                comparison_combined_df['eccentric_total'] = comparison_combined_df['eccentric_decel'] + comparison_combined_df['eccentric_brake']
                comparison_combined_df['ec_ratio'] = comparison_combined_df['eccentric_total'] / comparison_combined_df['concentric']
                comparison_combined_df['session'] = 'Dec 2025'
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    all_x = list(combined_df['concentric'].values)
    all_y = list(combined_df['eccentric_total'].values)
    if comparison_combined_df is not None and len(comparison_combined_df) > 0:
        all_x.extend(list(comparison_combined_df['concentric'].values))
        all_y.extend(list(comparison_combined_df['eccentric_total'].values))
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    x_center = (all_x.min() + all_x.max()) / 2
    y_center = (all_y.min() + all_y.max()) / 2
    
    ax.axhline(y=y_center, color='white', linestyle='-', alpha=0.8, linewidth=2)
    ax.axvline(x=x_center, color='white', linestyle='-', alpha=0.8, linewidth=2)
    
    if comparison_combined_df is not None and len(comparison_combined_df) > 0:
        for _, row in comparison_combined_df.iterrows():
            ec_ratio = row['ec_ratio']
            color = 'orange' if ec_ratio < 0.8 else ('yellow' if ec_ratio > 1.2 else 'lightgreen')
            ax.scatter(row['concentric'], row['eccentric_total'], c=color, s=150, edgecolors='gray', linewidth=3, alpha=0.6, zorder=4)
            ax.annotate(f'Dec\nE:C: {ec_ratio:.2f}', (row['concentric'], row['eccentric_total']),
                        xytext=(-50, -30), textcoords='offset points', color='gray', fontsize=9, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    for _, row in combined_df.iterrows():
        ec_ratio = row['ec_ratio']
        color = 'orange' if ec_ratio < 0.8 else ('yellow' if ec_ratio > 1.2 else 'lightgreen')
        ax.scatter(row['concentric'], row['eccentric_total'], c=color, s=200, edgecolors='white', linewidth=3, alpha=0.9, zorder=5)
        ax.annotate(f'Feb\nE:C: {ec_ratio:.2f}', (row['concentric'], row['eccentric_total']),
                    xytext=(15, 15), textcoords='offset points', color='white', fontsize=10, fontweight='bold')
    
    x_range = all_x.max() - all_x.min()
    y_range = all_y.max() - all_y.min()
    x_pad = max(x_range * 0.3, 30)
    y_pad = max(y_range * 0.3, 30)
    ax.set_xlim(all_x.min() - x_pad, all_x.max() + x_pad)
    ax.set_ylim(all_y.min() - y_pad, all_y.max() + y_pad)
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_left = xlim[0] + (x_center - xlim[0]) * 0.5
    x_right = x_center + (xlim[1] - x_center) * 0.5
    y_bottom = ylim[0] + (y_center - ylim[0]) * 0.5
    y_top = y_center + (ylim[1] - y_center) * 0.5
    
    label_style = dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7, edgecolor='white')
    ax.text(x_left, y_top, 'ECCENTRIC\nDEFICIENT', ha='center', va='center', color='yellow', fontsize=12, fontweight='bold', bbox=label_style)
    ax.text(x_right, y_top, 'MAINTENANCE', ha='center', va='center', color='lightgreen', fontsize=12, fontweight='bold', bbox=label_style)
    ax.text(x_left, y_bottom, 'LOW OVERALL\nCAPABILITY', ha='center', va='center', color='lightcoral', fontsize=12, fontweight='bold', bbox=label_style)
    ax.text(x_right, y_bottom, 'CONCENTRIC\nDEFICIENT', ha='center', va='center', color='orange', fontsize=12, fontweight='bold', bbox=label_style)
    
    ax.set_xlabel(f"Concentric Impulse ({concentric_data['units']})", color='white', fontsize=14, fontweight='bold')
    ax.set_ylabel(f"Combined Eccentric Impulse ({eccentric_decel_data['units']})", color='white', fontsize=14, fontweight='bold')
    ax.set_title(f'{player_name} - CMJ E:C Ratio Analysis', color='white', fontsize=16, fontweight='bold', pad=20)
    ax.tick_params(axis='both', colors='white', labelsize=12)
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.grid(True, alpha=0.3, color='white', linestyle=':', linewidth=0.5)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10,
               markeredgecolor='gray', markeredgewidth=2, label='December 2025', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white', markersize=10,
               markeredgecolor='white', markeredgewidth=2, label='February 2026', linestyle='None'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', framealpha=0.9)
    
    return fig


def display_selected_exercise_analysis(perf_df, player_name, exercise_code, exercise_display_name, comparison_perf_df=None):
    """Display analysis for the selected exercise with session comparison support"""
    if perf_df.empty:
        st.warning(f"No force plate data available for {player_name}")
        return
    
    perf_df['testType'] = perf_df['testType'].replace('SLJ', 'SJ')
    exercise_data = perf_df[perf_df['testType'] == exercise_code].copy()
    
    if exercise_data.empty:
        st.warning(f"No {exercise_display_name} data found for {player_name}")
        st.info(f"Available test types: {list(perf_df['testType'].unique())}")
        return
    
    st.subheader(f"{exercise_display_name} Analysis")
    
    key_metrics = {
        'CMJ': ['Jump Height (Flight Time)', 'Peak Power', 'Peak Force', 'RSI-modified'],
        'SJ': ['Jump Height (Flight Time)', 'Peak Power', 'Peak Force', 'Takeoff Peak Force'],
        'PPU': ['Peak Power', 'Peak Force', 'Flight Time'],
        'HJ': ['Jump Height (Flight Time)', 'Peak Force', 'Landing RFD', 'Time to Peak Force']
    }
    
    available_metrics = exercise_data['metric_name'].unique()
    target_metrics = key_metrics.get(exercise_code, [])
    
    matched_metrics = []
    for target in target_metrics:
        for available in available_metrics:
            if target.lower() in available.lower():
                matched_metrics.append(available)
                break
    
    # Get comparison exercise data
    comparison_exercise_data = None
    if comparison_perf_df is not None and not comparison_perf_df.empty:
        comparison_perf_df_copy = comparison_perf_df.copy()
        comparison_perf_df_copy['testType'] = comparison_perf_df_copy['testType'].replace('SLJ', 'SJ')
        comparison_exercise_data = comparison_perf_df_copy[comparison_perf_df_copy['testType'] == exercise_code].copy()
        if not comparison_exercise_data.empty:
            comparison_exercise_data['test_date'] = pd.to_datetime(comparison_exercise_data['recordedUTC']).dt.date
    
    if exercise_code == 'CMJ':
        quadrant_chart = create_cmj_quadrant_analysis(exercise_data, player_name, comparison_exercise_data)
        if quadrant_chart:
            st.pyplot(quadrant_chart)
            plt.close()
            st.markdown("### Quadrant Interpretation:")
            st.markdown("""
            - **Maintenance**: Good strength profile, focus on skill refinement
            - **Concentric Deficient**: Develop explosive power and rate of force development  
            - **Eccentric Deficient**: Build eccentric strength and force absorption capacity
            - **Foundationally Deficient**: Build fundamental strength in both phases
            """)
    
    if exercise_code == 'HJ':
        for available in available_metrics:
            if 'jump height' in available.lower() and available not in matched_metrics:
                matched_metrics.append(available)
            elif 'peak force' in available.lower() and available not in matched_metrics:
                matched_metrics.append(available)
            elif 'landing rfd' in available.lower() and available not in matched_metrics:
                matched_metrics.append(available)
        matched_metrics = [m for m in matched_metrics if 'best' not in m.lower() and 'mean' not in m.lower() and 'fatigue' not in m.lower()]
    
    if not matched_metrics:
        st.warning(f"No key metrics found for {exercise_display_name}")
        st.write(f"Available metrics: {list(available_metrics)[:5]}...")
        return
    
    exercise_data['test_date'] = pd.to_datetime(exercise_data['recordedUTC']).dt.date
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        all_test_days = set(exercise_data['test_date'].unique())
        if comparison_exercise_data is not None and not comparison_exercise_data.empty:
            all_test_days = all_test_days.union(set(comparison_exercise_data['test_date'].unique()))
        st.metric("Test Days", len(all_test_days))
    with col2:
        if exercise_code == 'HJ':
            total_repeats = len(exercise_data[exercise_data['limb'] == 'Trial']['repeat'].unique())
            st.metric("Total Repeats", total_repeats)
        else:
            all_trials = set(exercise_data['trialId'].unique())
            if comparison_exercise_data is not None and not comparison_exercise_data.empty:
                all_trials = all_trials.union(set(comparison_exercise_data['trialId'].unique()))
            st.metric("Total Tests", len(all_trials))
    with col3:
        total_reps = len(exercise_data)
        if comparison_exercise_data is not None and not comparison_exercise_data.empty:
            total_reps += len(comparison_exercise_data)
        st.metric("Total Reps", total_reps)
    with col4:
        latest_date = exercise_data['test_date'].max()
        st.metric("Latest Test", latest_date.strftime('%m/%d/%y') if pd.notna(latest_date) else "N/A")
    
    # Fatigue metrics for HJ
    if exercise_code == 'HJ':
        fatigue_metrics = [m for m in available_metrics if 'fatigue' in m.lower()
                           and 'hops/reps' not in m.lower() and 'peak power' not in m.lower()]
        
        if fatigue_metrics:
            st.subheader("Fatigue Analysis")
            if comparison_exercise_data is not None and not comparison_exercise_data.empty:
                st.caption("Showing change vs December 2025 baseline")
            
            fatigue_cols = st.columns(min(len(fatigue_metrics), 4))
            for i, metric in enumerate(fatigue_metrics):
                fatigue_data = exercise_data[(exercise_data['metric_name'] == metric) & (exercise_data['limb'] == 'Trial')].copy()
                if not fatigue_data.empty:
                    units = fatigue_data.iloc[0].get('units', '')
                    latest_value = fatigue_data.iloc[-1]['value']
                    delta = None
                    if comparison_exercise_data is not None and not comparison_exercise_data.empty:
                        comp_fatigue = comparison_exercise_data[(comparison_exercise_data['metric_name'] == metric) & (comparison_exercise_data['limb'] == 'Trial')]
                        if not comp_fatigue.empty:
                            baseline_value = comp_fatigue.iloc[-1]['value']
                            delta = latest_value - baseline_value
                    with fatigue_cols[i % 4]:
                        if delta is not None:
                            st.metric(metric.replace('Fatigue', '').strip(), f"{latest_value:.2f} {units}", delta=f"{delta:+.2f} vs Dec")
                        else:
                            st.metric(metric.replace('Fatigue', '').strip(), f"{latest_value:.2f} {units}", f"Avg: {fatigue_data['value'].mean():.2f}")
    
    # Current performance metrics (non-HJ)
    if exercise_code != 'HJ':
        st.subheader("Current Performance Metrics")
        cols = st.columns(min(len(matched_metrics), 4))
        for i, metric in enumerate(matched_metrics[:4]):
            metric_data = exercise_data[(exercise_data['metric_name'] == metric) & (exercise_data['limb'] == 'Trial')].copy()
            if not metric_data.empty:
                units = metric_data.iloc[0].get('units', '')
                best_value = metric_data['value'].max()
                delta = None
                if comparison_exercise_data is not None and not comparison_exercise_data.empty:
                    comp_metric = comparison_exercise_data[(comparison_exercise_data['metric_name'] == metric) & (comparison_exercise_data['limb'] == 'Trial')]
                    if not comp_metric.empty:
                        delta = best_value - comp_metric['value'].max()
                with cols[i % 4]:
                    if delta is not None:
                        st.metric(metric.replace('(Flight Time)', ''), f"{best_value:.2f} {units}", delta=f"{delta:+.2f} vs Dec")
                    else:
                        recent_value = metric_data.iloc[-1]['value']
                        st.metric(metric.replace('(Flight Time)', ''), f"{best_value:.2f} {units}", f"Recent: {recent_value:.2f}")
        
        # Comparison summary table
        if comparison_exercise_data is not None and not comparison_exercise_data.empty:
            st.markdown("---")
            st.subheader("Performance Comparison vs December Baseline")
            comparison_table_data = []
            for metric in matched_metrics:
                current_metric_data = exercise_data[(exercise_data['metric_name'] == metric) & (exercise_data['limb'] == 'Trial')]
                comp_metric_data = comparison_exercise_data[(comparison_exercise_data['metric_name'] == metric) & (comparison_exercise_data['limb'] == 'Trial')]
                if not current_metric_data.empty:
                    units = current_metric_data.iloc[0].get('units', '')
                    current_best = current_metric_data['value'].max()
                    if not comp_metric_data.empty:
                        baseline_best = comp_metric_data['value'].max()
                        change = current_best - baseline_best
                        pct_change = ((current_best - baseline_best) / baseline_best) * 100 if baseline_best != 0 else 0
                        comparison_table_data.append({
                            'Metric': metric.replace('(Flight Time)', '').strip(),
                            'Dec Best': f"{baseline_best:.2f} {units}",
                            'Feb Best': f"{current_best:.2f} {units}",
                            'Change': f"{change:+.2f}",
                            '% Change': f"{pct_change:+.1f}%"
                        })
                    else:
                        comparison_table_data.append({
                            'Metric': metric.replace('(Flight Time)', '').strip(),
                            'Dec Best': "N/A",
                            'Feb Best': f"{current_best:.2f} {units}",
                            'Change': "N/A",
                            '% Change': "N/A"
                        })
            if comparison_table_data:
                st.dataframe(pd.DataFrame(comparison_table_data), hide_index=True, use_container_width=True)
    
    # Progression charts
    st.subheader("Performance by Repeat Number" if exercise_code == 'HJ' else "Performance Progression (Daily Max Values)")
    
    comparison_daily_data_cache = {}
    if comparison_exercise_data is not None and not comparison_exercise_data.empty and exercise_code != 'HJ':
        for metric in matched_metrics:
            comparison_daily_data_cache[metric] = get_daily_values_for_metric(comparison_exercise_data, metric, exercise_code)
    
    for metric in matched_metrics:
        daily_data = get_daily_values_for_metric(exercise_data, metric, exercise_code)
        if daily_data.empty:
            continue
        
        metric_sample = exercise_data[(exercise_data['metric_name'] == metric) & (exercise_data['limb'] == 'Trial')]
        units = metric_sample.iloc[0].get('units', '') if not metric_sample.empty else ''
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        if exercise_code == 'HJ':
            metric_specific_data = exercise_data[(exercise_data['metric_name'] == metric) & (exercise_data['limb'] == 'Trial')].copy()
            comparison_metric_specific_data = None
            if comparison_exercise_data is not None and not comparison_exercise_data.empty:
                comparison_metric_specific_data = comparison_exercise_data[(comparison_exercise_data['metric_name'] == metric) & (comparison_exercise_data['limb'] == 'Trial')].copy()
            
            if not metric_specific_data.empty:
                repeats = metric_specific_data['repeat'].values
                values = metric_specific_data['value'].values
                sorted_indices = np.argsort(repeats)
                repeats = repeats[sorted_indices]
                values = values[sorted_indices]
                
                if comparison_metric_specific_data is not None and not comparison_metric_specific_data.empty:
                    comp_repeats = comparison_metric_specific_data['repeat'].values
                    comp_values = comparison_metric_specific_data['value'].values
                    comp_sorted = np.argsort(comp_repeats)
                    comp_repeats = comp_repeats[comp_sorted]
                    comp_values = comp_values[comp_sorted]
                    ax.scatter(comp_repeats, comp_values, s=100, color='#808080', edgecolors='white', linewidth=2, alpha=0.6, zorder=3, label='Dec 2025')
                    ax.plot(comp_repeats, comp_values, color='#808080', linewidth=2, alpha=0.5, linestyle='--')
                    comp_avg = np.mean(comp_values)
                    ax.axhline(y=comp_avg, color='#808080', linestyle=':', alpha=0.5, linewidth=2, label=f'Dec Avg: {comp_avg:.1f}')
                
                ax.scatter(repeats, values, s=150, color='#C41E3A', edgecolors='white', linewidth=3, alpha=0.9, zorder=4, label='Feb 2026')
                ax.plot(repeats, values, color='#C41E3A', linewidth=2, alpha=0.7)
                for repeat_num, value in zip(repeats, values):
                    ax.annotate(f'{value:.1f}', (repeat_num, value), textcoords="offset points", xytext=(0, 15), ha='center', color='white', fontsize=11, fontweight='bold')
                
                avg_value = np.mean(values)
                ax.axhline(y=avg_value, color='white', linestyle='--', alpha=0.7, linewidth=2, label=f'Feb Avg: {avg_value:.1f}')
                
                if comparison_metric_specific_data is not None and not comparison_metric_specific_data.empty:
                    change = avg_value - comp_avg
                    pct_change = ((avg_value - comp_avg) / comp_avg) * 100 if comp_avg != 0 else 0
                    change_color = '#00cc00' if change > 0 else '#ff4444' if change < 0 else 'white'
                    ax.text(0.5, 0.98, f"Avg Change: {change:+.1f} ({pct_change:+.1f}%)", transform=ax.transAxes,
                            ha='center', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e1e1e', alpha=0.9, edgecolor=change_color),
                            color=change_color, fontsize=12, fontweight='bold')
                
                ax.set_xlabel('Repeat Number', color='white', fontsize=14)
                title_suffix = "Session Comparison" if (comparison_metric_specific_data is not None and not comparison_metric_specific_data.empty) else "All Repeats"
                ax.set_title(f'{metric} - {title_suffix}', color='white', fontweight='bold', fontsize=16, pad=20)
                
                all_repeats = sorted(set(list(repeats)) | (set(comp_repeats) if comparison_metric_specific_data is not None and not comparison_metric_specific_data.empty else set()))
                ax.set_xticks(all_repeats)
                ax.set_xticklabels([f'R{int(r)}' for r in all_repeats])
        
        else:
            comparison_daily_data = comparison_daily_data_cache.get(metric)
            all_values = []
            all_labels = []
            all_colors = []
            
            if comparison_daily_data is not None and not comparison_daily_data.empty:
                all_values.append(comparison_daily_data['value'].max())
                all_labels.append('Dec 2025')
                all_colors.append('#808080')
            
            all_values.append(daily_data['value'].max())
            all_labels.append('Feb 2026')
            all_colors.append('#C41E3A')
            
            if len(all_values) >= 2:
                x_positions = np.arange(len(all_labels))
                bars = ax.bar(x_positions, all_values, color=all_colors, edgecolor='white', linewidth=2, width=0.6)
                for bar, value in zip(bars, all_values):
                    ax.annotate(f'{value:.2f}', (bar.get_x() + bar.get_width()/2, bar.get_height()),
                                textcoords="offset points", xytext=(0, 10), ha='center', color='white', fontsize=12, fontweight='bold')
                
                baseline_best = all_values[0]
                current_best = all_values[1]
                change = current_best - baseline_best
                pct_change = ((current_best - baseline_best) / baseline_best) * 100 if baseline_best != 0 else 0
                change_color = '#00cc00' if change > 0 else '#ff4444' if change < 0 else 'white'
                ax.text(0.5, 0.95, f"Change: {change:+.2f} ({pct_change:+.1f}%)", transform=ax.transAxes,
                        ha='center', va='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='#1e1e1e', alpha=0.9, edgecolor=change_color),
                        color=change_color, fontsize=14, fontweight='bold')
                ax.plot([0, 1], all_values, color='white', linestyle='--', linewidth=2, alpha=0.5, zorder=1)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(all_labels, fontsize=14)
                ax.set_title(f'{metric} - Session Comparison', color='white', fontweight='bold', fontsize=16, pad=20)
                y_min = min(all_values) * 0.9
                y_max = max(all_values) * 1.1
                ax.set_ylim(y_min, y_max)
            else:
                bars = ax.bar([0], all_values, color=all_colors, edgecolor='white', linewidth=2, width=0.4)
                ax.annotate(f'{all_values[0]:.2f}', (0, all_values[0]), textcoords="offset points", xytext=(0, 10), ha='center', color='white', fontsize=12, fontweight='bold')
                ax.set_xticks([0])
                ax.set_xticklabels(all_labels, fontsize=14)
                ax.set_title(f'{metric} - Current Performance', color='white', fontweight='bold', fontsize=16, pad=20)
                ax.set_ylim(all_values[0] * 0.9, all_values[0] * 1.1)
        
        ax.set_ylabel(f'{metric} ({units})', color='white', fontsize=14)
        ax.grid(True, alpha=0.3, color='white', linestyle='-', linewidth=0.5)
        ax.tick_params(axis='both', colors='white', labelsize=12)
        for spine in ax.spines.values():
            spine.set_color('white')
        
        if exercise_code == 'HJ':
            legend = ax.legend(loc='lower right', framealpha=0.9)
            legend.get_frame().set_facecolor('#1e1e1e')
            legend.get_frame().set_edgecolor('white')
            for text in legend.get_texts():
                text.set_color('white')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        if exercise_code != 'HJ' and comparison_daily_data_cache.get(metric) is not None and not comparison_daily_data_cache[metric].empty:
            baseline_best = comparison_daily_data_cache[metric]['value'].max()
            current_best = daily_data['value'].max()
            improvement = ((current_best - baseline_best) / baseline_best) * 100 if baseline_best != 0 else 0
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("December Best", f"{baseline_best:.2f} {units}")
            with col2:
                st.metric("February Best", f"{current_best:.2f} {units}")
            with col3:
                st.metric("Improvement", f"{improvement:+.1f}%")
        
        st.markdown("---")
    
    # Recent test history table
    st.subheader("Recent Test History")
    recent_trial_data = exercise_data[exercise_data['limb'] == 'Trial'].copy()
    
    if not recent_trial_data.empty:
        recent_trial_data = recent_trial_data.sort_values('recordedUTC', ascending=False)
        
        if exercise_code == 'HJ':
            unique_trials = recent_trial_data['trialId'].unique()[:10]
            table_data = []
            for trial_id in unique_trials:
                trial_metrics = recent_trial_data[recent_trial_data['trialId'] == trial_id]
                if trial_metrics.empty:
                    continue
                first_row = trial_metrics.iloc[0]
                test_datetime = pd.to_datetime(first_row['recordedUTC'])
                row_data = {'Date': test_datetime.strftime('%m/%d/%y'), 'Trial ID': trial_id}
                for metric_name in trial_metrics['metric_name'].unique():
                    metric_data = trial_metrics[trial_metrics['metric_name'] == metric_name]
                    if not metric_data.empty:
                        values = metric_data['value'].values
                        units = metric_data.iloc[0].get('units', '')
                        clean_metric = metric_name.replace('(Flight Time)', '').strip()
                        row_data[f'{clean_metric} (Best)'] = f"{values.max():.2f} {units}".strip()
                        row_data[f'{clean_metric} (Mean)'] = f"{values.mean():.2f} {units}".strip()
                table_data.append(row_data)
        else:
            unique_trials = recent_trial_data['trialId'].unique()[:20]
            table_data = []
            for trial_id in unique_trials:
                trial_metrics = recent_trial_data[recent_trial_data['trialId'] == trial_id]
                if trial_metrics.empty:
                    continue
                first_row = trial_metrics.iloc[0]
                test_datetime = pd.to_datetime(first_row['recordedUTC'])
                row_data = {'Date': test_datetime.strftime('%m/%d/%y'), 'Trial ID': trial_id}
                for display_name, metric_name in {'Flight Time': 'Flight Time', 'Peak Power / BM': 'Peak Power / BM', 'Takeoff Peak Force': 'Takeoff Peak Force', 'RSI-modified': 'RSI-modified'}.items():
                    metric_data = trial_metrics[trial_metrics['metric_name'].str.contains(metric_name, case=False, na=False)]
                    if not metric_data.empty:
                        value = metric_data.iloc[0]['value']
                        units = metric_data.iloc[0].get('units', '')
                        row_data[display_name] = f"{value:.2f} {units}".strip()
                    else:
                        row_data[display_name] = "N/A"
                table_data.append(row_data)
        
        if table_data:
            st.dataframe(pd.DataFrame(table_data), hide_index=True, use_container_width=True)
        else:
            st.info("No trial data available")
    else:
        st.info("No trial data available for this exercise type")


def display_player_force_plate_section(player_name, current_session, comparison_session=None):
    """Display Force Plate section for the selected player"""
    st.markdown('<h3 class="section-header">Force Plate Performance Analysis</h3>', unsafe_allow_html=True)
    
    if 'vald_profiles' not in st.session_state:
        st.session_state.vald_profiles = {}
    if 'vald_profiles_loaded' not in st.session_state:
        st.session_state.vald_profiles_loaded = False
    
    if not st.session_state.vald_profiles_loaded:
        with st.spinner("Loading VALD profiles..."):
            st.session_state.vald_profiles = fetch_all_vald_profiles()
            st.session_state.vald_profiles_loaded = True
    
    player_profile_id = find_player_vald_profile_id(player_name, st.session_state.vald_profiles)
    
    if not player_profile_id:
        st.warning(f"No VALD profile found for {player_name}")
        with st.expander("Available VALD Profiles (for debugging)", expanded=False):
            profile_names = [profile['fullName'] for profile in st.session_state.vald_profiles.values()]
            st.write(f"Found {len(profile_names)} profiles:")
            for name in sorted(profile_names)[:20]:
                st.write(f"- {name}")
        return
    
    st.success(f"Found VALD profile for {player_name}")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if current_session == "February 2026":
            default_date = date(2026, 2, 15)
        else:
            default_date = date(2025, 12, 6)

        
        force_plate_date = st.date_input(
            "Load Force Plate Data From:",
            value=default_date,
            min_value=date(2025, 12, 6),
            max_value=date.today(),
            key=f"fp_date_{player_name}_{current_session}"
        )
        load_fp_data = st.button("Load Data", type="primary", key=f"load_fp_{player_name}_{current_session}")
    
    with col2:
        exercise_options = {
            "CMJ Performance": "CMJ",
            "Squat Jump Performance": "SJ",
            "Hop Jump Performance": "HJ",
            "Plyo Pushup Performance": "PPU"
        }
        selected_exercise = st.selectbox("Select Exercise Type:", options=list(exercise_options.keys()), key=f"exercise_select_{player_name}")
        exercise_code = exercise_options[selected_exercise]
    
    if load_fp_data:
        team_id = get_vald_team_id()
        with st.spinner("Loading force plate data..."):
            df = fetch_player_forcedecks_tests(player_profile_id, force_plate_date.strftime('%Y-%m-%d'))
            if not df.empty:
                st.success(f"Found {len(df)} tests for {player_name}")
                test_ids = df['testId'].unique().tolist()
                trials_df = fetch_test_trials_for_player(team_id, test_ids)
                if not trials_df.empty:
                    perf_df = extract_player_performance_metrics(trials_df, df)
                    if not perf_df.empty:
                        st.session_state[f'fp_data_{player_name}_{current_session}'] = perf_df
                        st.success(f"Loaded {len(perf_df)} performance measurements")
                        
                        if comparison_session is not None:
                            with st.spinner("Loading December baseline data for comparison..."):
                                dec_df = fetch_player_forcedecks_tests(player_profile_id, "2025-12-06")  # Dec 06, 2025
                                if not dec_df.empty:
                                    dec_test_ids = dec_df['testId'].unique().tolist()
                                    dec_trials_df = fetch_test_trials_for_player(team_id, dec_test_ids)
                                    if not dec_trials_df.empty:
                                        dec_perf_df = extract_player_performance_metrics(dec_trials_df, dec_df)
                                        if not dec_perf_df.empty:
                                            st.session_state[f'fp_data_{player_name}_comparison'] = dec_perf_df
                                            st.success("Loaded December baseline data for comparison")
                    else:
                        st.error("No performance metrics extracted from trial data")
                else:
                    st.error("No trial data found")
            else:
                st.warning(f"No test data found for {player_name} from {force_plate_date}")
    
    if f'fp_data_{player_name}_{current_session}' in st.session_state:
        perf_df = st.session_state[f'fp_data_{player_name}_{current_session}']
        comparison_perf_df = None
        if comparison_session is not None and f'fp_data_{player_name}_comparison' in st.session_state:
            comparison_perf_df = st.session_state[f'fp_data_{player_name}_comparison']
        display_selected_exercise_analysis(perf_df, player_name, exercise_code, selected_exercise, comparison_perf_df)


@st.cache_data(ttl=600)
def fetch_player_dynamo_tests(profile_id, modified_from_date):
    """Fetch Dynamo test data for a specific player"""
    if not profile_id:
        return pd.DataFrame()
    
    token = get_access_token()
    if not token:
        return pd.DataFrame()
    
    headers = {"Authorization": f"Bearer {token}"}
    from datetime import datetime
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
            params = {"modifiedFromUtc": modified_from, "testFromUtc": test_from, "testToUtc": test_to, "includeRepSummaries": "true", "page": page}
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 204:
                break
            if response.ok:
                data = response.json()
                items = data.get("items", [])
                total_pages = data.get("totalPages", 1)
                if items:
                    filtered_tests = [test for test in items if test.get('athleteId') == profile_id]
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
    except Exception:
        return pd.DataFrame()


def extract_player_dynamo_metrics(dynamo_df):
    """Extract key metrics from Dynamo test data for a single player"""
    if dynamo_df.empty:
        return pd.DataFrame()
    
    performance_data = []
    for _, test in dynamo_df.iterrows():
        test_type = f"{test.get('bodyRegion', '')} {test.get('movement', '')} - {test.get('position', '')}"
        for rep in test.get('repetitionTypeSummaries', []):
            record = {
                'testId': test.get('id'),
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
            }
            performance_data.append(record)
    
    return pd.DataFrame(performance_data) if performance_data else pd.DataFrame()


def display_player_rotational_analysis(player_name, profile_id):
    st.markdown('<h3 class="section-header">Rotational Ability & Arm Care Analysis</h3>', unsafe_allow_html=True)
    
    if not profile_id:
        st.warning(f"No VALD profile found for {player_name} - cannot load Dynamo data")
        return
    
    col1, _ = st.columns([1, 1])
    with col1:
        dynamo_date = st.date_input(
            "Load Dynamo Data From:",
            value=date(2026, 2, 16),
            min_value=date(2025, 12, 6),
            max_value=date.today(),
            key=f"dynamo_date_{player_name}"
        )
        load_dynamo = st.button("Load Rotational & Arm Care Data", type="primary", key=f"load_dynamo_{player_name}")
    
    if load_dynamo:
        with st.spinner("Loading Dynamo data..."):
            dynamo_df = fetch_player_dynamo_tests(profile_id, dynamo_date.strftime('%Y-%m-%d'))
            if not dynamo_df.empty:
                dynamo_perf_df = extract_player_dynamo_metrics(dynamo_df)
                if not dynamo_perf_df.empty:
                    st.session_state[f'dynamo_data_{player_name}'] = dynamo_perf_df
                    st.success(f"Loaded {len(dynamo_perf_df)} Dynamo measurements for {player_name}")
                else:
                    st.warning("No performance metrics extracted from Dynamo data")
            else:
                st.warning(f"No Dynamo test data found for {player_name} from {dynamo_date}")
    
    if f'dynamo_data_{player_name}' in st.session_state:
        display_player_dynamo_analysis(st.session_state[f'dynamo_data_{player_name}'], player_name)


def display_player_dynamo_analysis(dynamo_perf_df, player_name, team_dynamo_df=None):
    if dynamo_perf_df.empty:
        st.warning("No Dynamo data available")
        return
    tab1, tab2, tab3 = st.tabs(["Trunk Rotation", "Arm Care (ER/IR)", "All Tests Summary"])
    with tab1:
        display_trunk_rotation_analysis(dynamo_perf_df, player_name)
    with tab2:
        display_arm_care_analysis(dynamo_perf_df, player_name, team_dynamo_df)
    with tab3:
        display_all_dynamo_tests(dynamo_perf_df, player_name)


def display_trunk_rotation_analysis(dynamo_perf_df, player_name):
    trunk_data = dynamo_perf_df[
        (dynamo_perf_df['bodyRegion'] == 'Trunk') &
        (dynamo_perf_df['movement'].str.contains('Rotation', case=False, na=False))
    ].copy()
    
    if trunk_data.empty:
        st.info("No Trunk Rotation data found for this player")
        return
    
    st.subheader("Trunk Rotation Performance")
    left_data = trunk_data[trunk_data['laterality'] == 'LeftSide']
    right_data = trunk_data[trunk_data['laterality'] == 'RightSide']
    
    metrics_to_show = [
        ('maxForceNewtons', 'Peak Force', 'N'),
        ('avgForceNewtons', 'Avg Force', 'N'),
        ('maxImpulseNewtonSeconds', 'Peak Impulse', 'N·s'),
        ('maxRateOfForceDevelopmentNewtonsPerSecond', 'Peak RFD', 'N/s'),
    ]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tests", len(trunk_data['testId'].unique()))
    with col2:
        st.metric("Left Peak Force", f"{left_data['maxForceNewtons'].max():.1f} N" if not left_data.empty else "N/A")
    with col3:
        st.metric("Right Peak Force", f"{right_data['maxForceNewtons'].max():.1f} N" if not right_data.empty else "N/A")
    
    if not left_data.empty and not right_data.empty:
        left_max = left_data['maxForceNewtons'].max()
        right_max = right_data['maxForceNewtons'].max()
        if left_max > 0 and right_max > 0:
            asymmetry = abs(left_max - right_max) / max(left_max, right_max) * 100
            dominant_side = "Left" if left_max > right_max else "Right"
            col1, col2 = st.columns(2)
            with col1:
                if asymmetry > 15:
                    st.error(f"⚠️ Asymmetry: {asymmetry:.1f}% - Significant imbalance")
                elif asymmetry > 10:
                    st.warning(f"Asymmetry: {asymmetry:.1f}% - Moderate imbalance")
                else:
                    st.success(f"Asymmetry: {asymmetry:.1f}% - Within normal range")
            with col2:
                st.info(f"Dominant Side: {dominant_side}")
    
    if not left_data.empty or not right_data.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        metric_names = []
        left_values = []
        right_values = []
        
        for col, name, unit in metrics_to_show:
            if col in trunk_data.columns:
                metric_names.append(f"{name}\n({unit})")
                left_val = left_data[col].max() if not left_data.empty and col in left_data.columns else 0
                right_val = right_data[col].max() if not right_data.empty and col in right_data.columns else 0
                left_values.append(left_val if pd.notna(left_val) else 0)
                right_values.append(right_val if pd.notna(right_val) else 0)
        
        if metric_names:
            x = np.arange(len(metric_names))
            width = 0.35
            bars1 = ax.bar(x - width/2, left_values, width, label='Left', color='#4A90A4', alpha=0.8)
            bars2 = ax.bar(x + width/2, right_values, width, label='Right', color='#C41E3A', alpha=0.8)
            
            ax.set_ylabel('Value', color='white', fontsize=12)
            ax.set_title(f'{player_name} - Trunk Rotation: Left vs Right', color='white', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metric_names, color='white', fontsize=10)
            ax.tick_params(colors='white')
            ax.legend(facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.3, axis='y')
            for spine in ax.spines.values():
                spine.set_color('white')
            for bar in list(bars1) + list(bars2):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.0f}', ha='center', va='bottom', color='white', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    if not left_data.empty and not right_data.empty:
        left_max = left_data['maxForceNewtons'].max()
        right_max = right_data['maxForceNewtons'].max()
        asymmetry = abs(left_max - right_max) / max(left_max, right_max) * 100 if max(left_max, right_max) > 0 else 0
        st.markdown("---")
        st.markdown("### Training Recommendations")
        if asymmetry > 15:
            weaker_side = "Left" if left_max < right_max else "Right"
            st.markdown(f"""
            **Significant Rotational Asymmetry Detected ({asymmetry:.1f}%)**
            - Focus on **unilateral rotational exercises** to strengthen the {weaker_side} side
            - Implement **anti-rotation holds** (Pallof press variations)
            - Address potential **hip or thoracic mobility restrictions** on the weaker side
            - Consider **manual therapy** assessment for tissue quality imbalances
            - **Monitor closely** - asymmetry >15% may increase injury risk for pitchers
            """)
        elif asymmetry > 10:
            st.markdown(f"""
            **Moderate Rotational Asymmetry ({asymmetry:.1f}%)**
            - Include **bilateral rotational training** with slight emphasis on weaker side
            - Maintain **mobility work** for thoracic spine and hips
            - Continue monitoring asymmetry over time
            """)
        else:
            st.markdown(f"""
            **Good Rotational Balance ({asymmetry:.1f}%)**
            - Continue **balanced rotational training** program
            - Focus on **progressive overload** for continued development
            - Maintain current mobility and stability work
            """)


def display_arm_care_analysis(dynamo_perf_df, player_name, team_dynamo_df=None):
    shoulder_data = dynamo_perf_df[dynamo_perf_df['bodyRegion'] == 'Shoulder'].copy()
    
    if shoulder_data.empty:
        st.info("No shoulder data found for this player")
        return
    
    er_data = shoulder_data[shoulder_data['movement'].str.contains('ExternalRotation', case=False, na=False)]
    ir_data = shoulder_data[shoulder_data['movement'].str.contains('InternalRotation', case=False, na=False)]
    
    st.subheader("Arm Care Analysis (Shoulder ER/IR)")
    
    if er_data.empty and ir_data.empty:
        st.info("No External/Internal Rotation data found")
        available_movements = shoulder_data['movement'].unique()
        if len(available_movements) > 0:
            st.write("Available shoulder tests:", list(available_movements))
        return
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("ER Tests", len(er_data['testId'].unique()) if not er_data.empty else 0)
    with col2:
        st.metric("IR Tests", len(ir_data['testId'].unique()) if not ir_data.empty else 0)
    with col3:
        st.metric("ER Peak Force", f"{er_data['maxForceNewtons'].max():.1f} N" if not er_data.empty else "N/A")
    with col4:
        st.metric("IR Peak Force", f"{ir_data['maxForceNewtons'].max():.1f} N" if not ir_data.empty else "N/A")
    
    if not er_data.empty and not ir_data.empty:
        er_max = er_data['maxForceNewtons'].max()
        ir_max = ir_data['maxForceNewtons'].max()
        
        if ir_max > 0:
            er_ir_ratio = er_max / ir_max
            st.markdown("---")
            st.subheader("ER/IR Ratio Analysis")
            
            team_median_ratio = team_median_er = team_median_ir = None
            if team_dynamo_df is not None and not team_dynamo_df.empty:
                team_shoulder = team_dynamo_df[team_dynamo_df['bodyRegion'] == 'Shoulder'].copy()
                team_er = team_shoulder[team_shoulder['movement'].str.contains('ExternalRotation', case=False, na=False)]
                team_ir = team_shoulder[team_shoulder['movement'].str.contains('InternalRotation', case=False, na=False)]
                if not team_er.empty and not team_ir.empty:
                    player_er_best = team_er.groupby('athleteId')['maxForceNewtons'].max()
                    player_ir_best = team_ir.groupby('athleteId')['maxForceNewtons'].max()
                    common_players = player_er_best.index.intersection(player_ir_best.index)
                    if len(common_players) > 0:
                        player_ratios = player_er_best[common_players] / player_ir_best[common_players]
                        team_median_ratio = player_ratios.median()
                        team_median_er = player_er_best.median()
                        team_median_ir = player_ir_best.median()
            
            if team_median_ratio is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    vs_median = ((er_ir_ratio - team_median_ratio) / team_median_ratio) * 100
                    st.metric("ER/IR Ratio", f"{er_ir_ratio:.2f}", f"{vs_median:+.1f}% vs team median")
                with col2:
                    er_vs_median = ((er_max - team_median_er) / team_median_er) * 100
                    st.metric("ER vs Team Median", f"{er_max:.1f} N", f"{er_vs_median:+.1f}%")
                with col3:
                    ir_vs_median = ((ir_max - team_median_ir) / team_median_ir) * 100
                    st.metric("IR vs Team Median", f"{ir_max:.1f} N", f"{ir_vs_median:+.1f}%")
            else:
                st.metric("ER/IR Ratio", f"{er_ir_ratio:.2f}")
            
            st.markdown("""
            **Interpretation (Isometric Testing):**
            - Professional pitcher throwing arm average: ~0.75-0.83
            - Non-throwing arm average: ~0.99
            - Ratios <0.70 may indicate increased injury risk
            - Ratios near 1.0 suggest balanced strength
            """)
            
            fig = create_er_ir_comparison_chart(er_max, ir_max, er_ir_ratio, player_name, team_median_ratio, team_median_er, team_median_ir)
            st.pyplot(fig)
            plt.close()
    
    st.markdown("---")
    st.subheader("Left vs Right Shoulder Comparison")
    comparison_data = []
    for movement, data, label in [('ExternalRotation', er_data, 'External Rotation'), ('InternalRotation', ir_data, 'Internal Rotation')]:
        if not data.empty:
            left = data[data['laterality'] == 'LeftSide']['maxForceNewtons'].max()
            right = data[data['laterality'] == 'RightSide']['maxForceNewtons'].max()
            if pd.notna(left) and pd.notna(right) and left > 0 and right > 0:
                asymmetry = abs(left - right) / max(left, right) * 100
                comparison_data.append({'Movement': label, 'Left (N)': round(left, 1), 'Right (N)': round(right, 1), 'Asymmetry (%)': round(asymmetry, 1), 'Dominant': 'Left' if left > right else 'Right'})
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        for _, row in comparison_df.iterrows():
            if row['Asymmetry (%)'] > 15:
                st.error(f"⚠️ {row['Movement']}: {row['Asymmetry (%)']:.1f}% asymmetry - significant imbalance")
            elif row['Asymmetry (%)'] > 10:
                st.warning(f"⚠️ {row['Movement']}: {row['Asymmetry (%)']:.1f}% asymmetry - monitor closely")


def create_er_ir_comparison_chart(er_value, ir_value, ratio, player_name,
                                   team_median_ratio=None, team_median_er=None, team_median_ir=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1e1e1e')
    
    ax1.set_facecolor('#1e1e1e')
    x = np.arange(2)
    width = 0.35
    player_bars = ax1.bar(x - width/2, [er_value, ir_value], width, label=player_name, color=['#4A90A4', '#C41E3A'], alpha=0.9, edgecolor='white', linewidth=2)
    if team_median_er is not None and team_median_ir is not None:
        median_bars = ax1.bar(x + width/2, [team_median_er, team_median_ir], width, label='Team Median', color=['#4A90A4', '#C41E3A'], alpha=0.4, edgecolor='white', linewidth=2, hatch='//')
        for bar, val in zip(median_bars, [team_median_er, team_median_ir]):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f}', ha='center', va='bottom', color='gray', fontsize=10)
    for bar, val in zip(player_bars, [er_value, ir_value]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, f'{val:.1f} N', ha='center', va='bottom', color='white', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Peak Force (Newtons)', color='white', fontsize=12)
    ax1.set_title('Rotator Strength vs Team', color='white', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['External\nRotation', 'Internal\nRotation'], color='white', fontsize=11)
    ax1.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('white')
    ax1.set_ylim(0, max(er_value, ir_value, team_median_er or 0, team_median_ir or 0) * 1.25)
    if team_median_er is not None:
        ax1.legend(loc='upper right', facecolor='#1e1e1e', edgecolor='white', labelcolor='white')
    
    ax2.set_facecolor('#1e1e1e')
    ax2.barh(['This Player'], [ratio], height=0.3, color='#C41E3A', alpha=0.9, edgecolor='white', linewidth=2)
    ax2.axvline(x=0.70, color='#FF0000', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axvline(x=0.83, color='#00CED1', linestyle='--', linewidth=2, alpha=0.8)
    ax2.axvline(x=1.0, color='white', linestyle=':', linewidth=2, alpha=0.5)
    if team_median_ratio is not None:
        ax2.axvline(x=team_median_ratio, color='#32CD32', linestyle='-', linewidth=3, alpha=0.9)
        ax2.text(team_median_ratio, 0.55, f'Team\nMedian\n({team_median_ratio:.2f})', ha='center', va='bottom', color='#32CD32', fontsize=10, fontweight='bold')
    ax2.text(0.70, 0.55, 'Injury Risk\nThreshold\n(0.70)', ha='center', va='bottom', color='#FF0000', fontsize=9, fontweight='bold')
    ax2.text(0.83, 0.55, 'Pro Pitcher\nAvg\n(0.83)', ha='center', va='bottom', color='#00CED1', fontsize=9, fontweight='bold')
    ax2.text(1.0, 0.55, 'Balanced\n(1.0)', ha='center', va='bottom', color='white', fontsize=9, alpha=0.7)
    ax2.text(min(ratio + 0.04, 1.25), 0, f'{ratio:.2f}', ha='left', va='center', color='white', fontsize=18, fontweight='bold')
    ax2.plot(ratio, 0, marker='D', markersize=14, color='white', zorder=10)
    ax2.set_xlim(0.4, 1.35)
    ax2.set_ylim(-0.4, 0.9)
    ax2.set_xlabel('ER/IR Ratio', color='white', fontsize=12)
    ax2.set_title('Ratio vs Research & Team References', color='white', fontsize=14, fontweight='bold')
    ax2.tick_params(colors='white', labelleft=False)
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_color('white')
    
    plt.suptitle(f'{player_name} - Shoulder ER/IR Analysis', color='white', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def display_all_dynamo_tests(dynamo_perf_df, player_name):
    st.subheader(f"All Dynamo Tests for {player_name}")
    
    test_summary = dynamo_perf_df.groupby(['bodyRegion', 'movement', 'position']).agg({
        'testId': 'nunique', 'maxForceNewtons': 'max', 'avgForceNewtons': 'mean', 'test_date': ['min', 'max']
    }).reset_index()
    test_summary.columns = ['Body Region', 'Movement', 'Position', 'Test Count', 'Peak Force (N)', 'Avg Force (N)', 'First Test', 'Last Test']
    test_summary['Peak Force (N)'] = test_summary['Peak Force (N)'].round(1)
    test_summary['Avg Force (N)'] = test_summary['Avg Force (N)'].round(1)
    st.dataframe(test_summary, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("Test Distribution by Body Region")
    region_counts = dynamo_perf_df.groupby('bodyRegion')['testId'].nunique().reset_index()
    region_counts.columns = ['Body Region', 'Test Count']
    
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    bars = ax.bar(region_counts['Body Region'], region_counts['Test Count'], color='#C41E3A', alpha=0.8, edgecolor='white')
    ax.set_ylabel('Number of Tests', color='white')
    ax.set_title(f'{player_name} - Tests by Body Region', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom', color='white', fontweight='bold')
    for spine in ax.spines.values():
        spine.set_color('white')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def biomechanics_display(player_name, biomech_dir):
    """Display biomechanical chart for the selected player."""
    st.subheader("Kinematics Sequence & Key Metrics Chart Display:")
    formatted_name = player_name.replace(" ", "")
    
    if not os.path.exists(biomech_dir):
        st.error("Biomechanics Reports directory not found")
        return
    
    available_images = []
    for ext in ['jpg', 'jpeg', 'png', 'gif', 'bmp']:
        for file_path in glob.glob(os.path.join(biomech_dir, f"{formatted_name}*.{ext}")):
            filename = os.path.basename(file_path)
            name_without_ext = filename.rsplit('.', 1)[0]
            if len(name_without_ext) >= 6:
                date_part = name_without_ext[-6:]
                if date_part.isdigit():
                    display_date = f"{date_part[:2]}/{date_part[2:4]}/{date_part[4:]}"
                    available_images.append({'file_path': file_path, 'date_code': date_part, 'display_date': display_date})
    
    if available_images:
        available_images.sort(key=lambda x: x['date_code'], reverse=True)
        latest_image = available_images[0]
        st.image(latest_image['file_path'], caption=f"Biomechanical Analysis for {player_name} - {latest_image['display_date']}")
    else:
        st.warning(f"No biomechanical chart found for {player_name}")


def _find_assessment_player_row(df, first_name, last_name):
    """Find a player row in an assessment dataframe by name."""
    if 'First Name' not in df.columns or 'Last Name' not in df.columns:
        return None
    exact = df[
        (df['First Name'].str.strip().str.lower() == first_name.lower()) &
        (df['Last Name'].str.strip().str.lower() == last_name.lower())
    ]
    if not exact.empty:
        return exact.iloc[0]
    partial = df[
        (df['First Name'].str.contains(first_name, case=False, na=False)) &
        (df['Last Name'].str.contains(last_name, case=False, na=False))
    ]
    if not partial.empty:
        return partial.iloc[0]
    return None


def _assessment_fill_rate(player_row, columns):
    """Return (filled, total) numeric metric counts for a player row."""
    skip = {'First Name', 'Last Name', 'Name', 'Player', 'ID'}
    numeric_cols = [c for c in columns if c not in skip]
    filled = sum(1 for c in numeric_cols if pd.notna(player_row[c]) and str(player_row[c]).strip() != '')
    return filled, len(numeric_cols)


def display_player_assessment_data(player_name, assessment_file, comparison_file=None):
    """Display the player's assessment data with outlier highlighting.

    Loads assessment_file (S2) first. If the player has < 25% of metrics
    populated (e.g. due to injury), falls back to KatsBaseballTableAssessment1.xlsx.
    """
    st.markdown('<h3 class="section-header">Player Assessment Data</h3>', unsafe_allow_html=True)

    FALLBACK_FILE = "data/KatsBaseballTableAssessment.xlsx"

    if not os.path.exists(assessment_file):
        st.error(f"Assessment table file not found at {assessment_file}")
        return

    try:
        df = pd.read_excel(assessment_file, engine='openpyxl')
        df = df.dropna(how='all')

        name_parts = player_name.split()
        if len(name_parts) >= 2:
            first_name = name_parts[0]
            last_name = name_parts[-1]
        else:
            st.warning(f"Could not parse player name: {player_name}")
            return

        player_row = _find_assessment_player_row(df, first_name, last_name)
        used_fallback = False

        # Fall back to S1 if player has < 25% of metrics in S2 (e.g. injury)
        if player_row is not None:
            assessment_columns_tmp = df.columns[:22] if len(df.columns) >= 22 else df.columns
            filled, total = _assessment_fill_rate(player_row, assessment_columns_tmp)
            if total > 0 and (filled / total) < 0.25:
                player_row = None  # trigger fallback

        if player_row is None and os.path.exists(FALLBACK_FILE):
            fallback_df = pd.read_excel(FALLBACK_FILE, engine='openpyxl')
            fallback_df = fallback_df.dropna(how='all')
            fallback_row = _find_assessment_player_row(fallback_df, first_name, last_name)
            if fallback_row is not None:
                df = fallback_df
                player_row = fallback_row
                used_fallback = True

        if player_row is None:
            st.info(f"No assessment data found for {player_name}")
            return

        # Re-derive columns from whichever dataframe we settled on
        comparison_row = None  # retained for future use

        if player_row is not None:
            st.success(f"Found assessment data for {player_name}")
            if used_fallback:
                st.info("\u26a0\ufe0f Showing Session 1 (December) assessment data \u2014 insufficient Session 2 metrics on file (possible injury).")
            assessment_columns = df.columns[:22] if len(df.columns) >= 22 else df.columns
            stats_info = {}
            for column in assessment_columns:
                numeric_col = pd.to_numeric(df[column], errors='coerce')
                if numeric_col.notna().sum() >= 3:
                    median = numeric_col.median()
                    std = numeric_col.std()
                    if std > 0:
                        stats_info[column] = {'median': median, 'std': std, 'lower_bound': median - 2 * std, 'upper_bound': median + 2 * std}
            
            assessment_data = player_row[assessment_columns]
            
            st.markdown("""
            <div style='padding: 10px; background-color: #000000; border-radius: 5px; margin-bottom: 10px; color: white;'>
                <strong>Legend:</strong> 
                <span style='background-color: #cc0000; color: white; padding: 2px 8px; margin: 0 5px; border-radius: 3px; font-weight: bold;'>Mobility Deficiency (< Median - 2 SD)</span>
                <span style='background-color: #006400; color: white; padding: 2px 8px; margin: 0 5px; border-radius: 3px; font-weight: bold;'>Hyper Mobility (> Median + 2 SD)</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Key Assessment Metrics")
            skip_columns = ['First Name', 'Last Name', 'Name', 'Player', 'ID']
            cols = st.columns(4)
            col_idx = 0
            
            for column, value in assessment_data.items():
                if column not in skip_columns and pd.notna(value) and str(value).strip() != '':
                    is_outlier = False
                    outlier_type = None
                    if column in stats_info:
                        try:
                            numeric_value = float(value)
                            if numeric_value < stats_info[column]['lower_bound']:
                                is_outlier = True
                                outlier_type = 'low'
                            elif numeric_value > stats_info[column]['upper_bound']:
                                is_outlier = True
                                outlier_type = 'high'
                        except (ValueError, TypeError):
                            pass
                    
                    delta = None
                    if comparison_row is not None and column in comparison_row.index:
                        try:
                            current_val = float(value)
                            previous_val = float(comparison_row[column])
                            if pd.notna(previous_val):
                                delta = current_val - previous_val
                        except (ValueError, TypeError):
                            delta = None
                    
                    with cols[col_idx % 4]:
                        if is_outlier:
                            delta_text = f"<div style='font-size: 0.7em; margin-top: 3px;'>Change: {delta:+.1f}</div>" if delta is not None else ""
                            if outlier_type == 'low':
                                st.markdown(f"""
                                <div style='background-color: #cc0000; color: white; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;'>
                                    <div style='font-size: 0.8em; margin-bottom: 5px;'>{column}</div>
                                    <div style='font-size: 1.5em;'>{value}</div>
                                    <div style='font-size: 0.7em; margin-top: 5px;'>Mobility Deficiency</div>
                                    {delta_text}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style='background-color: #006400; color: white; padding: 10px; border-radius: 5px; font-weight: bold; text-align: center;'>
                                    <div style='font-size: 0.8em; margin-bottom: 5px;'>{column}</div>
                                    <div style='font-size: 1.5em;'>{value}</div>
                                    <div style='font-size: 0.7em; margin-top: 5px;'>Hyper Mobility</div>
                                    {delta_text}
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            if delta is not None:
                                st.metric(column, str(value), delta=f"{delta:+.1f}")
                            else:
                                st.metric(column, str(value))
                    
                    col_idx += 1
                    if col_idx % 4 == 0:
                        cols = st.columns(4)
            
            # Training Translation Guide
            outliers_exist = any(
                column in stats_info and pd.notna(value) and (float(value) < stats_info[column]['lower_bound'] or float(value) > stats_info[column]['upper_bound'])
                for column, value in assessment_data.items()
                if column in stats_info and pd.notna(value) and str(value).strip() != ''
                for _ in [None]
                if not (lambda: (_ for _ in ()) if True else (_ for _ in ()))()
            ) if False else False
            
            for column, value in assessment_data.items():
                if column in stats_info and pd.notna(value):
                    try:
                        numeric_value = float(value)
                        if numeric_value < stats_info[column]['lower_bound'] or numeric_value > stats_info[column]['upper_bound']:
                            outliers_exist = True
                            break
                    except (ValueError, TypeError):
                        pass
            
            if outliers_exist:
                st.markdown("---")
                st.markdown('<h3 class="section-header">Training Translation Guide</h3>', unsafe_allow_html=True)
                st.markdown("""
                **Mobility Deficiency (Low Values):**
                - Increase mobility drills and dynamic warm-ups
                - Address tissue quality (foam rolling, soft tissue work)
                - Focus on controlled articular rotations (CARs)
                - Gradually expand range of motion through progressive stretching
                
                **Hyper Mobility (High Values):**
                - Emphasize strength training throughout full range of motion
                - Focus on eccentric control and end-range strength
                - Implement tempo work and isometric holds
                - Develop motor control to utilize available mobility effectively
                """)
            
            # Comments section
            if len(df.columns) > 22:
                st.markdown("---")
                st.subheader("Player Comments & Notes")
                comment_columns = df.columns[22:]
                comment_data = player_row[comment_columns]
                has_comments = False
                for column, value in comment_data.items():
                    if pd.notna(value) and str(value).strip() != '':
                        has_comments = True
                        display_name = "Further Comments" if (pd.isna(column) or str(column).startswith('Unnamed:') or str(column).strip() == '') else str(column)
                        with st.expander(f"{display_name}", expanded=True):
                            st.write(str(value))
                if not has_comments:
                    st.info("No additional comments or notes available for this player.")
        else:
            st.warning(f"No assessment data found for {player_name}")
            st.info(f"Searched for: First Name = '{first_name}', Last Name = '{last_name}'")
    
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        st.info("Please ensure the file is a valid Excel file (.xlsx format)")


# Main application
def main():
    st.markdown('<h1 class="main-header">ECC Kats Baseball</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header"> Total Player Profile Display </p>', unsafe_allow_html=True)
    
    # Sidebar
    try:
        st.sidebar.image("images/liquid_logo.png", width=250)
        st.sidebar.image("images/logo.png", width=250)
    except FileNotFoundError:
        st.sidebar.warning("Logo not found at images/logo.png")
    
    st.sidebar.title("Data Selection")
    st.sidebar.markdown("---")
    
    # Session selector
    selected_session = st.sidebar.selectbox(
        "Select Data Collection Period",
        list(SESSION_CONFIG.keys()),
        index=1  # Default to February 2026
    )
    
    current_config = SESSION_CONFIG[selected_session]
    show_comparison = selected_session == "February 2026"
    comparison_config = SESSION_CONFIG["December 2025"] if show_comparison else None
    
    if show_comparison:
        st.sidebar.info("Showing changes vs December 2025 baseline")
    
    st.sidebar.markdown("---")
    st.sidebar.title("Player Selection")
    
    try:
        all_players = load_all_player_data(current_config["bullpen_dir"])
    except Exception as e:
        st.error(f"Failed to load player data: {str(e)}")
        st.stop()
    
    if not all_players:
        st.error(f"No player data found in {current_config['bullpen_dir']}. Please check your data directory.")
        st.stop()
    
    player_names = sorted(all_players.keys())
    selected_player = st.sidebar.selectbox("Select Player", player_names)
    
    if selected_player:
        player_data = all_players[selected_player]
        pitch_data = player_data['pitch_data']
        handedness = player_data['handedness']
        
        st.markdown(f'<h2 class="player-header">{selected_player} ({handedness}) - {selected_session}</h2>', unsafe_allow_html=True)
        
        pitch_stuff = calculate_player_stuff_plus(pitch_data)
        
        if not pitch_stuff:
            st.warning("No pitch types with sufficient data for Stuff+ calculation.")
            st.stop()
        
        # Load comparison pitch data if viewing February
        comparison_pitch_stuff = None
        if show_comparison:
            try:
                comparison_players = load_all_player_data(comparison_config["bullpen_dir"])
                if selected_player in comparison_players:
                    comparison_pitch_stuff = calculate_player_stuff_plus(comparison_players[selected_player]['pitch_data'])
            except Exception:
                comparison_pitch_stuff = None
        
        # Overall Stuff+ Summary
        st.markdown('<h3 class="section-header">Kats Stuff+ Overview</h3>', unsafe_allow_html=True)
        
        overall_stuff_plus = np.mean([data['stuff_plus'] for data in pitch_stuff.values()])
        total_pitches = sum([data['count'] for data in pitch_stuff.values()])
        pitch_types_count = len(pitch_stuff)
        comparison_overall = np.mean([data['stuff_plus'] for data in comparison_pitch_stuff.values()]) if comparison_pitch_stuff else None
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if comparison_overall is not None:
                delta = overall_stuff_plus - comparison_overall
                st.metric("Overall Stuff+", f"{overall_stuff_plus:.1f}", delta=f"{delta:+.1f}")
            else:
                st.metric("Overall Stuff+", f"{overall_stuff_plus:.1f}")
        with col2:
            st.metric("Pitch Types", f"{pitch_types_count}")
        with col3:
            st.metric("Total Pitches", f"{total_pitches:,}")
        with col4:
            best_pitch = max(pitch_stuff.keys(), key=lambda x: pitch_stuff[x]['stuff_plus'])
            st.metric("Best Pitch", f"{best_pitch}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<h3 class="section-header">Stuff+ Radar Chart</h3>', unsafe_allow_html=True)
            radar_chart = create_stuff_plus_radar_chart(pitch_stuff, selected_player)
            if radar_chart:
                st.pyplot(radar_chart)
        with col2:
            st.markdown('<h3 class="section-header">Stuff+ by Pitch Type</h3>', unsafe_allow_html=True)
            bar_chart = create_stuff_plus_bar_chart(pitch_stuff, selected_player)
            if bar_chart:
                st.pyplot(bar_chart)
        
        # Stuff+ Comparison Section (February view only)
        if show_comparison and comparison_pitch_stuff:
            st.markdown('<h3 class="section-header">Stuff+ Progress vs December Baseline</h3>', unsafe_allow_html=True)
            
            comparison_data = []
            all_pitch_types = set(list(pitch_stuff.keys()) + list(comparison_pitch_stuff.keys()))
            
            for pitch_type in all_pitch_types:
                current = pitch_stuff.get(pitch_type, {})
                baseline = comparison_pitch_stuff.get(pitch_type, {})
                current_stuff = current.get('stuff_plus', None)
                baseline_stuff = baseline.get('stuff_plus', None)
                current_velo = current.get('avg_velocity', None)
                baseline_velo = baseline.get('avg_velocity', None)
                stuff_change = (current_stuff - baseline_stuff) if (current_stuff is not None and baseline_stuff is not None) else None
                velo_change = (current_velo - baseline_velo) if (current_velo is not None and baseline_velo is not None) else None
                comparison_data.append({'Pitch Type': pitch_type, 'Dec Stuff+': baseline_stuff, 'Feb Stuff+': current_stuff, 'Stuff+ Change': stuff_change, 'Velo Change': velo_change})
            
            if comparison_data:
                comp_df = pd.DataFrame(comparison_data)
                
                def color_change(val):
                    if pd.isna(val) or val is None:
                        return ''
                    return 'color: #00cc00' if val > 0 else ('color: #ff4444' if val < 0 else '')
                
                styled_df = comp_df.style.applymap(color_change, subset=['Stuff+ Change', 'Velo Change']).format({
                    'Dec Stuff+': lambda v: f"{v:.1f}" if pd.notna(v) and v is not None else 'N/A',
                    'Feb Stuff+': lambda v: f"{v:.1f}" if pd.notna(v) and v is not None else 'N/A',
                    'Stuff+ Change': lambda v: f"{v:+.1f}" if pd.notna(v) and v is not None else 'N/A',
                    'Velo Change': lambda v: f"{v:+.1f} mph" if pd.notna(v) and v is not None else 'N/A',
                })
                st.dataframe(styled_df, hide_index=True, use_container_width=True)
                
                valid_stuff_changes = [row['Stuff+ Change'] for row in comparison_data if row['Stuff+ Change'] is not None]
                col1, col2, col3 = st.columns(3)
                with col1:
                    improvements = sum(1 for val in valid_stuff_changes if val > 0)
                    st.metric("Pitches Improved", f"{improvements}/{len(valid_stuff_changes)}")
                with col2:
                    if valid_stuff_changes:
                        st.metric("Avg Stuff+ Change", f"{np.mean(valid_stuff_changes):+.1f}")
                with col3:
                    if valid_stuff_changes:
                        best_idx = np.argmax(valid_stuff_changes)
                        best_pitch_type = [row['Pitch Type'] for row in comparison_data if row['Stuff+ Change'] is not None][best_idx]
                        st.metric("Most Improved", f"{best_pitch_type} ({max(valid_stuff_changes):+.1f})")
        
        # Detailed pitch analysis tabs
        st.markdown('<h3 class="section-header">Detailed Pitch Analysis</h3>', unsafe_allow_html=True)
        pitch_types = list(pitch_stuff.keys())
        if len(pitch_types) > 0:
            tabs = st.tabs(pitch_types)
            for i, pitch_type in enumerate(pitch_types):
                with tabs[i]:
                    display_pitch_stuff_details(pitch_stuff, pitch_type)
        
        # Movement analysis and development reports
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<h3 class="section-header">Movement Profile</h3>', unsafe_allow_html=True)
            movement_chart = create_movement_chart(pitch_data)
            if movement_chart:
                st.pyplot(movement_chart)
            else:
                st.info("Movement data not available")
        with col2:
            display_pitch_development_report_section(selected_player, current_config["dev_reports_dir"])
        
        # Stuff+ Summary Table
        st.markdown('<h3 class="section-header">Kats Stuff+ Summary</h3>', unsafe_allow_html=True)
        
        summary_data = []
        all_pitch_types_in_data = pitch_data['Pitch Type'].unique()
        valid_pitch_types = [pt for pt in all_pitch_types_in_data if pt not in ['-', '', None] and pd.notna(pt)]
        
        for pitch_type in valid_pitch_types:
            pitch_subset = pitch_data[pitch_data['Pitch Type'] == pitch_type]
            if len(pitch_subset) > 0:
                def remove_outliers_for_summary(data):
                    if len(data) >= 3:
                        Q1 = data.quantile(0.25)
                        Q3 = data.quantile(0.75)
                        IQR = Q3 - Q1
                        if IQR > 0:
                            lower_bound = Q1 - 2.0 * IQR
                            upper_bound = Q3 + 2.0 * IQR
                            filtered = data[(data >= lower_bound) & (data <= upper_bound)]
                            return filtered if len(filtered) >= 1 else data
                    return data
                
                hb_clean = remove_outliers_for_summary(pd.to_numeric(pitch_subset['HB (trajectory)'].dropna(), errors='coerce').dropna())
                vb_clean = remove_outliers_for_summary(pd.to_numeric(pitch_subset['VB (trajectory)'].dropna(), errors='coerce').dropna())
                velocity_clean = remove_outliers_for_summary(pd.to_numeric(pitch_subset['Velocity'].dropna(), errors='coerce').dropna())
                spin_clean = remove_outliers_for_summary(pd.to_numeric(pitch_subset['Total Spin'].dropna(), errors='coerce').dropna())
                
                speed_diff = 0
                if pitch_type != 'Fastball':
                    fastball_data = pitch_data[pitch_data['Pitch Type'].str.contains('Fastball', case=False, na=False)]
                    if len(fastball_data) > 0:
                        fb_velo_data = pd.to_numeric(fastball_data['Velocity'], errors='coerce').dropna()
                        fb_velo = fb_velo_data.mean() if len(fb_velo_data) > 0 else 0
                        pitch_velo = velocity_clean.mean() if len(velocity_clean) > 0 else 0
                        speed_diff = fb_velo - pitch_velo if fb_velo > 0 and pitch_velo > 0 else 0
                
                stuff_plus_val = pitch_stuff.get(pitch_type, {}).get('stuff_plus', None)
                if stuff_plus_val is None and len(velocity_clean) > 0:
                    fastball_data = pitch_data[pitch_data['Pitch Type'].str.contains('Fastball', case=False, na=False)]
                    player_fastball_velocity = None
                    if len(fastball_data) > 0:
                        fb_velo_data = pd.to_numeric(fastball_data['Velocity'], errors='coerce').dropna()
                        player_fastball_velocity = fb_velo_data.mean() if len(fb_velo_data) > 0 else None
                    try:
                        stuff_plus_val = calculate_Kats_stuff_plus_for_pitch_type(pitch_subset, pitch_type, player_fastball_velocity)
                    except:
                        stuff_plus_val = None
                
                summary_data.append({
                    'Pitch Type': pitch_type,
                    'Stuff+': stuff_plus_val if stuff_plus_val is not None else 'N/A',
                    'Count': len(pitch_subset),
                    'Avg Velocity': round(velocity_clean.mean(), 1) if len(velocity_clean) > 0 else 0,
                    'Avg Spin': round(spin_clean.mean(), 0) if len(spin_clean) > 0 else 0,
                    'H-Break': round(abs(hb_clean.mean()), 1) if len(hb_clean) > 0 else 0,
                    'V-Break': round(vb_clean.mean(), 1) if len(vb_clean) > 0 else 0,
                    'Speed Diff': round(speed_diff, 1)
                })
        
        summary_df = pd.DataFrame(summary_data)
        df_copy = summary_df.copy()
        df_copy['stuff_plus_numeric'] = pd.to_numeric(df_copy['Stuff+'], errors='coerce')
        summary_df = df_copy.sort_values('stuff_plus_numeric', ascending=False, na_position='last').drop('stuff_plus_numeric', axis=1)
        
        st.dataframe(summary_df, hide_index=True, use_container_width=True, column_config={
            "Stuff+": st.column_config.NumberColumn("Stuff+", format="%.1f"),
            "Count": st.column_config.NumberColumn("Count", format="%.0f"),
            "Avg Velocity": st.column_config.NumberColumn("Velocity", format="%.1f mph"),
            "Avg Spin": st.column_config.NumberColumn("Spin Rate", format="%.0f rpm"),
            "H-Break": st.column_config.NumberColumn("H-Break", format="%.1f in"),
            "V-Break": st.column_config.NumberColumn("V-Break", format="%.1f in"),
            "Speed Diff": st.column_config.NumberColumn("Speed Diff vs FB", format="%.1f mph")
        })
        
        # Recent pitch data
        st.markdown('<h3 class="section-header">Recent Pitch Data</h3>', unsafe_allow_html=True)
        display_cols = ['Date', 'Pitch Type', 'Velocity', 'Total Spin']
        for col in ['HB (trajectory)', 'VB (trajectory)', 'Release Height', 'Release Side', 'Is Strike']:
            if col in pitch_data.columns:
                display_cols.append(col)
        
        st.dataframe(pitch_data[display_cols].head(20), hide_index=True, use_container_width=True, column_config={
            "Velocity": st.column_config.NumberColumn("Velocity", format="%.1f mph"),
            "Total Spin": st.column_config.NumberColumn("Spin Rate", format="%.0f rpm"),
            "HB (trajectory)": st.column_config.NumberColumn("H-Break", format="%.1f in"),
            "VB (trajectory)": st.column_config.NumberColumn("V-Break", format="%.1f in"),
            "Release Height": st.column_config.NumberColumn("Release Height", format="%.2f ft"),
            "Release Side": st.column_config.NumberColumn("Release Side", format="%.2f ft"),
            "Is Strike": st.column_config.TextColumn("Strike")
        })
        
        # Force plate section
        if show_comparison:
            display_player_force_plate_section(selected_player, selected_session, comparison_session="December 2025")
        else:
            display_player_force_plate_section(selected_player, selected_session)
        
        # VALD profile for rotational analysis
        if 'vald_profiles' not in st.session_state or not st.session_state.vald_profiles:
            st.session_state.vald_profiles = fetch_all_vald_profiles()
        
        player_profile_id = find_player_vald_profile_id(selected_player, st.session_state.vald_profiles)
        display_player_rotational_analysis(selected_player, player_profile_id)
        
        # Biomechanics
        st.markdown('<h3 class="section-header">Biomechanical Analysis</h3>', unsafe_allow_html=True)
        biomechanics_display(selected_player, current_config["biomech_dir"])
        
        # Assessment data
        display_player_assessment_data(selected_player, current_config["assessment_file"])
    
    st.markdown("---")
    st.markdown("*ECC Kats Player Lookup | Made by Liquid Sports Lab*")


if __name__ == "__main__":
    main()