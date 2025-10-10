# Home.py

import streamlit as st

st.set_page_config(
    page_title="Novamol- home",
    page_icon="💊",
    layout="wide",
)


# Purple Theme CSS - Add this to your existing pipeline UI
st.markdown("""
<style>
    /* Color Variables */
    :root {
        --primary-purple: #667eea;
        --secondary-purple: #764ba2;
        --dark-purple: #4a2c85;
        --light-purple: #9c88ff;
        --very-light-purple: #f8f6ff;
        --text-dark: #202124;
        --text-medium: #5f6368;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar arrow visibility fix */
    button[kind="header"] {
        background-color: var(--primary-purple) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
    }
    
    button[kind="header"]:hover {
        background-color: var(--secondary-purple) !important;
    }
    
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #f8f6ff 0%, #ffffff 50%, #f0efff 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, var(--primary-purple) 0%, var(--secondary-purple) 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white;
    }
    
    /* Headers - Purple */
    h1, h2, h3, h4, h5, h6 {
        color: var(--dark-purple) !important;
    }
    
    /* All text - Dark for visibility */
    p, div, span, li, label {
        color: var(--text-dark) !important;
    }
    
    /* Buttons - Purple gradient */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-purple), var(--secondary-purple)) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Input fields - Light purple background */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stMultiSelect > div > div {
        background: var(--very-light-purple) !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 8px !important;
        color: var(--text-dark) !important;
    }
    
    /* Progress bar - Purple gradient */
    .stProgress > div > div > div {
        background: var(--light-purple);
    }
    
    /* Success messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(255, 255, 255, 0.9) 100%) !important;
        border-left: 4px solid #28a745 !important;
        color: var(--text-dark) !important;
    }
    
    /* Info messages */
    .stInfo {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(255, 255, 255, 0.9) 100%) !important;
        border-left: 4px solid var(--primary-purple) !important;
        color: var(--text-dark) !important;
    }
    
    /* Warning messages */
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 255, 255, 0.9) 100%) !important;
        border-left: 4px solid #ffc107 !important;
        color: var(--text-dark) !important;
    }
    
    /* Error messages */
    .stError {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(255, 255, 255, 0.9) 100%) !important;
        border-left: 4px solid #dc3545 !important;
        color: var(--text-dark) !important;
    }
    
    /* Metrics - White cards with purple accent */
    [data-testid="metric-container"] {
        background: white !important;
        padding: 1.5rem !important;
        border-radius: 15px !important;
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.1) !important;
        border: 2px solid rgba(102, 126, 234, 0.1) !important;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.2) !important;
        border-color: var(--light-purple) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: var(--primary-purple) !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-dark) !important;
        font-weight: 500 !important;
    }
    
    /* Dataframe - Purple border */
    .stDataFrame {
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 10px !important;
    }
    
    /* Tabs - Purple styling */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--very-light-purple);
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: var(--primary-purple) !important;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary-purple), var(--secondary-purple));
        color: white !important;
        border-radius: 8px;
    }
    
    /* Expander - Purple header */
    .streamlit-expanderHeader {
        background: var(--very-light-purple) !important;
        color: var(--dark-purple) !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    /* File uploader - Purple border */
    .stFileUploader {
        background: var(--very-light-purple) !important;
        border: 2px dashed rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
    }
    
    /* Download button - Purple */
    .stDownloadButton > button {
        background: linear-gradient(135deg, var(--primary-purple), var(--secondary-purple)) !important;
        color: white !important;
        border-radius: 10px !important;
    }
    
    /* Slider - Purple track */
    .stSlider > div > div > div > div {
        background: var(--primary-purple) !important;
    }
    
    /* Checkbox/Radio - Purple when selected */
    .stCheckbox > label > div[data-checked="true"],
    .stRadio > label > div[data-checked="true"] {
        background-color: var(--primary-purple) !important;
    }
    
            
    /* Sidebar purple background */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar - make labels white */
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    /* Sidebar selectbox - WHITE BACKGROUND with BLACK TEXT */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: white !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        color: black !important;
        # background-color: var(--dark-purple) !important;
    }
    
    /* Dropdown menu options - WHITE BACKGROUND with BLACK TEXT */
    [data-testid="stSidebar"] [role="listbox"] {
        background-color: white !important;
    }
    
    [data-testid="stSidebar"] [role="option"] {
        background-color: white !important;
        color: black !important;
    }
    
    [data-testid="stSidebar"] [role="option"]:hover {
        background-color: #f0f0f0 !important;
        color: black !important;
    }
    
    /* Sidebar button - white with purple text */
    [data-testid="stSidebar"] .stButton button {
        background-color: white !important;
        color: var(--primary-purple) !important;
        border: none !important;
        font-weight: 600 !important;
    }
            
    .stSidebar p, div, span, li, label {
        color: white!important;
    }
    
    .stMainBlockContainer p, div, li, label {
        color: var(--text-dark)!important;
    }
            
    /* Progress bar - PURPLE */
    .stProgress > div > div > div > div {
        background-color: var(--primary-purple) !important;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Progress bar background - light purple */
    .stProgress > div > div {
        background-color: rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Main content - ensure text is visible */
    .main h1, .main h2, .main h3 {
        color: var(--dark-purple) !important;
    }
    
    .main p, .main div, .main span, .main li {
        color: #202124 !important;
    }
    
    /* Buttons purple gradient */
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
    }
    
    /* Download button purple */
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
    }
    
    /* Metrics purple accent */
    [data-testid="stMetricValue"] {
        color: var(--primary-purple) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
with st.container():
    st.subheader("Accelerating Drug Discovery with AI")
    st.title("NovaMol")
    st.markdown("""
    An innovative AI engine that generates novel, drug-like molecules and predicts their chemical properties. NovaMol uses a dual-architecture approach to explore new chemical space, helping researchers discover promising candidates for a wide range of diseases.
    """)
    st.info("Select **'🧪 Launch NovaMol App'** from the sidebar to get started!")


# --- Stats Section ---
with st.container():
    st.write("---")
    st.header("Our Impact at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Truly Novel Molecules Generated", "500+")
    with col2:
        st.metric("Molecules found in PubChem", "1400+")
    with col3:
        st.metric("Potency and Solubility Error", "~0.5")
    with col4:
        st.metric("Molecular Weight Error", "~33")


# --- Quote Section ---
with st.container():
    st.write("---")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 1rem; margin: 2rem 0;">
        <h2 style='text-align: center; font-style: italic; font-weight: 300;'>
           "NovaMol's ability to generate chemically valid, novel structures and provide instant property predictions is a game-changer. It allows us to explore new therapeutic hypotheses at a speed that was previously unimaginable."
        </h2>
    </div>
    """, unsafe_allow_html=True)


# --- Features Section ---
with st.container():
    st.write("---")
    st.header("The NovaMol Architecture")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🧬 Drug Discovery")
        st.write(
            """
            At its core, NovaMol uses a Recurrent Neural Network (RNN) trained on vast chemical libraries. It learns the fundamental 'language' of molecular structures to build new, chemically valid molecules from scratch.
            """
        )

    with col2:
        st.subheader(" 🤖 Predictive Analytics")
        st.write(
            """
            A powerful Graph Neural Network (GNN) analyzes the graph structure of each generated molecule. It predicts critical drug-like properties such as bioactivity (pChEMBL), solubility (logP), and molecular weight, providing instant feedback on a candidate's potential.
            """
        )
    
    with col3:
        st.subheader(" 🎯 Targeted Discovery")
        st.write(
            """
            By training on data for specific biological targets (like EGFR for lung cancer or ABL1 for leukemia), the entire pipeline can be focused, generating and evaluating molecules with a higher probability of being effective against a particular disease.
            """
        )