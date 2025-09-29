# Home.py

import streamlit as st

st.set_page_config(
    page_title="ProteinAI - Home",
    page_icon="💊",
    layout="wide",
)

# --- Hero Section ---
with st.container():
    st.subheader("Accelerating breakthroughs in biology with AI")
    st.title("ProteinAI")
    st.markdown("""
    Revolutionary AI system that predicts protein structures and interactions with unprecedented accuracy, 
    helping scientists understand how life's molecules work and accelerating discoveries that could solve 
    humanity's biggest challenges.
    """)
    st.info("Select **'🧪 Launch NovaMol App'** from the sidebar to get started!")


# --- Stats Section ---
with st.container():
    st.write("---")
    st.header("Our Impact at a Glance")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Structures Predicted", "200M+")
    with col2:
        st.metric("Researchers Engaged", "2M+")
    with col3:
        st.metric("Countries Reached", "190")
    with col4:
        st.metric("Prediction Accuracy", "99%")


# --- Quote Section ---
with st.container():
    st.write("---")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 2rem; border-radius: 1rem; margin: 2rem 0;">
        <h2 style='text-align: center; font-style: italic; font-weight: 300;'>
            "What took us months and years to do, ProteinAI was able to do in a weekend."
        </h2>
        <p style='text-align: center; margin-top: 1rem;'>
            – Dr. Sarah Johnson, Director of Structural Biology Institute
        </p>
    </div>
    """, unsafe_allow_html=True)


# --- Features Section ---
with st.container():
    st.write("---")
    st.header("How ProteinAI is Making a Difference")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🧬 Drug Discovery")
        st.write(
            """
            Accelerate the development of new medicines by understanding how potential drugs 
            interact with target proteins at the molecular level. Our NovaMol tool is a prime example of this capability.
            """
        )

    with col2:
        st.subheader("🦠 Disease Research")
        st.write(
            """
            Unlock new treatments for cancer, Alzheimer's, and other diseases by understanding 
            the protein structures involved in their mechanisms.
            """
        )
    
    with col3:
        st.subheader("🌱 Environmental Solutions")
        st.write(
            """
            Design novel enzymes that can break down plastic pollution and develop sustainable solutions 
            to environmental challenges using protein engineering.
            """
        )