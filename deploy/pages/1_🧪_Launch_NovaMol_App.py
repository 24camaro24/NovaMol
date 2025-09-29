# app.py (Upgraded with PubChem count in summary)

import streamlit as st
import pandas as pd
import pipeline
from concurrent.futures import ProcessPoolExecutor
import os
import time
import json
import uuid





# --- Page Configuration ---
st.set_page_config(
    page_title="NovaMol AI Drug Discovery",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Session State Initialization ---
if 'job_future' not in st.session_state:
    st.session_state.job_future = None
if 'job_started' not in st.session_state:
    st.session_state.job_started = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'error' not in st.session_state:
    st.session_state.error = None
if 'progress_file' not in st.session_state:
    st.session_state.progress_file = ""

# --- Main App UI ---

st.title("💊 NovaMol: AI Drug Discovery Pipeline")
st.write("Select a biological target, and the pipeline will train predictive and generative models to discover novel, drug-like molecules.")

st.sidebar.header("Configuration")

TARGET_DATASETS = {
    "Lung Cancer (EGFR)": "CHEMBL203",
    "Colon Cancer (VEGFR2)": "CHEMBL240",
    "Breast Cancer (ERα)": "CHEMBL226",
    "Leukemia (BCR-ABL)": "CHEMBL4800",
    "Liver Cancer (c-Met)": "CHEMBL1978",
    "CNS / Alzheimer's (BACE1)": "CHEMBL2094",
    "Antibacterial (DHFR)": "CHEMBL202",
    "Antiviral (HIV-1 Protease)": "CHEMBL246",
}

selected_target_name = st.sidebar.selectbox(
    "Select a Target Dataset:",
    options=list(TARGET_DATASETS.keys())
)
target_id_input = TARGET_DATASETS[selected_target_name]

executor = ProcessPoolExecutor(max_workers=1)

if st.sidebar.button("🚀 Run Discovery Pipeline"):
    st.session_state.job_started = True
    st.session_state.results = None
    st.session_state.error = None
    
    st.session_state.progress_file = f"progress_{uuid.uuid4()}.log"
    
    output_dir = f'results_{target_id_input}'
    st.session_state.job_future = executor.submit(
        pipeline.run_complete_pipeline, 
        target_id=target_id_input,
        output_dir=output_dir,
        progress_file=st.session_state.progress_file
    )
    st.info(f"Pipeline started for {selected_target_name} ({target_id_input}). You can see detailed progress below.")

if st.session_state.job_started:
    future = st.session_state.job_future
    
    progress_bar = st.progress(0)
    progress_text = st.empty()

    while not future.done():
        if os.path.exists(st.session_state.progress_file):
            with open(st.session_state.progress_file, 'r') as f:
                try:
                    progress = json.load(f)
                    percent_complete = int((progress['current'] / progress['total']) * 100) if progress['total'] > 0 else 0
                    progress_bar.progress(percent_complete)
                    
                    if 'Epoch' in progress['step']:
                        progress_text.text(f"Status: {progress['step']} (Epoch {progress['current']}/{progress['total']})")
                    else:
                         progress_text.text(f"Status: {progress['step']}")
                
                except (json.JSONDecodeError, KeyError):
                    pass
        time.sleep(0.5)

    try:
        st.session_state.results = future.result()
        progress_bar.progress(100)
        progress_text.text("Pipeline Complete!")
    except Exception as e:
        st.session_state.error = e
        progress_text.error("An error occurred during the pipeline.")
    
    if os.path.exists(st.session_state.progress_file):
        os.remove(st.session_state.progress_file)

    if st.session_state.results:
        st.success(f"✅ Pipeline completed successfully for {selected_target_name}!")
        
        run_stats, performance_report, proof_df, csv_path = st.session_state.results

        # --- MODIFIED: Updated Run Summary Section ---
        st.header("1. Run Summary")
        
        truly_novel_count = run_stats.get('molecules_novel_vs_training_set', 0) - run_stats.get('molecules_found_in_pubchem', 0)

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("GNN Epochs", run_stats.get('gnn_epochs', 'N/A'))
        col2.metric("RNN Epochs", run_stats.get('rnn_epochs', 'N/A'))
        col3.metric("Novel (vs. Training Set)", run_stats.get('molecules_novel_vs_training_set', 'N/A'))
        col4.metric("Found in PubChem", run_stats.get('molecules_found_in_pubchem', 'N/A'))
        col5.metric("Truly Novel", truly_novel_count)


        st.header("2. Model Performance Report")
        st.subheader("Aggregate Performance (Mean Absolute Error)")
        
        cols = st.columns(len(performance_report))
        for i, (metric, value) in enumerate(performance_report.items()):
            cols[i].metric(label=metric, value=f"{value:.4f}")

        st.subheader("Sample Predictions on Test Set (Proof)")
        st.dataframe(proof_df)

        st.header("3. Generated Novel Drug Candidates")
        try:
            df_results = pd.read_csv(csv_path)
            st.dataframe(df_results)
            
            with open(csv_path, "rb") as f:
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=f,
                    file_name=os.path.basename(csv_path),
                    mime="text/csv",
                )
        except FileNotFoundError:
            st.error(f"Could not find the results file at {csv_path}.")
        except pd.errors.EmptyDataError:
             st.warning("The pipeline ran successfully, but no valid novel molecules were generated in this run.")
        
    elif st.session_state.error:
        st.error("An error occurred while running the pipeline:")
        st.exception(st.session_state.error)
