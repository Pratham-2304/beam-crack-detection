import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import tempfile
import base64

# -------------------------------------------------
# Fix Python path so `ml` is visible
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from ml.inference import predict_from_excel

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Beam Crack Detection System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -------------------------------------------------
# Custom CSS for Professional Styling
# -------------------------------------------------
st.markdown("""
<style>
    /* Main background gradient - more vibrant */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        animation: gradientShift 15s ease infinite;
        background-size: 200% 200%;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main container styling - enhanced */
    .main-container {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        padding: 3rem;
        border-radius: 25px;
        box-shadow: 0 25px 80px rgba(0,0,0,0.4);
        margin: 2rem auto;
        max-width: 1200px;
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Header styling - more dramatic */
    .header-container {
        text-align: center;
        margin-bottom: 3rem;
        padding-bottom: 2rem;
        border-bottom: 3px solid transparent;
        border-image: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        border-image-slice: 1;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
        letter-spacing: -1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #555;
        font-weight: 500;
        letter-spacing: 0.5px;
    }
    
    /* Icon styling - animated */
    .header-icon {
        font-size: 5rem;
        margin-bottom: 1rem;
        animation: float 3s ease-in-out infinite;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.2));
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Upload section - more appealing */
    .upload-section {
        background: linear-gradient(135deg, #e3f2fd 0%, #e1bee7 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin: 2rem 0;
        border: 3px dashed #667eea;
        position: relative;
        overflow: hidden;
    }
    
    .upload-section::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(102,126,234,0.1) 0%, transparent 70%);
        animation: rotate 20s linear infinite;
    }
    
    @keyframes rotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Feature boxes - enhanced */
    .feature-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }
    
    .feature-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        opacity: 0;
        transition: opacity 0.4s ease;
    }
    
    .feature-box:hover::before {
        opacity: 1;
    }
    
    .feature-box:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 10px 30px rgba(102,126,234,0.3);
        border-left-color: #764ba2;
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
        display: block;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    
    .feature-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 0.7rem;
        position: relative;
        z-index: 1;
    }
    
    .feature-desc {
        color: #666;
        font-size: 1rem;
        line-height: 1.6;
        position: relative;
        z-index: 1;
    }
    
    /* Results section */
    .results-container {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    /* Buttons - more vibrant */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.9rem 2.5rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.7);
        background: linear-gradient(135deg, #764ba2 0%, #f093fb 100%);
    }
    
    /* File uploader */
    .uploadedFile {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* Success/Error messages - enhanced */
    .stSuccess {
        background: linear-gradient(135deg, #00c853 0%, #64dd17 100%);
        color: white;
        border-radius: 15px;
        padding: 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 200, 83, 0.3);
    }
    
    .stError {
        background: linear-gradient(135deg, #ff1744 0%, #f50057 100%);
        color: white;
        border-radius: 15px;
        padding: 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(255, 23, 68, 0.3);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Info boxes - more colorful */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        box-shadow: 0 3px 10px rgba(33, 150, 243, 0.2);
        font-size: 1.05rem;
    }
    
    /* Stats cards - premium look */
    .stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.15);
        border: 2px solid transparent;
        border-image: linear-gradient(135deg, #667eea, #764ba2);
        border-image-slice: 1;
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.2);
    }
    
    .stat-number {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #666;
        font-size: 1.1rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    
    /* Section headers */
    h3 {
        color: #333;
        font-weight: 700;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1.5rem;
        padding-left: 1rem;
        border-left: 5px solid #667eea;
    }
    
    h4 {
        color: #444;
        font-weight: 600;
        font-size: 1.4rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None
if "uploaded_files_cache" not in st.session_state:
    st.session_state.uploaded_files_cache = None

# -------------------------------------------------
# Header Section
# -------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-icon">🏗️</div>
    <h1 class="main-title">Beam Crack Detection System</h1>
    <p class="subtitle">Advanced AI-Powered Structural Analysis</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------
# Reset Button
# -------------------------------------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔄 Reset / Start New Prediction", use_container_width=True):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# -------------------------------------------------
# Feature Highlights (if no results yet)
# -------------------------------------------------
if st.session_state.results is None:
    st.markdown("### 🎯 System Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">⚡</div>
            <div class="feature-title">Fast Analysis</div>
            <div class="feature-desc">Process multiple Excel files in seconds with our optimized algorithm</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Accurate Detection</div>
            <div class="feature-desc">Machine learning models trained on thousands of crack patterns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Detailed Reports</div>
            <div class="feature-desc">Comprehensive analysis with downloadable results</div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------
# Upload Section
# -------------------------------------------------
st.markdown("### 📤 Upload Your Data")

st.markdown("""
<div class="info-box">
    <strong>📋 Instructions:</strong> Upload one or multiple Excel files (.xlsx) containing beam damage data. 
    Our system will analyze each file and predict crack locations.
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Choose Excel files",
    type=["xlsx"],
    accept_multiple_files=True,
    help="Maximum file size: 200MB per file",
    key="file_uploader"
)

# -------------------------------------------------
# RUN PREDICTION (ONLY ON NEW UPLOAD)
# -------------------------------------------------
if uploaded_files and st.session_state.results is None:
    # Store current uploaded files to track changes
    current_files = [f.name for f in uploaded_files]
    
    results = []
    
    # Show progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("🔍 Analyzing beam structures..."):
        for idx, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                pred = predict_from_excel(tmp_path)
                pred["file"] = uploaded_file.name
                results.append(pred)
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

    progress_bar.empty()
    status_text.empty()
    
    if results:
        st.session_state.results = pd.DataFrame(results)
        st.session_state.uploaded_files_cache = current_files

# -------------------------------------------------
# DISPLAY RESULTS (ONLY FROM STATE)
# -------------------------------------------------
if st.session_state.results is not None:
    st.markdown("### 📊 Analysis Results")
    
    # Success message
    st.success("✅ Analysis completed successfully!")
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{len(st.session_state.results)}</div>
            <div class="stat-label">Files Analyzed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'prediction' in st.session_state.results.columns:
            crack_count = st.session_state.results['prediction'].notna().sum()
        else:
            crack_count = len(st.session_state.results)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{crack_count}</div>
            <div class="stat-label">Predictions Made</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">✓</div>
            <div class="stat-label">Status: Complete</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Results table
    st.markdown("#### 📈 Detailed Results")
    st.dataframe(
        st.session_state.results,
        use_container_width=True,
        height=400
    )
    
    # Download section
    st.markdown("#### 💾 Export Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        csv = st.session_state.results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Results as CSV",
            data=csv,
            file_name="beam_crack_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.button("📊 Generate Report", use_container_width=True):
            # Generate a detailed HTML report
            report_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Beam Crack Detection Report</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    }}
                    .container {{
                        background: white;
                        padding: 40px;
                        border-radius: 15px;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                    }}
                    h1 {{
                        color: #1a1a1a;
                        text-align: center;
                        border-bottom: 3px solid #667eea;
                        padding-bottom: 20px;
                        margin-bottom: 30px;
                    }}
                    h2 {{
                        color: #667eea;
                        margin-top: 30px;
                        border-left: 5px solid #667eea;
                        padding-left: 15px;
                    }}
                    .summary {{
                        background: linear-gradient(135deg, #e3f2fd 0%, #e1bee7 100%);
                        padding: 20px;
                        border-radius: 10px;
                        margin: 20px 0;
                    }}
                    .stat-box {{
                        display: inline-block;
                        background: white;
                        padding: 15px 30px;
                        margin: 10px;
                        border-radius: 10px;
                        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                        text-align: center;
                    }}
                    .stat-number {{
                        font-size: 2.5rem;
                        font-weight: bold;
                        color: #667eea;
                    }}
                    .stat-label {{
                        color: #666;
                        font-size: 0.9rem;
                    }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    th {{
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 15px;
                        text-align: left;
                        font-weight: 600;
                    }}
                    td {{
                        padding: 12px 15px;
                        border-bottom: 1px solid #ddd;
                    }}
                    tr:hover {{
                        background-color: #f5f5f5;
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 40px;
                        padding-top: 20px;
                        border-top: 2px solid #ddd;
                        color: #666;
                    }}
                    .timestamp {{
                        color: #999;
                        font-size: 0.9rem;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🏗️ Beam Crack Detection Analysis Report</h1>
                    
                    <div class="summary">
                        <h2>📊 Summary Statistics</h2>
                        <div style="text-align: center;">
                            <div class="stat-box">
                                <div class="stat-number">{len(st.session_state.results)}</div>
                                <div class="stat-label">Files Analyzed</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-number">{st.session_state.results.shape[1]}</div>
                                <div class="stat-label">Data Columns</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-number">✓</div>
                                <div class="stat-label">Analysis Complete</div>
                            </div>
                        </div>
                    </div>
                    
                    <h2>📋 Detailed Results</h2>
                    {st.session_state.results.to_html(index=False, classes='data-table')}
                    
                    <div class="footer">
                        <p><strong>Beam Crack Detection System v1.0</strong></p>
                        <p>⚡ Powered by Advanced Machine Learning • 🏗️ Built for Structural Engineers</p>
                        <p class="timestamp">Report generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Offer download of the HTML report
            st.download_button(
                label="⬇️ Download HTML Report",
                data=report_html.encode('utf-8'),
                file_name=f"beam_crack_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html",
                use_container_width=True
            )
            
            st.success("✅ Report generated! Click the button above to download.")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; background: rgba(255,255,255,0.95); padding: 2rem; border-radius: 15px; margin: 2rem 1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
    <p style="font-size: 1.2rem; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">
        Beam Crack Detection System v1.0
    </p>
    <p style="color: #555; font-size: 0.95rem;">
        ⚡ Powered by Advanced Machine Learning • 🏗️ Built for Structural Engineers
    </p>
</div>
""", unsafe_allow_html=True)