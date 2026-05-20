"""
Complete UI Styling CSS for Streamlit
Modern, professional, responsive design system
"""

STREAMLIT_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    /* ============================================================
       BASE STYLES
       ============================================================ */
    
    * {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecf1 100%);
        color: #1f2937;
    }
    
    /* ============================================================
       MAIN CONTAINER
       ============================================================ */
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecf1 100%);
        padding: 2.5rem 2rem;
    }
    
    /* ============================================================
       SIDEBAR
       ============================================================ */
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2937 0%, #0f172a 100%);
        color: #f3f4f6;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f3f4f6 !important;
        border-bottom: none !important;
    }
    
    /* ============================================================
       TYPOGRAPHY
       ============================================================ */
    
    h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        font-family: 'Poppins', sans-serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        position: relative;
        padding-bottom: 0.75rem;
    }
    
    h2::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 2px;
    }
    
    h3 {
        font-family: 'Poppins', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #667eea;
        margin: 1.5rem 0 0.75rem 0;
    }
    
    p {
        color: #6b7280;
        line-height: 1.6;
    }
    
    /* ============================================================
       CARDS & CONTAINERS
       ============================================================ */
    
    .stMetric {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.08),
                    0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stMetric:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15),
                    0 1px 3px rgba(0, 0, 0, 0.1);
        border-color: rgba(102, 126, 234, 0.2);
    }
    
    .stContainer {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    
    /* ============================================================
       FORMS & INPUTS
       ============================================================ */
    
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input,
    .stSelectbox select {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 2px solid #e5e7eb !important;
        border-radius: 8px !important;
        padding: 0.9rem 1.1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        color: #1f2937 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stNumberInput input:focus,
    .stSelectbox select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1) !important;
        background: white !important;
        outline: none !important;
    }
    
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
        font-style: italic;
    }
    
    /* ============================================================
       BUTTONS
       ============================================================ */
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.85rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        cursor: pointer;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.9);
        color: #667eea;
        border: 2px solid #667eea;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2);
    }
    
    /* ============================================================
       TABS
       ============================================================ */
    
    [data-baseweb="tab-list"] {
        border-bottom: 2px solid #e5e7eb;
        gap: 0.5rem;
        background: rgba(255, 255, 255, 0.4);
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    [data-baseweb="tab"] {
        color: #6b7280;
        padding: 0.9rem 1.5rem;
        font-weight: 500;
        border-radius: 6px;
        transition: all 0.3s ease;
        background: transparent;
        border-bottom: none !important;
        font-family: 'Inter', sans-serif;
    }
    
    [data-baseweb="tab"][aria-selected="true"] {
        color: #667eea;
        background: rgba(102, 126, 234, 0.1);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
    }
    
    [data-baseweb="tab"]:hover {
        color: #667eea;
    }
    
    /* ============================================================
       EXPANDERS
       ============================================================ */
    
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.03) 100%);
        border-radius: 8px;
        border: 1px solid rgba(102, 126, 234, 0.1);
        padding: 1rem;
        font-weight: 600;
        color: #1f2937;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.08) 100%);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateX(2px);
    }
    
    /* ============================================================
       ALERTS & MESSAGES
       ============================================================ */
    
    .stAlert {
        border-radius: 8px;
        padding: 1.25rem;
        border: 1px solid rgba(102, 126, 234, 0.2);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.02) 100%);
        backdrop-filter: blur(5px);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(245, 158, 11, 0.05) 100%) !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* ============================================================
       DIVIDERS
       ============================================================ */
    
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* ============================================================
       LOADING STATES
       ============================================================ */
    
    .stSpinner > div {
        border-color: rgba(102, 126, 234, 0.2);
        border-top-color: #667eea;
    }
    
    /* ============================================================
       SCROLLBAR
       ============================================================ */
    
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5568d3 0%, #6b3b8f 100%);
    }
    
    /* ============================================================
       RESPONSIVE DESIGN
       ============================================================ */
    
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem;
        }
        
        h2 {
            font-size: 1.4rem;
        }
        
        h3 {
            font-size: 1.05rem;
        }
        
        .main {
            padding: 1.5rem 1rem;
        }
        
        [data-baseweb="tab"] {
            padding: 0.7rem 1rem;
            font-size: 0.9rem;
        }
    }
    
    @media (max-width: 480px) {
        h1 {
            font-size: 1.5rem;
        }
        
        h2 {
            font-size: 1.2rem;
        }
        
        .stMetric {
            padding: 1rem;
        }
        
        .stButton > button {
            padding: 0.7rem 1.2rem;
            font-size: 0.85rem;
        }
    }
</style>
"""


def apply_streamlit_styling(st_obj):
    """
    Apply custom styling to Streamlit app
    
    Usage:
        import streamlit as st
        from ui_styling import apply_streamlit_styling
        
        apply_streamlit_styling(st)
    """
    st_obj.markdown(STREAMLIT_CSS, unsafe_allow_html=True)
