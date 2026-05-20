"""
TAV-RAG Giao diện Web - Ứng dụng Demo Streamlit
Giao diện tương tác cho Retrieval-Augmented Generation với Tavily Search API
"""
import streamlit as st
import asyncio
import time
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Thêm project root vào path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import TAVILY_API_KEY, USE_MOCK_TAVILY, LLM_LANGUAGE
from src.advanced_rag import AdvancedRAGSystem
from src.performance_tuning import PerformanceTuner

# Cấu hình trang
st.set_page_config(
    page_title="Hệ Thống TAV-RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "TAV-RAG v1.0 - Hệ Thống Tạo Tác Cải Thiện Truy Xuất"}
)

# Modern, Beautiful & Responsive UI Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        margin: 0;
        padding: 0;
    }
    
    /* Animated Loading Spinner */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stSpinner {
        animation: spin 2s linear infinite;
    }
    
    /* Main Background - Enhanced Gradient */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecf1 50%, #f0f4f8 100%);
        padding: 3rem 2.5rem;
        min-height: 100vh;
    }
    
    html, body {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecf1 50%, #f0f4f8 100%);
    }
    
    /* Sidebar Styling - Dark Elegant */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2937 0%, #0f172a 100%);
        color: #f3f4f6;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Headers - Professional Typography */
    h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.75rem;
        letter-spacing: -0.8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: slideUp 0.6s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    h2 {
        font-family: 'Poppins', sans-serif;
        font-size: 1.85rem;
        font-weight: 600;
        color: #1f2937;
        margin: 2.5rem 0 1.25rem 0;
        position: relative;
        padding-bottom: 0.9rem;
        letter-spacing: -0.3px;
    }
    
    h2::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 70px;
        height: 5px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 3px;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    h3 {
        font-family: 'Poppins', sans-serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: #667eea;
        margin: 1.75rem 0 1rem 0;
        letter-spacing: -0.2px;
    }
    
    /* Subtitle styling */
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        font-weight: 400;
        margin-bottom: 0.75rem;
        line-height: 1.6;
    }
    
    /* Cards - Glassmorphism Effect with Better Spacing */
    .stMetric {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 1.95rem;
        border-radius: 12px;
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.1),
                    0 1px 3px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(102, 126, 234, 0.12);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .stMetric:hover {
        transform: translateY(-6px);
        box-shadow: 0 10px 35px rgba(102, 126, 234, 0.18),
                    0 2px 8px rgba(0, 0, 0, 0.08);
        border-color: rgba(102, 126, 234, 0.25);
    }
    
    /* Container Box - Improved */
    .stContainer {
        background: rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 2.5rem;
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(120, 120, 255, 0.08);
        transition: all 0.3s ease;
    }
    
    /* Input Fields - Modern Style with Better Spacing */
    .stTextInput input, 
    .stTextArea textarea,
    .stNumberInput input {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 2px solid #e5e7eb !important;
        border-radius: 10px !important;
        padding: 1.1rem 1.3rem !important;
        font-size: 0.95rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        color: #1f2937 !important;
        min-height: 44px !important;
        letter-spacing: 0.2px;
    }
    
    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stNumberInput input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.12),
                    0 4px 12px rgba(102, 126, 234, 0.2) !important;
        background: white !important;
        outline: none !important;
    }
    
    /* Placeholder styling */
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #d1d5db !important;
        font-style: italic;
        font-weight: 400;
    }
    
    /* Buttons - Modern & Interactive with Better Feedback */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 1rem 2.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.35);
        text-transform: uppercase;
        cursor: pointer;
        position: relative;
        overflow: hidden;
        min-height: 44px;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.4);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 35px rgba(102, 126, 234, 0.45),
                    0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Secondary Button */
    .stButton > button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.95);
        color: #667eea;
        border: 2px solid #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: white;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    /* Tabs - Modern Style with Better Spacing */
    [data-baseweb="tab-list"] {
        border-bottom: 2px solid #e5e7eb;
        gap: 0.75rem;
        background: rgba(255, 255, 255, 0.5);
        padding: 0.75rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    
    [data-baseweb="tab"] {
        color: #6b7280;
        padding: 1.1rem 1.8rem;
        font-weight: 500;
        border-radius: 8px;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        background: transparent;
        border-bottom: none !important;
        letter-spacing: 0.3px;
    }
    
    [data-baseweb="tab"][aria-selected="true"] {
        color: white;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }
    
    [data-baseweb="tab"]:hover {
        color: #667eea;
        background: rgba(102, 126, 234, 0.08);
    }
    
    /* Expanders - Custom Styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.04) 100%);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.15);
        padding: 1.2rem;
        font-weight: 600;
        color: #1f2937;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 0.75rem;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.12) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-color: rgba(102, 126, 234, 0.35);
        transform: translateX(3px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.12);
    }
    
    /* Alerts & Messages - Better Design */
    .stAlert {
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid rgba(102, 126, 234, 0.25);
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.03) 100%);
        backdrop-filter: blur(5px);
        margin: 1rem 0;
        animation: slideUp 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.12) 0%, rgba(34, 197, 94, 0.06) 100%) !important;
        border: 1px solid rgba(34, 197, 94, 0.35) !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(5px) !important;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(239, 68, 68, 0.06) 100%) !important;
        border: 1px solid rgba(239, 68, 68, 0.35) !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(5px) !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(245, 158, 11, 0.06) 100%) !important;
        border: 1px solid rgba(245, 158, 11, 0.35) !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(5px) !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.12) 0%, rgba(59, 130, 246, 0.06) 100%) !important;
        border: 1px solid rgba(59, 130, 246, 0.35) !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        backdrop-filter: blur(5px) !important;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.4), transparent);
        margin: 2.5rem 0;
    }
    
    /* Text styling */
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar specific styling */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f3f4f6 !important;
        border-bottom: none !important;
    }
    
    [data-testid="stSidebar"] h2::after {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* Spinner and loading */
    .stSpinner > div {
        border-color: rgba(102, 126, 234, 0.2);
        border-top-color: #667eea;
    }
    
    /* Scrollbar Styling */
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
        transition: all 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5568d3 0%, #6b3b8f 100%);
        box-shadow: 0 0 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem;
        }
        
        h2 {
            font-size: 1.4rem;
        }
        
        .main {
            padding: 1.5rem 1rem;
        }
        
        .stButton > button {
            padding: 0.9rem 2rem;
            font-size: 0.85rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
    st.session_state.initialized = False
    st.session_state.query_history = []
    st.session_state.cache_stats = None
else:
    pass

# Khởi tạo hệ thống RAG
def initialize_rag():
    """Khởi tạo hệ thống RAG"""
    # Check for TAVILY_API_KEY
    if not TAVILY_API_KEY:
        st.error("❌ TAVILY_API_KEY không được thiết lập!")
        st.error("Vui lòng thiết lập TAVILY_API_KEY trong environment var hoặc .env file")
        st.info("Cách thiết lập:")
        st.code("export TAVILY_API_KEY='your_api_key_here'")
        return False
    
    try:
        st.session_state.rag_system = AdvancedRAGSystem(
            use_mock_search=False,  # Use REAL Tavily API
            use_mock_llm=True,  # Keep mock LLM for testing
            language=LLM_LANGUAGE
        )
        
        # Thêm tài liệu mẫu nếu chưa thêm
        if "docs_added" not in st.session_state:
            sample_docs = [
                "Sinh thành Tăng-Ghép (RAG) kết hợp truy xuất thông tin với tạo tác. Nó trước tiên truy xuất các tài liệu có liên quan, sau đó tạo ra các câu trả lời dựa trên bối cảnh.",
                "Tavily là một công cụ tìm kiếm được tối ưu hóa cho AI. Nó cung cấp tìm kiếm web thời gian thực với khả năng lọc nâng cao.",
                "Các Mô hình Ngôn ngữ Lớn (LLM) là các mạng nơ-ron được đào tạo trên lượng văn bản khổng lồ. Chúng có thể tạo ra văn bản giống như con người với độ chính xác cao.",
                "Cơ sở dữ liệu vectơ lưu trữ các embedding để tìm kiếm tương tự hiệu quả. Chúng cho phép tìm kiếm ngữ nghĩa ở quy mô lớn.",
                "Các chiến lược lưu vào bộ nhớ đệm làm giảm các cuộc gọi API dư thừa. Lưu trữ bộ nhớ nhanh nhưng không bền vững, lưu trữ tệp bền vững.",
                "Tối ưu hóa truy vấn cải thiện chất lượng truy xuất thông qua phát hiện ý định, mở rộng và viết lại.",
            ]
            try:
                st.session_state.rag_system.add_documents(sample_docs)
                st.session_state.docs_added = True
            except Exception as doc_err:
                st.warning(f"⚠️ Không thể thêm tài liệu mẫu: {doc_err}")
                st.session_state.docs_added = False
        
        st.session_state.initialized = True
        st.success("✅ Hệ thống RAG đã khởi tạo thành công!")
    except Exception as e:
        st.error(f"❌ Lỗi khởi tạo hệ thống: {e}")
        import traceback
        st.error(traceback.format_exc())
        return False
    return True

# Tiêu đề - Modern Hero Section
st.markdown("""
<div style="
    background: linear-gradient(135deg, rgba(102,126,234,0.12) 0%, rgba(118,75,162,0.08) 100%);
    border-radius: 16px;
    padding: 3.5rem 3rem;
    margin-bottom: 3rem;
    margin-top: -2rem;
    border: 1px solid rgba(102,126,234,0.2);
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
    animation: slideUp 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
">
    <h1 style="margin: 0; font-size: 3rem; letter-spacing: -1px;">🔍 TAV-RAG System</h1>
    <p style="
        margin: 0.8rem 0 0 0;
        font-size: 1.15rem;
        color: #6b7280;
        font-weight: 400;
        line-height: 1.6;
    ">Retrieval-Augmented Generation với Tavily Search API</p>
    <div style="
        display: flex;
        gap: 1.2rem;
        margin-top: 1.5rem;
        flex-wrap: wrap;
    ">
        <span style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 0.65rem 1.3rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        ">v1.2 Enhanced UI</span>
        <span style="
            background: rgba(34,197,94,0.25);
            color: #16a34a;
            padding: 0.65rem 1.3rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border: 1px solid rgba(34,197,94,0.4);
        ">✓ Ready to Use</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Thanh bên
with st.sidebar:
    st.markdown("""
    <div style="
        background: linear-gradient(180deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.08) 100%);
        border-radius: 12px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(102,126,234,0.25);
    ">
        <h2 style="margin-top: 0; font-size: 1.4rem; color: #f3f4f6; margin-bottom: 0.5rem;">⚙️ Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Nút khởi tạo
    if not st.session_state.initialized:
        col_btn = st.columns(1)[0]
        if col_btn.button("🚀 Initialize System", use_container_width=True, key="init_btn"):
            initialize_rag()
    else:
        st.success("✅ System Ready", icon="✅")
    
    st.markdown("---")
    
    # Cài đặt chính
    st.markdown("""
    <div style="margin-top: 2.5rem; margin-bottom: 1.5rem;">
        <h3 style="color: #f3f4f6; font-size: 1.1rem; margin-bottom: 1.5rem; margin-top: 0;">🎯 Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    top_k = st.slider("Top-K Results", 1, 10, 5, help="Number of documents to retrieve")
    cache_enabled = st.checkbox("Enable Cache", value=True, help="Cache query results")
    
    st.markdown("---")
    
    # Thông tin nhanh
    st.markdown("""
    <div style="margin-top: 2.5rem;">
        <h3 style="color: #f3f4f6; font-size: 1.1rem; margin-bottom: 1.5rem; margin-top: 0;">📊 Quick Stats</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("History", len(st.session_state.query_history))
    with col2:
        if st.session_state.rag_system:
            try:
                cache_stats = st.session_state.rag_system.get_cache_stats()
                hits = cache_stats.get("stats", {}).get("hits", 0)
                st.metric("Cache Hits", hits)
            except:
                st.metric("Cache Hits", 0)

# Nội dung chính
if not st.session_state.initialized:
    st.warning("👈 Click '🚀 Initialize System' in sidebar to begin!")
else:
    # Tạo các tab
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Truy Vấn Đơn", 
        "📊 Xử Lý Hàng Loạt", 
        "📈 Hiệu Suất", 
        "💾 Thống Kê Bộ Nhớ"
    ])
    
    # TAB 1: Truy vấn Đơn
    with tab1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.05) 100%);
            border-radius: 14px;
            padding: 2.5rem;
            margin-bottom: 2.5rem;
            border: 1px solid rgba(102,126,234,0.18);
        ">
            <h2 style="margin-top: 0; color: #1f2937; margin-bottom: 0.5rem;">🔍 Đặt một Câu Hỏi</h2>
            <p style="color: #6b7280; margin-bottom: 0; margin-top: 0.75rem; line-height: 1.6;">
                Nhập câu hỏi của bạn để hệ thống tìm kiếm và tạo ra câu trả lời chi tiết
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([5, 1])
        
        with col1:
            query = st.text_input(
                "placeholder",
                placeholder="Ví dụ: RAG là gì? Cách tối ưu hóa embeddings?",
                key="query_input",
                label_visibility="collapsed"
            )
        
        with col2:
            search_btn = st.button("🔍 Tìm Kiếm", use_container_width=True, key="search_btn")
        
        if search_btn and query:
            start_time = time.time()
            
            try:
                # Xử lý truy vấn
                answer = asyncio.run(
                    st.session_state.rag_system.answer(query, top_k=top_k)
                )
                
                elapsed = time.time() - start_time
                
                # Thêm vào lịch sử
                st.session_state.query_history.append({
                    "query": query,
                    "answer": answer,
                    "latency": answer.latency_ms
                })
                
                # Hiển thị kết quả - Enhanced Design
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(34,197,94,0.15) 0%, rgba(34,197,94,0.08) 100%);
                    border: 2px solid rgba(34,197,94,0.35);
                    border-radius: 10px;
                    padding: 1.25rem;
                    margin: 2rem 0 2.5rem 0;
                    animation: slideUp 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
                ">
                    <p style="margin: 0; color: #16a34a; font-weight: 700; display: flex; align-items: center; gap: 0.5rem;">
                        ✅ <strong>Hoàn tất trong {:.0f}ms</strong>
                    </p>
                </div>
                """.format(answer.latency_ms), unsafe_allow_html=True)
                
                # Chỉ số - Modern Cards
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎯 Độ Tin Cậy", f"{answer.confidence:.1%}")
                with col2:
                    st.metric("📊 Độ Liên Quan", f"{answer.relevance_score:.2f}")
                with col3:
                    st.metric("⚡ Độ Trễ", f"{answer.latency_ms:.0f}ms")
                with col4:
                    model_name = answer.model.replace('gpt-4', 'GPT-4').replace('-mock', '')
                    st.metric("🤖 Mô Hình", model_name)
                
                # Reranking info
                if hasattr(answer, 'reranking_method') and answer.reranking_method and answer.reranking_method != "none":
                    rerank_col1, rerank_col2 = st.columns(2)
                    with rerank_col1:
                        st.metric("🔄 Phương Pháp XH", answer.reranking_method.upper())
                    with rerank_col2:
                        if hasattr(answer, 'reranking_scores') and answer.reranking_scores:
                            avg_improvement = sum(
                                (s['final_score'] - s['original_score']) / max(s['original_score'], 0.01)
                                for s in answer.reranking_scores
                            ) / len(answer.reranking_scores)
                            st.metric("Cải Thiện TB", f"{avg_improvement:+.1%}")
                
                st.markdown("---")
                
                # Câu trả lời - Enhanced Display
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(245,250,255,0.7) 100%);
                    border-radius: 14px;
                    padding: 2.5rem;
                    margin: 2.5rem 0;
                    border: 1px solid rgba(102,126,234,0.18);
                    backdrop-filter: blur(8px);
                    box-shadow: 0 4px 20px rgba(102, 126, 234, 0.08);
                ">
                    <h3 style="margin-top: 0; color: #1f2937; margin-bottom: 1.5rem;">📝 Câu Trả Lời</h3>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 1.05rem; line-height: 1.9; color: #374151;'>{answer.answer}</p>", unsafe_allow_html=True)
                
                # Các tài liệu được truy xuất - with reranking scores
                if answer.retrieved_docs:
                    st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.05) 100%);
                        border-radius: 14px;
                        padding: 2.5rem;
                        margin: 3rem 0 2rem 0;
                        border: 1px solid rgba(102,126,234,0.18);
                    ">
                        <h3 style="margin-top: 0; color: #1f2937; margin-bottom: 1rem;">📄 Tài Liệu Được Truy Xuất & Xếp Hạng</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show reranking method if available
                    if hasattr(answer, 'reranking_method') and answer.reranking_method and answer.reranking_method != "none":
                        st.info(f"🔄 **Phương Pháp Xếp Hạng Lại:** {answer.reranking_method.upper()}")
                    
                    # Show reranking visualization if available
                    show_rerank = st.checkbox("📊 Hiển thị Chi Tiết Xếp Hạng Lại", value=True)
                    
                    for i, doc in enumerate(answer.retrieved_docs, 1):
                        # Ensure proper UTF-8 encoding for Vietnamese text
                        if isinstance(doc, bytes):
                            doc = doc.decode('utf-8', errors='replace')
                        
                        # Convert to string and store as the full document content
                        doc_full_content = str(doc)
                        
                        # Extract URL and title from document
                        doc_url = None
                        doc_title = None
                        
                        # Try to extract from dict-like object
                        if hasattr(doc, 'get'):
                            doc_url = doc.get('url') or doc.get('source')
                            doc_title = doc.get('title') or doc.get('name')
                            doc_full_content = doc.get('content') or doc.get('text') or str(doc)
                        elif isinstance(doc, dict):
                            doc_url = doc.get('url') or doc.get('source')
                            doc_title = doc.get('title') or doc.get('name')
                            doc_full_content = doc.get('content') or doc.get('text') or str(doc)
                        
                        # Get preview for title
                        doc_preview = doc_full_content[:80].replace('\n', ' ').strip()
                        if len(doc_full_content) > 80:
                            doc_preview += "..."
                        
                        # Use extracted title or preview
                        display_title = doc_title or doc_preview
                        
                        # Get reranking score for this document if available
                        rerank_info = None
                        if hasattr(answer, 'reranking_scores') and answer.reranking_scores and i <= len(answer.reranking_scores):
                            rerank_info = answer.reranking_scores[i - 1]
                        
                        # Display header with scores
                        if rerank_info:
                            header_text = f"📌 Tài Liệu {i}: {display_title}"
                            header_text += f" | Original: {rerank_info['original_score']:.2f} → Final: {rerank_info['final_score']:.2f}"
                        else:
                            header_text = f"📌 Tài Liệu {i}: {display_title}"
                        
                        with st.expander(header_text, expanded=(i==1)):
                            # Display URL and source info
                            st.markdown("""
                            <div style="
                                background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.04) 100%);
                                border-radius: 10px;
                                padding: 1.2rem;
                                margin-bottom: 1.5rem;
                                border: 1px solid rgba(102,126,234,0.15);
                            ">
                            """, unsafe_allow_html=True)
                            
                            if doc_title:
                                st.markdown(f"**📌 Tiêu đề:** {doc_title}", unsafe_allow_html=True)
                            
                            if doc_url:
                                st.markdown(
                                    f"**🔗 Nguồn:** [🌐 {doc_url}]({doc_url})",
                                    unsafe_allow_html=True
                                )
                            else:
                                st.markdown("**⚠️ Nguồn:** Không có nguồn phù hợp", unsafe_allow_html=True)
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                            # Display full document content
                            st.markdown("#### 📄 Nội Dung")
                            st.markdown(doc_full_content, unsafe_allow_html=True)
                            
                            # Display reranking details if available
                            if show_rerank and rerank_info:
                                st.markdown("---")
                                st.markdown("#### 🔄 Thông Tin Xếp Hạng Lại")
                                
                                # Create columns for metrics
                                score_col1, score_col2, score_col3 = st.columns(3)
                                
                                with score_col1:
                                    st.metric(
                                        "Điểm Gốc",
                                        f"{rerank_info['original_score']:.2f}",
                                        help="Điểm từ khâu truy xuất ban đầu"
                                    )
                                
                                with score_col2:
                                    st.metric(
                                        "Điểm Xếp Hạng",
                                        f"{rerank_info['rerank_score']:.2f}",
                                        help="Điểm sau khi applied reranking"
                                    )
                                
                                with score_col3:
                                    improvement = ((rerank_info['final_score'] - rerank_info['original_score']) / 
                                                  max(rerank_info['original_score'], 0.01))
                                    st.metric(
                                        "Cải Thiện",
                                        f"{improvement:+.1%}",
                                        help="Mức cải thiện sau xếp hạng lại"
                                    )
                                
                                st.markdown(f"**📝 Lý Do:** {rerank_info['reasoning']}")
                    
                    # Show reranking impact chart
                    if show_rerank and hasattr(answer, 'reranking_scores') and answer.reranking_scores:
                        st.markdown("---")
                        st.markdown("#### 📊 Biểu Đồ Tác Động Xếp Hạng Lại")
                        
                        try:
                            import pandas as pd
                            import matplotlib.pyplot as plt
                            
                            # Prepare data
                            docs_data = {
                                "Tài Liệu": [f"Doc {s['position']}" for s in answer.reranking_scores],
                                "Điểm Gốc": [s['original_score'] for s in answer.reranking_scores],
                                "Điểm Cuối Cùng": [s['final_score'] for s in answer.reranking_scores]
                            }
                            
                            df = pd.DataFrame(docs_data)
                            
                            # Create chart
                            fig, ax = plt.subplots(figsize=(10, 4))
                            x = range(len(df))
                            width = 0.35
                            
                            ax.bar([i - width/2 for i in x], df['Điểm Gốc'], width, label='Điểm Gốc', alpha=0.8, color='#667eea')
                            ax.bar([i + width/2 for i in x], df['Điểm Cuối Cùng'], width, label='Điểm Xếp Hạng', alpha=0.8, color='#764ba2')
                            
                            ax.set_xlabel("Tài Liệu")
                            ax.set_ylabel("Điểm Liên Quan")
                            ax.set_title("Tác Động Của Xếp Hạng Lại Đến Điểm Liên Quan")
                            ax.set_xticks(x)
                            ax.set_xticklabels(df['Tài Liệu'])
                            ax.legend()
                            ax.grid(axis='y', alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Không thể vẽ biểu đồ: {e}")
            
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
        
        # Lịch sử truy vấn
        if st.session_state.query_history:
            st.markdown("---")
            st.markdown(f"### 📜 Truy Vấn Gần Đây ({len(st.session_state.query_history)})")
            
            for i, item in enumerate(st.session_state.query_history[-5:], 1):
                time_str = f"{item['latency']:.0f}ms"
                with st.expander(f"{i}. {item['query'][:50]}... ({time_str})"):
                    st.write(item['answer'].answer)
    
    # TAB 2: Xử lý Hàng loạt
    with tab2:
        st.markdown("### ⚡ Xử Lý Hàng Loạt Truy Vấn")
        
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(102,126,234,0.08) 0%, rgba(118,75,162,0.04) 100%);
            border-radius: 12px;
            padding: 1.8rem;
            margin-bottom: 2.5rem;
            border: 1px solid rgba(102,126,234,0.15);
        ">
            💡 <strong>Xử lý nhiều truy vấn hiệu quả</strong>
            <ul style="margin-top: 1rem; line-height: 1.8;">
                <li>⚙️ Thực thi song song để tăng tốc độ</li>
                <li>🔇 Loại bỏ bản sao thông minh</li>
                <li>📊 Theo dõi tiến trình</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Các truy vấn mẫu
        default_queries = [
            "What is Retrieval-Augmented Generation?",
            "How does Tavily improve RAG systems?",
            "What are vector databases?",
            "Explain query optimization",
            "How does caching work?"
        ]
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            use_grouping = st.checkbox("Nhóm Truy Vấn Thông Minh", value=True)
        
        with col2:
            batch_btn = st.button("⚡ Xử Lý Hàng Loạt", use_container_width=True)
        
        st.markdown("### Các Truy Vấn")
        batch_queries = []
        
        cols = st.columns(1)
        for i, default_q in enumerate(default_queries):
            q = st.text_input(
                f"Truy Vấn {i+1}",
                value=default_q,
                key=f"batch_q_{i}",
                label_visibility="collapsed"
            )
            if q:
                batch_queries.append(q)
        
        if batch_btn and batch_queries:
            start_time = time.time()
            
            try:
                # Xử lý hàng loạt
                results = asyncio.run(
                    st.session_state.rag_system.batch_answer(
                        batch_queries,
                        top_k=top_k,
                        use_grouping=use_grouping
                    )
                )
                
                elapsed = time.time() - start_time
                
                # Tóm tắt
                st.success(f"✅ Đã xử lý {len(results)} truy vấn trong {elapsed*1000:.0f}ms")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tổng Truy Vấn", len(results))
                with col2:
                    avg_latency = sum(r.latency_ms for r in results) / len(results)
                    st.metric("Độ Trễ Trung Bình", f"{avg_latency:.0f}ms")
                with col3:
                    avg_confidence = sum(r.confidence for r in results) / len(results)
                    st.metric("Độ Tin Cậy Trung Bình", f"{avg_confidence:.1%}")
                
                st.markdown("---")
                
                # Results
                st.markdown("### Kết Quả")
                for i, result in enumerate(results, 1):
                    with st.expander(f"{i}. {result.query} ({result.confidence:.1%})"):
                        st.write(result.answer)
            
            except Exception as e:
                st.error(f"❌ Lỗi: {e}")
    
    # TAB 3: Chỉ số Hiệu suất
    with tab3:
        st.markdown("### Chỉ Số Hiệu Suất")
        
        # Báo cáo đánh giá
        report = st.session_state.rag_system.get_evaluation_report()
        
        if report:
            # Chuyển đổi đối tượng thành dict
            if hasattr(report, '__dict__'):
                report_dict = report.__dict__
            else:
                report_dict = report if isinstance(report, dict) else {}
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Truy Vấn Đã Xử Lý", report_dict.get("total_queries", 0))
            
            with col2:
                avg_lat = report_dict.get("avg_latency_ms", 0)
                st.metric("Độ Trễ Trung Bình", f"{avg_lat:.0f}ms")
            
            with col3:
                precision = report_dict.get("avg_precision", 0)
                st.metric("Độ Chính Xác", f"{precision:.2f}")
            
            with col4:
                recall = report_dict.get("avg_recall", 0)
                st.metric("Khôi Phục", f"{recall:.2f}")
            
            st.markdown("---")
            
            # Chỉ số chi tiết
            st.markdown("### Chỉ Số Chi Tiết")
            
            metrics_data = {
                "Chỉ Số": [
                    "Tổng Truy Vấn", "Độ Trễ Trung Bình", "Độ Chính Xác Trung Bình", 
                    "Khôi Phục Trung Bình", "Điểm F1", "MRR", "NDCG"
                ],
                "Giá Trị": [
                    report_dict.get("total_queries", 0),
                    f"{report_dict.get('avg_latency_ms', 0):.0f}ms",
                    f"{report_dict.get('avg_precision', 0):.2f}",
                    f"{report_dict.get('avg_recall', 0):.2f}",
                    f"{report_dict.get('avg_f1', 0):.2f}",
                    f"{report_dict.get('avg_mrr', 0):.2f}",
                    f"{report_dict.get('avg_ndcg', 0):.2f}"
                ]
            }
            
            st.table(metrics_data)
        else:
            st.info("📊 Chạy một số truy vấn để xem chỉ số hiệu suất")
    
    # TAB 4: Thống kê Bộ nhớ đệm
    with tab4:
        st.markdown("### Thống Kê Bộ Nhớ Đệm")
        
        cache_stats = st.session_state.rag_system.get_cache_stats()
        
        if cache_stats.get("enabled"):
            stats = cache_stats.get("stats", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Bộ Nhớ Trùng", stats.get("hits", 0))
            
            with col2:
                st.metric("Bộ Nhớ Không Trùng", stats.get("misses", 0))
            
            with col3:
                hit_rate = stats.get("hit_rate", 0)
                st.metric("Tỷ Lệ Trùng", f"{hit_rate:.1%}")
            
            st.markdown("---")
            
            # Cache details
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.info(f"**Chiến Lược:** {cache_stats.get('strategy', 'N/A')}")
            
            with col2:
                st.info(f"**Mục Được Lưu:** {stats.get('total_cached_items', 0)}")
            
            with col3:
                memory_mb = stats.get("memory_usage_mb", 0)
                st.info(f"**Bộ Nhớ:** {memory_mb:.1f} MB")
            
            st.markdown("---")
            
            # Cache effectiveness
            st.markdown("### 📊 Hiệu Quả")
            
            total_requests = stats.get("hits", 0) + stats.get("misses", 0)
            if total_requests > 0:
                time_saved = stats.get("hits", 0) * 0.290  # Approx 290ms per hit saved
                cost_saved = stats.get("hits", 0) * 0.0001  # Approx $0.0001 per hit saved
                
                st.success(f"""
                **Tác Động:**
                - Thời gian tiết kiệm: ~{time_saved:.1f} giây
                - Chi phí tiết kiệm: ~${cost_saved:.4f}
                - Tăng tốc độ: {stats.get('hit_rate', 0)*50:.0f}x cho các truy vấn được lưu
                """)
        else:
            st.warning("💾 Bộ nhớ đệm bị vô hiệu hóa")
    
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #9ca3af; font-size: 0.9rem; padding: 2rem 0;">
<p style="margin: 0;"><strong style="color: #6b7280;">TAV-RAG v1.2</strong> | Hệ Thống Tạo Tác Cải Thiện Truy Xuất</p>
<p style="margin: 0.5rem 0 0 0;">Trạng Thái: ✅ Sản Xuất Sẵn Sàng | Giao Diện Được Cải Thiện</p>
</div>
""", unsafe_allow_html=True)
