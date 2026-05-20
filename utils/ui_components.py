"""
UI Styling Utilities - Custom Components & Styles
Reusable styling functions untuk Streamlit
"""
import streamlit as st


def render_metric_card(label: str, value: str, icon: str = "📊", color: str = "blue"):
    """
    Render một metric card đẹp
    
    Args:
        label: Tên metric
        value: Giá trị
        icon: Icon (emoji)
        color: Màu (blue, green, red, orange, purple)
    """
    colors = {
        "blue": "#667eea",
        "green": "#22c55e",
        "red": "#ef4444",
        "orange": "#f59e0b",
        "purple": "#a855f7"
    }
    
    color_hex = colors.get(color, colors["blue"])
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba({int(color_hex[1:3], 16)},{int(color_hex[3:5], 16)},{int(color_hex[5:7], 16)},0.1) 0%, rgba({int(color_hex[1:3], 16)},{int(color_hex[3:5], 16)},{int(color_hex[5:7], 16)},0.05) 100%);
        border: 1px solid rgba({int(color_hex[1:3], 16)},{int(color_hex[3:5], 16)},{int(color_hex[5:7], 16)},0.2);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        backdrop-filter: blur(5px);
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-size: 0.85rem; color: #9ca3af; font-weight: 600; text-transform: uppercase; margin-bottom: 0.5rem;">{label}</div>
        <div style="font-size: 1.75rem; color: {color_hex}; font-weight: 700;">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def render_hero_section(title: str, subtitle: str, badges: list = None):
    """
    Render một hero section đẹp
    
    Args:
        title: Tiêu đề chính
        subtitle: Tiêu đề phụ
        badges: List của badge {"label": "...", "color": "..."}
    """
    badge_html = ""
    if badges:
        for badge in badges:
            colors = {
                "success": "linear-gradient(135deg, #22c55e 0%, #16a34a 100%); color: white;",
                "info": "linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;",
                "warning": "linear-gradient(135deg, #f59e0b 0%, #d97706 100%); color: white;",
                "danger": "linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: white;",
                "light": "background: rgba(255,255,255,0.8); color: #667eea; border: 2px solid #667eea;",
            }
            color_style = colors.get(badge.get("color", "info"), colors["info"])
            badge_html += f"""
            <span style="
                background: {color_style};
                padding: 0.5rem 1rem;
                border-radius: 6px;
                font-size: 0.85rem;
                font-weight: 600;
                margin-right: 0.5rem;
            ">{badge['label']}</span>
            """
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.05) 100%);
        border-radius: 12px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(102,126,234,0.15);
        backdrop-filter: blur(10px);
    ">
        <h1 style="margin: 0; font-size: 2.5rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">{title}</h1>
        <p style="margin: 0.75rem 0 0 0; font-size: 1.1rem; color: #6b7280; font-weight: 400;">{subtitle}</p>
        <div style="margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap;">
            {badge_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_info_box(content: str, type: str = "info"):
    """
    Render một info box đẹp
    
    Args:
        content: Nội dung
        type: info, success, warning, error
    """
    styles = {
        "info": {
            "bg": "rgba(59,130,246,0.1)",
            "border": "rgba(59,130,246,0.3)",
            "icon": "ℹ️"
        },
        "success": {
            "bg": "rgba(34,197,94,0.1)",
            "border": "rgba(34,197,94,0.3)",
            "icon": "✅"
        },
        "warning": {
            "bg": "rgba(245,158,11,0.1)",
            "border": "rgba(245,158,11,0.3)",
            "icon": "⚠️"
        },
        "error": {
            "bg": "rgba(239,68,68,0.1)",
            "border": "rgba(239,68,68,0.3)",
            "icon": "❌"
        }
    }
    
    style = styles.get(type, styles["info"])
    
    st.markdown(f"""
    <div style="
        background: {style['bg']};
        border: 1px solid {style['border']};
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    ">
        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{style['icon']}</span>
        <span style="color: #374151;">{content}</span>
    </div>
    """, unsafe_allow_html=True)


def render_code_block(code: str, language: str = "python"):
    """
    Render một code block đẹp
    
    Args:
        code: Đoạn code
        language: Ngôn ngữ lập trình
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(31,41,55,0.95) 0%, rgba(15,23,42,0.95) 100%);
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102,126,234,0.2);
        overflow-x: auto;
    ">
        <div style="color: #9ca3af; font-size: 0.85rem; margin-bottom: 0.5rem; font-weight: 600;">{language.upper()}</div>
        <pre style="
            color: #e5e7eb;
            margin: 0;
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.6;
        ">{code}</pre>
    </div>
    """, unsafe_allow_html=True)


def render_document_card(title: str, preview: str, score: float, source: str = "Unknown"):
    """
    Render một document card đẹp
    
    Args:
        title: Tiêu đề tài liệu
        preview: Xem trước nội dung
        score: Điểm liên quan (0-1)
        source: Nguồn tải liệu
    """
    # Tính màu dựa trên score
    if score >= 0.8:
        score_color = "#22c55e"
        score_label = "Rất Liên Quan"
    elif score >= 0.6:
        score_color = "#3b82f6"
        score_label = "Liên Quan"
    elif score >= 0.4:
        score_color = "#f59e0b"
        score_label = "Có Liên Quan"
    else:
        score_color = "#ef4444"
        score_label = "Ít Liên Quan"
    
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(255,255,255,0.7) 0%, rgba(240,245,255,0.5) 100%);
        border: 2px solid {score_color}33;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    ">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
            <div style="flex: 1;">
                <h4 style="margin: 0 0 0.5rem 0; color: #1f2937; font-size: 1.1rem;">{title}</h4>
                <p style="margin: 0; color: #6b7280; font-size: 0.9rem;">{source}</p>
            </div>
            <div style="
                background: {score_color};
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                text-align: center;
                margin-left: 1rem;
                min-width: 80px;
            ">
                <div style="font-weight: 700; font-size: 1.1rem;">{score:.0%}</div>
                <div style="font-size: 0.75rem; opacity: 0.9;">{score_label}</div>
            </div>
        </div>
        <p style="
            margin: 0;
            color: #374151;
            font-size: 0.95rem;
            line-height: 1.6;
            border-top: 1px solid rgba(0,0,0,0.05);
            padding-top: 1rem;
        ">{preview}</p>
    </div>
    """, unsafe_allow_html=True)


def render_stats_row(stats: dict):
    """
    Render một dòng thống kê với nhiều chỉ số
    
    Args:
        stats: Dict key=label, value=value hoặc {"label": "...", "value": "...", "icon": "..."}
    """
    cols_html = ""
    for label, data in stats.items():
        if isinstance(data, dict):
            value = data.get("value", "N/A")
            icon = data.get("icon", "📊")
        else:
            value = data
            icon = "📊"
        
        cols_html += f"""
        <div style="
            flex: 1;
            background: rgba(255,255,255,0.5);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            border: 1px solid rgba(102,126,234,0.1);
        ">
            <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="font-size: 0.85rem; color: #9ca3af; font-weight: 600; margin-bottom: 0.5rem; text-transform: uppercase;">{label}</div>
            <div style="font-size: 1.3rem; color: #667eea; font-weight: 700;">{value}</div>
        </div>
        """
    
    st.markdown(f"""
    <div style="
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
        flex-wrap: wrap;
    ">
        {cols_html}
    </div>
    """, unsafe_allow_html=True)


def render_divider(spacing: str = "2rem"):
    """
    Render một divider đẹp
    
    Args:
        spacing: Khoảng cách trên/dưới (default: 2rem)
    """
    st.markdown(f"""
    <div style="
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.5), transparent);
        margin: {spacing} 0;
    "></div>
    """, unsafe_allow_html=True)


# Color constants
COLORS = {
    "primary": "#667eea",
    "primary_dark": "#764ba2",
    "success": "#22c55e",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "info": "#3b82f6",
    "gray_light": "#f3f4f6",
    "gray_dark": "#1f2937",
    "text": "#374151",
    "text_muted": "#6b7280",
}

# Gradient constants
GRADIENTS = {
    "primary": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "success": "linear-gradient(135deg, #22c55e 0%, #16a34a 100%)",
    "warning": "linear-gradient(135deg, #f59e0b 0%, #d97706 100%)",
    "danger": "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)",
    "background": "linear-gradient(135deg, #f5f7fa 0%, #e9ecf1 100%)",
}
