"""
Theme Manager for Streamlit Application
Supports multiple color themes and dynamic switching
"""

from dataclasses import dataclass
from typing import Dict
import streamlit as st


@dataclass
class ThemeColor:
    """Color palette for a theme"""
    primary_start: str
    primary_end: str
    secondary_start: str
    secondary_end: str
    success: str
    warning: str
    danger: str
    info: str
    bg_light: str
    bg_dark: str
    text_primary: str
    text_secondary: str
    border_light: str
    

# ============================================================
# THEME DEFINITIONS
# ============================================================

THEME_MODERN = ThemeColor(
    primary_start="#667eea",
    primary_end="#764ba2",
    secondary_start="#667eea",
    secondary_end="#764ba2",
    success="#22c55e",
    warning="#f59e0b",
    danger="#ef4444",
    info="#3b82f6",
    bg_light="#f5f7fa",
    bg_dark="#0f172a",
    text_primary="#1f2937",
    text_secondary="#6b7280",
    border_light="#e5e7eb"
)

THEME_OCEAN = ThemeColor(
    primary_start="#0ea5e9",
    primary_end="#0369a1",
    secondary_start="#06b6d4",
    secondary_end="#0891b2",
    success="#10b981",
    warning="#f59e0b",
    danger="#ef4444",
    info="#3b82f6",
    bg_light="#ecf0f1",
    bg_dark="#0f4c75",
    text_primary="#1e293b",
    text_secondary="#64748b",
    border_light="#cbd5e1"
)

THEME_SUNSET = ThemeColor(
    primary_start="#f97316",
    primary_end="#dc2626",
    secondary_start="#f59e0b",
    secondary_end="#ea580c",
    success="#16a34a",
    warning="#ea580c",
    danger="#dc2626",
    info="#06b6d4",
    bg_light="#fef9f3",
    bg_dark="#431407",
    text_primary="#292524",
    text_secondary="#78716c",
    border_light="#e7e5e4"
)

THEME_FOREST = ThemeColor(
    primary_start="#059669",
    primary_end="#047857",
    secondary_start="#10b981",
    secondary_end="#059669",
    success="#16a34a",
    warning="#f59e0b",
    danger="#ef4444",
    info="#0ea5e9",
    bg_light="#ecfdf5",
    bg_dark="#064e3b",
    text_primary="#1b4332",
    text_secondary="#52796f",
    border_light="#d1e7dd"
)

THEME_DARK = ThemeColor(
    primary_start="#a78bfa",
    primary_end="#c4b5fd",
    secondary_start="#a78bfa",
    secondary_end="#e9d5ff",
    success="#34d399",
    warning="#fbbf24",
    danger="#ff6b6b",
    info="#60a5fa",
    bg_light="#1f2937",
    bg_dark="#111827",
    text_primary="#f3f4f6",
    text_secondary="#d1d5db",
    border_light="#4b5563"
)

THEMES: Dict[str, ThemeColor] = {
    "Modern": THEME_MODERN,
    "Ocean": THEME_OCEAN,
    "Sunset": THEME_SUNSET,
    "Forest": THEME_FOREST,
    "Dark": THEME_DARK,
}


class ThemeManager:
    """Manage themes and generate CSS dynamically"""
    
    @staticmethod
    def get_theme() -> ThemeColor:
        """Get current theme from session state"""
        if "theme" not in st.session_state:
            st.session_state.theme = THEME_MODERN
        return st.session_state.theme
    
    @staticmethod
    def set_theme(theme_name: str) -> None:
        """Set current theme by name"""
        if theme_name in THEMES:
            st.session_state.theme = THEMES[theme_name]
    
    @staticmethod
    def generate_css(theme: ThemeColor) -> str:
        """Generate CSS for given theme"""
        return f"""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Inter:wght@400;500;600&display=swap');
            
            :root {{
                --color-primary-start: {theme.primary_start};
                --color-primary-end: {theme.primary_end};
                --color-secondary-start: {theme.secondary_start};
                --color-secondary-end: {theme.secondary_end};
                --color-success: {theme.success};
                --color-warning: {theme.warning};
                --color-danger: {theme.danger};
                --color-info: {theme.info};
                --color-bg-light: {theme.bg_light};
                --color-bg-dark: {theme.bg_dark};
                --color-text-primary: {theme.text_primary};
                --color-text-secondary: {theme.text_secondary};
                --color-border-light: {theme.border_light};
            }}
            
            * {{
                font-family: 'Inter', 'Segoe UI', sans-serif;
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            html, body {{
                background: linear-gradient(135deg, {theme.bg_light} 0%, {theme.bg_dark}20 100%);
                color: {theme.text_primary};
            }}
            
            .main {{
                background: linear-gradient(135deg, {theme.bg_light} 0%, {theme.bg_dark}20 100%);
                padding: 2.5rem 2rem;
            }}
            
            /* Typography */
            h1 {{
                font-family: 'Poppins', sans-serif;
                font-size: 3rem;
                font-weight: 700;
                color: {theme.text_primary};
                margin-bottom: 0.5rem;
                letter-spacing: -0.5px;
                background: linear-gradient(135deg, {theme.primary_start} 0%, {theme.primary_end} 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }}
            
            h2 {{
                font-family: 'Poppins', sans-serif;
                font-size: 1.75rem;
                font-weight: 600;
                color: {theme.text_primary};
                margin: 2rem 0 1rem 0;
                position: relative;
                padding-bottom: 0.75rem;
            }}
            
            h2::after {{
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                width: 60px;
                height: 4px;
                background: linear-gradient(90deg, {theme.primary_start} 0%, {theme.primary_end} 100%);
                border-radius: 2px;
            }}
            
            h3 {{
                font-family: 'Poppins', sans-serif;
                font-size: 1.25rem;
                font-weight: 600;
                color: {theme.primary_start};
                margin: 1.5rem 0 0.75rem 0;
            }}
            
            /* Cards */
            .stMetric {{
                background: rgba(255, 255, 255, 0.7);
                backdrop-filter: blur(10px);
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 4px 20px rgba(102, 126, 234, 0.08),
                            0 1px 3px rgba(0, 0, 0, 0.05);
                border: 1px solid {theme.border_light};
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            
            .stMetric:hover {{
                transform: translateY(-4px);
                box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15),
                            0 1px 3px rgba(0, 0, 0, 0.1);
            }}
            
            /* Inputs */
            .stTextInput input,
            .stTextArea textarea {{
                background: rgba(255, 255, 255, 0.8) !important;
                border: 2px solid {theme.border_light} !important;
                border-radius: 8px !important;
                padding: 0.9rem 1.1rem !important;
                color: {theme.text_primary} !important;
                transition: all 0.2s ease !important;
            }}
            
            .stTextInput input:focus,
            .stTextArea textarea:focus {{
                border-color: {theme.primary_start} !important;
                box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1) !important;
            }}
            
            /* Buttons */
            .stButton > button {{
                background: linear-gradient(135deg, {theme.primary_start} 0%, {theme.primary_end} 100%);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 0.85rem 2rem;
                font-weight: 600;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }}
            
            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
            }}
            
            /* Alerts */
            .stSuccess {{
                background: linear-gradient(135deg, rgba({theme.success}22, 0.1) 0%, rgba({theme.success}22, 0.05) 100%) !important;
                border: 1px solid {theme.success}40 !important;
                border-radius: 8px !important;
            }}
            
            .stError {{
                background: linear-gradient(135deg, rgba({theme.danger}22, 0.1) 0%, rgba({theme.danger}22, 0.05) 100%) !important;
                border: 1px solid {theme.danger}40 !important;
                border-radius: 8px !important;
            }}
            
            .stWarning {{
                background: linear-gradient(135deg, rgba({theme.warning}22, 0.1) 0%, rgba({theme.warning}22, 0.05) 100%) !important;
                border: 1px solid {theme.warning}40 !important;
                border-radius: 8px !important;
            }}
            
            /* Scrollbar */
            ::-webkit-scrollbar {{
                width: 8px;
                height: 8px;
            }}
            
            ::-webkit-scrollbar-thumb {{
                background: linear-gradient(135deg, {theme.primary_start} 0%, {theme.primary_end} 100%);
                border-radius: 10px;
            }}
            
            /* Responsive */
            @media (max-width: 768px) {{
                h1 {{ font-size: 2rem; }}
                h2 {{ font-size: 1.4rem; }}
                h3 {{ font-size: 1.05rem; }}
            }}
        </style>
        """
    
    @staticmethod
    def apply_theme(theme: ThemeColor) -> None:
        """Apply theme to Streamlit app"""
        css = ThemeManager.generate_css(theme)
        st.markdown(css, unsafe_allow_html=True)
    
    @staticmethod
    def render_theme_selector() -> None:
        """Render theme selector in sidebar"""
        current_theme_name = None
        for name, theme in THEMES.items():
            if st.session_state.get("theme") == theme:
                current_theme_name = name
                break
        
        selected = st.selectbox(
            "🎨 Chọn Theme",
            list(THEMES.keys()),
            index=list(THEMES.keys()).index(current_theme_name) if current_theme_name else 0,
            key="theme_selector"
        )
        
        if selected:
            ThemeManager.set_theme(selected)
            st.rerun()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def apply_modern_theme():
    """Apply modern theme"""
    ThemeManager.set_theme("Modern")
    ThemeManager.apply_theme(ThemeManager.get_theme())

def apply_ocean_theme():
    """Apply ocean theme"""
    ThemeManager.set_theme("Ocean")
    ThemeManager.apply_theme(ThemeManager.get_theme())

def apply_sunset_theme():
    """Apply sunset theme"""
    ThemeManager.set_theme("Sunset")
    ThemeManager.apply_theme(ThemeManager.get_theme())

def apply_forest_theme():
    """Apply forest theme"""
    ThemeManager.set_theme("Forest")
    ThemeManager.apply_theme(ThemeManager.get_theme())

def apply_dark_theme():
    """Apply dark theme"""
    ThemeManager.set_theme("Dark")
    ThemeManager.apply_theme(ThemeManager.get_theme())


# ============================================================
# EXAMPLE USAGE
# ============================================================

"""
Usage in your Streamlit app:

    import streamlit as st
    from utils.theme_manager import ThemeManager
    
    # In your app, early in the execution:
    
    # Apply styling
    theme = ThemeManager.get_theme()
    ThemeManager.apply_theme(theme)
    
    # In sidebar, add theme selector
    with st.sidebar:
        ThemeManager.render_theme_selector()
        
    # Rest of your app...
    st.title("Your App Title")
    st.write("Your content here")
"""
