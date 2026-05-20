"""
UI System Initialization & Bootstrap
One-click setup for all UI utilities

Usage:
    from utils.ui_init import init_ui_system
    
    # In your Streamlit app, at the very beginning:
    init_ui_system(theme="Modern", show_theme_selector=True)
    
    # Now your app has beautiful styling!
"""

import streamlit as st
from utils.theme_manager import ThemeManager
from typing import Literal, Optional


def init_ui_system(
    theme: Literal["Modern", "Ocean", "Sunset", "Forest", "Dark"] = "Modern",
    show_theme_selector: bool = True,
    page_title: Optional[str] = None,
    page_icon: Optional[str] = None,
    layout: Literal["centered", "wide"] = "wide"
) -> None:
    """
    Initialize entire UI system with one function call.
    
    Parameters:
    -----------
    theme : str
        Initial theme to apply. Options: "Modern", "Ocean", "Sunset", "Forest", "Dark"
        Default: "Modern"
    
    show_theme_selector : bool
        Whether to show theme selector in sidebar
        Default: True
    
    page_title : str, optional
        Page title for browser tab
        Default: None (uses Streamlit default)
    
    page_icon : str, optional
        Page icon for browser tab
        Default: None
    
    layout : str
        Page layout. Options: "centered", "wide"
        Default: "wide"
    
    Example:
    --------
    import streamlit as st
    from utils.ui_init import init_ui_system
    
    # Initialize UI system
    init_ui_system(
        theme="Modern",
        show_theme_selector=True,
        page_title="TAV-RAG System",
        page_icon="🔍",
        layout="wide"
    )
    
    # Now use Streamlit normally
    st.title("My Beautiful App")
    """
    
    # Set page config
    config_kwargs = {"layout": layout}
    if page_title:
        config_kwargs["page_title"] = page_title
    if page_icon:
        config_kwargs["page_icon"] = page_icon
    
    st.set_page_config(**config_kwargs)
    
    # Apply theme
    if "ui_system_initialized" not in st.session_state:
        ThemeManager.set_theme(theme)
        st.session_state.ui_system_initialized = True
    
    current_theme = ThemeManager.get_theme()
    ThemeManager.apply_theme(current_theme)
    
    # Optional: Add theme selector to sidebar
    if show_theme_selector:
        with st.sidebar:
            st.divider()
            st.markdown("**🎨 Appearance**")
            ThemeManager.render_theme_selector()
            st.divider()


def init_ui_system_minimal(theme: str = "Modern") -> None:
    """
    Ultra-minimal UI initialization (no theme selector, no page config).
    
    Use this if you already have st.set_page_config() elsewhere.
    
    Parameters:
    -----------
    theme : str
        Initial theme to apply
    
    Example:
    --------
    import streamlit as st
    
    st.set_page_config(page_title="My App")  # Your own config
    
    from utils.ui_init import init_ui_system_minimal
    init_ui_system_minimal("Ocean")  # Just apply theme
    """
    
    ThemeManager.set_theme(theme)
    current_theme = ThemeManager.get_theme()
    ThemeManager.apply_theme(current_theme)


def create_ui_navbar(
    title: str,
    items: Optional[dict] = None,
    show_theme_selector: bool = True
) -> None:
    """
    Create a navigation bar with logo and items.
    
    Parameters:
    -----------
    title : str
        App title/logo
    
    items : dict, optional
        Navigation items. Format: {"label": "url"}
        Example: {"Home": "/", "Docs": "/docs", "About": "/about"}
    
    show_theme_selector : bool
        Show theme selector in navbar
    
    Example:
    --------
    create_ui_navbar(
        "🔍 TAV-RAG",
        {"Home": "#", "Docs": "#docs", "API": "#api"},
        show_theme_selector=True
    )
    """
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### {title}")
    
    with col2:
        if items:
            tabs = st.tabs(list(items.keys()))
            for tab, (label, url) in zip(tabs, items.items()):
                with tab:
                    st.markdown(f"[{label}]({url})")
    
    with col3:
        if show_theme_selector:
            themes = ["Modern", "Ocean", "Sunset", "Forest", "Dark"]
            current = None
            for theme_name in themes:
                if st.session_state.get("theme_name") == theme_name:
                    current = theme_name
                    break
            
            selected = st.selectbox(
                "Theme",
                themes,
                index=themes.index(current) if current else 0,
                key="navbar_theme_selector"
            )
            
            if selected and st.session_state.get("theme_name") != selected:
                ThemeManager.set_theme(selected)
                st.session_state.theme_name = selected
                st.rerun()


def create_ui_footer(
    left_text: str = "© 2024",
    center_text: str = "Made with ❤️",
    right_text: str = "v1.0"
) -> None:
    """
    Create a professional footer.
    
    Example:
    --------
    create_ui_footer(
        "© 2024 TAV-RAG",
        "Made with ❤️ and Streamlit",
        "v2.0 | Production Ready"
    )
    """
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption(left_text)
    with col2:
        st.caption(center_text)
    with col3:
        st.caption(right_text)


def create_ui_sidebar_config(
    app_title: str = "TAV-RAG System",
    app_version: str = "2.0",
    show_theme_selector: bool = True,
    additional_sections: Optional[dict] = None
) -> None:
    """
    Create a complete sidebar configuration.
    
    Parameters:
    -----------
    app_title : str
        Application title
    
    app_version : str
        Application version
    
    show_theme_selector : bool
        Show theme selector
    
    additional_sections : dict, optional
        Additional sidebar sections
        Format: {"Section Title": [items]}
    
    Example:
    --------
    create_ui_sidebar_config(
        app_title="TAV-RAG",
        app_version="2.0",
        show_theme_selector=True,
        additional_sections={
            "Settings": ["Query Settings", "Model Settings"],
            "Help": ["Documentation", "FAQ"]
        }
    )
    """
    
    with st.sidebar:
        # Header
        st.markdown(f"### {app_title}")
        st.caption(f"v{app_version}")
        st.divider()
        
        # Theme selector
        if show_theme_selector:
            st.markdown("**🎨 Theme**")
            ThemeManager.render_theme_selector()
            st.divider()
        
        # Additional sections
        if additional_sections:
            for section_title, items in additional_sections.items():
                st.markdown(f"**{section_title}**")
                for item in items:
                    st.write(f"• {item}")
                st.divider()
        
        # About
        st.markdown("**About**")
        st.write("TAV-RAG System - Advanced Retrieval-Augmented Generation")


# ====================================================================
# QUICK TEMPLATES
# ====================================================================

def template_simple_app() -> None:
    """
    Simple app template with UI system.
    Copy-paste to get started quickly.
    """
    
    code = '''
import streamlit as st
from utils.ui_init import init_ui_system
from utils.ui_components import render_metric_card

# Step 1: Initialize UI system
init_ui_system(
    theme="Modern",
    page_title="My App",
    page_icon="🚀"
)

# Step 2: Create app content
st.title("Welcome!")

# Step 3: Use components
col1, col2 = st.columns(2)
with col1:
    render_metric_card("Users", "1500", "👥", "#667eea")
with col2:
    render_metric_card("Active", "1200", "🟢", "#22c55e")

# That's it!
    '''
    
    return code


def template_advanced_app() -> None:
    """
    Advanced app template with all features.
    """
    
    code = '''
import streamlit as st
from utils.ui_init import init_ui_system, create_ui_sidebar_config, create_ui_footer
from utils.ui_components import render_hero_section, render_metric_card
from utils.response_formatter import display_answer
from utils.animations import AnimationLibrary as AL

# Initialize system
init_ui_system(
    theme="Modern",
    page_title="TAV-RAG System",
    page_icon="🔍"
)

# Sidebar configuration
create_ui_sidebar_config(
    app_title="TAV-RAG",
    app_version="2.0",
    additional_sections={
        "Settings": ["Query", "Model"],
        "Help": ["Docs", "FAQ"]
    }
)

# Hero section
st.markdown(f'<div style="{AL.FADE_IN_UP}"><h1>Welcome</h1></div>', unsafe_allow_html=True)

# Main content with tabs
tab1, tab2, tab3 = st.tabs(["Search", "Results", "Settings"])

with tab1:
    query = st.text_input("Ask me anything...")
    if st.button("Search"):
        # Show metrics
        cols = st.columns(4)
        metrics = [
            ("Confidence", "95%", "🎯", "#667eea"),
            ("Relevance", "0.92", "📊", "#22c55e"),
            ("Latency", "245ms", "⚡", "#f59e0b"),
            ("Sources", "3", "📚", "#3b82f6")
        ]
        for col, (title, value, icon, color) in zip(cols, metrics):
            with col:
                render_metric_card(title, value, icon, color)

# Footer
create_ui_footer(
    "© 2024 TAV-RAG",
    "Made with ❤️",
    "v2.0"
)
    '''
    
    return code


# ====================================================================
# PREDEFINED CONFIGURATIONS
# ====================================================================

THEME_CONFIGS = {
    "professional": {
        "theme": "Modern",
        "layout": "wide",
        "show_theme_selector": True,
    },
    "creative": {
        "theme": "Sunset",
        "layout": "centered",
        "show_theme_selector": True,
    },
    "minimal": {
        "theme": "Ocean",
        "layout": "wide",
        "show_theme_selector": False,
    },
    "dark": {
        "theme": "Dark",
        "layout": "wide",
        "show_theme_selector": True,
    }
}


def apply_preset_config(preset: Literal["professional", "creative", "minimal", "dark"]) -> None:
    """
    Apply predefined UI configuration.
    
    Parameters:
    -----------
    preset : str
        Preset name: "professional", "creative", "minimal", "dark"
    
    Example:
    --------
    from utils.ui_init import apply_preset_config
    
    apply_preset_config("professional")
    """
    
    if preset in THEME_CONFIGS:
        config = THEME_CONFIGS[preset]
        init_ui_system(**config)


# ====================================================================
# HELPER FUNCTIONS
# ====================================================================

def get_current_theme_name() -> str:
    """Get name of currently applied theme."""
    from utils.theme_manager import THEMES
    
    current_theme = st.session_state.get("theme")
    for name, theme in THEMES.items():
        if theme == current_theme:
            return name
    return "Modern"


def switch_theme(theme_name: str) -> None:
    """Switch to a different theme."""
    ThemeManager.set_theme(theme_name)
    st.rerun()


# ====================================================================
# QUICK START INSTRUCTIONS
# ====================================================================

QUICK_START = """
# 🚀 UI System Quick Start

## 1. Minimal Setup (30 seconds)
```python
import streamlit as st
from utils.ui_init import init_ui_system

init_ui_system()
st.title("Hello, beautiful app!")
```

## 2. With Theme Selector
```python
from utils.ui_init import init_ui_system

init_ui_system(theme="Modern", show_theme_selector=True)
```

## 3. With Components
```python
from utils.ui_init import init_ui_system
from utils.ui_components import render_metric_card

init_ui_system()

col1, col2 = st.columns(2)
with col1:
    render_metric_card("Title", "Value", "🎯", "#667eea")
```

## 4. Complete Setup
```python
from utils.ui_init import (
    init_ui_system,
    create_ui_sidebar_config,
    create_ui_footer
)

# Initialize
init_ui_system("Modern")

# Configure sidebar
create_ui_sidebar_config("My App", "1.0")

# Your content here
st.title("Content")

# Footer
create_ui_footer()
```

---

**Available themes:** Modern, Ocean, Sunset, Forest, Dark
**Available presets:** professional, creative, minimal, dark

Choose what works best for you!
"""
