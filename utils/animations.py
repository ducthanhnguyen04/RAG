"""
Animation & Effects Helper for Streamlit
Provides reusable animation classes and effects
"""

from typing import Literal


class AnimationLibrary:
    """Collections of CSS animations and effects"""
    
    # ============================================================
    # FADE ANIMATIONS
    # ============================================================
    
    FADE_IN = """
    animation: fadeIn 0.6s ease-in-out;
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    """
    
    FADE_OUT = """
    animation: fadeOut 0.6s ease-in-out;
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; }
    }
    """
    
    FADE_IN_UP = """
    animation: fadeInUp 0.6s ease-out;
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    """
    
    FADE_IN_DOWN = """
    animation: fadeInDown 0.6s ease-out;
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    """
    
    FADE_IN_LEFT = """
    animation: fadeInLeft 0.6s ease-out;
    @keyframes fadeInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    """
    
    FADE_IN_RIGHT = """
    animation: fadeInRight 0.6s ease-out;
    @keyframes fadeInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    """
    
    # ============================================================
    # SLIDE ANIMATIONS
    # ============================================================
    
    SLIDE_IN_RIGHT = """
    animation: slideInRight 0.5s ease-out;
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    """
    
    SLIDE_IN_LEFT = """
    animation: slideInLeft 0.5s ease-out;
    @keyframes slideInLeft {
        from {
            transform: translateX(-100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    """
    
    SLIDE_IN_DOWN = """
    animation: slideInDown 0.5s ease-out;
    @keyframes slideInDown {
        from {
            transform: translateY(-100%);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    """
    
    SLIDE_IN_UP = """
    animation: slideInUp 0.5s ease-out;
    @keyframes slideInUp {
        from {
            transform: translateY(100%);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    """
    
    # ============================================================
    # SCALE ANIMATIONS
    # ============================================================
    
    SCALE_UP = """
    animation: scaleUp 0.4s ease-out;
    @keyframes scaleUp {
        from {
            transform: scale(0.95);
            opacity: 0;
        }
        to {
            transform: scale(1);
            opacity: 1;
        }
    }
    """
    
    SCALE_DOWN = """
    animation: scaleDown 0.4s ease-out;
    @keyframes scaleDown {
        from {
            transform: scale(1.05);
            opacity: 0;
        }
        to {
            transform: scale(1);
            opacity: 1;
        }
    }
    """
    
    # ============================================================
    # BOUNCE ANIMATIONS
    # ============================================================
    
    BOUNCE_IN = """
    animation: bounceIn 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    @keyframes bounceIn {
        0% {
            opacity: 0;
            transform: scale(0.3);
        }
        50% {
            opacity: 1;
            transform: scale(1.05);
        }
        70% {
            transform: scale(0.9);
        }
        100% {
            transform: scale(1);
        }
    }
    """
    
    BOUNCE_UP = """
    animation: bounceUp 0.7s cubic-bezier(0.68, -0.55, 0.265, 1.55);
    @keyframes bounceUp {
        0% {
            opacity: 0;
            transform: translateY(50px);
        }
        60% {
            opacity: 1;
            transform: translateY(-10px);
        }
        80% {
            transform: translateY(5px);
        }
        100% {
            transform: translateY(0);
        }
    }
    """
    
    # ============================================================
    # PULSE ANIMATIONS
    # ============================================================
    
    PULSE = """
    animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    """
    
    PULSE_SCALE = """
    animation: pulseScale 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    @keyframes pulseScale {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.8;
            transform: scale(1.05);
        }
    }
    """
    
    # ============================================================
    # ROTATE ANIMATIONS
    # ============================================================
    
    ROTATE = """
    animation: rotate 2s linear infinite;
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    """
    
    SPIN_SLOW = """
    animation: spin 3s linear infinite;
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    """
    
    SPIN_FAST = """
    animation: spinFast 1s linear infinite;
    @keyframes spinFast {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    """
    
    # ============================================================
    # SHAKE ANIMATIONS
    # ============================================================
    
    SHAKE = """
    animation: shake 0.5s;
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
        20%, 40%, 60%, 80% { transform: translateX(5px); }
    }
    """
    
    WIGGLE = """
    animation: wiggle 0.7s;
    @keyframes wiggle {
        0%, 100% { transform: rotate(0deg); }
        25% { transform: rotate(-2deg); }
        50% { transform: rotate(2deg); }
        75% { transform: rotate(-2deg); }
    }
    """
    
    # ============================================================
    # GLOW ANIMATIONS
    # ============================================================
    
    GLOW = """
    animation: glow 2s ease-in-out infinite;
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 1); }
    }
    """
    
    GLOW_TEXT = """
    animation: glowText 2s ease-in-out infinite;
    @keyframes glowText {
        0%, 100% { text-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
        50% { text-shadow: 0 0 20px rgba(102, 126, 234, 1); }
    }
    """
    
    # ============================================================
    # FLIP ANIMATIONS
    # ============================================================
    
    FLIP_IN_X = """
    animation: flipInX 0.6s ease-in-out;
    @keyframes flipInX {
        from {
            opacity: 0;
            transform: perspective(400px) rotateX(90deg);
        }
        to {
            opacity: 1;
            transform: perspective(400px) rotateX(0deg);
        }
    }
    """
    
    FLIP_IN_Y = """
    animation: flipInY 0.6s ease-in-out;
    @keyframes flipInY {
        from {
            opacity: 0;
            transform: perspective(400px) rotateY(90deg);
        }
        to {
            opacity: 1;
            transform: perspective(400px) rotateY(0deg);
        }
    }
    """


class HoverEffects:
    """CSS hover effects"""
    
    LIFT = """
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    &:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    """
    
    GLOW_ON_HOVER = """
    transition: all 0.3s ease;
    &:hover {
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.6);
    }
    """
    
    COLOR_SHIFT = """
    transition: all 0.3s ease;
    &:hover {
        filter: brightness(1.1) saturate(1.1);
    }
    """
    
    SCALE_HOVER = """
    transition: all 0.3s ease;
    &:hover {
        transform: scale(1.05);
    }
    """
    
    UNDERLINE_EXPAND = """
    position: relative;
    transition: all 0.3s ease;
    &::after {
        content: '';
        position: absolute;
        bottom: -2px;
        left: 50%;
        width: 0;
        height: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: all 0.3s ease;
        transform: translateX(-50%);
    }
    &:hover::after {
        width: 100%;
    }
    """


class TransitionLibrary:
    """Smooth transition styles"""
    
    SMOOTH = "transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);"
    
    FAST = "transition: all 0.15s ease;"
    
    SLOW = "transition: all 0.6s ease;"
    
    VERY_SLOW = "transition: all 1s ease;"
    
    ELASTIC = "transition: all 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);"
    
    EASE_IN_OUT = "transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);"
    
    EASE_OUT = "transition: all 0.3s cubic-bezier(0, 0, 0.2, 1);"
    
    EASE_IN = "transition: all 0.3s cubic-bezier(0.4, 0, 1, 1);"


def get_animation(name: str) -> str:
    """Get animation by name"""
    animations = {
        'fade_in': AnimationLibrary.FADE_IN,
        'fade_in_up': AnimationLibrary.FADE_IN_UP,
        'fade_in_down': AnimationLibrary.FADE_IN_DOWN,
        'fade_in_left': AnimationLibrary.FADE_IN_LEFT,
        'fade_in_right': AnimationLibrary.FADE_IN_RIGHT,
        'slide_in_right': AnimationLibrary.SLIDE_IN_RIGHT,
        'slide_in_left': AnimationLibrary.SLIDE_IN_LEFT,
        'slide_in_down': AnimationLibrary.SLIDE_IN_DOWN,
        'slide_in_up': AnimationLibrary.SLIDE_IN_UP,
        'scale_up': AnimationLibrary.SCALE_UP,
        'bounce_in': AnimationLibrary.BOUNCE_IN,
        'bounce_up': AnimationLibrary.BOUNCE_UP,
        'pulse': AnimationLibrary.PULSE,
        'glow': AnimationLibrary.GLOW,
        'flip_in_x': AnimationLibrary.FLIP_IN_X,
        'flip_in_y': AnimationLibrary.FLIP_IN_Y,
    }
    return animations.get(name.lower(), AnimationLibrary.FADE_IN)


def apply_animation(element_id: str, animation: str) -> str:
    """Apply animation to element and return HTML"""
    return f"""
    <div id="{element_id}" style="{animation}">
        {content}
    </div>
    """
