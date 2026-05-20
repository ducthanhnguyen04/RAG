"""
Response Formatter for Streamlit
Format and display LLM responses, documents, and data beautifully
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import streamlit as st


@dataclass
class FormattedResponse:
    """Container for formatted response"""
    answer: str
    sources: List[str]
    metadata: Dict[str, Any]
    confidence: float
    latency_ms: float


class ResponseFormatter:
    """Format LLM responses and documents for display"""
    
    @staticmethod
    def format_answer(answer: str, format_type: str = "default") -> str:
        """Format answer text with proper styling"""
        
        if format_type == "markdown":
            return answer
        
        elif format_type == "highlighted":
            # Highlight key terms
            important_words = [
                "important", "note", "critical", "warning",
                "success", "error", "conclusion", "summary"
            ]
            for word in important_words:
                answer = answer.replace(
                    word.capitalize(),
                    f"<strong style='color: #667eea'>{word.capitalize()}</strong>"
                )
            return answer
        
        elif format_type == "structured":
            # Add structure to answer
            lines = answer.split("\n")
            formatted_lines = []
            
            for line in lines:
                if line.startswith("•") or line.startswith("-"):
                    formatted_lines.append(f"<li>{line[1:].strip()}</li>")
                elif line.strip() and len(line) > 30:
                    formatted_lines.append(f"<p>{line}</p>")
                else:
                    formatted_lines.append(line)
            
            return "\n".join(formatted_lines)
        
        return answer
    
    @staticmethod
    def format_source_citation(source: str, url: Optional[str] = None) -> str:
        """Format source citation with styling"""
        
        if url:
            return f"<a href='{url}' target='_blank' style='color: #667eea; text-decoration: none;'>📄 {source}</a>"
        else:
            return f"<span style='color: #667eea;'>📄 {source}</span>"
    
    @staticmethod
    def format_metadata(metadata: Dict[str, Any]) -> str:
        """Format metadata for display"""
        
        html = "<div style='color: #6b7280; font-size: 0.9em;'>"
        
        for key, value in metadata.items():
            # Format key name
            display_key = key.replace("_", " ").title()
            
            # Format value based on type
            if isinstance(value, (int, float)):
                if isinstance(value, float) and value <= 1.0:
                    display_value = f"{value:.1%}"
                else:
                    display_value = str(value)
            elif isinstance(value, bool):
                display_value = "✓" if value else "✗"
            elif isinstance(value, list):
                display_value = ", ".join(str(v) for v in value)
            else:
                display_value = str(value)
            
            html += f"<strong>{display_key}:</strong> {display_value}<br>"
        
        html += "</div>"
        return html
    
    @staticmethod
    def create_answer_card(
        answer: str,
        sources: List[str],
        confidence: float,
        latency_ms: float,
        model: str = "Gemini"
    ) -> str:
        """Create formatted answer card"""
        
        return f"""
        <div style="
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.02) 100%);
            border: 1px solid rgba(102, 126, 234, 0.2);
            border-radius: 12px;
            padding: 2rem;
            backdrop-filter: blur(10px);
            animation: fadeInUp 0.6s ease-out;
        ">
            <div style="margin-bottom: 1.5rem;">
                {answer}
            </div>
            
            <div style="
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 1rem;
                margin-top: 1.5rem;
                padding-top: 1.5rem;
                border-top: 1px solid rgba(102, 126, 234, 0.1);
                font-size: 0.85rem;
                color: #6b7280;
            ">
                <div><strong style="color: #667eea;">🎯 Confidence</strong><br>{confidence:.1%}</div>
                <div><strong style="color: #22c55e;">⚡ Latency</strong><br>{latency_ms:.0f}ms</div>
                <div><strong style="color: #3b82f6;">🤖 Model</strong><br>{model}</div>
                <div><strong style="color: #f59e0b;">📚 Sources</strong><br>{len(sources)} found</div>
            </div>
            
            {f'''
            <div style="
                margin-top: 1.5rem;
                padding-top: 1.5rem;
                border-top: 1px solid rgba(102, 126, 234, 0.1);
            ">
                <strong style="color: #667eea;">Sources:</strong><br>
                {chr(10).join(f"• {source}" for source in sources[:3])}
            </div>
            ''' if sources else ''}
        </div>
        """
    
    @staticmethod
    def create_document_snippet(
        title: str,
        content: str,
        url: str,
        score: float,
        source_type: str = "Web"
    ) -> str:
        """Create formatted document snippet"""
        
        # Truncate content if too long
        max_length = 200
        if len(content) > max_length:
            content = content[:max_length].rsplit(" ", 1)[0] + "..."
        
        return f"""
        <div style="
            background: rgba(255, 255, 255, 0.6);
            border: 1px solid rgba(102, 126, 234, 0.1);
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(5px);
            transition: all 0.3s ease;
        " onmouseover="this.style.boxShadow='0 4px 20px rgba(102, 126, 234, 0.15)'" 
           onmouseout="this.style.boxShadow='0 0 0 rgba(0,0,0,0)'">
            
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 0.75rem;">
                <a href="{url}" target="_blank" style="
                    color: #667eea;
                    text-decoration: none;
                    font-weight: 600;
                    font-size: 1rem;
                ">
                    📄 {title}
                </a>
                <span style="
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.05) 100%);
                    padding: 0.25rem 0.75rem;
                    border-radius: 4px;
                    font-size: 0.85rem;
                    font-weight: 600;
                    color: #667eea;
                ">
                    Score: {score:.2f}
                </span>
            </div>
            
            <p style="color: #6b7280; margin-bottom: 0.75rem; line-height: 1.5;">
                {content}
            </p>
            
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #9ca3af;">
                <span>📁 {source_type}</span>
                <span>⭐ {score * 100:.0f}% Relevant</span>
            </div>
        </div>
        """
    
    @staticmethod
    def create_error_message(
        error_type: str,
        message: str,
        suggestion: Optional[str] = None
    ) -> str:
        """Create formatted error message"""
        
        suggestion_html = ""
        if suggestion:
            suggestion_html = f"""
            <div style="
                margin-top: 1rem;
                padding-top: 1rem;
                border-top: 1px solid rgba(239, 68, 68, 0.2);
            ">
                <strong style="color: #ef4444;">💡 Suggestion:</strong><br>
                <span style="color: #6b7280;">{suggestion}</span>
            </div>
            """
        
        return f"""
        <div style="
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%);
            border: 1px solid rgba(239, 68, 68, 0.3);
            border-radius: 8px;
            padding: 1.25rem;
            backdrop-filter: blur(5px);
        ">
            <strong style="color: #ef4444;">❌ {error_type}</strong><br>
            <span style="color: #6b7280;">{message}</span>
            {suggestion_html}
        </div>
        """
    
    @staticmethod
    def create_success_message(message: str, duration_ms: Optional[float] = None) -> str:
        """Create formatted success message"""
        
        duration_html = ""
        if duration_ms:
            duration_html = f"<span style='color: #9ca3af;'> • Completed in {duration_ms:.0f}ms</span>"
        
        return f"""
        <div style="
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(34, 197, 94, 0.05) 100%);
            border: 1px solid rgba(34, 197, 94, 0.3);
            border-radius: 8px;
            padding: 1.25rem;
            backdrop-filter: blur(5px);
        ">
            <strong style="color: #22c55e;">✅ Success</strong>{duration_html}<br>
            <span style="color: #6b7280;">{message}</span>
        </div>
        """
    
    @staticmethod
    def create_loading_message(message: str = "Processing...") -> str:
        """Create formatted loading message"""
        
        return f"""
        <div style="
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
            border: 1px solid rgba(59, 130, 246, 0.3);
            border-radius: 8px;
            padding: 1.25rem;
            backdrop-filter: blur(5px);
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        ">
            <strong style="color: #3b82f6;">⏳ Loading</strong><br>
            <span style="color: #6b7280;">{message}</span>
        </div>
        """
    
    @staticmethod
    def create_comparison_table(
        rows: List[Dict[str, str]],
        highlight_col: Optional[int] = None
    ) -> str:
        """Create formatted comparison table"""
        
        if not rows or not rows[0]:
            return ""
        
        header_keys = list(rows[0].keys())
        
        html = """
        <table style="
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        ">
        """
        
        # Header
        html += "<thead><tr style='border-bottom: 2px solid #667eea;'>"
        for i, key in enumerate(header_keys):
            style = "background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.05) 100%);" if i == highlight_col else ""
            html += f"<th style='padding: 0.75rem; text-align: left; font-weight: 600; color: #667eea; {style}'>{key}</th>"
        html += "</tr></thead>"
        
        # Body
        html += "<tbody>"
        for row in rows:
            html += "<tr style='border-bottom: 1px solid rgba(102, 126, 234, 0.1);'>"
            for i, key in enumerate(header_keys):
                style = "background: rgba(102, 126, 234, 0.05);" if i == highlight_col else ""
                html += f"<td style='padding: 0.75rem; color: #6b7280; {style}'>{row.get(key, '')}</td>"
            html += "</tr>"
        html += "</tbody>"
        
        html += "</table>"
        return html
    
    @staticmethod
    def display_with_streamlit(
        title: str,
        answer: str,
        sources: List[str],
        confidence: float,
        latency_ms: float,
        documents: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Display formatted response using Streamlit"""
        
        # Title
        st.markdown(f"### {title}", unsafe_allow_html=True)
        
        # Answer card
        card_html = ResponseFormatter.create_answer_card(
            answer,
            sources,
            confidence,
            latency_ms
        )
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Documents (if provided)
        if documents:
            st.markdown("---")
            st.markdown("#### 📚 Source Documents")
            
            for doc in documents:
                doc_html = ResponseFormatter.create_document_snippet(
                    title=doc.get("title", "Untitled"),
                    content=doc.get("content", ""),
                    url=doc.get("url", "#"),
                    score=doc.get("score", 0.0),
                    source_type=doc.get("source", "Web")
                )
                st.markdown(doc_html, unsafe_allow_html=True)


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def display_answer(
    answer: str,
    sources: List[str],
    confidence: float,
    latency_ms: float
) -> None:
    """Quick function to display answer"""
    formatter = ResponseFormatter()
    card = formatter.create_answer_card(answer, sources, confidence, latency_ms)
    st.markdown(card, unsafe_allow_html=True)


def display_error(error_type: str, message: str, suggestion: Optional[str] = None) -> None:
    """Quick function to display error"""
    formatter = ResponseFormatter()
    error_html = formatter.create_error_message(error_type, message, suggestion)
    st.markdown(error_html, unsafe_allow_html=True)


def display_success(message: str, duration_ms: Optional[float] = None) -> None:
    """Quick function to display success"""
    formatter = ResponseFormatter()
    success_html = formatter.create_success_message(message, duration_ms)
    st.markdown(success_html, unsafe_allow_html=True)


def display_document(
    title: str,
    content: str,
    url: str,
    score: float
) -> None:
    """Quick function to display document"""
    formatter = ResponseFormatter()
    doc_html = formatter.create_document_snippet(title, content, url, score)
    st.markdown(doc_html, unsafe_allow_html=True)
