import streamlit as st
import time
import uuid
import os
import plotly.express as px
import pandas as pd
import streamlit.components.v1 as components
from xhtml2pdf import pisa
from io import BytesIO
import base64
import markdown
import plotly.io as pio
import cairosvg
import requests
import zlib
import urllib.parse
import json
import nest_asyncio
nest_asyncio.apply()
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from agents import (
    ResearcherAgent, AnalystAgent, FactCheckerAgent, BiasDetectorAgent,
    CitationVerifierAgent, WriterAgent, QualityControlAgent, GapAnalystAgent,
    SynthesizerAgent, FormatterAgent, PlannerAgent, ChatAgent,
    get_token_usage, reset_token_usage, get_agent_token_breakdown
)
from graph import build_research_graph, ResearchState
import experiment_logger


def sync_astream(graph, *args, **kwargs):
    """Iterate a LangGraph async stream synchronously via nest_asyncio."""
    loop = asyncio.get_event_loop()
    aiter = graph.astream(*args, **kwargs).__aiter__()
    while True:
        try:
            yield loop.run_until_complete(aiter.__anext__())
        except StopAsyncIteration:
            break


# Page Config
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Enhanced Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1d3a 50%, #0a0e27 100%);
        color: #e8eaf6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    /* Enhanced Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 12px 32px;
        border: none;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }

    /* Report Container with Glassmorphism */
    .report-container {
        background: rgba(38, 39, 48, 0.7);
        backdrop-filter: blur(10px);
        padding: 2.5rem;
        border-radius: 16px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        line-height: 1.8;
    }

    /* Gradient Headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    h2, h3 {
        color: #b0b8ff;
        font-weight: 600;
    }

    /* Agent Status Cards */
    .agent-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .agent-card:hover {
        transform: translateX(5px);
        border-color: rgba(102, 126, 234, 0.6);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }

    /* Progress Bar Enhancement */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    }

    /* Input Fields */
    .stTextInput>div>div>input {
        background-color: rgba(38, 39, 48, 0.6);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        color: #e8eaf6;
        padding: 12px;
        font-size: 15px;
    }

    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }

    /* Sidebar Enhancement */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(26, 29, 58, 0.95) 0%, rgba(10, 14, 39, 0.95) 100%);
        border-right: 1px solid rgba(102, 126, 234, 0.2);
    }

    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.3);
        margin: 0.5rem 0;
    }

    /* Success/Warning/Error Messages */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 10px;
        padding: 1rem;
        backdrop-filter: blur(10px);
    }

    /* Chat Messages */
    .stChatMessage {
        background: rgba(38, 39, 48, 0.5);
        border-radius: 12px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(38, 39, 48, 0.6);
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.2);
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(38, 39, 48, 0.3);
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# Session State - Enhanced with more tracking
if "report" not in st.session_state:
    st.session_state.report = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "svg_image" not in st.session_state:
    st.session_state.svg_image = None
if "research_history" not in st.session_state:
    st.session_state.research_history = []
if "agent_status" not in st.session_state:
    st.session_state.agent_status = {}
if "current_context" not in st.session_state:
    st.session_state.current_context = None
if "error_logs" not in st.session_state:
    st.session_state.error_logs = []
if "theme" not in st.session_state:
    st.session_state.theme = "dark"  # Default dark theme
if "token_usage" not in st.session_state:
    st.session_state.token_usage = {"total": 0, "sessions": []}
if "agent_metrics" not in st.session_state:
    st.session_state.agent_metrics = {}
if "research_data" not in st.session_state:
    st.session_state.research_data = None
if "references_section" not in st.session_state:
    st.session_state.references_section = ""
if "grounding_stats" not in st.session_state:
    st.session_state.grounding_stats = {}
if "token_stats" not in st.session_state:
    st.session_state.token_stats = {}
# HITL state
if "hitl_pending" not in st.session_state:
    st.session_state.hitl_pending = False
if "hitl_thread_config" not in st.session_state:
    st.session_state.hitl_thread_config = None
if "hitl_running_state" not in st.session_state:
    st.session_state.hitl_running_state = {}
if "hitl_context" not in st.session_state:
    st.session_state.hitl_context = {}
if "hitl_agent_objects" not in st.session_state:
    st.session_state.hitl_agent_objects = {}
if "ready_to_write" not in st.session_state:
    st.session_state.ready_to_write = False
if "research_start_time" not in st.session_state:
    st.session_state.research_start_time = 0.0
# Helper: Citation Formatters


def format_citation(source, citation_format="APA 7th"):
    """Format a citation based on the selected style"""
    if citation_format == "APA 7th":
        return format_apa_citation(source)
    elif citation_format == "MLA 9th":
        return format_mla_citation(source)
    elif citation_format == "IEEE":
        return format_ieee_citation(source)
    elif citation_format == "Chicago":
        return format_chicago_citation(source)
    elif citation_format == "Harvard":
        return format_harvard_citation(source)
    else:
        return format_apa_citation(source)  # Default to APA


def format_apa_citation(source):
    """Format citation in APA 7th edition style"""
    source_type = source.get('type', 'web')

    if source_type == 'arxiv':
        authors = source.get('authors', [])
        if len(authors) == 1:
            author_text = authors[0]
        elif len(authors) == 2:
            author_text = f"{authors[0]} & {authors[1]}"
        elif len(authors) > 2:
            author_text = f"{authors[0]} et al."
        else:
            author_text = "Unknown Author"

        year = source.get('published', 'n.d.')[:4]
        title = source.get('title', 'Untitled')
        url = source.get('url', '')

        return f"{author_text}. ({year}). *{title}*. arXiv. {url}"

    elif source_type == 'web':
        title = source.get('title', 'Untitled')
        url = source.get('href', source.get('url', ''))
        year = source.get('year', 'n.d.')

        return f"{title}. ({year}). Retrieved from {url}"

    elif source_type == 'wikipedia':
        title = source.get('title', 'Wikipedia')
        year = source.get('year', 'n.d.')

        return f"Wikipedia contributors. ({year}). *{title}*. Wikipedia. https://en.wikipedia.org"

    return f"{source.get('title', 'Unknown source')}. Retrieved from {source.get('url', 'N/A')}"


def format_mla_citation(source):
    """Format citation in MLA 9th edition style"""
    source_type = source.get('type', 'web')

    if source_type == 'arxiv':
        authors = source.get('authors', [])
        if len(authors) >= 1:
            author_text = f"{authors[0]}"
            if len(authors) > 1:
                author_text += ", et al"
        else:
            author_text = "Unknown Author"

        title = source.get('title', 'Untitled')
        year = source.get('published', 'n.d.')[:4]
        url = source.get('url', '')

        return f'{author_text}. "{title}." *arXiv*, {year}, {url}.'

    elif source_type == 'web':
        title = source.get('title', 'Untitled')
        url = source.get('href', source.get('url', ''))

        return f'"{title}." Web. {url}.'

    elif source_type == 'wikipedia':
        title = source.get('title', 'Wikipedia')

        return f'"Wikipedia contributors. "{title}." *Wikipedia*, https://en.wikipedia.org.'

    return f'"{source.get("title", "Unknown source")}." {source.get("url", "N/A")}.'


def format_ieee_citation(source):
    """Format citation in IEEE style"""
    source_type = source.get('type', 'web')

    if source_type == 'arxiv':
        authors = source.get('authors', [])
        if len(authors) == 1:
            author_text = authors[0]
        elif len(authors) == 2:
            author_text = f"{authors[0]} and {authors[1]}"
        elif len(authors) > 2:
            author_text = f"{authors[0]} et al."
        else:
            author_text = "Unknown"

        title = source.get('title', 'Untitled')
        year = source.get('published', 'n.d.')[:4]
        url = source.get('url', '')

        return f'{author_text}, "{title}," arXiv, {year}. [Online]. Available: {url}'

    elif source_type == 'web':
        title = source.get('title', 'Untitled')
        url = source.get('href', source.get('url', ''))

        return f'"{title}." [Online]. Available: {url}'

    return f'"{source.get("title", "Unknown")}," [Online]. Available: {source.get("url", "N/A")}'


def format_chicago_citation(source):
    """Format citation in Chicago style"""
    source_type = source.get('type', 'web')

    if source_type == 'arxiv':
        authors = source.get('authors', [])
        if len(authors) >= 1:
            author_text = authors[0]
        else:
            author_text = "Unknown Author"

        title = source.get('title', 'Untitled')
        year = source.get('published', 'n.d.')[:4]
        url = source.get('url', '')

        return f'{author_text}. "{title}." arXiv ({year}). {url}.'

    elif source_type == 'web':
        title = source.get('title', 'Untitled')
        url = source.get('href', source.get('url', ''))

        return f'"{title}." Accessed via {url}.'

    return f'{source.get("title", "Unknown source")}. {source.get("url", "N/A")}.'


def format_harvard_citation(source):
    """Format citation in Harvard style"""
    source_type = source.get('type', 'web')

    if source_type == 'arxiv':
        authors = source.get('authors', [])
        if len(authors) >= 1:
            author_text = authors[0]
            if len(authors) > 1:
                author_text += " et al."
        else:
            author_text = "Unknown"

        year = source.get('published', 'n.d.')[:4]
        title = source.get('title', 'Untitled')
        url = source.get('url', '')

        return f"{author_text} ({year}) *{title}*, arXiv. Available at: {url}"

    elif source_type == 'web':
        title = source.get('title', 'Untitled')
        url = source.get('href', source.get('url', ''))

        return f"*{title}*. Available at: {url}"

    return f'{source.get("title", "Unknown")}. Available at: {source.get("url", "N/A")}'


# Helper: Mermaid Renderer (Web)


def mermaid(code):
    html_code = f"""
    <div class="mermaid">
    {code}
    </div>
    <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true, theme: 'dark' }});
    </script>
    """
    components.html(html_code, height=400, scrolling=True)

# Helper: Mermaid to Image (for PDF)


def get_mermaid_image(code):
    """Generates Mermaid diagrams using a robust POST request to avoid URL limits."""
    # Clean the code just in case the LLM wrapped it in markdown
    code = code.replace("```mermaid", "").replace("```", "").strip()
    
    try:
        # Most reliable method: POST raw text to Kroki
        response = requests.post(
            "https://kroki.io/mermaid/png", 
            data=code.encode('utf-8'),
            headers={'Content-Type': 'text/plain'},
            timeout=15
        )
        if response.status_code == 200:
            return base64.b64encode(response.content).decode('utf-8')
        else:
            print(f"Kroki API failed with status {response.status_code}")
    except Exception as e:
        print(f"Mermaid conversion failed: {e}")
    return None

# Helper: SVG to PNG (for PDF)


def get_svg_image(svg_code):
    try:
        png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
        return base64.b64encode(png_data).decode('utf-8')
    except Exception as e:
        print(f"SVG conversion failed: {e}")
        return None

# Helper: Generate Chart Image (Base64)


def get_chart_image(viz):
    """Generates charts with Kaleido, falling back to QuickChart POST API."""
    try:
        df = pd.DataFrame({
            'Label': viz.get('labels', []),
            'Value': viz.get('values', [])
        })
        chart_type = viz.get('chart_type', 'bar')

        if chart_type == 'line': fig = px.line(df, x='Label', y='Value', title=viz.get('title'))
        elif chart_type == 'pie': fig = px.pie(df, names='Label', values='Value', title=viz.get('title'))
        elif chart_type == 'scatter': fig = px.scatter(df, x='Label', y='Value', title=viz.get('title'))
        elif chart_type == 'area': fig = px.area(df, x='Label', y='Value', title=viz.get('title'))
        else: fig = px.bar(df, x='Label', y='Value', title=viz.get('title'))

        # Attempt 1: Local Kaleido render
        try:
            img_bytes = pio.to_image(fig, format='png', width=600, height=400, scale=2)
            return base64.b64encode(img_bytes).decode('utf-8')
        except Exception as kaleido_err:
            print(f"Kaleido engine crashed ({kaleido_err}). Falling back to QuickChart POST API...")
            
            # Attempt 2: API Fallback via POST (Fixes JSON error & URL limits)
            qc_type = chart_type if chart_type in ['bar', 'line', 'pie', 'scatter'] else 'bar'
            chart_config = {
                "type": qc_type,
                "data": {
                    "labels": viz.get('labels', []),
                    "datasets": [{"label": "Value", "data": viz.get('values', []), "backgroundColor": "#667eea"}]
                },
                "options": {
                    "plugins": {"title": {"display": True, "text": viz.get('title', '')}}
                }
            }
            
            response = requests.post(
                "https://quickchart.io/chart",
                json={
                    "chart": chart_config,
                    "width": 600,
                    "height": 400,
                    "format": "png",
                    "backgroundColor": "white"
                },
                timeout=15
            )
            
            if response.status_code == 200:
                return base64.b64encode(response.content).decode('utf-8')
            else:
                print(f"QuickChart API Error: {response.text}")
                
    except Exception as e:
        print(f"Chart generation completely failed: {e}")
    return None

# Helper: PDF Generator
def create_pdf(report_text, analysis_data, svg_code, topic="Research Report", references_text=""):
    # Convert Markdown to HTML and split by sections
    report_sections = report_text.split('##')

    # Filter visualizations
    viz_list = []
    if analysis_data and analysis_data.get('visualizations'):
        for viz in analysis_data['visualizations']:
            title = viz.get('title', '').lower()
            description = viz.get('description', '').lower()
            skip_keywords = ['publication', 'timeline', 'year', 'research areas', 'paper', 'distribution', 'author', 'source']
            if not any(keyword in title or keyword in description for keyword in skip_keywords):
                viz_list.append(viz)

    content_html = ""
    viz_index = 0

    if svg_code:
        b64_svg = get_svg_image(svg_code)
        if b64_svg:
            content_html += f"""<div align="center"><img src="data:image/png;base64,{b64_svg}" width="150"/></div>"""

    for idx, section in enumerate(report_sections):
        if section.strip():
            section_text = f"##{section}" if idx > 0 else section
            section_html = markdown.markdown(section_text, extensions=['tables', 'fenced_code'])
            content_html += section_html

            # Insert visualization
            if viz_index < len(viz_list) and idx > 0 and idx <= len(viz_list):
                viz = viz_list[viz_index]
                content_html += f"""
                <div style="text-align: center; border: 1px solid #ddd; padding: 10px; margin-bottom: 15px;">
                    <h3>📊 {viz.get('title')}</h3>
                    <p style='font-size: 10pt; color: #555;'><i>{viz.get('description')}</i></p>
                """

                if viz.get('type') == 'chart':
                    b64_img = get_chart_image(viz)
                    if b64_img:
                        # Fixed for xhtml2pdf compatibility (removed complex CSS, used align/width)
                        content_html += f"""<br/><img src="data:image/png;base64,{b64_img}" width="450" align="middle"/>"""
                    else:
                        content_html += f"<p style='color:red;'><i>[Chart generation failed. Check Kaleido dependency.]</i></p>"

                elif viz.get('type') == 'mermaid':
                    b64_mermaid = get_mermaid_image(viz.get('code'))
                    if b64_mermaid:
                        content_html += f"""<br/><img src="data:image/png;base64,{b64_mermaid}" width="450" align="middle"/>"""
                    else:
                        content_html += f"<p><i>[Diagram rendering unavailable]</i></p>"

                content_html += "</div>"
                viz_index += 1

    if references_text:
        ref_html = markdown.markdown(references_text, extensions=['tables', 'fenced_code'])
        content_html += f"<div style='page-break-before: always;'></div>"
        content_html += ref_html

    full_html = f"""
    <html>
    <head>
        <style>
            @page {{ size: A4; margin: 2cm; }}
            body {{ font-family: Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.5; color: #333; }}
            h1 {{ color: #1a237e; font-size: 20pt; border-bottom: 2px solid #1a237e; padding-bottom: 5px; }}
            h2 {{ color: #283593; font-size: 16pt; margin-top: 15px; border-bottom: 1px solid #eee; }}
            h3 {{ color: #303f9f; font-size: 13pt; margin-top: 10px; }}
            p {{ margin-bottom: 8px; text-align: justify; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; font-size: 10pt; }}
            th, td {{ border: 1px solid #ccc; padding: 6px; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Research Report: {topic.title()}</h1>
        {content_html}
    </body>
    </html>
    """

    pdf_file = BytesIO()
    pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)
    if pisa_status.err:
        return None
    return pdf_file.getvalue()

# Header with Enhanced Layout
col_title, col_stats = st.columns([2, 1])

with col_title:
    st.title("🔬 Multi-Agent Research System")
    st.markdown(
        "### Deploy a team of AI agents to research any topic comprehensively"
    )

with col_stats:
    if st.session_state.research_history:
        total_research = len(st.session_state.research_history)
        st.markdown(f"""
        <div class='metric-card'>
            <h4 style='margin:0; color: #667eea;'>📊 Statistics</h4>
            <p style='font-size: 24px; font-weight: 700; margin: 5px 0;'>{total_research}</p>
            <p style='font-size: 14px; color: #9fa8da; margin:0;'>Research Reports</p>
        </div>
        """, unsafe_allow_html=True)

# Sidebar with Enhanced Features
with st.sidebar:
    st.header("⚙️ Configuration")

    # Report Configuration
    with st.expander("📋 Report Settings", expanded=True):
        report_type = st.selectbox(
            "Report Type",
            ["Comprehensive Report", "Literature Review",
                "Market Analysis", "Technical Deep Dive", "Comparative Analysis"]
        )

        report_length = st.select_slider(
            "Report Depth",
            options=["Concise", "Standard", "Detailed", "Exhaustive"],
            value="Standard"
        )

        include_citations = st.checkbox("Include Citations", value=True)

        citation_format = st.selectbox(
            "Citation Format",
            ["APA 7th", "MLA 9th", "IEEE", "Chicago", "Harvard"],
            index=0,
            help="Select citation style for references"
        )

        include_visualizations = st.checkbox("Generate Visualizations", value=True)
        enable_hitl = st.checkbox("Enable Human-in-the-Loop (HITL)", value=True, help="Pause after research to allow manual review and prompt editing before generating the final report.")

    st.divider()

    # Knowledge Base
    with st.expander("📂 Knowledge Base", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload PDFs for Context",
            type=["pdf"],
            accept_multiple_files=True,
            help="Add your own research papers or documents"
        )

        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded")

    st.divider()

    # Research History
    with st.expander("📚 Research History", expanded=False):
        if st.session_state.research_history:
            st.markdown("#### Recent Research")
            for idx, item in enumerate(reversed(st.session_state.research_history[-5:])):
                if st.button(f"📄 {item['topic'][:30]}...", key=f"history_{idx}"):
                    st.session_state.report = item.get('report')
                    st.session_state.analysis = item.get('analysis')
        else:
            st.info("No research history yet")

    st.divider()

    # System Info
    _langsmith_active = bool(os.getenv("LANGCHAIN_API_KEY"))
    _langsmith_badge  = "🟢 LangSmith Active" if _langsmith_active else "⚪ LangSmith Off"
    st.markdown(f"""
    <div style='padding: 1rem; background: rgba(102, 126, 234, 0.1); border-radius: 10px; border: 1px solid rgba(102, 126, 234, 0.3);'>
        <h4 style='margin: 0 0 10px 0; color: #667eea;'>🚀 System Features</h4>
        <ul style='margin: 0; padding-left: 20px; font-size: 13px;'>
            <li>Multi-Agent AI Research</li>
            <li>LangGraph QC Loop</li>
            <li>Human-in-the-Loop Review</li>
            <li>Token Cost Tracking</li>
            <li>Dynamic Visualizations</li>
            <li>PDF/JSON/MD Export</li>
            <li>Source Grounding</li>
            <li>Bias & Fact Checking</li>
        </ul>
        <hr style='border-color: rgba(102,126,234,0.2); margin: 8px 0;'>
        <span style='font-size: 12px; color: #9fa8da;'>{_langsmith_badge}</span>
    </div>
    """, unsafe_allow_html=True)

# Main Input
col1, col2 = st.columns([3, 1])
with col1:
    topic = st.text_input("Enter Research Topic",
                          placeholder="e.g., The Future of Solid State Batteries")
with col2:
    st.write("")  # Spacer
    st.write("")
    start_btn = st.button("Start Research", width="stretch")

# Main Research Logic
if start_btn and topic:
    # Clear previous state
    st.session_state.report = None
    st.session_state.analysis = None
    st.session_state.chat_history = []
    st.session_state.svg_image = None
    st.session_state.references_section = ""
    st.session_state.grounding_stats = {}
    st.session_state.hitl_pending = False
    st.session_state.ready_to_write = False
    st.session_state.research_start_time = time.time()
    reset_token_usage()
    experiment_logger.start_run(topic, {
        "report_length": report_length,
        "citation_format": citation_format,
        "enable_hitl": enable_hitl,
    })

    researcher_obj = ResearcherAgent()
    analyst_obj = AnalystAgent()
    gap_obj = GapAnalystAgent()
    bias_obj = BiasDetectorAgent()
    fact_obj = FactCheckerAgent()
    synth_obj = SynthesizerAgent()
    qc_obj = QualityControlAgent()
    planner_obj = PlannerAgent()
    fmt_obj = FormatterAgent()

    display_agents = [
        researcher_obj, analyst_obj,
        gap_obj, bias_obj, fact_obj,
        synth_obj, qc_obj, planner_obj, fmt_obj
    ]

    st.markdown("---")
    st.subheader("🤖 Agent Orchestration Dashboard")

    agent_cols = st.columns(4)
    agent_status_placeholders = {}
    for idx, agent in enumerate(display_agents):
        with agent_cols[idx % 4]:
            agent_status_placeholders[agent.name] = st.empty()
            agent_status_placeholders[agent.name].markdown(f"""
            <div class='agent-card' style='opacity: 0.5;'>
                <div style='font-size: 24px; text-align: center;'>{agent.icon}</div>
                <div style='font-size: 12px; text-align: center; margin-top: 8px;'>{agent.name}</div>
                <div style='font-size: 10px; text-align: center; color: #9fa8da;'>Pending</div>
            </div>
            """, unsafe_allow_html=True)

    log_container = st.expander("📋 View Live Agent Logs", expanded=False)

    context = {
        "topic": topic,
        "type": report_type,
        "uploaded_files": uploaded_files,
        "report_length": report_length,
        "include_citations": include_citations,
        "citation_format": citation_format,
        "include_visualizations": include_visualizations
    }
    st.session_state.current_context = context

    def mark_working(agent):
        agent_status_placeholders[agent.name].markdown(f"""
        <div class='agent-card' style='border-color: #667eea; box-shadow: 0 0 20px rgba(102,126,234,0.4);'>
            <div style='font-size: 24px; text-align: center;'>{agent.icon}</div>
            <div style='font-size: 12px; text-align: center; margin-top: 8px; font-weight: 600;'>{agent.name}</div>
            <div style='font-size: 10px; text-align: center; color: #667eea;'>⚡ Working...</div>
        </div>
        """, unsafe_allow_html=True)

    def mark_done(agent, success=True):
        color, label = ("#4ade80", "✓ Complete") if success else ("#f87171", "✗ Error")
        agent_status_placeholders[agent.name].markdown(f"""
        <div class='agent-card' style='border-color: {color};'>
            <div style='font-size: 24px; text-align: center;'>{agent.icon}</div>
            <div style='font-size: 12px; text-align: center; margin-top: 8px;'>{agent.name}</div>
            <div style='font-size: 10px; text-align: center; color: {color};'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    initial_state = ResearchState(
        topic=topic, report_config=context, research_data={}, research_chunks={},
        analysis={}, bias_report="", validation_report="", gaps="",
        qc_score=0, qc_passed=True, qc_feedback="", qc_iterations=0,
        planner_strategy={}, logs=[]
    )

    node_agent_map = {
        "researcher": researcher_obj, "analyst": analyst_obj,
        "parallel_critics": [gap_obj, bias_obj, fact_obj],
        "synthesizer": synth_obj, "qc": qc_obj, "planner": planner_obj, "formatter": fmt_obj
    }

    thread_id = str(uuid.uuid4())
    thread_config = {"configurable": {"thread_id": thread_id}}
    qc_loop_count = 0
    running_state = dict(initial_state)
    running_state["logs"] = []

    # DYNAMIC GRAPH COMPILATION
    research_graph = build_research_graph(interrupt=enable_hitl)

    with st.status("🤖 Deploying AI Research Team...", expanded=True) as status:
        for event in sync_astream(research_graph, initial_state, config=thread_config, stream_mode="updates"):
            for node_name, updates in event.items():
                if isinstance(updates, tuple): updates = updates[0] if updates else {}

                for k, v in updates.items():
                    if k == "logs": running_state["logs"] = running_state.get("logs", []) + v
                    else: running_state[k] = v

                agent_or_agents = node_agent_map.get(node_name)
                if isinstance(agent_or_agents, list):
                    for a in agent_or_agents: mark_done(a)
                elif agent_or_agents: mark_done(agent_or_agents)

                if node_name == "qc":
                    experiment_logger.record_qc_event(
                        score=updates.get("qc_score", 0),
                        passed=updates.get("qc_passed", True),
                        iteration=running_state.get("qc_iterations", 0),
                    )
                if node_name == "qc" and not updates.get("qc_passed", True):
                    qc_loop_count += 1
                    status.update(label=f"⚠️ QC score {updates.get('qc_score')}/100 — retrying (attempt {qc_loop_count}/2)...")
                    for a in ([node_agent_map["researcher"], node_agent_map["analyst"]] + node_agent_map["parallel_critics"]):
                        mark_working(a)
                elif node_name == "qc":
                    status.update(label=f"✅ QC passed ({updates.get('qc_score')}/100) — finalizing...")
                else:
                    status.update(label=f"{node_name.replace('_', ' ').title()} complete...")

                if "logs" in updates:
                    with log_container:
                        for log in updates["logs"]: st.markdown(log)

        for pipeline_agent in [researcher_obj, analyst_obj, gap_obj, bias_obj, fact_obj, synth_obj, qc_obj, planner_obj, fmt_obj]:
            mark_done(pipeline_agent)

        context.update({k: v for k, v in running_state.items() if k != "logs"})
        st.session_state.analysis = running_state.get("analysis", {})
        st.session_state.token_stats = get_token_usage()

        if enable_hitl:
            status.update(label="⏸️ Awaiting human review — approve to generate report...", expanded=False)
        else:
            status.update(label="✅ Research complete. Moving to report generation...", expanded=False)

    if qc_loop_count > 0:
        st.info(f"🔄 QC triggered {qc_loop_count} research iteration(s). Final score: {running_state.get('qc_score')}/100")

    # DYNAMIC ROUTING: Pause for review OR skip straight to writing
    if enable_hitl:
        st.session_state.hitl_pending = True
        st.session_state.hitl_thread_config = thread_config
        st.session_state.hitl_running_state = dict(running_state)
        st.session_state.hitl_context = dict(context)
        st.rerun() 
    else:
        st.session_state.hitl_pending = False
        st.session_state.hitl_context = dict(context)
        st.session_state.ready_to_write = True
        st.rerun() # Refresh to trigger the writer block smoothly


# ── Human-in-the-Loop Review Panel ───────────────────────────────────────────
if st.session_state.hitl_pending and not st.session_state.report:
    rs  = st.session_state.hitl_running_state
    ctx = st.session_state.hitl_context

    st.markdown("---")
    st.subheader("🧑‍💼 Human-in-the-Loop Review")
    st.markdown("The AI research team has completed its work. Review the findings below, then **approve** to generate the full report — or add instructions and **revise**.")

    analysis, research = rs.get("analysis", {}), rs.get("research_data", {})
    themes, summary = analysis.get("key_themes", []), analysis.get("summary", "No summary available.")
    gaps, bias = rs.get("gaps", "No gap analysis available."), rs.get("bias_report", "No bias report available.")
    n_arxiv, n_web = len(research.get("arxiv", [])), len(research.get("web", []))
    wiki_ok, n_chunks = bool(research.get("wiki")), len(rs.get("research_chunks", {}))
    qc_score = rs.get("qc_score", 0)
    
    ts = st.session_state.token_stats
    est_cost, total_tokens = ts.get("estimated_cost_usd", 0.0), ts.get("total_tokens", 0)

    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.markdown(f"**📝 Research Summary**\n\n{summary}")
        if themes:
            st.markdown("**🔑 Key Themes**")
            for t in themes: st.markdown(f"- {t}")
    with col_r:
        st.markdown(f"""
        <div class='metric-card' style='text-align:center;'>
            <div style='font-size:13px; color:#9fa8da;'>QC Score</div>
            <div style='font-size:32px; font-weight:700; color:{"#4ade80" if qc_score >= 85 else "#fbbf24"};'>{qc_score}</div>
            <div style='font-size:11px; color:#9fa8da;'>/ 100</div>
            <hr style='border-color:rgba(102,126,234,0.2); margin:8px 0;'>
            <div style='font-size:12px;'>💰 ${est_cost:.4f}</div>
            <div style='font-size:12px;'>🪙 {total_tokens:,} toks</div>
            <hr style='border-color:rgba(102,126,234,0.2); margin:8px 0;'>
            <div style='font-size:12px;'>📄 {n_arxiv} papers</div>
            <div style='font-size:12px;'>🌐 {n_web} web results</div>
            <div style='font-size:12px;'>📚 {"Wiki ✓" if wiki_ok else "Wiki ✗"}</div>
            <div style='font-size:12px;'>📎 {n_chunks} chunks</div>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 Gap Analysis", expanded=False): st.markdown(gaps)
    with st.expander("⚖️ Bias Assessment", expanded=False): st.markdown(bias)

    st.markdown("---")
    extra_instructions = st.text_area("Additional instructions for the Writer (optional)", placeholder="e.g., Focus more on regulatory implications. Add a section on cost comparison.", height=80)

    hitl_col1, hitl_col2, hitl_col3 = st.columns([1, 1, 1])
    approve_btn = hitl_col1.button("✅ Approve & Generate Report", type="primary")
    retry_btn   = hitl_col2.button("🔄 Feedback & Retry Research")
    cancel_btn  = hitl_col3.button("❌ Cancel")

    if cancel_btn:
        st.session_state.hitl_pending = False
        st.rerun()

    if retry_btn:
        if not extra_instructions.strip():
            st.warning("Please provide feedback/instructions for the retry.")
        else:
            research_graph = build_research_graph(interrupt=True)
            research_graph.update_state(st.session_state.hitl_thread_config, {"qc_feedback": extra_instructions, "qc_passed": False, "qc_iterations": 0}, as_node="qc")
            with st.status("🔄 Resuming research based on feedback...", expanded=True) as status:
                for event in sync_astream(research_graph, None, config=st.session_state.hitl_thread_config, stream_mode="updates"):
                    for node_name, updates in event.items():
                        if isinstance(updates, tuple): updates = updates[0] if updates else {}
                        if "logs" in updates:
                            for log in updates["logs"]: st.write(log)
            
            snapshot = research_graph.get_state(st.session_state.hitl_thread_config)
            new_state = snapshot.values
            st.session_state.hitl_running_state = new_state
            st.session_state.analysis = new_state.get("analysis", {})
            st.session_state.token_stats = get_token_usage()
            # Refresh hitl_context with new research data so the writer gets updated chunks
            refreshed_ctx = dict(st.session_state.hitl_context)
            for k in ("research_data", "research_chunks", "analysis", "bias_report",
                      "validation_report", "gaps", "qc_score", "qc_passed", "qc_feedback"):
                if k in new_state:
                    refreshed_ctx[k] = new_state[k]
            st.session_state.hitl_context = refreshed_ctx
            st.rerun()

    if approve_btn:
        thread_config = st.session_state.hitl_thread_config
        context       = dict(st.session_state.hitl_context)

        if extra_instructions.strip():
            context["extra_writer_instructions"] = extra_instructions.strip()

        with st.status("🎨 Formatter processing...", expanded=False) as fmt_status:
            research_graph = build_research_graph(interrupt=True)
            for event in sync_astream(research_graph, None, config=thread_config, stream_mode="updates"):
                pass
            fmt_status.update(label="🎨 Formatter complete.", state="complete")

        st.session_state.hitl_context = context
        st.session_state.hitl_pending = False
        st.session_state.ready_to_write = True
        st.rerun()

# ── Report Generation (Shared Flow for both HITL and Auto-Bypass) ────────────
if st.session_state.ready_to_write and not st.session_state.report:
    context = st.session_state.hitl_context
    writer = WriterAgent()
    
    st.markdown("---")
    st.markdown("#### 📡 Live Report Generation")
    stream_placeholder = st.empty()
    full_report = ""
    
    try:
        for chunk in writer.stream_task(context):
            full_report += chunk
            stream_placeholder.markdown(full_report + "▌")

        for fence in ("```markdown", "```"):
            if full_report.startswith(fence):
                full_report = full_report.replace(fence, "", 1)
                break
        if full_report.endswith("```"):
            full_report = full_report.rsplit("```", 1)[0]
        full_report = full_report.strip()

        stream_placeholder.empty()
        context["draft_report"] = full_report

        # ── Citation Verifier ─────────────────────────────────────────────
        with st.spinner("📝 Verifying citations and formatting document..."):
            citation_verifier = CitationVerifierAgent()
            cv_result = citation_verifier.perform_task(context)
            st.session_state.references_section = cv_result.get("references_section", "")
            full_report = cv_result.get("draft_report", full_report)
            
            st.session_state.grounding_stats = {
                "invalid": cv_result.get("invalid_citations_flagged", 0),
                "used":    cv_result.get("used_sources_count", 0),
                "total":   len(context.get("research_chunks", {}))
            }

        # ── Finalize State ────────────────────────────────────────────────
        st.session_state.token_stats = get_token_usage()
        st.session_state.report   = full_report
        st.session_state.analysis = context.get("analysis", {})
        
        total_time = time.time() - st.session_state.research_start_time
        experiment_logger.finish_run(
            token_stats=get_token_usage(),
            grounding_stats=st.session_state.grounding_stats,
            agent_breakdown=get_agent_token_breakdown(),
            research_chunks=context.get("research_chunks", {}),
            wall_time_s=total_time,
        )
        st.success(f"✅ Done in {total_time:.1f}s")

        current_topic = context.get("topic", "Research Report")
        st.session_state.research_history.append({
            "topic":     current_topic,
            "report":    full_report,
            "analysis":  st.session_state.analysis,
            "timestamp": time.time(),
            "duration":  total_time
        })

        st.session_state.ready_to_write = False
        # Notice: No st.rerun() here! This allows it to cleanly fall through and render the visual report below.

    except Exception as e:
        st.error(f"⚠️ Writer failed: {e}")
        st.session_state.ready_to_write = False


# Display Results
if st.session_state.report:
    st.divider()

    # Get topic from current context or use a default
    current_topic = st.session_state.current_context.get('topic', 'Research Report') if st.session_state.current_context else 'Research Report'

    # Grounding stats banner
    gs = st.session_state.grounding_stats
    ts = st.session_state.token_stats
    if gs:
        invalid = gs.get('invalid', 0)
        used    = gs.get('used', 0)
        total   = gs.get('total', 0)
        
        est_cost = ts.get("estimated_cost_usd", 0.0)
        total_tokens = ts.get("total_tokens", 0)
        
        badge_color = "#4ade80" if invalid == 0 else "#f87171"
        st.markdown(f"""
        <div style='background: rgba(102,126,234,0.1); border: 1px solid rgba(102,126,234,0.3);
                    border-radius: 10px; padding: 0.75rem 1.25rem; margin-bottom: 1rem;
                    display: flex; gap: 2rem; align-items: center; flex-wrap: wrap;'>
            <span>💰 <b>${est_cost:.4f}</b></span>
            <span>🪙 <b>{total_tokens:,}</b> tokens</span>
            <span style='border-left: 1px solid rgba(102,126,234,0.3); height: 20px; margin: 0 10px;'></span>
            <span>📎 <b>{total}</b> source chunks indexed</span>
            <span>🔗 <b>{used}</b> unique sources cited</span>
            <span style='color:{badge_color};'>{"✅" if invalid == 0 else "⚠️"} <b>{invalid}</b> hallucinated tag{"s" if invalid != 1 else ""} detected</span>
        </div>
        """, unsafe_allow_html=True)

    # Per-agent token breakdown
    from_breakdown = get_agent_token_breakdown()
    if from_breakdown:
        with st.expander("🪙 Token Cost Breakdown by Agent"):
            bd_cols = st.columns(min(len(from_breakdown), 4))
            for i, (agent_name, usage) in enumerate(from_breakdown.items()):
                with bd_cols[i % min(len(from_breakdown), 4)]:
                    st.metric(
                        agent_name,
                        f"{usage['total_tokens']:,} tok",
                        f"${usage['estimated_cost_usd']:.4f}"
                    )

    # 1. Report with integrated visualizations
    st.subheader(f"📄 Research Report: {current_topic.title()}")

    # Split the report into sections for better visualization integration
    report_sections = st.session_state.report.split('##')

    # Process visualizations - filter out unwanted charts
    viz_list = []
    if st.session_state.analysis and st.session_state.analysis.get('visualizations'):
        for viz in st.session_state.analysis['visualizations']:
            title = viz.get('title', '').lower()
            description = viz.get('description', '').lower()

            # Skip metadata/source-related charts
            skip_keywords = [
                'publication', 'timeline', 'year', 'research areas',
                'paper', 'distribution of research', 'author', 'source',
                'number of papers', 'document'
            ]

            should_skip = any(keyword in title or keyword in description for keyword in skip_keywords)

            if not should_skip:
                viz_list.append(viz)

    viz_index = 0

    # Display report with interspersed visualizations
    for idx, section in enumerate(report_sections):
        if section.strip():
            # Add the ## back for proper header formatting
            section_text = f"##{section}" if idx > 0 else section
            st.markdown(section_text, unsafe_allow_html=True)

            # Insert relevant visualization after key sections
            if viz_index < len(viz_list) and idx > 0 and idx <= len(viz_list):
                viz = viz_list[viz_index]

                st.markdown("---")
                st.markdown(f"**📊 {viz.get('title')}**")
                st.caption(viz.get('description', ''))

                if viz.get('type') == 'chart':
                    try:
                        df = pd.DataFrame({
                            'Label': viz['labels'],
                            'Value': viz['values']
                        })
                        chart_type = viz.get('chart_type', 'bar')

                        # Create chart
                        if chart_type == 'line':
                            fig = px.line(df, x='Label', y='Value', template='plotly_dark')
                        elif chart_type == 'pie':
                            fig = px.pie(df, names='Label', values='Value', template='plotly_dark')
                        elif chart_type == 'scatter':
                            fig = px.scatter(df, x='Label', y='Value', template='plotly_dark', size='Value')
                        elif chart_type == 'area':
                            fig = px.area(df, x='Label', y='Value', template='plotly_dark')
                        else:
                            fig = px.bar(df, x='Label', y='Value', template='plotly_dark')

                        fig.update_layout(
                            showlegend=True,
                            hovermode='x unified',
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#e8eaf6'),
                            height=400
                        )

                        st.plotly_chart(fig, key=f"inline_chart_{viz_index}", config={
                            'displayModeBar': True,
                            'displaylogo': False
                        })
                    except Exception as e:
                        st.caption(f"Chart unavailable: {e}")

                elif viz.get('type') == 'mermaid':
                    mermaid(viz.get('code'))

                st.markdown("---")
                viz_index += 1

    # Verified References Section
    if st.session_state.references_section:
        st.divider()
        st.markdown(st.session_state.references_section)

    # 3. Enhanced Export Options
    st.subheader("📥 Export Options")
    
    # CRITICAL FIX: Combine report and references for exports
    full_export_text = st.session_state.report
    if st.session_state.references_section:
        full_export_text += "\n\n" + st.session_state.references_section

    export_cols = st.columns(4)

    with export_cols[0]:
        if st.button("📄 Download PDF", width="stretch"):
            with st.spinner("Generating PDF..."):
                try:
                    pdf_bytes = create_pdf(
                        st.session_state.report,                 # Pass main report
                        st.session_state.analysis,
                        st.session_state.svg_image,
                        current_topic,
                        st.session_state.references_section      # Pass references explicitly
                    )
                    if pdf_bytes:
                        st.download_button(
                            label="💾 Save PDF",
                            data=pdf_bytes,
                            file_name=f"research_{current_topic.replace(' ', '_')[:30]}.pdf",
                            mime="application/pdf",
                            width="stretch"
                        )
                    else:
                        st.error("PDF Generation failed.")
                except Exception as e:
                    st.error(f"PDF Generation Error: {e}")

    with export_cols[1]:
        st.download_button(
            label="📝 Download MD",
            data=full_export_text,
            file_name=f"research_{current_topic.replace(' ', '_')[:30]}.md",
            mime="text/markdown",
            width="stretch"
        )

    with export_cols[2]:
        export_data = {
            "topic": current_topic,
            "report_content": full_export_text,
            "analysis": st.session_state.analysis,
            "timestamp": time.time()
        }
        json_content = json.dumps(export_data, indent=2)
        st.download_button(
            label="📊 Download JSON",
            data=json_content,
            file_name=f"research_{current_topic.replace(' ', '_')[:30]}.json",
            mime="application/json",
            width="stretch"
        )

    with export_cols[3]:
        txt_content = f"Research Report: {current_topic}\n\n{full_export_text}"
        st.download_button(
            label="📃 Download TXT",
            data=txt_content,
            file_name=f"research_{current_topic.replace(' ', '_')[:30]}.txt",
            mime="text/plain",
            width="stretch"
        )

    # 4. Chat Interface
    st.divider()
    st.subheader("💬 Chat with Researcher")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about this report..."):
        st.session_state.chat_history.append(
            {"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            chat_agent = ChatAgent()
            response = chat_agent.perform_task({
                "report": st.session_state.report,
                "question": prompt
            })
            st.markdown(response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": response})

elif start_btn and not topic:
    st.warning("Please enter a research topic.")
