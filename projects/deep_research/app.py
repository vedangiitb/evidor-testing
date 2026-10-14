"""Streamlit Frontend for Evidor Deep Research Agent with Arize Phoenix Telemetry."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys
import time
import urllib.parse

# Ensure repository root is on sys.path regardless of execution directory
_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

_project_dir = Path(__file__).resolve().parent
if str(_project_dir) not in sys.path:
    sys.path.insert(0, str(_project_dir))

from dotenv import load_dotenv
import streamlit as st

load_dotenv()

from evidor import GeminiProvider, OpenAIProvider

try:
    from projects.deep_research.engine import DeepResearchEngine
    from projects.deep_research.models import ResearchReport
    from projects.deep_research.search_provider import TavilySearchProvider, create_search_provider
except ModuleNotFoundError:
    from engine import DeepResearchEngine
    from models import ResearchReport
    from search_provider import TavilySearchProvider, create_search_provider

st.set_page_config(
    page_title="Evidor Deep Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .phoenix-badge {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .evidor-badge {
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 6px;
    }
    .langfuse-badge {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []
if "current_report" not in st.session_state:
    st.session_state.current_report = None
if "activity_logs" not in st.session_state:
    st.session_state.activity_logs = []

# Sidebar Configuration
with st.sidebar:
    st.title("⚙️ Research Settings")

    provider_choice = st.selectbox(
        "Model Provider",
        options=["Google Gemini", "OpenAI"],
        index=0,
    )

    if provider_choice == "Google Gemini":
        default_model = "gemini-2.5-flash"
        model_options = ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
    else:
        default_model = "gpt-4o-mini"
        model_options = ["gpt-4o-mini", "gpt-4o", "o3-mini"]

    model_name = st.selectbox("Model Identifier", options=model_options, index=0)

    st.markdown("---")
    st.subheader("🔍 Web Search (Tavily)")
    tavily_key = st.text_input(
        "Tavily API Key",
        value=os.getenv("TAVILY_API_KEY", ""),
        type="password",
        help="Reads TAVILY_API_KEY from environment/.env by default.",
    )
    max_tool_iterations = st.slider("Max Research Iterations", min_value=10, max_value=40, value=25, step=5)

    st.markdown("---")
    st.subheader("📊 Observability & Telemetry")
    enable_phoenix = st.checkbox("Enable Arize Phoenix Tracing", value=True)
    phoenix_endpoint = st.text_input(
        "Phoenix Collector Endpoint",
        value=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006/v1/traces"),
    )
    phoenix_project = st.text_input("Phoenix Project Name", value="deep-research")

    if enable_phoenix:
        st.markdown(
            """
            <div style="background-color: #FEF3C7; border-left: 4px solid #F59E0B; padding: 10px; border-radius: 4px; font-size: 0.85rem; color: #92400E;">
                <strong>Arize Phoenix Active:</strong> OpenInference traces for agent execution, LLM turns, and tool calls are captured in real time.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"[🔗 Open Phoenix Dashboard (Port 6006)](http://localhost:6006)")

    has_langfuse_keys = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
    enable_langfuse = st.checkbox("Enable Langfuse Tracing", value=has_langfuse_keys)
    langfuse_base = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
    if enable_langfuse:
        st.markdown(
            f"""
            <div style="background-color: #ECFDF5; border-left: 4px solid #10B981; padding: 10px; border-radius: 4px; font-size: 0.85rem; color: #065F46;">
                <strong>Langfuse Active:</strong> Agent runs, generations, and tool traces are streamed to Langfuse.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(f"[🔗 Open Langfuse Cloud]({langfuse_base})")

    # Saved reports browser in sidebar
    st.markdown("---")
    st.subheader("📂 Past Research Reports")
    reports_dir = Path(__file__).parent / "output" / "reports"
    alt_reports_dir = Path.cwd() / "output" / "reports"
    all_files: list[Path] = []
    if reports_dir.exists():
        all_files.extend(reports_dir.glob("*.md"))
    file_map = {f.name: f for f in all_files}
    saved_files = sorted(list(file_map.values()), key=lambda p: p.stat().st_mtime, reverse=True)
    if saved_files:
        selected_file = st.selectbox(
            "Load Saved Report",
            options=["-- Select --"] + [f.name for f in saved_files[:15]],
        )
        if selected_file != "-- Select --":
            chosen_path = file_map[selected_file]
            if st.button("Load Report"):
                content = chosen_path.read_text(encoding="utf-8")
                st.session_state.current_report = ResearchReport(
                    topic=selected_file.replace("report_", "").replace(".md", "").replace("_", " ").title(),
                    markdown_report=content,
                    saved_file_path=str(chosen_path),
                )
                st.rerun()
    else:
        st.caption("No past reports yet.")

# Main Header
st.markdown(
    """
    <div>
        <span class="evidor-badge">Evidor Agent Runtime</span>
        <span class="phoenix-badge">Arize Phoenix OpenInference</span>
        <span class="langfuse-badge">Langfuse Telemetry</span>
    </div>
    <div class="main-header">Autonomous Deep Research Agent</div>
    <div class="sub-header">
        Investigate complex engineering questions, cross-examine web sources, compare architectural trade-offs, and produce cited publication-grade reports.
    </div>
    """,
    unsafe_allow_html=True,
)

# Example Chips
st.markdown("**Quick Topics & Examples:**")
col_ex1, col_ex2, col_ex3 = st.columns(3)
example_query = ""

with col_ex1:
    if st.button("🐍 Python Agent Frameworks Trade-offs", use_container_width=True):
        example_query = "Compare the main Python agent frameworks and identify their trade-offs for production deployments."

with col_ex2:
    if st.button("🏗️ LangGraph vs CrewAI vs Evidor", use_container_width=True):
        example_query = "Compare LangGraph, CrewAI, AutoGen, and Evidor for building production multi-agent systems."

with col_ex3:
    if st.button("⚡ Vector Databases for Enterprise RAG", use_container_width=True):
        example_query = "Compare Qdrant, Pinecone, Milvus, and pgvector for enterprise-scale multi-tenant RAG."

# Topic Query Input
query_input = st.text_area(
    "Research Topic / Question:",
    value=example_query if example_query else (st.session_state.get("last_query") or "Compare the main Python agent frameworks and identify their trade-offs for production deployments."),
    height=100,
    help="Enter an engineering topic, comparison, or problem to research across the web.",
)

col_btn1, col_btn2 = st.columns([1, 4])
with col_btn1:
    run_btn = st.button("🚀 Start Deep Research", type="primary", use_container_width=True)

# Run Research Flow
if run_btn and query_input.strip():
    st.session_state.last_query = query_input.strip()
    st.session_state.activity_logs = []

    # Prepare provider
    if provider_choice == "Google Gemini":
        gemini_key = os.getenv("GEMINI_API_KEY")
        if not gemini_key:
            st.error("Missing `GEMINI_API_KEY` in environment or `.env`. Please provide it to run Gemini.")
            st.stop()
        provider_instance = GeminiProvider(model=model_name)
    else:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            st.error("Missing `OPENAI_API_KEY` in environment or `.env`. Please provide it to run OpenAI.")
            st.stop()
        provider_instance = OpenAIProvider(model=model_name)

    search_prov = create_search_provider(api_key=tavily_key.strip() if tavily_key.strip() else None)

    # Engine initialization
    engine = DeepResearchEngine(
        provider=provider_instance,
        model_name=model_name,
        search_provider=search_prov,
        enable_phoenix=enable_phoenix,
        phoenix_endpoint=phoenix_endpoint,
        phoenix_project_name=phoenix_project,
        enable_langfuse=enable_langfuse,
        max_tool_iterations=max_tool_iterations,
    )

    progress_placeholder = st.empty()
    activity_expander = st.expander("🔍 Live Research Activity Feed", expanded=True)
    activity_container = activity_expander.container()

    status_card = progress_placeholder.info(f"⏳ **Phase 1/4**: Formulating research angles and launching queries for '{query_input[:60]}...'")

    def ui_progress_callback(action: str, details: dict):
        if action == "start":
            activity_container.write(f"🚀 Initialized agent with model `{details.get('model')}`")
        elif action == "complete":
            activity_container.write(f"✅ Research synthesis finished in {details.get('elapsed', 0):.1f}s")

    try:
        with st.spinner("Agent is actively searching the web, reading primary sources, and cross-comparing claims..."):
            report = engine.research(query_input.strip(), progress_callback=ui_progress_callback)
            st.session_state.current_report = report

        if report.saved_file_path:
            progress_placeholder.success(f"✨ **Research Complete!** Full report synthesized and saved to `{report.saved_file_path}`.")
        else:
            progress_placeholder.success("✨ **Research Complete!** Full report synthesized.")

    except Exception as exc:
        progress_placeholder.error(f"❌ Error during research execution: {exc}")
    finally:
        engine.close()

# Display Current Report
report: ResearchReport | None = st.session_state.current_report

if report is not None:
    st.markdown("---")

    # Metrics Summary Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(report.sources)}</div>
                <div class="metric-label">Sources Read</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(report.findings)}</div>
                <div class="metric-label">Claims / Findings</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{len(report.comparisons)}</div>
                <div class="metric-label">Cross-Comparisons</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{report.telemetry.total_tool_calls}</div>
                <div class="metric-label">Tool Calls</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m_col5:
        tokens_display = f"{report.telemetry.total_tokens:,}" if report.telemetry.total_tokens else "N/A"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-value">{tokens_display}</div>
                <div class="metric-label">Tokens Used</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Result Tabs
    tab_report, tab_comparisons, tab_findings, tab_sources, tab_telemetry = st.tabs([
        "📄 Cited Research Report",
        "⚖️ Claim Comparisons & Trade-offs",
        "📌 Evidence & Findings",
        "🌐 Discovered Sources",
        "📊 Arize Phoenix Telemetry",
    ])

    with tab_report:
        col_down1, col_down2 = st.columns([4, 1])
        with col_down2:
            st.download_button(
                label="📥 Download Report (.md)",
                data=report.markdown_report,
                file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.markdown(report.markdown_report)

    with tab_comparisons:
        st.subheader("⚖️ Cross-Source Architectural Comparisons")
        if report.comparisons:
            for comp in report.comparisons:
                with st.container():
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #E2E8F0; border-radius: 8px; padding: 16px; margin-bottom: 12px; background: #FFFFFF;">
                            <div style="font-weight: 700; font-size: 1.1rem; color: #1E293B;">
                                🔍 {comp.topic_aspect} <span style="font-size: 0.8rem; background: #EEF2F6; padding: 2px 8px; border-radius: 4px; margin-left: 8px;">{comp.comparison_type.upper()}</span>
                            </div>
                            <div style="margin-top: 8px; display: flex; gap: 16px;">
                                <div style="flex: 1; background: #F8FAFC; padding: 10px; border-radius: 6px;">
                                    <strong>Source A:</strong> <code>{comp.source_a}</code><br>
                                    <em>"{comp.claim_a}"</em>
                                </div>
                                <div style="flex: 1; background: #F8FAFC; padding: 10px; border-radius: 6px;">
                                    <strong>Source B:</strong> <code>{comp.source_b}</code><br>
                                    <em>"{comp.claim_b}"</em>
                                </div>
                            </div>
                            <div style="margin-top: 10px; color: #334155;">
                                <strong>Analytical Synthesis:</strong> {comp.analysis}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("No explicit claim comparisons were recorded for this session. (The agent incorporated comparisons directly in the report text).")

    with tab_findings:
        st.subheader("📌 Extracted Evidence & Claims")
        if report.findings:
            for f in report.findings:
                st.markdown(
                    f"- **[{f.id}] ({f.category})**: {f.claim}\n"
                    f"  - **Source**: [{f.source_url}]({f.source_url})\n"
                    f"  - **Supporting Excerpt**: *\"{f.supporting_quote}\"*" if f.supporting_quote else ""
                )
        else:
            st.info("Findings are embedded in the main cited report.")

    with tab_sources:
        st.subheader("🌐 Analyzed Primary Sources")
        if report.sources:
            for idx, doc in enumerate(report.sources, 1):
                with st.expander(f"[{idx}] {doc.title or doc.url} ({doc.word_count} words)"):
                    st.markdown(f"**URL:** [{doc.url}]({doc.url})")
                    st.markdown(f"**Domain:** `{doc.domain}`")
                    st.text_area("Content Excerpt", value=doc.content[:1500] + ("..." if len(doc.content) > 1500 else ""), height=150)
        else:
            st.info("No full sources cached.")

    with tab_telemetry:
        st.subheader("📊 Observability & Phoenix OpenInference Tracing")
        st.markdown(
            """
            Evidor uses non-blocking actor workers to emit OpenInference spans for Agent Runs, LLM Turns, and Tool Invocations directly into **Arize Phoenix**.
            """
        )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            st.json(report.telemetry.to_dict())

        with col_t2:
            phoenix_url = report.telemetry.phoenix_url or "http://localhost:6006"
            st.markdown(
                f"""
                <div style="padding: 20px; border: 2px dashed #CBD5E1; border-radius: 8px; text-align: center;">
                    <h4>🚀 Live Arize Phoenix Tracing Dashboard</h4>
                    <p style="color: #64748B;">Inspect span latencies, token consumption, and parent-child tool call hierarchies in the Phoenix UI.</p>
                    <a href="{phoenix_url}" target="_blank" style="background: #2563EB; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: 600; display: inline-block;">
                        Open Phoenix Dashboard (Port 6006) ↗
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if report.telemetry.langfuse_enabled:
                langfuse_url = report.telemetry.langfuse_url or "https://cloud.langfuse.com"
                st.markdown(
                    f"""
                    <div style="padding: 20px; border: 2px dashed #10B981; border-radius: 8px; text-align: center; margin-top: 14px;">
                        <h4>⚡ Live Langfuse Cloud Traces</h4>
                        <p style="color: #64748B;">Inspect production LLM generations, tool executions, latency waterfalls, and cost attribution in Langfuse.</p>
                        <a href="{langfuse_url}" target="_blank" style="background: #10B981; color: white; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-weight: 600; display: inline-block;">
                            Open Langfuse Cloud ↗
                        </a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("Built with 💙 using Evidor, Arize Phoenix, and Streamlit.")

