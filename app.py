"""
FactCheck AI - Advanced Claim Verification System
Uses NLI + Semantic Similarity + Multi-Source Evidence
"""
import sys
from pathlib import Path

# Ensure imports work when running directly
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import streamlit as st
from src.scraper import fetch_evidence, get_entities
from src.analyzer import verify_claim, Verdict, get_analyzer

# Page configuration
st.set_page_config(
    page_title="FactCheck AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .verdict-true { 
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 20px; border-radius: 15px; color: white;
        text-align: center; font-size: 1.5em; font-weight: bold;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    .verdict-false { 
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 20px; border-radius: 15px; color: white;
        text-align: center; font-size: 1.5em; font-weight: bold;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
    }
    .verdict-mixed { 
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 20px; border-radius: 15px; color: white;
        text-align: center; font-size: 1.5em; font-weight: bold;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
    }
    .verdict-unknown { 
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        padding: 20px; border-radius: 15px; color: white;
        text-align: center; font-size: 1.5em; font-weight: bold;
        box-shadow: 0 4px 15px rgba(107, 114, 128, 0.3);
    }
    .evidence-card {
        background: #f8fafc; padding: 15px; border-radius: 10px;
        margin: 10px 0; border-left: 4px solid #3b82f6;
    }
    .metric-card {
        background: white; padding: 15px; border-radius: 10px;
        text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/3d-fluency/94/search.png", width=80)
    st.title("⚙️ Settings")
    
    st.markdown("---")
    
    use_all_sources = st.checkbox("🌐 Use all sources", value=True, 
                                   help="Include DuckDuckGo and Google Fact Check API")
    show_debug = st.checkbox("🔧 Show debug info", value=False)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.markdown("""
    - **NLI Model**: RoBERTa-large-MNLI
    - **Embedding**: all-MiniLM-L6-v2
    - **Sources**: Wikipedia, DuckDuckGo, Google Fact Check
    """)
    
    st.markdown("---")
    st.markdown("### 📖 How it works")
    st.markdown("""
    1. **Entity Extraction**: Identifies key entities in your claim
    2. **Evidence Gathering**: Scrapes multiple reliable sources
    3. **NLI Analysis**: Uses Natural Language Inference to detect entailment/contradiction
    4. **Semantic Matching**: Finds most relevant evidence
    5. **Ensemble Verdict**: Combines scores for final decision
    """)

# Main content
st.title("🔍 FactCheck AI")
st.markdown("### Advanced Claim Verification using NLI & Multi-Source Evidence")
st.markdown("---")

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'selected_example' not in st.session_state:
    st.session_state.selected_example = None

# Check if an example was selected
default_claim = ""
if st.session_state.selected_example:
    default_claim = st.session_state.selected_example
    st.session_state.selected_example = None  # Reset after use

# Pre-load models button
col1, col2 = st.columns([3, 1])
with col1:
    user_claim = st.text_area(
        "Enter a claim to verify:",
        value=default_claim,
        placeholder="Example: Barack Obama was born in Hawaii",
        height=100
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if not st.session_state.model_loaded:
        if st.button("🚀 Load Models", use_container_width=True):
            with st.spinner("Loading AI models... (first time may take a minute)"):
                try:
                    get_analyzer()
                    st.session_state.model_loaded = True
                    st.success("Models loaded!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading models: {e}")
    else:
        st.success("✅ Models Ready")

# Example claims
st.markdown("**Try these examples:**")
example_cols = st.columns(4)
examples = [
    "Barack Obama was born in Hawaii",
    "The Earth is flat",
    "Python was created by Guido van Rossum",
    "Einstein invented the telephone"
]
for i, example in enumerate(examples):
    with example_cols[i]:
        if st.button(example[:25] + "...", key=f"ex_{i}", use_container_width=True):
            st.session_state.selected_example = example
            st.rerun()

st.markdown("---")

# Verify button
if st.button("🔎 Verify Claim", type="primary", use_container_width=True, disabled=not user_claim):
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the models first by clicking 'Load Models'")
    elif user_claim:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Entity extraction
        status_text.text("🔍 Extracting entities from claim...")
        progress_bar.progress(10)
        entities = get_entities(user_claim)
        
        # Step 2: Evidence gathering
        status_text.text("📚 Gathering evidence from multiple sources...")
        progress_bar.progress(30)
        evidence_data = fetch_evidence(user_claim, use_all_sources=use_all_sources)
        
        if evidence_data:
            # Step 3: NLI Analysis
            status_text.text("🧠 Running NLI analysis...")
            progress_bar.progress(60)
            
            # Step 4: Verification
            status_text.text("⚖️ Computing final verdict...")
            progress_bar.progress(80)
            result = verify_claim(user_claim, evidence_data)
            
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            st.session_state.last_result = result
            
            # Display Results
            st.markdown("## 📋 Verification Results")
            
            # Verdict display
            verdict = result.verdict
            if verdict in [Verdict.TRUE, Verdict.MOSTLY_TRUE]:
                verdict_class = "verdict-true"
                icon = "✅"
            elif verdict in [Verdict.FALSE, Verdict.MOSTLY_FALSE]:
                verdict_class = "verdict-false"
                icon = "❌"
            elif verdict == Verdict.MIXED:
                verdict_class = "verdict-mixed"
                icon = "⚠️"
            else:
                verdict_class = "verdict-unknown"
                icon = "❓"
            
            st.markdown(f"""
            <div class="{verdict_class}">
                {icon} {verdict.value}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Metrics row
            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("Confidence", f"{result.confidence*100:.0f}%")
            with metric_cols[1]:
                st.metric("NLI Score", f"{(result.nli_score+1)/2*100:.0f}%")
            with metric_cols[2]:
                st.metric("Semantic Match", f"{result.semantic_score*100:.0f}%")
            with metric_cols[3]:
                st.metric("Evidence Found", len(evidence_data))
            
            st.markdown("---")
            
            # Explanation
            st.markdown("### 📝 Analysis")
            st.info(result.explanation)
            
            # Evidence sections
            col_support, col_contra = st.columns(2)
            
            with col_support:
                st.markdown("### ✅ Supporting Evidence")
                if result.supporting_evidence:
                    for i, ev in enumerate(result.supporting_evidence[:3], 1):
                        evidence = ev['evidence']
                        with st.expander(f"Evidence {i}: {evidence['source'][:50]}...", expanded=(i==1)):
                            st.markdown(f"**Source:** [{evidence['source']}]({evidence['url']})")
                            st.markdown(f"**Reliability:** {evidence.get('reliability', 0.5)*100:.0f}%")
                            st.markdown(f"**Text:** {evidence['text'][:500]}...")
                            st.markdown(f"**Entailment Score:** {ev['nli_scores']['entailment']*100:.0f}%")
                else:
                    st.markdown("_No strong supporting evidence found_")
            
            with col_contra:
                st.markdown("### ❌ Contradicting Evidence")
                if result.contradicting_evidence:
                    for i, ev in enumerate(result.contradicting_evidence[:3], 1):
                        evidence = ev['evidence']
                        with st.expander(f"Evidence {i}: {evidence['source'][:50]}...", expanded=(i==1)):
                            st.markdown(f"**Source:** [{evidence['source']}]({evidence['url']})")
                            st.markdown(f"**Reliability:** {evidence.get('reliability', 0.5)*100:.0f}%")
                            st.markdown(f"**Text:** {evidence['text'][:500]}...")
                            st.markdown(f"**Contradiction Score:** {ev['nli_scores']['contradiction']*100:.0f}%")
                else:
                    st.markdown("_No contradicting evidence found_")
            
            # Debug info
            if show_debug:
                st.markdown("---")
                with st.expander("🔧 Debug Information"):
                    st.markdown("**Extracted Entities:**")
                    st.json(entities)
                    st.markdown(f"**Total evidence pieces:** {len(evidence_data)}")
                    st.markdown("**All evidence sources:**")
                    sources = list(set([e['source'] for e in evidence_data]))
                    for src in sources[:10]:
                        st.markdown(f"- {src}")
        else:
            progress_bar.empty()
            status_text.empty()
            st.warning("⚠️ Could not find any evidence to verify this claim. Try rephrasing or using a more specific claim.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.9em;">
    <p>🔍 FactCheck AI uses advanced NLI models and multi-source evidence gathering</p>
    <p>⚠️ This is an AI-powered tool and should not be used as the sole source for fact verification</p>
</div>
""", unsafe_allow_html=True)