import streamlit as st
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.chatbot import CybersecurityChatbot
from utils.config import Config

# Page configuration
st.set_page_config(
    page_title="🛡️ Cybersecurity Chatbot",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .user-message {
        background-color: #e8f4fd;
        border-left: 4px solid #1f77b4;
    }
    .bot-message {
        background-color: #f0f8f0;
        border-left: 4px solid #28a745;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .source-tag {
        display: inline-block;
        background-color: #6c757d;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        margin: 0.1rem;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_chatbot():
    """Initialize chatbot (cached for performance)"""
    try:
        return CybersecurityChatbot()
    except Exception as e:
        st.error(f"Failed to initialize chatbot: {e}")
        return None

def display_message(message: str, is_user: bool = False, metadata: Dict = None):
    """Display a chat message with styling"""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🔍 You:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        confidence_class = ""
        if metadata and metadata.get("confidence"):
            confidence = metadata["confidence"]
            if confidence >= 0.8:
                confidence_class = "confidence-high"
            elif confidence >= 0.6:
                confidence_class = "confidence-medium"
            else:
                confidence_class = "confidence-low"
        
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 Cybersecurity Assistant:</strong><br>
            {message}
        </div>
        """, unsafe_allow_html=True)
        
        # Display metadata if available
        if metadata:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if metadata.get("confidence"):
                    st.markdown(f"""
                    <span class="{confidence_class}">
                        📊 Confidence: {metadata['confidence']} ({metadata.get('confidence_level', 'Unknown')})
                    </span>
                    """, unsafe_allow_html=True)
            
            with col2:
                if metadata.get("response_time"):
                    st.markdown(f"⏱️ Response: {metadata['response_time']}s")
            
            with col3:
                if metadata.get("sources_used"):
                    sources_html = ""
                    for source in metadata["sources_used"]:
                        sources_html += f'<span class="source-tag">{source}</span>'
                    st.markdown(f"Sources: {sources_html}", unsafe_allow_html=True)

def display_source_documents(documents: List):
    """Display source documents in an expandable section"""
    if not documents:
        return
    
    with st.expander(f"📄 View Source Documents ({len(documents)} found)"):
        for i, doc in enumerate(documents, 1):
            st.markdown(f"**Source {i}: {doc.metadata.get('source', 'Unknown')}**")
            
            # Display metadata
            metadata_cols = st.columns(4)
            with metadata_cols[0]:
                if doc.metadata.get('cve_id'):
                    st.markdown(f"CVE: `{doc.metadata['cve_id']}`")
                elif doc.metadata.get('technique_id'):
                    st.markdown(f"MITRE: `{doc.metadata['technique_id']}`")
            
            with metadata_cols[1]:
                if doc.metadata.get('severity'):
                    st.markdown(f"Severity: `{doc.metadata['severity']}`")
                elif doc.metadata.get('risk_level'):
                    st.markdown(f"Risk: `{doc.metadata['risk_level']}`")
            
            with metadata_cols[2]:
                if doc.metadata.get('doc_type'):
                    st.markdown(f"Type: `{doc.metadata['doc_type']}`")
            
            with metadata_cols[3]:
                if doc.metadata.get('cvss_score'):
                    st.markdown(f"CVSS: `{doc.metadata['cvss_score']}`")
            
            # Display content
            st.text_area(
                f"Content {i}:",
                value=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                height=100,
                key=f"doc_content_{i}_{time.time()}"
            )
            
            st.markdown("---")

def create_statistics_dashboard(chatbot: CybersecurityChatbot):
    """Create statistics dashboard"""
    st.subheader("📊 Session Statistics")
    
    stats = chatbot.session_stats
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Queries Processed", stats["queries_processed"])
    
    with col2:
        uptime = datetime.now() - stats["session_start"]
        st.metric("Session Duration", str(uptime).split('.')[0])
    
    with col3:
        st.metric("Avg Response Time", f"{stats['average_response_time']:.2f}s")
    
    with col4:
        st.metric("Sources Accessed", len(stats["sources_used"]))
    
    # Sources usage chart
    if stats["sources_used"]:
        st.subheader("📚 Data Sources Used")
        
        # Create a simple bar chart of sources
        sources_df = pd.DataFrame({
            'Source': list(stats["sources_used"]),
            'Used': [1] * len(stats["sources_used"])
        })
        
        fig = px.bar(
            sources_df, 
            x='Source', 
            y='Used',
            title="Sources Accessed This Session",
            color='Source'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Conversation history summary
    if chatbot.conversation_history:
        st.subheader("💬 Conversation Analysis")
        
        # Confidence distribution
        confidences = [entry.get('confidence', 0) for entry in chatbot.conversation_history]
        if confidences:
            fig = px.histogram(
                x=confidences,
                nbins=10,
                title="Response Confidence Distribution",
                labels={'x': 'Confidence Score', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)

def create_knowledge_base_info(chatbot: CybersecurityChatbot):
    """Display knowledge base information"""
    st.subheader("📖 Knowledge Base Information")
    
    try:
        vector_stats = chatbot.rag_chain.vector_store.get_statistics()
        
        # Total documents
        total_docs = vector_stats.get("total_documents", 0)
        st.metric("Total Documents", total_docs)
        
        # Sources breakdown
        sources = vector_stats.get("sources", {})
        if sources:
            sources_df = pd.DataFrame([
                {"Source": source, "Documents": count}
                for source, count in sources.items()
            ])
            
            fig = px.pie(
                sources_df,
                values='Documents',
                names='Source',
                title="Knowledge Base Composition"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Model information
        st.markdown("**Configuration:**")
        st.markdown(f"- Embedding Model: `{Config.EMBEDDING_MODEL}`")
        st.markdown(f"- LLM Model: `{Config.LLM_MODEL}`")
        st.markdown(f"- Vector Store: `{Config.VECTOR_STORE_TYPE}`")
        
    except Exception as e:
        st.error(f"Could not load knowledge base statistics: {e}")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🛡️ Cybersecurity Chatbot</h1>', unsafe_allow_html=True)
    st.markdown("**Your AI-powered cybersecurity knowledge assistant**")
    
    # Initialize chatbot
    chatbot = initialize_chatbot()
    if not chatbot:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation"):
            if 'messages' in st.session_state:
                st.session_state.messages = []
            st.success("Conversation cleared!")
            st.rebalance()
        
        # Export conversation
        if st.button("💾 Export Conversation"):
            if chatbot.conversation_history:
                export_data = chatbot.export_conversation(format="json")
                st.download_button(
                    label="Download JSON",
                    data=export_data,
                    file_name=f"cybersec_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.info("No conversation to export")
        
        st.markdown("---")
        
        # Quick actions
        st.header("⚡ Quick Actions")
        
        if st.button("📊 Show Statistics"):
            st.session_state.show_stats = True
        
        if st.button("📖 Knowledge Base Info"):
            st.session_state.show_kb_info = True
        
        if st.button("💡 Show Examples"):
            st.session_state.show_examples = True
        
        st.markdown("---")
        
        # Filter options
        st.header("🔍 Filter Options")
        source_filter = st.selectbox(
            "Filter by Source:",
            ["All", "CVE", "OWASP", "MITRE", "Exploits"],
            index=0
        )
        
        doc_type_filter = st.selectbox(
            "Filter by Type:",
            ["All", "vulnerability", "security_guidance", "attack_technique", "exploit_technique"],
            index=0
        )
    
    # Main content area
    main_tab, stats_tab, kb_tab = st.tabs(["💬 Chat", "📊 Statistics", "📖 Knowledge Base"])
    
    with main_tab:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            display_message(
                message["content"], 
                message["role"] == "user",
                message.get("metadata")
            )
            
            if message["role"] == "assistant" and message.get("source_documents"):
                display_source_documents(message["source_documents"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about cybersecurity..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_message(prompt, is_user=True)
            
            # Get bot response
            with st.spinner("🤔 Thinking..."):
                # Apply filters if selected
                if source_filter != "All" or doc_type_filter != "All":
                    source = source_filter if source_filter != "All" else None
                    doc_type = doc_type_filter if doc_type_filter != "All" else None
                    response = chatbot.rag_chain.query_with_filters(
                        prompt, 
                        source_filter=source,
                        doc_type_filter=doc_type
                    )
                else:
                    response = chatbot.chat(prompt)
            
            # Display bot response
            display_message(response["answer"], metadata=response)
            
            # Add to session state
            assistant_message = {
                "role": "assistant",
                "content": response["answer"],
                "metadata": response,
                "source_documents": response.get("source_documents", [])
            }
            st.session_state.messages.append(assistant_message)
            
            # Display source documents
            if response.get("source_documents"):
                display_source_documents(response["source_documents"])
            
            # Suggest follow-up questions
            suggestions = chatbot.suggest_follow_up_questions(response)
            if suggestions:
                st.markdown("**💡 Suggested follow-up questions:**")
                for suggestion in suggestions:
                    if st.button(suggestion, key=f"suggestion_{hash(suggestion)}"):
                        st.session_state.follow_up = suggestion
                        st.rerun()
        
        # Handle follow-up questions
        if hasattr(st.session_state, 'follow_up'):
            st.chat_input = st.session_state.follow_up
            delattr(st.session_state, 'follow_up')
    
    with stats_tab:
        create_statistics_dashboard(chatbot)
    
    with kb_tab:
        create_knowledge_base_info(chatbot)
    
    # Handle sidebar state
    if hasattr(st.session_state, 'show_stats') and st.session_state.show_stats:
        create_statistics_dashboard(chatbot)
        del st.session_state.show_stats
    
    if hasattr(st.session_state, 'show_kb_info') and st.session_state.show_kb_info:
        create_knowledge_base_info(chatbot)
        del st.session_state.show_kb_info
    
    if hasattr(st.session_state, 'show_examples') and st.session_state.show_examples:
        st.info("""
        **Example Questions to Try:**
        
        🔍 **CVE Queries:**
        - "Tell me about CVE-2023-1234"
        - "What are SQL injection vulnerabilities?"
        
        🛡️ **OWASP Questions:**
        - "What is cross-site scripting?"
        - "How to prevent injection attacks?"
        
        ⚔️ **MITRE ATT&CK:**
        - "What is privilege escalation?"
        - "How do lateral movement attacks work?"
        
        💥 **Exploit Techniques:**
        - "How do buffer overflow attacks work?"
        - "What are common countermeasures?"
        """)
        del st.session_state.show_examples

if __name__ == "__main__":
    main()