"""
Task 2: FAQ Chatbot - Streamlit Conversational Interface.

An intelligent customer assistant powered by TF-IDF vectorization and Cosine Similarity,
featuring interactive conversation history, category explorer, confidence scoring, and export tools.
"""

import json
import streamlit as st
from chatbot_model import FAQChatbot

# Page configuration
st.set_page_config(
    page_title="AI FAQ Chatbot | CodeAlpha",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling chat UI
st.markdown("""
<style>
    .chat-header {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .chat-subtitle {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 1.2rem;
    }
    .confidence-badge {
        display: inline-block;
        padding: 3px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background: #E0E7FF;
        color: #3730A3;
        margin-top: 6px;
    }
    .suggestion-btn {
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "bot_instance" not in st.session_state:
    st.session_state.bot_instance = FAQChatbot()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I am the **CodeAlpha AI Support Assistant**. How can I help you today? Ask me anything about orders, deliveries, returns, payments, or account settings!",
            "confidence": 1.0,
            "category": "Welcome",
            "suggestions": [
                "How do I track my order status?",
                "What is your return policy?",
                "What payment methods do you accept?",
                "How do I reset my password?"
            ]
        }
    ]

# Header
st.markdown('<div class="chat-header">🤖 Intelligent FAQ Conversational Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="chat-subtitle">CodeAlpha Artificial Intelligence Internship — Task 2: NLP Pipeline with TF-IDF Vectorization & Cosine Similarity</div>', unsafe_allow_html=True)

# Sidebar: Controls & Category Explorer
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/bot.png", width=70)
    st.title("FAQ Knowledge Base")
    
    # Confidence Threshold Controller
    threshold = st.slider(
        "Cosine Confidence Threshold",
        min_value=0.10,
        max_value=0.80,
        value=0.25,
        step=0.05,
        help="Higher values require closer text matches before triggering FAQ answers."
    )
    st.session_state.bot_instance.confidence_threshold = threshold
    
    st.divider()
    st.markdown("### 📚 Browse by Category")
    
    categories = st.session_state.bot_instance.get_categories()
    selected_category = st.selectbox("Select Topic:", ["All Categories"] + categories)
    
    if selected_category == "All Categories":
        display_faqs = st.session_state.bot_instance.faqs_data[:8]
    else:
        display_faqs = st.session_state.bot_instance.get_faqs_by_category(selected_category)
        
    st.caption("Click any question below to test the bot:")
    for faq in display_faqs:
        if st.button(f"❓ {faq['question']}", key=f"q_{faq['id']}", use_container_width=True):
            st.session_state.prompt_to_submit = faq['question']
            st.rerun()

    st.divider()
    st.markdown("### ⚙️ Session Actions")
    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Conversation restarted. How may I assist you?",
                    "confidence": 1.0,
                    "category": "System",
                    "suggestions": st.session_state.bot_instance.get_top_questions(3)
                }
            ]
            st.rerun()
            
    with col_export:
        chat_export_data = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            label="💾 Export",
            data=chat_export_data,
            file_name="faq_chat_history.json",
            mime="application/json",
            use_container_width=True
        )

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display meta tags if assistant message
        if message["role"] == "assistant":
            conf = message.get("confidence")
            cat = message.get("category")
            if conf is not None and cat:
                st.caption(f"🏷️ Category: **{cat}** | 🎯 Confidence: **{int(conf * 100)}%**")
                
            # Render suggestion chips if available
            suggestions = message.get("suggestions", [])
            if suggestions:
                st.markdown("**Suggested Questions:**")
                cols = st.columns(min(len(suggestions), 2))
                for idx, sugg in enumerate(suggestions[:4]):
                    col_target = cols[idx % len(cols)]
                    if col_target.button(sugg, key=f"sugg_{message.get('category', '')}_{idx}_{sugg[:10]}"):
                        st.session_state.prompt_to_submit = sugg
                        st.rerun()

# Handle pre-selected prompt or user text input
user_query = None

if "prompt_to_submit" in st.session_state and st.session_state.prompt_to_submit:
    user_query = st.session_state.prompt_to_submit
    del st.session_state.prompt_to_submit
else:
    user_query = st.chat_input("Type your question here (e.g., 'How long does delivery take?')...")

# Process User Message
if user_query:
    # 1. Append User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Get Bot Response via NLP pipeline
    with st.spinner("Analyzing query semantics..."):
        result = st.session_state.bot_instance.get_response(user_query)

    # 3. Append Assistant Message
    bot_msg = {
        "role": "assistant",
        "content": result["response"],
        "confidence": result["confidence"],
        "category": result["category"] or "Unmatched",
        "suggestions": result["suggestions"]
    }
    st.session_state.messages.append(bot_msg)
    
    # Rerun to render nicely
    st.rerun()
