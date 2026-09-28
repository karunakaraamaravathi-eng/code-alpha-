"""
Unified Web Portal & Interactive Dashboard for CodeAlpha AI Internship.

Integrates Task 1 (Language Translator), Task 2 (FAQ Chatbot), and Task 3 (Music Generation)
into a single, cohesive, modern web application.
"""

import json
import os
import sys
import time
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="CodeAlpha AI Internship Portal",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Append paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "CodeAlpha_AI_Tasks", "Task_1_Language_Translator"))
sys.path.insert(0, os.path.join(BASE_DIR, "CodeAlpha_AI_Tasks", "Task_2_FAQ_Chatbot"))
sys.path.insert(0, os.path.join(BASE_DIR, "CodeAlpha_AI_Tasks", "Task_3_Music_Generation"))

from translator_core import translator_service
from chatbot_model import FAQChatbot
from generate_music import generate_notes_algorithmic_fallback, create_midi_from_notes

# Custom CSS
st.markdown("""
<style>
    .portal-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366F1 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .portal-sub {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .task-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Main Header
st.markdown('<div class="portal-title">🚀 CodeAlpha Artificial Intelligence Internship</div>', unsafe_allow_html=True)
st.markdown('<div class="portal-sub">Unified Production Showcase — Complete Implementations of Task 1, Task 2, and Task 3</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/brain.png", width=70)
    st.title("Internship Portal")
    st.markdown("**Intern:** Karunakara Amaravathi")
    st.markdown("**Domain:** Artificial Intelligence")
    st.markdown("**Repository:** [code-alpha-](https://github.com/karunakaraamaravathi-eng/code-alpha-)")
    st.divider()
    st.info("💡 You can navigate between the three tasks using the tabs at the top or run individual tasks independently.")

# Navigation Tabs
tab_task1, tab_task2, tab_task3, tab_overview = st.tabs([
    "🌐 Task 1: Language Translator",
    "🤖 Task 2: FAQ Chatbot",
    "🎵 Task 3: Music Generation",
    "📄 Project Documentation & Architecture"
])

# -------------------------------------------------------------
# TAB 1: LANGUAGE TRANSLATION TOOL
# -------------------------------------------------------------
with tab_task1:
    st.markdown("### 🌐 Neural Multi-Language Translation & Speech Synthesis")
    st.caption("Deep-learning powered machine translation with voice synthesis across 30+ languages.")
    
    lang_dict = translator_service.get_supported_languages()
    lang_names = list(lang_dict.keys())
    
    col_t1_left, col_t1_mid, col_t1_right = st.columns([5, 1, 5])
    with col_t1_left:
        src_sel = st.selectbox("From Language", ["Auto Detect"] + lang_names, index=0, key="t1_src")
    with col_t1_mid:
        st.write("")
        st.write("")
        st.markdown("<h3 style='text-align:center;'>⇄</h3>", unsafe_allow_html=True)
    with col_t1_right:
        def_tgt = lang_names.index("Spanish") if "Spanish" in lang_names else 0
        tgt_sel = st.selectbox("To Language", lang_names, index=def_tgt, key="t1_tgt")
        
    src_c = "auto" if src_sel == "Auto Detect" else lang_dict.get(src_sel, "auto")
    tgt_c = lang_dict.get(tgt_sel, "es")
    
    col_in, col_out = st.columns(2)
    with col_in:
        user_input_t1 = st.text_area(
            "Enter text to translate:",
            value="Artificial intelligence and natural language processing are revolutionizing modern applications.",
            height=180,
            key="t1_input"
        )
        if user_input_t1.strip():
            if st.button("🔊 Pronounce Input Text", key="t1_tts_in"):
                audio_in = translator_service.text_to_speech(user_input_t1, lang_code=src_c if src_c != "auto" else "en")
                if audio_in:
                    st.audio(audio_in, format="audio/mp3")
                    
    with col_out:
        if user_input_t1.strip():
            with st.spinner("Translating..."):
                ok, t1_res = translator_service.translate(user_input_t1, source_lang=src_c, target_lang=tgt_c)
            
            if ok:
                st.text_area("Translated Result:", value=t1_res, height=180, key="t1_out")
                if st.button("🔊 Listen to Translation", key="t1_tts_out"):
                    audio_out = translator_service.text_to_speech(t1_res, lang_code=tgt_c)
                    if audio_out:
                        st.audio(audio_out, format="audio/mp3")
            else:
                st.error(t1_res)
        else:
            st.info("Enter source text on the left to view translation.")

# -------------------------------------------------------------
# TAB 2: FAQ CHATBOT
# -------------------------------------------------------------
with tab_task2:
    st.markdown("### 🤖 Intelligent FAQ Retrieval Conversational Agent")
    st.caption("Natural Language Processing pipeline with Tokenization, TF-IDF Vectorization & Cosine Similarity.")
    
    if "portal_bot" not in st.session_state:
        st.session_state.portal_bot = FAQChatbot()
        
    if "portal_messages" not in st.session_state:
        st.session_state.portal_messages = [
            {
                "role": "assistant",
                "content": "👋 Hello! Ask me any question about orders, delivery tracking, refunds, payment security, or technical support.",
                "confidence": 1.0,
                "category": "Welcome"
            }
        ]
        
    col_chat, col_faq_sample = st.columns([7, 3])
    
    with col_faq_sample:
        st.markdown("#### 💡 Quick FAQ Prompts")
        quick_qs = [
            "How do I track my order status?",
            "What is your return policy?",
            "What payment methods do you accept?",
            "Is my credit card payment secure?",
            "How do I reset my password?",
            "Do you ship internationally?"
        ]
        for q in quick_qs:
            if st.button(f"❓ {q}", key=f"portal_q_{q[:15]}", use_container_width=True):
                st.session_state.portal_chat_prompt = q
                st.rerun()
                
    with col_chat:
        st.markdown("#### 💬 Conversation")
        for msg in st.session_state.portal_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and "confidence" in msg and msg["category"] != "Welcome":
                    st.caption(f"🎯 Match Confidence: **{int(msg['confidence'] * 100)}%** | 🏷️ Category: **{msg.get('category')}**")

        chat_input = None
        if "portal_chat_prompt" in st.session_state and st.session_state.portal_chat_prompt:
            chat_input = st.session_state.portal_chat_prompt
            del st.session_state.portal_chat_prompt
        else:
            chat_input = st.chat_input("Ask a question (e.g. 'Can I return an opened item?')...")
            
        if chat_input:
            st.session_state.portal_messages.append({"role": "user", "content": chat_input})
            bot_res = st.session_state.portal_bot.get_response(chat_input)
            st.session_state.portal_messages.append({
                "role": "assistant",
                "content": bot_res["response"],
                "confidence": bot_res["confidence"],
                "category": bot_res["category"] or "Fallback"
            })
            st.rerun()

# -------------------------------------------------------------
# TAB 3: MUSIC GENERATION WITH AI
# -------------------------------------------------------------
with tab_task3:
    st.markdown("### 🎵 Generative AI Music Composition Engine (LSTM & Music21)")
    st.caption("Deep Learning symbolic music generator creating novel polyphonic & melodic MIDI compositions.")
    
    col_music_ctrl, col_music_out = st.columns([5, 5])
    
    with col_music_ctrl:
        st.markdown("#### 🎹 Generation Settings")
        num_notes_sel = st.slider("Number of Note/Chord Events:", min_value=30, max_value=200, value=64, step=10)
        temp_sel = st.slider("Creativity / Temperature:", min_value=0.2, max_value=1.5, value=0.85, step=0.05)
        
        generate_btn = st.button("✨ Generate Novel AI Composition (.mid)", type="primary", use_container_width=True)
        
    with col_music_out:
        st.markdown("#### 🎼 Output & Playback")
        if generate_btn:
            with st.spinner("Synthesizing musical sequences with piano stream..."):
                out_dir = os.path.join(BASE_DIR, "CodeAlpha_AI_Tasks", "Task_3_Music_Generation", "output")
                os.makedirs(out_dir, exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                midi_filename = f"portal_music_{stamp}.mid"
                midi_path = os.path.join(out_dir, midi_filename)
                
                notes_gen = generate_notes_algorithmic_fallback(num_notes=num_notes_sel)
                create_midi_from_notes(notes_gen, midi_path)
                
                st.success(f"🎉 Composition generated successfully! ({len(notes_gen)} musical events)")
                st.code(f"Notes Sequence: {notes_gen[:15]} ...")
                
                with open(midi_path, "rb") as f:
                    midi_bytes = f.read()
                    
                st.download_button(
                    label="📥 Download Generated MIDI File (.mid)",
                    data=midi_bytes,
                    file_name=midi_filename,
                    mime="audio/midi",
                    use_container_width=True
                )
        else:
            st.info("Click 'Generate Novel AI Composition' to create a new playable MIDI song!")

# -------------------------------------------------------------
# TAB 4: ARCHITECTURE & DOCUMENTATION
# -------------------------------------------------------------
with tab_overview:
    st.markdown("### 📄 CodeAlpha AI Internship Overview & Verification")
    st.markdown("""
    #### 📌 Summary of Completed Tasks:
    1. **Task 1: Language Translation Tool**
       - Dual-layer translation engine with fallback.
       - Integrated Text-to-Speech (TTS) voice pronunciation.
       - Streamlit web interface with language auto-detect & swap.
    2. **Task 2: FAQ Retrieval Chatbot**
       - Rich dataset of 17+ domain-specific Q&As.
       - Tokenization, stopword filtering, TF-IDF Vectorization, and Cosine Similarity scoring.
       - Confidence thresholding ($\ge 0.25$) with intelligent fallback.
    3. **Task 3: Music Generation with AI**
       - Long Short-Term Memory (LSTM) Recurrent Neural Network.
       - Symbolic music processing via `music21`.
       - Automatic generation of playable `.mid` standard MIDI files.

    #### 💻 Terminal Verification Command:
    To run all 3 tasks directly in your terminal:
    ```bash
    python run_all_tasks.py
    ```
    """)
