"""
Task 1: Language Translation Tool - Streamlit Web Application.

A modern, responsive multilingual translation tool with text-to-speech,
instant language swapping, audio playback, and history tracking.
"""

import json
import streamlit as st
from translator_core import translator_service

# Page Configuration
st.set_page_config(
    page_title="AI Language Translator | CodeAlpha",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for a modern, sleek interface
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 12px;
    }
    .stTextArea textarea {
        font-size: 1.05rem;
        border-radius: 10px;
    }
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 6px;
        background: #EEF2FF;
        color: #4338CA;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "history" not in st.session_state:
    st.session_state.history = []

if "source_lang_idx" not in st.session_state:
    st.session_state.source_lang_idx = 0  # Auto Detect

if "target_lang_idx" not in st.session_state:
    st.session_state.target_lang_idx = 1  # Spanish default

# Load Languages
lang_dict = translator_service.get_supported_languages()
language_names = list(lang_dict.keys())
source_options = ["Auto Detect"] + language_names
target_options = language_names

# Header section
st.markdown('<div class="main-header">🌐 Neural AI Language Translator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">CodeAlpha Artificial Intelligence Internship — Task 1: Real-time Multi-Language Translation & Voice Synthesis</div>', unsafe_allow_html=True)

# Sidebar: History & Settings
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/translation.png", width=70)
    st.title("Settings & History")
    
    st.markdown("### 📋 Translation History")
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-8:])):
            with st.expander(f"🔹 {item['source_lang']} ➔ {item['target_lang']} ({item['timestamp']})"):
                st.markdown(f"**Original:** {item['source_text']}")
                st.markdown(f"**Translated:** {item['translated_text']}")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
            
        # Download History
        history_json = json.dumps(st.session_state.history, indent=2)
        st.download_button(
            label="💾 Export History (JSON)",
            data=history_json,
            file_name="translation_history.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.info("No translations performed yet. Your session history will appear here.")
        
    st.divider()
    st.markdown("### ℹ️ About Task 1")
    st.caption("Engineered using **Deep-Translator API**, **gTTS Speech Engine**, and **Streamlit** for CodeAlpha AI Internship.")

# Layout: 2 Columns for Source and Target
col_src_header, col_swap_btn, col_tgt_header = st.columns([5, 1, 5])

with col_src_header:
    src_choice = st.selectbox(
        "Source Language",
        options=source_options,
        index=0,
        key="src_lang_select"
    )

with col_swap_btn:
    st.write("")
    st.write("")
    swap_clicked = st.button("⇄", help="Swap source and target languages", use_container_width=True)

with col_tgt_header:
    default_tgt_index = language_names.index("Spanish") if "Spanish" in language_names else 0
    tgt_choice = st.selectbox(
        "Target Language",
        options=target_options,
        index=default_tgt_index,
        key="tgt_lang_select"
    )

# Resolve language codes
src_code = "auto" if src_choice == "Auto Detect" else lang_dict.get(src_choice, "auto")
tgt_code = lang_dict.get(tgt_choice, "es")

# Main translation input/output workspace
col_src, col_tgt = st.columns(2)

with col_src:
    st.markdown("#### 📝 Source Text")
    input_text = st.text_area(
        "Enter text to translate",
        height=220,
        placeholder="Type or paste text here (e.g., 'Artificial intelligence is transforming the future of software engineering.')...",
        label_visibility="collapsed"
    )
    
    # Text Metrics
    word_count = len(input_text.split()) if input_text.strip() else 0
    char_count = len(input_text)
    st.caption(f"📊 **Words:** {word_count} | **Characters:** {char_count}")
    
    # Source Audio TTS
    if input_text.strip():
        if st.button("🔊 Pronounce Source Text", key="tts_src"):
            with st.spinner("Generating speech..."):
                audio_bytes = translator_service.text_to_speech(input_text, lang_code=src_code if src_code != "auto" else "en")
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.warning("Audio synthesis not available for this language.")

with col_tgt:
    st.markdown("#### 🎯 Translated Output")
    
    translated_result = ""
    is_success = False
    
    if input_text.strip():
        with st.spinner("Translating with Neural Engine..."):
            is_success, translated_result = translator_service.translate(
                text=input_text,
                source_lang=src_code,
                target_lang=tgt_code
            )
            
        if is_success:
            st.text_area(
                "Translation Result",
                value=translated_result,
                height=220,
                label_visibility="collapsed"
            )
            
            # Save to history if distinct from last entry
            import datetime
            current_time = datetime.datetime.now().strftime("%H:%M:%S")
            last_entry = st.session_state.history[-1] if st.session_state.history else None
            
            if not last_entry or last_entry.get("source_text") != input_text or last_entry.get("target_lang") != tgt_choice:
                st.session_state.history.append({
                    "source_lang": src_choice,
                    "target_lang": tgt_choice,
                    "source_text": input_text,
                    "translated_text": translated_result,
                    "timestamp": current_time
                })
        else:
            st.error(translated_result)
    else:
        st.info("👈 Enter text in the source box to see live translation.")

# Target Action Row: Audio & Download
if input_text.strip() and is_success and translated_result:
    col_audio, col_copy, col_download = st.columns([4, 3, 3])
    
    with col_audio:
        if st.button("🔊 Listen to Translation", key="tts_tgt"):
            with st.spinner("Generating voice audio..."):
                tgt_audio = translator_service.text_to_speech(translated_result, lang_code=tgt_code)
                if tgt_audio:
                    st.audio(tgt_audio, format="audio/mp3")
                else:
                    st.warning("Speech pronunciation unavailable for this language.")
                    
    with col_download:
        st.download_button(
            label="📥 Download as TXT",
            data=f"Original ({src_choice}):\n{input_text}\n\nTranslated ({tgt_choice}):\n{translated_result}",
            file_name=f"translation_{tgt_code}.txt",
            mime="text/plain",
            use_container_width=True
        )

st.divider()
st.markdown("""
<div style="text-align: center; color: #94A3B8; font-size: 0.85rem;">
    CodeAlpha Artificial Intelligence Internship Program • Task 1: Language Translation Tool • Developed with Streamlit & Deep Translator
</div>
""", unsafe_allow_html=True)
