# 🚀 CodeAlpha Artificial Intelligence Internship Projects

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30.0+-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0+-FF6F00.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0+-F7931E.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Internship](https://img.shields.io/badge/CodeAlpha-AI%20Internship-purple.svg)](https://codealpha.tech/)

A comprehensive, production-grade Artificial Intelligence portfolio containing complete, modular, and fully functional solutions for **Task 1**, **Task 2**, and **Task 3** developed for the **CodeAlpha Artificial Intelligence Internship Program**.

---

## 📑 Table of Contents
- [📌 Project Architecture & Directory Structure](#-project-architecture--directory-structure)
- [🛠️ Tech Stack & Dependencies](#️-tech-stack--dependencies)
- [⚡ Quick Start & Installation](#-quick-start--installation)
- [🌐 Task 1: Language Translation Tool](#-task-1-language-translation-tool)
- [🤖 Task 2: FAQ Retrieval Chatbot](#-task-2-faq-retrieval-chatbot)
- [🎵 Task 3: Music Generation with AI (LSTM)](#-task-3-music-generation-with-ai-lstm)
- [📦 GitHub Repository Setup & Push Guide](#-github-repository-setup--push-guide)
- [👨‍💻 Author & Acknowledgements](#-author--acknowledgements)

---

## 📌 Project Architecture & Directory Structure

```text
CodeAlpha_AI_Tasks/
├── Task_1_Language_Translator/
│   ├── app.py                     # Streamlit frontend with speech synthesis & history
│   ├── translator_core.py         # GoogleTranslator & gTTS translation engine
│   └── requirements.txt           # Task 1 specific dependencies
│
├── Task_2_FAQ_Chatbot/
│   ├── app.py                     # Streamlit conversational interface with category explorer
│   ├── chatbot_model.py           # NLP pipeline (Lemmatization, TF-IDF, Cosine Similarity)
│   ├── faqs.json                  # Knowledge base with 17+ domain-specific FAQs
│   └── requirements.txt           # Task 2 specific dependencies
│
├── Task_3_Music_Generation/
│   ├── generate_music.py          # Music inference script generating playable .mid files
│   ├── train_model.py             # LSTM Neural Network training pipeline with music21
│   ├── sample_data/               # Directory for training MIDI files (.mid)
│   ├── output/                    # Generated novel MIDI compositions
│   └── requirements.txt           # Task 3 specific dependencies
│
├── .gitignore                     # Git ignore rules for virtualenvs, cache, and models
├── README.md                      # Master project documentation
└── requirements.txt               # Unified project dependencies
```

---

## 🛠️ Tech Stack & Dependencies

| Task | Domain | Core Technologies & Frameworks |
| :--- | :--- | :--- |
| **Task 1** | Machine Translation & Audio | `Streamlit`, `deep-translator`, `gTTS`, `langdetect` |
| **Task 2** | NLP & Information Retrieval | `NLTK`, `Scikit-Learn`, `TF-IDF`, `Cosine Similarity`, `Streamlit` |
| **Task 3** | Deep Learning & Generative Music | `TensorFlow/Keras`, `LSTM RNN`, `music21`, `NumPy`, `SciPy` |

---

## ⚡ Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/karunakaraamaravathi-eng/code-alpha-.git
cd code-alpha-
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# Windows (PowerShell / Command Prompt)
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Master Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🌐 Task 1: Language Translation Tool

### Overview
A modern web application that enables bidirectional real-time text translation across 30+ global languages with integrated voice pronunciation (Text-to-Speech), character metrics, instant language swap, and session history export.

### Key Features:
- **Neural Machine Translation**: Powered by Google Translator API through `deep-translator`.
- **Text-to-Speech (TTS)**: Native pronunciation playback using `gTTS` audio synthesis.
- **Language Auto-Detection & Swapping**: Automatically identifies input language and supports one-click reversal.
- **Session History & Export**: Tracks past translations and allows downloading history as JSON or text.
- **Robust Error Handling**: Handles network timeouts, empty strings, and long paragraph chunking gracefully.

### Running Task 1:
```bash
cd CodeAlpha_AI_Tasks/Task_1_Language_Translator
streamlit run app.py
```
*The web interface will automatically open in your default browser at `http://localhost:8501`.*

---

## 🤖 Task 2: FAQ Retrieval Chatbot

### Overview
An intelligent conversational agent built to assist users with e-commerce, technical support, order management, and platform security questions using a complete Natural Language Processing (NLP) pipeline.

### NLP Pipeline Architecture:
1. **Preprocessing**: Lowercasing, punctuation stripping, NLTK tokenization, and `WordNetLemmatizer` normalization.
2. **Feature Extraction**: Sublinear `TfidfVectorizer` extracting unigrams and bigrams.
3. **Similarity Scoring**: Computes mathematical `Cosine Similarity` between user query and FAQ patterns.
4. **Confidence Thresholding**: Triggers exact answers if confidence $\ge 0.25$, or falls back to smart suggested questions if confidence is low.

### Key Features:
- **Interactive Chat Interface**: Modern chat bubbles with real-time response generation.
- **Confidence Badge & Category Tag**: Shows similarity percentage and topic classification.
- **Category Explorer Sidebar**: Browse and click frequently asked questions to test instant answers.
- **Adjustable Threshold Slider**: Dynamically adjust the confidence sensitivity from the UI.

### Running Task 2:
```bash
cd CodeAlpha_AI_Tasks/Task_2_FAQ_Chatbot
streamlit run app.py
```
*The web interface will automatically open at `http://localhost:8501`.*

---

## 🎵 Task 3: Music Generation with AI (LSTM)

### Overview
A Deep Learning generative model utilizing **Long Short-Term Memory (LSTM)** neural networks and **music21** symbolic musicology toolkit to learn musical pitch and chord patterns from MIDI files and synthesize original polyphonic and monophonic compositions.

### Model Architecture:
- **Input**: Lookback window of 32 sequential musical note/chord tokens normalized by vocabulary size.
- **Layer 1**: `LSTM(256, return_sequences=True)` + `Dropout(0.3)` + `BatchNormalization()`
- **Layer 2**: `LSTM(256)` + `Dropout(0.3)` + `BatchNormalization()`
- **Layer 3**: `Dense(128, activation='relu')` + `Dropout(0.2)`
- **Output Layer**: `Dense(vocab_size, activation='softmax')`
- **Sampling**: Temperature-scaled multinomial sampling for controlled harmonic creativity.

### 1. Train the Neural Model:
```bash
cd CodeAlpha_AI_Tasks/Task_3_Music_Generation
python train_model.py --epochs 15 --batch_size 32
```
*(Note: If `sample_data/` is empty, the training script automatically creates synthetic MIDI melodies to ensure immediate, zero-setup execution).*

### 2. Generate Playable AI Music (.mid):
```bash
python generate_music.py --num_notes 120 --temperature 0.85
```
*Generated `.mid` files will be saved in the `output/` folder and can be played with any media player (Windows Media Player, VLC) or opened in any Digital Audio Workstation (GarageBand, FL Studio).*

---

## 📦 GitHub Repository Setup & Push Guide

To commit and push this entire project to your GitHub repository ([`karunakaraamaravathi-eng/code-alpha-`](https://github.com/karunakaraamaravathi-eng/code-alpha-)), run the following commands in your project root:

```bash
# 1. Initialize Git
git init

# 2. Add Remote Origin
git remote add origin https://github.com/karunakaraamaravathi-eng/code-alpha-.git

# 3. Stage All Files
git add .

# 4. Commit Changes
git commit -m "feat: Complete CodeAlpha AI Internship Tasks 1, 2, and 3"

# 5. Push to GitHub Main Branch
git branch -M main
git push -u origin main --force
```

---

## 👨‍💻 Author & Acknowledgements
- **Intern**: Karunakara Amaravathi
- **Internship Program**: CodeAlpha Artificial Intelligence Internship
- **Repository**: [https://github.com/karunakaraamaravathi-eng/code-alpha-](https://github.com/karunakaraamaravathi-eng/code-alpha-)

---
*Built with ❤️ for CodeAlpha Artificial Intelligence Internship.*
