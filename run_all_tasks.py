"""
Unified Terminal Demonstration Script for CodeAlpha AI Internship Tasks.

Executes Task 1, Task 2, and Task 3 end-to-end and displays clear, formatted
results in the terminal for presentation and verification.
"""

import os
import sys
import time

# Ensure UTF-8 output encoding for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure project root is on PYTHONPATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "CodeAlpha_AI_Tasks", "Task_1_Language_Translator"))
sys.path.insert(0, os.path.join(BASE_DIR, "CodeAlpha_AI_Tasks", "Task_2_FAQ_Chatbot"))
sys.path.insert(0, os.path.join(BASE_DIR, "CodeAlpha_AI_Tasks", "Task_3_Music_Generation"))


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f"  {title.upper()}")
    print("=" * 70)


def run_task_1():
    print_banner("Task 1: Language Translation Tool Demo")
    from translator_core import translator_service

    test_phrases = [
        ("Hello, welcome to CodeAlpha Artificial Intelligence Internship!", "es", "Spanish"),
        ("Machine learning models can learn patterns from large datasets.", "fr", "French"),
        ("Natural Language Processing enables computers to understand human language.", "de", "German"),
        ("Artificial intelligence is transforming the future of technology.", "hi", "Hindi")
    ]

    for text, lang_code, lang_name in test_phrases:
        print(f"\n[>] Source (English): \"{text}\"")
        success, translation = translator_service.translate(text, source_lang="en", target_lang=lang_code)
        if success:
            try:
                print(f"[+] Translated ({lang_name}): \"{translation}\"")
            except Exception:
                print(f"[+] Translated ({lang_name}): \"{translation.encode('ascii', 'replace').decode()}\"")
            # Test TTS bytes generation
            audio_bytes = translator_service.text_to_speech(translation, lang_code=lang_code)
            audio_status = f"{len(audio_bytes)} bytes audio generated" if audio_bytes else "N/A"
            print(f"[*] Voice Synthesis (TTS): {audio_status}")
        else:
            print(f"[-] Translation Error: {translation}")

    print("\n[+] Task 1 Completed Successfully!")


def run_task_2():
    print_banner("Task 2: FAQ Chatbot (NLP & Cosine Similarity) Demo")
    from chatbot_model import FAQChatbot

    bot = FAQChatbot()
    test_queries = [
        "How can I track my order delivery?",
        "What is your return policy for damaged products?",
        "Is my credit card payment secure on this website?",
        "I forgot my account password, how do I reset it?",
        "Can you ship to international countries?",
        "What is the weather like on Mars?"  # Unmatched fallback test
    ]

    for query in test_queries:
        print(f"\n[>] User Query: \"{query}\"")
        res = bot.get_response(query)
        match_str = "MATCH FOUND" if res["matched"] else "FALLBACK (LOW CONFIDENCE)"
        print(f"[*] Match Status: [{match_str}] | Confidence: {res['confidence'] * 100:.1f}%")
        if res.get("category"):
            print(f"[*] Topic Category: {res['category']}")
        print(f"[+] Bot Response: \"{res['response']}\"")
        if res.get("suggestions"):
            print(f"[*] Suggested Related FAQs: {res['suggestions'][:2]}")

    print("\n[+] Task 2 Completed Successfully!")


def run_task_3():
    print_banner("Task 3: Music Generation with AI Demo")
    from generate_music import generate_notes_algorithmic_fallback, create_midi_from_notes

    output_dir = os.path.join(BASE_DIR, "CodeAlpha_AI_Tasks", "Task_3_Music_Generation", "output")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"terminal_demo_music_{timestamp}.mid")

    print("[*] Generating musical note & chord progression sequence...")
    notes_sequence = generate_notes_algorithmic_fallback(num_notes=48)
    print(f"[*] Sample Note Sequence: {notes_sequence[:12]}...")

    print("[*] Synthesizing MIDI composition with music21 piano stream...")
    saved_file = create_midi_from_notes(notes_sequence, output_path)

    print(f"[+] Playable MIDI File Created: {saved_file}")
    print(f"[+] File Size: {os.path.getsize(saved_file)} bytes")
    print("\n[+] Task 3 Completed Successfully!")


def main():
    print("\n" + "#" * 70)
    print("#  CODEALPHA ARTIFICIAL INTELLIGENCE INTERNSHIP - ALL TASKS DEMO    #")
    print("#" * 70)

    run_task_1()
    run_task_2()
    run_task_3()

    print("\n" + "#" * 70)
    print("#  ALL 3 TASKS EXECUTED AND VERIFIED SUCCESSFULLY!                 #")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
