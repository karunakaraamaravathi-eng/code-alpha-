"""
Translator Core Engine for Task 1: Language Translation Tool.

This module provides production-grade language translation and Text-to-Speech (TTS)
capabilities with multiple redundant backends:
1. Direct Google Translate GTX Engine (Fast, High availability, 0-latency)
2. Deep-Translator Google Engine
3. MyMemory Fallback Engine
"""

import io
import logging
import urllib.parse
from typing import Dict, List, Optional, Tuple
import requests
from deep_translator import GoogleTranslator, MyMemoryTranslator
from gtts import gTTS

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class LanguageTranslator:
    """
    Production-grade Language Translation and Text-to-Speech service.
    """

    POPULAR_LANGUAGES: Dict[str, str] = {
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Hindi": "hi",
        "Telugu": "te",
        "Tamil": "ta",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Bengali": "bn",
        "Gujarati": "gu",
        "Marathi": "mr",
        "Punjabi": "pa",
        "Japanese": "ja",
        "Chinese (Simplified)": "zh-CN",
        "Chinese (Traditional)": "zh-TW",
        "Russian": "ru",
        "Portuguese": "pt",
        "Italian": "it",
        "Arabic": "ar",
        "Korean": "ko",
        "Turkish": "tr",
        "Dutch": "nl",
        "Swedish": "sv",
        "Polish": "pl",
        "Greek": "el",
        "Vietnamese": "vi",
        "Thai": "th",
        "Indonesian": "id",
        "Urdu": "ur"
    }

    # Locale mapping for MyMemory
    MYMEMORY_LOCALES: Dict[str, str] = {
        "en": "en-US",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE",
        "hi": "hi-IN",
        "te": "te-IN",
        "ta": "ta-IN",
        "kn": "kn-IN",
        "ml": "ml-IN",
        "bn": "bn-IN",
        "gu": "gu-IN",
        "mr": "mr-IN",
        "pa": "pa-IN",
        "ja": "ja-JP",
        "zh-cn": "zh-CN",
        "zh-tw": "zh-TW",
        "ru": "ru-RU",
        "pt": "pt-PT",
        "it": "it-IT",
        "ar": "ar-SA",
        "ko": "ko-KR",
        "tr": "tr-TR",
        "nl": "nl-NL",
        "sv": "sv-SE",
        "pl": "pl-PL",
        "el": "el-GR",
        "vi": "vi-VN",
        "th": "th-TH",
        "id": "id-ID",
        "ur": "ur-PK"
    }

    TTS_SUPPORTED_CODES = {
        "en", "es", "fr", "de", "hi", "te", "ta", "kn", "ml", "bn", "gu",
        "mr", "ja", "zh-CN", "zh-TW", "ru", "pt", "it", "ar", "ko", "tr",
        "nl", "sv", "pl", "el", "vi", "th", "id", "ur"
    }

    def __init__(self):
        """Initialize supported languages dictionary."""
        self._supported_languages = dict(self.POPULAR_LANGUAGES)
        try:
            dynamic_langs = GoogleTranslator().get_supported_languages(as_dict=True)
            for name, code in dynamic_langs.items():
                self._supported_languages[name.title()] = code
        except Exception:
            pass

    def get_supported_languages(self) -> Dict[str, str]:
        """Returns sorted language names mapped to codes."""
        return {k: self._supported_languages[k] for k in sorted(self._supported_languages.keys())}

    def _translate_gtx(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Direct Google Translate GTX endpoint."""
        src = "auto" if source_lang == "auto" else source_lang
        tgt = target_lang
        url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl={src}&tl={tgt}&dt=t&q={urllib.parse.quote(text)}"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url, headers=headers, timeout=6)
        if response.status_code == 200:
            data = response.json()
            if data and isinstance(data, list) and len(data) > 0 and data[0]:
                translated_sentences = [seg[0] for seg in data[0] if seg and seg[0]]
                return "".join(translated_sentences)
        return None

    def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "en"
    ) -> Tuple[bool, str]:
        """
        Translates text with automatic multi-backend failover.
        """
        if not text or not text.strip():
            return False, "Input text cannot be empty."

        clean_text = text.strip()

        # 1. Primary: Direct GTX Engine
        try:
            result = self._translate_gtx(clean_text, source_lang, target_lang)
            if result:
                return True, result
        except Exception as e:
            logger.warning("GTX engine failed: %s. Trying deep_translator...", e)

        # 2. Secondary: Deep Translator GoogleTranslator
        try:
            max_chunk_size = 4000
            if len(clean_text) <= max_chunk_size:
                dt_translator = GoogleTranslator(source=source_lang, target=target_lang)
                res = dt_translator.translate(clean_text)
                if res:
                    return True, res
            else:
                chunks = [clean_text[i:i + max_chunk_size] for i in range(0, len(clean_text), max_chunk_size)]
                dt_translator = GoogleTranslator(source=source_lang, target=target_lang)
                res_chunks = [dt_translator.translate(chunk) for chunk in chunks]
                return True, " ".join(res_chunks)
        except Exception as e:
            logger.warning("DeepTranslator Google failed: %s. Trying MyMemory...", e)

        # 3. Tertiary: MyMemory Translator
        try:
            src_loc = self.MYMEMORY_LOCALES.get(source_lang.lower(), "en-US") if source_lang != "auto" else "en-US"
            tgt_loc = self.MYMEMORY_LOCALES.get(target_lang.lower(), f"{target_lang}-{target_lang.upper()}")
            mm_translator = MyMemoryTranslator(source=src_loc, target=tgt_loc)
            mm_res = mm_translator.translate(clean_text[:500])
            if mm_res:
                return True, mm_res
        except Exception as e:
            logger.error("MyMemory failed: %s", e)

        return False, "Translation service temporarily unavailable. Please try again."

    def text_to_speech(self, text: str, lang_code: str = "en") -> Optional[bytes]:
        """
        Converts text to speech audio bytes using gTTS.
        """
        if not text or not text.strip():
            return None

        normalized_code = lang_code.lower()
        if normalized_code not in self.TTS_SUPPORTED_CODES and "-" in normalized_code:
            normalized_code = normalized_code.split("-")[0]

        try:
            tts = gTTS(text=text[:1000], lang=normalized_code, slow=False)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception as e:
            logger.warning("TTS failed for '%s': %s. Retrying with default English...", lang_code, e)
            try:
                tts = gTTS(text=text[:1000], lang="en", slow=False)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp.read()
            except Exception:
                return None


# Module singleton
translator_service = LanguageTranslator()
