"""
Chatbot Model and NLP Pipeline for Task 2: FAQ Chatbot.

Implements text preprocessing, tokenization, stopword filtering,
TF-IDF vectorization, and Cosine Similarity matching with confidence
thresholding and fallback recommendations.
"""

import json
import logging
import os
import re
import string
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standard English stopwords for zero-latency, offline execution
STANDARD_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class FAQChatbot:
    """
    Intelligent FAQ Retrieval Chatbot utilizing TF-IDF Vectorization
    and Cosine Similarity scoring.
    """

    GREETINGS = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening", "howdy"]
    FAREWELLS = ["bye", "goodbye", "see you", "thank you", "thanks", "thx", "appreciate it"]

    def __init__(self, faqs_path: Optional[str] = None, confidence_threshold: float = 0.25):
        """
        Initialize the FAQ Chatbot with knowledge base and NLP pipeline.

        Args:
            faqs_path: Path to the JSON dataset of FAQs.
            confidence_threshold: Minimum cosine similarity score required for match.
        """
        self.confidence_threshold = confidence_threshold
        self.stop_words = STANDARD_STOPWORDS

        if faqs_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            faqs_path = os.path.join(base_dir, "faqs.json")

        self.faqs_path = faqs_path
        self.faqs_data: List[Dict[str, Any]] = []
        self.corpus: List[str] = []
        self.corpus_faq_map: List[int] = []  # Maps corpus index to faq index
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.tfidf_matrix = None

        self.load_faqs()

    def preprocess(self, text: str) -> str:
        """
        Cleans, tokenizes, removes stopwords and punctuation, and normalizes text.

        Args:
            text: Raw input string.

        Returns:
            Preprocessed normalized string.
        """
        if not text:
            return ""

        # Lowercase and clean special chars
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)

        tokens = text.split()

        # Simple stemming / normalization rule
        cleaned_tokens = []
        for word in tokens:
            if word not in self.stop_words and word not in string.punctuation and len(word) > 1:
                # Basic suffix reduction
                if word.endswith("ing") and len(word) > 5:
                    word = word[:-3]
                elif word.endswith("ed") and len(word) > 4:
                    word = word[:-2]
                elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
                    word = word[:-1]
                cleaned_tokens.append(word)

        return " ".join(cleaned_tokens) if cleaned_tokens else text

    def load_faqs(self) -> None:
        """Loads FAQ entries from JSON and builds the TF-IDF search index."""
        if not os.path.exists(self.faqs_path):
            logger.error("FAQ file not found at: %s", self.faqs_path)
            self.faqs_data = []
            return

        try:
            with open(self.faqs_path, "r", encoding="utf-8") as f:
                self.faqs_data = json.load(f)

            self.corpus = []
            self.corpus_faq_map = []

            for idx, item in enumerate(self.faqs_data):
                # Primary question
                primary_q = item.get("question", "")
                self.corpus.append(self.preprocess(primary_q))
                self.corpus_faq_map.append(idx)

                # Sample queries
                for sample in item.get("sample_queries", []):
                    self.corpus.append(self.preprocess(sample))
                    self.corpus_faq_map.append(idx)

                # Keywords
                keywords = " ".join(item.get("keywords", []))
                if keywords:
                    self.corpus.append(self.preprocess(keywords))
                    self.corpus_faq_map.append(idx)

            if self.corpus:
                self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
                logger.info("Indexed %d FAQ patterns across %d FAQs.", len(self.corpus), len(self.faqs_data))
        except Exception as e:
            logger.error("Error loading FAQs: %s", e, exc_info=True)

    def get_response(self, user_query: str) -> Dict[str, Any]:
        """
        Matches user query against the FAQ index using Cosine Similarity.

        Args:
            user_query: The user's input string.

        Returns:
            Dict containing match status, response text, confidence score, and suggestions.
        """
        if not user_query or not user_query.strip():
            return {
                "matched": False,
                "confidence": 0.0,
                "response": "Please type a question or choose one from the categories.",
                "matched_question": None,
                "category": None,
                "suggestions": self.get_top_questions(3)
            }

        cleaned_query = user_query.strip().lower()

        # Check for instant greetings
        if any(cleaned_query == g or cleaned_query.startswith(g + " ") for g in self.GREETINGS):
            return {
                "matched": True,
                "confidence": 1.0,
                "response": "Hello! 👋 I am your CodeAlpha AI Customer Assistant. How can I help you today? You can ask me about orders, shipping, returns, refunds, payment security, or technical support!",
                "matched_question": "Greeting",
                "category": "General",
                "suggestions": self.get_top_questions(3)
            }

        # Check for instant farewells / gratitude
        if any(cleaned_query == f or cleaned_query.startswith(f + " ") for f in self.FAREWELLS):
            return {
                "matched": True,
                "confidence": 1.0,
                "response": "You're very welcome! 😊 Feel free to ask if you have any other questions. Have a wonderful day!",
                "matched_question": "Farewell",
                "category": "General",
                "suggestions": []
            }

        if self.tfidf_matrix is None or len(self.corpus) == 0:
            return {
                "matched": False,
                "confidence": 0.0,
                "response": "Knowledge base is currently empty or indexing failed.",
                "matched_question": None,
                "category": None,
                "suggestions": []
            }

        # Preprocess and vectorize user query
        processed_query = self.preprocess(user_query)
        if not processed_query.strip():
            processed_query = user_query.lower()

        query_vector = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= self.confidence_threshold:
            faq_idx = self.corpus_faq_map[best_idx]
            matched_faq = self.faqs_data[faq_idx]
            
            # Find related questions in the same category
            related = [
                f["question"]
                for f in self.faqs_data
                if f["category"] == matched_faq["category"] and f["id"] != matched_faq["id"]
            ][:3]

            return {
                "matched": True,
                "confidence": round(best_score, 4),
                "response": matched_faq["answer"],
                "matched_question": matched_faq["question"],
                "category": matched_faq["category"],
                "suggestions": related if related else self.get_top_questions(3)
            }
        else:
            # Fallback response with suggested queries
            return {
                "matched": False,
                "confidence": round(best_score, 4),
                "response": (
                    "I'm sorry, I couldn't find a direct match for your question in our knowledge base. 🤔\n\n"
                    "Here are some frequently asked questions you can explore, or try rephrasing your question with keywords like *order*, *refund*, *shipping*, or *payment*."
                ),
                "matched_question": None,
                "category": None,
                "suggestions": self.get_top_questions(4)
            }

    def get_categories(self) -> List[str]:
        """Returns unique categories available in the knowledge base."""
        return sorted(list(set(item["category"] for item in self.faqs_data if "category" in item)))

    def get_faqs_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Returns all FAQs belonging to a specified category."""
        return [item for item in self.faqs_data if item.get("category") == category]

    def get_top_questions(self, limit: int = 4) -> List[str]:
        """Returns top general questions for recommendations."""
        return [item["question"] for item in self.faqs_data[:limit]]


# Singleton instance for quick app binding
chatbot_engine = FAQChatbot()
