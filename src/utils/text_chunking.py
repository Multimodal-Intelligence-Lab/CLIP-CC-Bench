"""
Shared text chunking utility for all embedding_model models.

Uses NLTK for sentence-level tokenization to support fine-grained embedding matching.
"""
import nltk
from typing import List
import logging

logger = logging.getLogger('text_chunking')


def ensure_nltk_data():
    """Ensure NLTK punkt tokenizer is available."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)
        logger.info("NLTK punkt tokenizer downloaded successfully")


def chunk_text_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using NLTK sent_tokenize.

    Args:
        text: Input text to chunk into sentences

    Returns:
        List of sentence strings (empty sentences are filtered out)

    Examples:
        >>> chunk_text_into_sentences("Hello world. How are you?")
        ['Hello world.', 'How are you?']

        >>> chunk_text_into_sentences("Single sentence")
        ['Single sentence']
    """
    if not text or not text.strip():
        return []

    # Ensure NLTK data is available
    ensure_nltk_data()

    # Tokenize into sentences
    sentences = nltk.sent_tokenize(text)

    # Filter out empty sentences and strip whitespace
    filtered_sentences = [s.strip() for s in sentences if s.strip()]

    return filtered_sentences
