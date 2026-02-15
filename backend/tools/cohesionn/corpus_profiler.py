"""
Corpus Profiler - Lightweight term extraction after ingest.

Analyzes ingested chunks to extract domain-specific vocabulary,
stores as corpus_profile.json, and provides terms for Phii context injection.
"""

import json
import logging
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Hardcoded English stop words (~200 most common).
# No external dependency needed.
STOP_WORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for",
    "from", "further", "get", "got", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how",
    "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "just", "let's", "like", "may",
    "me", "might", "more", "most", "mustn't", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "per", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd",
    "they'll", "they're", "they've", "this", "those", "through", "to",
    "too", "under", "until", "up", "upon", "us", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while",
    "who", "who's", "whom", "why", "why's", "will", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves",
    # Common filler words that aren't domain-specific
    "also", "based", "using", "used", "well", "much", "many", "often",
    "however", "therefore", "thus", "hence", "since", "although", "though",
    "within", "without", "another", "either", "neither", "rather",
    "whether", "yet", "still", "already", "always", "never", "ever",
    "even", "every", "several", "various", "specific", "particular",
    "general", "different", "similar", "following", "including",
    "according", "regarding", "typically", "generally", "usually",
    "commonly", "approximately", "respectively", "especially",
    "significantly", "relatively", "particularly", "essentially",
    # Common document words
    "figure", "table", "chapter", "section", "page", "reference",
    "references", "note", "notes", "see", "shown", "described",
    "provided", "present", "presented", "report", "document",
    "example", "examples", "data", "results", "result", "method",
    "methods", "analysis", "total", "average", "value", "values",
    "number", "new", "first", "second", "two", "three", "one",
})

# Regex for tokenization: alphanumeric sequences (including hyphens for compound terms)
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9-]*[a-zA-Z0-9]|[a-zA-Z]")


def _tokenize(text: str) -> List[str]:
    """Extract lowercased word tokens from text."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _is_valid_term(term: str) -> bool:
    """Check if a term is worth keeping."""
    # Too short
    if len(term) < 3:
        return False
    # Pure numbers
    if term.replace("-", "").replace(".", "").isdigit():
        return False
    # All same character
    if len(set(term.replace("-", ""))) <= 1:
        return False
    # Stop word
    if term in STOP_WORDS:
        return False
    return True


def extract_terms(
    chunk_texts: List[str],
    top_n: int = 100,
) -> List[Dict[str, Any]]:
    """Extract domain-specific terms from chunk texts.

    Tokenizes into unigrams, bigrams, and trigrams.
    Filters stop words and common English filler.
    Returns top N terms by frequency.

    Args:
        chunk_texts: List of chunk content strings
        top_n: Number of top terms to return

    Returns:
        List of {"term": str, "frequency": int} dicts, sorted by frequency desc
    """
    unigram_counts = Counter()
    bigram_counts = Counter()
    trigram_counts = Counter()

    for text in chunk_texts:
        tokens = _tokenize(text)

        # Unigrams
        for t in tokens:
            if _is_valid_term(t):
                unigram_counts[t] += 1

        # Bigrams
        for i in range(len(tokens) - 1):
            a, b = tokens[i], tokens[i + 1]
            if a not in STOP_WORDS and b not in STOP_WORDS and len(a) >= 2 and len(b) >= 2:
                bigram = f"{a} {b}"
                bigram_counts[bigram] += 1

        # Trigrams
        for i in range(len(tokens) - 2):
            a, b, c = tokens[i], tokens[i + 1], tokens[i + 2]
            # Allow one stop word in the middle (e.g. "factor of safety")
            outer_ok = a not in STOP_WORDS and c not in STOP_WORDS
            if outer_ok and len(a) >= 2 and len(c) >= 2:
                trigram = f"{a} {b} {c}"
                trigram_counts[trigram] += 1

    # Merge all n-gram counts into a single pool
    # Prefer multi-word terms: weight bigrams 2x and trigrams 3x
    combined = Counter()

    for term, count in unigram_counts.items():
        if count >= 2:  # Skip hapax legomena for unigrams
            combined[term] = count

    for term, count in bigram_counts.items():
        if count >= 2:
            combined[term] = count * 2

    for term, count in trigram_counts.items():
        if count >= 2:
            combined[term] = count * 3

    # Take top N
    top_terms = combined.most_common(top_n)

    return [{"term": term, "frequency": freq} for term, freq in top_terms]


def build_corpus_profile(
    chunk_texts: List[str],
    document_count: int,
    top_n: int = 100,
) -> Dict[str, Any]:
    """Build a complete corpus profile from chunk texts.

    Args:
        chunk_texts: List of chunk content strings
        document_count: Number of documents that were ingested
        top_n: Number of top terms to extract

    Returns:
        Complete profile dict ready for JSON serialization
    """
    terms = extract_terms(chunk_texts, top_n=top_n)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": document_count,
        "chunk_count": len(chunk_texts),
        "terms": terms,
    }


def save_corpus_profile(
    profile: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """Save corpus profile to disk.

    Args:
        profile: Profile dict from build_corpus_profile()
        output_path: Override path (uses config default if None)

    Returns:
        Path where profile was saved
    """
    if output_path is None:
        from config import runtime_config
        output_path = runtime_config.corpus_profile_path

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        f"Corpus profile saved: {len(profile.get('terms', []))} terms "
        f"from {profile.get('document_count', 0)} documents -> {path}"
    )
    return str(path)


def load_corpus_profile(profile_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load corpus profile from disk.

    Args:
        profile_path: Override path (uses config default if None)

    Returns:
        Profile dict, or None if file doesn't exist or is invalid
    """
    if profile_path is None:
        from config import runtime_config
        profile_path = runtime_config.corpus_profile_path

    path = Path(profile_path)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "terms" not in data:
            logger.warning(f"Invalid corpus profile format: {path}")
            return None
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load corpus profile: {e}")
        return None


def get_top_terms_text(top_n: int = 20, profile_path: Optional[str] = None) -> Optional[str]:
    """Get a comma-separated string of top domain terms.

    Convenience function for Phii context injection.

    Args:
        top_n: Number of terms to include
        profile_path: Override path (uses config default if None)

    Returns:
        # domain-specific terms extracted from corpus (e.g., technical abbreviations, named methods)
        String of comma-separated terms, or None if no profile exists
    """
    profile = load_corpus_profile(profile_path)
    if not profile:
        return None

    terms = profile.get("terms", [])[:top_n]
    if not terms:
        return None

    return ", ".join(t["term"] for t in terms)


def profile_after_ingest(
    chunks: List[Any],
    document_count: int,
) -> None:
    """Run corpus profiling after an ingest operation.

    Non-blocking: logs warnings on failure, never raises.

    Args:
        chunks: List of Chunk objects (from chunker) with .content attribute,
                or list of dicts with "content" key
        document_count: Number of documents ingested
    """
    from config import runtime_config

    if not runtime_config.corpus_profiling_enabled:
        return

    try:
        # Extract text from chunks (handle both Chunk objects and dicts)
        texts = []
        for chunk in chunks:
            if hasattr(chunk, "content"):
                texts.append(chunk.content)
            elif isinstance(chunk, dict) and "content" in chunk:
                texts.append(chunk["content"])

        if not texts:
            logger.debug("No chunk texts for corpus profiling")
            return

        profile = build_corpus_profile(texts, document_count)
        save_corpus_profile(profile)

    except Exception as e:
        logger.warning(f"Corpus profiling failed (non-fatal): {e}")
