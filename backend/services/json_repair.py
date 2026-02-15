"""
Shared JSON Repair Utility

Extracts, repairs, and parses JSON from LLM responses.
Vision models frequently produce malformed JSON - this module handles the 8 most
common failure modes in a deterministic repair pipeline.

Used by: GraphExtractor, LoggView, Mapperr (and any future structured extraction)
"""

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


def extract_json_from_response(text: str) -> Optional[str]:
    """
    Extract JSON string from an LLM response.

    Tries (in order):
    1. ```json fenced code blocks
    2. Raw JSON object (outermost { ... })

    Args:
        text: Raw LLM response text

    Returns:
        Extracted JSON string, or None if no JSON found
    """
    if not text:
        return None

    # Try ```json blocks first (most reliable)
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # Try raw JSON object
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        return json_match.group(0).strip()

    return None


def repair_json(json_str: str) -> str:
    """
    Attempt to repair malformed JSON from LLM output.

    Handles 8 common LLM JSON failure modes:
    1. Truncated output (unclosed braces from context limit)
    2. Python literals (None, True, False instead of null, true, false)
    3. Single quotes instead of double quotes
    4. Trailing commas before } or ]
    5. Empty values (missing value after colon)
    6. Units attached to numbers (e.g., "depth": 1.5m)
    7. Unquoted string values
    8. Unclosed braces/brackets from truncation

    Args:
        json_str: Malformed JSON string

    Returns:
        Repaired JSON string (may still be invalid in edge cases)
    """
    original = json_str

    # Step 1: Truncate at last complete brace (in case of truncated output)
    brace_count = 0
    last_valid_pos = 0
    for i, c in enumerate(json_str):
        if c == "{":
            brace_count += 1
        elif c == "}":
            brace_count -= 1
            if brace_count == 0:
                last_valid_pos = i + 1
    if last_valid_pos > 0 and last_valid_pos < len(json_str):
        json_str = json_str[:last_valid_pos]
        logger.debug(f"Truncated trailing content after position {last_valid_pos}")

    # Step 2: Fix Python-style values
    json_str = re.sub(r"\bNone\b", "null", json_str)
    json_str = re.sub(r"\bTrue\b", "true", json_str)
    json_str = re.sub(r"\bFalse\b", "false", json_str)

    # Step 3: Single quotes to double quotes (careful with apostrophes in text)
    # Only replace quotes that look like JSON delimiters
    json_str = re.sub(r"(?<=[{,:\[])\s*'([^']*?)'\s*(?=[},:\]])", r'"\1"', json_str)
    json_str = re.sub(r"'(\w+)':", r'"\1":', json_str)  # Keys with single quotes

    # Step 4: Remove trailing commas
    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    # Step 5: Fix empty values (e.g., "key": , -> "key": null,)
    json_str = re.sub(r":\s*,", ": null,", json_str)
    json_str = re.sub(r":\s*}", ": null}", json_str)

    # Step 6: Fix values with units attached (e.g., "depth": 1.5m -> "depth": 1.5)
    json_str = re.sub(r":\s*(\d+\.?\d*)\s*m\s*([,}])", r": \1\2", json_str)
    json_str = re.sub(r":\s*(\d+\.?\d*)\s*mm\s*([,}])", r": \1\2", json_str)

    # Step 7: Fix unquoted string values that should be quoted
    # Match patterns like: "key": value, where value is not a number, null, true, false
    json_str = re.sub(
        r":\s*([a-zA-Z][a-zA-Z0-9_\s]*[a-zA-Z0-9])\s*([,}])",
        lambda m: (
            f': "{m.group(1).strip()}"{m.group(2)}'
            if m.group(1).strip().lower() not in ("null", "true", "false")
            else m.group(0)
        ),
        json_str,
    )

    # Step 8: Close unclosed braces/brackets if truncated
    open_braces = json_str.count("{") - json_str.count("}")
    open_brackets = json_str.count("[") - json_str.count("]")
    if open_braces > 0 or open_brackets > 0:
        json_str += "]" * open_brackets + "}" * open_braces
        logger.debug(f"Closed {open_braces} braces and {open_brackets} brackets")

    if json_str != original:
        logger.info("Applied JSON repairs")

    return json_str


def parse_json_response(text: str) -> Optional[Any]:
    """
    Full pipeline: extract JSON from LLM response, repair, and parse.

    Args:
        text: Raw LLM response text

    Returns:
        Parsed Python object (dict/list), or None if extraction/parsing fails
    """
    json_str = extract_json_from_response(text)
    if json_str is None:
        logger.warning("No JSON found in response")
        return None

    # Try parsing as-is first (fast path)
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Apply repairs and retry
    repaired = repair_json(json_str)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed after all repairs: {e}")
        return None
