"""
ARCA Logging Configuration - Color-Coded Docker Logs

Provides:
- ColorFormatter: ANSI color-coded log output
- Helper functions: log_message_in, log_message_out, log_thinking, log_tool, log_llm
- setup_logging(): Configure application logging

Usage:
    from logging_config import setup_logging, log_message_in, log_tool
    setup_logging()
    logger = logging.getLogger(__name__)
    log_message_in(logger, "User question", search=True)
"""

import logging
import sys

# ANSI color codes
COLORS = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    # Event colors
    "MSG_IN": "\033[96m",  # Cyan - incoming message
    "MSG_OUT": "\033[92m",  # Green - outgoing response
    "THINKING": "\033[95m",  # Magenta - thinking mode
    "TOOL": "\033[93m",  # Yellow - tool calls
    "LLM": "\033[94m",  # Blue - LLM operations
    "ERROR": "\033[91m",  # Red - errors
    "WARN": "\033[33m",  # Orange/Yellow - warnings
    "DEBUG": "\033[90m",  # Gray - debug info
}


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    LEVEL_COLORS = {
        logging.DEBUG: COLORS["DEBUG"],
        logging.INFO: COLORS["RESET"],
        logging.WARNING: COLORS["WARN"],
        logging.ERROR: COLORS["ERROR"],
        logging.CRITICAL: COLORS["ERROR"] + COLORS["BOLD"],
    }

    def format(self, record: logging.LogRecord) -> str:
        # Apply level-based color
        color = self.LEVEL_COLORS.get(record.levelno, COLORS["RESET"])

        # Format: timestamp [LEVEL] message (no module name for compactness)
        timestamp = self.formatTime(record, "%H:%M:%S")
        level = record.levelname[:4]

        formatted = (
            f"{COLORS['DIM']}{timestamp}{COLORS['RESET']} "
            f"[{color}{level}{COLORS['RESET']}] "
            f"{record.getMessage()}"
        )

        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)

        return formatted


def setup_logging(level: int = logging.INFO) -> None:
    """Configure colored logging for the application."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColorFormatter())

    # Configure root logger
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)  # Suppress auth warnings
    logging.getLogger("tqdm").setLevel(logging.WARNING)


# =============================================================================
# COLORED LOG HELPER FUNCTIONS
# =============================================================================


def log_message_in(logger: logging.Logger, message: str, **context) -> None:
    """Log incoming user message.

    Args:
        logger: Logger instance
        message: User message text
        **context: Additional context (search, deep, think, phii, etc.)
    """
    preview = message[:80] + "..." if len(message) > 80 else message
    ctx = " ".join(f"{k}={v}" for k, v in context.items())
    logger.info(f"{COLORS['MSG_IN']}>>> MESSAGE{COLORS['RESET']} {preview} [{ctx}]")


def log_message_out(
    logger: logging.Logger,
    tools_used: list = None,
    citations: int = 0,
) -> None:
    """Log outgoing response.

    Args:
        logger: Logger instance
        tools_used: List of tool names used
        citations: Number of citations included
    """
    tools = ", ".join(tools_used) if tools_used else "none"
    logger.info(f"{COLORS['MSG_OUT']}<<< RESPONSE{COLORS['RESET']} " f"tools=[{tools}] citations={citations}")


def log_thinking(logger: logging.Logger, state: str, chars: int = 0) -> None:
    """Log thinking mode start/end.

    Args:
        logger: Logger instance
        state: 'start' or 'end'
        chars: Character count (for end state)
    """
    if state == "start":
        logger.info(f"{COLORS['THINKING']}... THINKING{COLORS['RESET']} started")
    else:
        logger.info(f"{COLORS['THINKING']}... THINKING{COLORS['RESET']} " f"done ({chars} chars)")


def log_tool(
    logger: logging.Logger,
    tool_name: str,
    state: str,
    **context,
) -> None:
    """Log tool execution.

    Args:
        logger: Logger instance
        tool_name: Name of the tool
        state: 'start' or 'end'
        **context: Additional context (query, results, etc.)
    """
    ctx = " ".join(f"{k}={v}" for k, v in context.items()) if context else ""
    if state == "start":
        logger.info(f"{COLORS['TOOL']}>>> TOOL{COLORS['RESET']} {tool_name} {ctx}")
    else:
        logger.info(f"{COLORS['TOOL']}<<< TOOL{COLORS['RESET']} {tool_name} {ctx}")


def log_llm(
    logger: logging.Logger,
    state: str,
    model: str = "",
    duration: float = 0,
) -> None:
    """Log LLM call.

    Args:
        logger: Logger instance
        state: 'start' or 'end'
        model: Model name
        duration: Call duration in seconds (for end state)
    """
    if state == "start":
        logger.info(f"{COLORS['LLM']}>>> LLM{COLORS['RESET']} calling {model}")
    else:
        logger.info(f"{COLORS['LLM']}<<< LLM{COLORS['RESET']} " f"{model} completed in {duration:.1f}s")
