"""
ARCA Chat Session - Conversation state management

Dataclass holding all conversation state for a WebSocket session.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class ChatSession:
    """Holds conversation state for a single WebSocket session.

    Attributes:
        session_id: Unique identifier for this session
        conversation_history: List of message dicts (role, content)
        project_name: Optional project name from context
        site_name: Optional site name from context
        last_analysis_category: Last used analysis category (e.g. material type)
        last_analysis_context: Last used analysis context (e.g. usage scenario)
        notes: Session notes accumulated during conversation

        Phii tracking (for implicit feedback):
        last_user_message: Previous user message
        last_assistant_response: Previous assistant response
        last_message_id: ID of last message exchange
        last_tools_used: Tools used in last response
    """

    session_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)

    # Project context
    project_name: Optional[str] = None
    site_name: Optional[str] = None
    last_analysis_category: Optional[str] = None
    last_analysis_context: Optional[str] = None
    notes: List[str] = field(default_factory=list)

    # Phii implicit feedback tracking
    last_user_message: str = ""
    last_assistant_response: str = ""
    last_message_id: str = ""
    last_tools_used: List[str] = field(default_factory=list)

    def update_from_analysis(self, category: str, context: str) -> None:
        """Update session with analysis parameters."""
        self.last_analysis_category = category
        self.last_analysis_context = context

    def add_note(self, note: str) -> None:
        """Add a note to the session."""
        self.notes.append(note)

    def get_notes_string(self) -> str:
        """Get formatted notes string for system prompt."""
        if not self.notes and not self.last_analysis_category:
            return ""
        parts = []
        if self.last_analysis_category:
            parts.append(f"Last analysis: {self.last_analysis_category}, {self.last_analysis_context}")
        if self.notes:
            parts.append("Notes: " + "; ".join(self.notes[-5:]))
        return "SESSION NOTES:\n" + "\n".join(parts)

    def add_exchange(self, user_message: str, assistant_response: str, tools_used: List[str]) -> None:
        """Record a message exchange for history and Phii tracking.

        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            tools_used: List of tools that were used
        """
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})

        # Update Phii tracking
        self.last_user_message = user_message
        self.last_assistant_response = assistant_response
        self.last_message_id = f"{self.session_id}_{len(self.conversation_history)}"
        self.last_tools_used = tools_used

        # Trim history if too long
        if len(self.conversation_history) > 30:
            self.conversation_history = self.conversation_history[-30:]

    def get_messages_for_llm(self, system_prompt: str, current_message: str) -> List[Dict[str, str]]:
        """Build message list for LLM call.

        Args:
            system_prompt: The system prompt to use
            current_message: The current user message

        Returns:
            List of message dicts ready for LLM
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": current_message})
        return messages

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary for Redis persistence.

        Returns:
            Dict representation of session state
        """
        return {
            "session_id": self.session_id,
            "conversation_history": self.conversation_history,
            "project_name": self.project_name,
            "site_name": self.site_name,
            "last_analysis_category": self.last_analysis_category,
            "last_analysis_context": self.last_analysis_context,
            "notes": self.notes,
            "last_user_message": self.last_user_message,
            "last_assistant_response": self.last_assistant_response,
            "last_message_id": self.last_message_id,
            "last_tools_used": self.last_tools_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatSession":
        """Deserialize session from dictionary.

        Args:
            data: Dict representation of session state

        Returns:
            ChatSession instance
        """
        return cls(
            session_id=data["session_id"],
            conversation_history=data.get("conversation_history", []),
            project_name=data.get("project_name"),
            site_name=data.get("site_name"),
            last_analysis_category=data.get("last_analysis_category"),
            last_analysis_context=data.get("last_analysis_context"),
            notes=data.get("notes", []),
            last_user_message=data.get("last_user_message", ""),
            last_assistant_response=data.get("last_assistant_response", ""),
            last_message_id=data.get("last_message_id", ""),
            last_tools_used=data.get("last_tools_used", []),
        )
