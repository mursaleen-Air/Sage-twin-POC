# memory.py
"""
Conversation Memory and Context Management
Stores chat history and derived insights for context-aware responses
"""

from datetime import datetime
from typing import List, Dict, Any
import json


class ConversationMemory:
    """Manages conversation history and context"""
    
    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.messages: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.company_data: Dict[str, Any] = {}
        self.insights: List[str] = []
        self.goals: List[str] = []
        self.decisions: List[Dict[str, Any]] = []
        
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.messages.append(message)
        
        # Trim history if needed
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
    
    def add_insight(self, insight: str):
        """Store a derived insight"""
        if insight not in self.insights:
            self.insights.append(insight)
            if len(self.insights) > 10:
                self.insights = self.insights[-10:]
    
    def add_goal(self, goal: str):
        """Track user goals"""
        if goal not in self.goals:
            self.goals.append(goal)
    
    def add_decision(self, decision: Dict[str, Any]):
        """Record a business decision made"""
        decision["timestamp"] = datetime.now().isoformat()
        self.decisions.append(decision)
    
    def set_company_data(self, data: Dict[str, Any]):
        """Store uploaded company data"""
        self.company_data = data
    
    def get_context_summary(self) -> str:
        """Generate a summary of current context for AI"""
        summary_parts = []
        
        if self.company_data:
            metrics = ", ".join([f"{k}: {v}" for k, v in self.company_data.items() 
                               if isinstance(v, (int, float))])
            summary_parts.append(f"Company Metrics: {metrics}")
        
        if self.goals:
            summary_parts.append(f"User Goals: {', '.join(self.goals[-3:])}")
        
        if self.insights:
            summary_parts.append(f"Key Insights: {'; '.join(self.insights[-3:])}")
        
        if self.decisions:
            recent = self.decisions[-2:]
            decisions_text = "; ".join([d.get("action", "unknown") for d in recent])
            summary_parts.append(f"Recent Decisions: {decisions_text}")
        
        return "\n".join(summary_parts) if summary_parts else "No prior context."
    
    def get_chat_history(self, limit: int = 10) -> List[Dict[str, str]]:
        """Get recent chat history for AI context"""
        recent = self.messages[-limit:]
        return [{"role": m["role"], "content": m["content"]} for m in recent]
    
    def clear(self):
        """Clear all memory"""
        self.messages = []
        self.context = {}
        self.insights = []
        self.goals = []
        self.decisions = []
        # Keep company data


# Global memory instance
conversation_memory = ConversationMemory()
