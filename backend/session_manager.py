# session_manager.py
"""
Session Management for Multi-User Support
Each user gets their own isolated Digital Twin state.
"""

import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from state_engine import BusinessState
from data_sources import DataSourceManager


from monitoring.drift_detector import DriftDetector
from ml.models.revenue_forecaster import RevenueForecaster
from ml.models.churn_predictor import ChurnPredictor
from multi_agents import MultiAgentEngine


class UserSession:
    """Represents a single user's session with their own state"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.business_state = BusinessState()
        self.data_source_manager = DataSourceManager()
        self.multi_agent_engine = MultiAgentEngine()  # Isolated simulation engine
        self.drift_detector = DriftDetector()
        self.revenue_forecaster = RevenueForecaster()
        self.churn_predictor = ChurnPredictor()
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
    
    def touch(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()
    
    def is_expired(self, max_age_hours: int = 24) -> bool:
        """Check if session has expired"""
        return datetime.now() - self.last_accessed > timedelta(hours=max_age_hours)


class SessionManager:
    """
    Manages multiple user sessions.
    Each session has its own BusinessState and DataSourceManager.
    """
    
    def __init__(self, max_sessions: int = 1000, session_ttl_hours: int = 24):
        self.sessions: Dict[str, UserSession] = {}
        self.max_sessions = max_sessions
        self.session_ttl_hours = session_ttl_hours
    
    def create_session(self) -> str:
        """Create a new session and return the session ID"""
        # Clean up expired sessions first
        self._cleanup_expired()
        
        # If at capacity, remove oldest session
        if len(self.sessions) >= self.max_sessions:
            self._remove_oldest_session()
        
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = UserSession(session_id)
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get a session by ID, returns None if not found or expired"""
        session = self.sessions.get(session_id)
        
        if session is None:
            return None
        
        if session.is_expired(self.session_ttl_hours):
            del self.sessions[session_id]
            return None
        
        session.touch()
        return session
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> tuple[str, UserSession]:
        """Get existing session or create new one"""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session_id, session
        
        # Create new session
        new_id = self.create_session()
        return new_id, self.sessions[new_id]
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata"""
        session = self.get_session(session_id)
        if not session:
            return None
        
        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "initialized": session.business_state.initialized,
            "data_sources": session.data_source_manager.get_combined_state()
        }
    
    def list_sessions(self) -> Dict[str, Any]:
        """List all active sessions (for admin/debugging)"""
        self._cleanup_expired()
        
        return {
            "total_sessions": len(self.sessions),
            "max_sessions": self.max_sessions,
            "sessions": [
                {
                    "session_id": sid,
                    "created_at": s.created_at.isoformat(),
                    "last_accessed": s.last_accessed.isoformat(),
                    "initialized": s.business_state.initialized
                }
                for sid, s in self.sessions.items()
            ]
        }
    
    def _cleanup_expired(self):
        """Remove all expired sessions"""
        expired = [
            sid for sid, session in self.sessions.items()
            if session.is_expired(self.session_ttl_hours)
        ]
        for sid in expired:
            del self.sessions[sid]
    
    def _remove_oldest_session(self):
        """Remove the oldest session"""
        if not self.sessions:
            return
        
        oldest_id = min(
            self.sessions.keys(),
            key=lambda sid: self.sessions[sid].last_accessed
        )
        del self.sessions[oldest_id]


# Global session manager
session_manager = SessionManager()
