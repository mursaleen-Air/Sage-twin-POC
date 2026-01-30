# state_engine.py
"""
Step 2: Business State Engine
Creates and maintains the living state of the digital twin
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import statistics


class BusinessState:
    """
    The Digital Twin's live internal memory.
    Stores current state, calculates baselines, and tracks history.
    """
    
    def __init__(self):
        # Core KPIs (Step 1 - What the twin represents)
        self.current_state: Dict[str, float] = {}
        self.baseline_state: Dict[str, float] = {}
        self.previous_state: Dict[str, float] = {}
        
        # Calculated metrics
        self.growth_rates: Dict[str, float] = {}
        self.volatility_scores: Dict[str, float] = {}
        self.averages: Dict[str, float] = {}
        
        # Historical data for trends
        self.history: List[Dict[str, Any]] = []
        
        # Business Health Index
        self.health_score: float = 50.0
        self.confidence_score: float = 0.0
        
        # Metadata
        self.last_updated: Optional[str] = None
        self.initialized: bool = False
    
    def initialize_from_csv(self, rows: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Process CSV data and build the structured state object.
        Returns initialization summary.
        """
        if not rows:
            return {"success": False, "error": "No data provided"}
        
        # Extract and normalize numeric values
        numeric_data: Dict[str, List[float]] = {}
        
        for row in rows:
            for key, value in row.items():
                clean_key = self._normalize_key(key)
                try:
                    num_val = float(str(value).replace(",", "").replace("$", "").replace("%", "").strip())
                    if clean_key not in numeric_data:
                        numeric_data[clean_key] = []
                    numeric_data[clean_key].append(num_val)
                except (ValueError, AttributeError):
                    pass
        
        # Calculate state metrics
        for key, values in numeric_data.items():
            # Current value (latest)
            self.current_state[key] = values[-1] if values else 0
            
            # Baseline (first value or average)
            self.baseline_state[key] = values[0] if values else 0
            
            # Average
            self.averages[key] = statistics.mean(values) if values else 0
            
            # Growth rate
            if len(values) >= 2 and values[0] != 0:
                self.growth_rates[key] = ((values[-1] - values[0]) / values[0]) * 100
            else:
                self.growth_rates[key] = 0
            
            # Volatility (standard deviation / mean)
            if len(values) >= 2 and self.averages[key] != 0:
                self.volatility_scores[key] = (statistics.stdev(values) / self.averages[key]) * 100
            else:
                self.volatility_scores[key] = 0
        
        # Store in history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "state": self.current_state.copy(),
            "type": "initialization"
        })
        
        # Calculate initial health score
        self._calculate_health_score()
        
        # Set metadata
        self.last_updated = datetime.now().isoformat()
        self.initialized = True
        self.confidence_score = min(100, len(rows) * 10)  # More data = more confidence
        
        return {
            "success": True,
            "metrics_count": len(self.current_state),
            "rows_processed": len(rows),
            "health_score": self.health_score,
            "confidence_score": self.confidence_score
        }
    
    def _normalize_key(self, key: str) -> str:
        """Normalize metric key names"""
        return key.strip().lower().replace(" ", "_").replace("-", "_")
    
    def _calculate_health_score(self):
        """
        Step 7: Calculate unified Business Health Index (0-100)
        Combines: Revenue growth, Sentiment, Risk, Stability
        """
        score_components = []
        
        # Revenue health (growth is good)
        revenue_growth = self.growth_rates.get("revenue", 0)
        revenue_score = min(100, max(0, 50 + revenue_growth))
        score_components.append(("revenue", revenue_score, 0.3))
        
        # Sentiment health (higher is better)
        sentiment = self.current_state.get("sentiment", 50)
        sentiment_score = min(100, max(0, sentiment))
        score_components.append(("sentiment", sentiment_score, 0.2))
        
        # Risk health (lower is better)
        risk = self.current_state.get("risk_score", 50)
        risk_score = max(0, 100 - risk)
        score_components.append(("risk", risk_score, 0.25))
        
        # Stability (low volatility is good)
        avg_volatility = statistics.mean(self.volatility_scores.values()) if self.volatility_scores else 0
        stability_score = max(0, 100 - avg_volatility)
        score_components.append(("stability", stability_score, 0.25))
        
        # Weighted average
        total_weight = sum(w for _, _, w in score_components)
        self.health_score = sum(s * w for _, s, w in score_components) / total_weight if total_weight > 0 else 50
        
        return self.health_score
    
    def update_state(self, new_state: Dict[str, float], reason: str = "simulation"):
        """Update current state and record in history"""
        self.previous_state = self.current_state.copy()
        self.current_state = new_state.copy()
        
        # Record in history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "state": new_state.copy(),
            "type": reason
        })
        
        # Keep history manageable
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        # Recalculate health
        self._calculate_health_score()
        self.last_updated = datetime.now().isoformat()
    
    def get_deltas(self) -> Dict[str, Dict[str, float]]:
        """Calculate before vs after comparison with delta indicators"""
        deltas = {}
        for key in self.current_state:
            current = self.current_state.get(key, 0)
            baseline = self.baseline_state.get(key, current)
            previous = self.previous_state.get(key, baseline)
            
            if baseline != 0:
                delta_from_baseline = ((current - baseline) / baseline) * 100
            else:
                delta_from_baseline = 0
            
            if previous != 0:
                delta_from_previous = ((current - previous) / previous) * 100
            else:
                delta_from_previous = 0
            
            deltas[key] = {
                "current": current,
                "baseline": baseline,
                "previous": previous,
                "delta_baseline_pct": round(delta_from_baseline, 2),
                "delta_previous_pct": round(delta_from_previous, 2),
                "direction": "up" if current > previous else "down" if current < previous else "stable"
            }
        
        return deltas
    
    def get_snapshot(self) -> Dict[str, Any]:
        """Get complete state snapshot"""
        return {
            "current_state": self.current_state,
            "baseline_state": self.baseline_state,
            "averages": self.averages,
            "growth_rates": self.growth_rates,
            "volatility_scores": self.volatility_scores,
            "health_score": round(self.health_score, 1),
            "confidence_score": round(self.confidence_score, 1),
            "last_updated": self.last_updated,
            "deltas": self.get_deltas()
        }


# Global state instance
business_state = BusinessState()
