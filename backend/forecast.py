# forecast.py
"""
Step 8: Forecast Layer
Lightweight projection for 3 months forward with trend line and risk trajectory
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta


def generate_forecast(
    current_state: Dict[str, float],
    adjustments: Dict[str, float],
    months: int = 3
) -> Dict[str, Any]:
    """
    Generate monthly projections for key metrics.
    Returns trend lines and risk trajectory.
    """
    
    # Key metrics to forecast
    forecast_metrics = ["revenue", "customers", "sentiment", "churn_rate", "risk_score", "costs"]
    
    # Monthly projections
    projections: Dict[str, List[Dict[str, Any]]] = {metric: [] for metric in forecast_metrics}
    
    # Current values
    working_state = current_state.copy()
    
    for month in range(months + 1):  # 0 = current, 1-3 = projections
        month_date = datetime.now() + timedelta(days=30 * month)
        
        for metric in forecast_metrics:
            if metric not in working_state:
                continue
                
            base_value = current_state.get(metric, 0)
            current_value = working_state.get(metric, base_value)
            
            if month == 0:
                # Current state
                projections[metric].append({
                    "month": month_date.strftime("%b %Y"),
                    "value": current_value,
                    "type": "current"
                })
            else:
                # Project forward based on adjustments and natural trends
                monthly_change = _calculate_monthly_trend(
                    metric, current_value, adjustments, month
                )
                projected_value = current_value * (1 + monthly_change / 100)
                
                # Dampen projections over time (uncertainty increases)
                confidence_factor = max(0.5, 1 - (month * 0.1))
                
                projections[metric].append({
                    "month": month_date.strftime("%b %Y"),
                    "value": round(projected_value, 2),
                    "type": "projection",
                    "confidence": round(confidence_factor * 100, 0),
                    "change_from_current": round(((projected_value - base_value) / base_value) * 100, 1) if base_value > 0 else 0
                })
                
                # Update working state for next iteration
                working_state[metric] = projected_value
    
    # Calculate trend direction for each metric
    trends = {}
    for metric, data in projections.items():
        if len(data) >= 2:
            first = data[0]["value"]
            last = data[-1]["value"]
            if first > 0:
                trend_pct = ((last - first) / first) * 100
                trends[metric] = {
                    "direction": "up" if trend_pct > 0 else "down" if trend_pct < 0 else "stable",
                    "change_pct": round(trend_pct, 1),
                    "outlook": _get_outlook(metric, trend_pct)
                }
    
    # Risk trajectory
    risk_trajectory = _calculate_risk_trajectory(projections, current_state)
    
    # Overall forecast summary
    summary = _generate_forecast_summary(projections, trends, risk_trajectory)
    
    return {
        "projections": projections,
        "trends": trends,
        "risk_trajectory": risk_trajectory,
        "summary": summary,
        "forecast_months": months,
        "generated_at": datetime.now().isoformat()
    }


def _calculate_monthly_trend(
    metric: str, 
    current_value: float, 
    adjustments: Dict[str, float],
    month: int
) -> float:
    """Calculate monthly trend based on metric type and adjustments"""
    
    # Base growth rates (% per month)
    base_rates = {
        "revenue": 2.0,      # Slight natural growth
        "customers": 1.5,    # Customer base tends to grow
        "sentiment": -0.5,   # Sentiment naturally drifts down without effort
        "churn_rate": 0.2,   # Churn tends to creep up
        "risk_score": 0.3,   # Risk tends to accumulate
        "costs": 1.0,        # Costs naturally rise
    }
    
    base_rate = base_rates.get(metric, 0)
    
    # Adjustment impacts
    adjustment_impact = 0
    
    if metric == "revenue":
        adjustment_impact += adjustments.get("marketing_spend", 0) * 0.05
        adjustment_impact += adjustments.get("price", 0) * 0.03
        adjustment_impact -= adjustments.get("churn_rate", 0) * 0.08
    
    elif metric == "customers":
        adjustment_impact += adjustments.get("marketing_spend", 0) * 0.04
        adjustment_impact -= adjustments.get("price", 0) * 0.02
        adjustment_impact -= adjustments.get("delivery_delay", 0) * 0.05
    
    elif metric == "sentiment":
        adjustment_impact -= adjustments.get("delivery_delay", 0) * 0.5
        adjustment_impact -= adjustments.get("price", 0) * 0.03
        adjustment_impact += adjustments.get("marketing_spend", 0) * 0.02
    
    elif metric == "churn_rate":
        adjustment_impact += adjustments.get("price", 0) * 0.02
        adjustment_impact += adjustments.get("delivery_delay", 0) * 0.1
        adjustment_impact -= adjustments.get("marketing_spend", 0) * 0.01
    
    elif metric == "risk_score":
        adjustment_impact += adjustments.get("delivery_delay", 0) * 0.3
        adjustment_impact += adjustments.get("costs", 0) * 0.02
        adjustment_impact -= adjustments.get("revenue", 0) * 0.02
    
    # Diminishing returns over time
    time_factor = 1.0 / (1 + month * 0.1)
    
    return (base_rate + adjustment_impact) * time_factor


def _get_outlook(metric: str, trend_pct: float) -> str:
    """Determine outlook based on metric and trend"""
    
    positive_metrics = ["revenue", "customers", "sentiment"]
    negative_metrics = ["churn_rate", "risk_score", "costs", "delivery_delay"]
    
    if metric in positive_metrics:
        if trend_pct > 10:
            return "strong"
        elif trend_pct > 0:
            return "positive"
        elif trend_pct > -10:
            return "cautious"
        else:
            return "concerning"
    else:  # negative metrics - lower is better
        if trend_pct < -10:
            return "strong"
        elif trend_pct < 0:
            return "positive"
        elif trend_pct < 10:
            return "cautious"
        else:
            return "concerning"


def _calculate_risk_trajectory(
    projections: Dict[str, List[Dict]], 
    current_state: Dict[str, float]
) -> List[Dict[str, Any]]:
    """Calculate risk trajectory over forecast period"""
    
    risk_projection = projections.get("risk_score", [])
    
    trajectory = []
    for i, point in enumerate(risk_projection):
        risk_value = point["value"]
        
        if risk_value > 70:
            level = "critical"
            action = "Immediate intervention required"
        elif risk_value > 50:
            level = "high"
            action = "Action needed within 30 days"
        elif risk_value > 30:
            level = "moderate"
            action = "Monitor closely"
        else:
            level = "low"
            action = "Maintain current strategy"
        
        trajectory.append({
            "month": point["month"],
            "risk": round(risk_value, 1),
            "level": level,
            "action": action
        })
    
    return trajectory


def _generate_forecast_summary(
    projections: Dict[str, List[Dict]],
    trends: Dict[str, Dict],
    risk_trajectory: List[Dict]
) -> Dict[str, Any]:
    """Generate executive summary of forecast"""
    
    positive_trends = sum(1 for t in trends.values() if t.get("outlook") in ["strong", "positive"])
    concerning_trends = sum(1 for t in trends.values() if t.get("outlook") == "concerning")
    
    # Overall outlook
    if concerning_trends >= 2:
        outlook = "challenging"
        outlook_description = "Multiple metrics show concerning trends. Strategic intervention recommended."
    elif positive_trends >= 3:
        outlook = "favorable"
        outlook_description = "Most metrics showing positive momentum. Consider growth investments."
    else:
        outlook = "stable"
        outlook_description = "Mixed signals. Focus on optimization and risk monitoring."
    
    # Final risk level
    final_risk = risk_trajectory[-1] if risk_trajectory else {"level": "unknown", "risk": 50}
    
    return {
        "outlook": outlook,
        "description": outlook_description,
        "positive_trends": positive_trends,
        "concerning_trends": concerning_trends,
        "projected_risk_level": final_risk["level"],
        "recommendation": final_risk.get("action", "Monitor situation")
    }
