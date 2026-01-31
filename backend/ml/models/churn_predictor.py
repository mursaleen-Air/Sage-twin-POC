"""
Customer Churn Prediction Model
Predicts probability of customer churn based on behavioral and engagement signals.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ChurnRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChurnPrediction:
    """Churn prediction result"""
    churn_probability: float
    risk_level: ChurnRisk
    contributing_factors: Dict[str, float]
    retention_recommendations: List[str]
    expected_revenue_at_risk: float
    confidence: float


class ChurnPredictor:
    """
    Customer churn prediction model.
    Uses behavioral signals and engagement metrics to predict churn risk.
    """
    
    def __init__(self):
        self.is_trained = False
        self.feature_weights: Dict[str, float] = {}
        self.thresholds = {
            "low": 0.2,
            "medium": 0.4,
            "high": 0.6,
            "critical": 0.8
        }
        
        # Default churn factors (negative = increases churn)
        self.churn_factors = {
            "sentiment": -0.25,           # Lower sentiment = higher churn
            "delivery_delay": 0.15,       # Higher delay = higher churn
            "price_increase": 0.20,       # Price increases drive churn
            "support_tickets": 0.10,      # More tickets = higher churn
            "days_since_purchase": 0.12,  # Longer gaps = higher churn
            "engagement_score": -0.20,    # Lower engagement = higher churn
            "nps_score": -0.15,           # Lower NPS = higher churn
            "product_usage": -0.18,       # Lower usage = higher churn
            "competitor_mentions": 0.08,  # More competitor talk = higher churn
        }
        
        self.baseline_churn_rate = 0.05  # 5% baseline monthly churn
    
    def train(
        self,
        historical_data: List[Dict[str, Any]],
        churned_labels: Optional[List[bool]] = None
    ) -> Dict[str, Any]:
        """
        Train the churn predictor.
        
        Args:
            historical_data: Customer metrics over time
            churned_labels: Optional labels indicating if customer churned
            
        Returns:
            Training results
        """
        if len(historical_data) < 10:
            return {
                "success": False,
                "error": "Need at least 10 data points for training"
            }
        
        # If we have labels, learn from them
        if churned_labels and len(churned_labels) == len(historical_data):
            self._learn_from_labels(historical_data, churned_labels)
        else:
            # Use heuristic learning based on churn_rate metric
            self._learn_from_churn_rate(historical_data)
        
        self.is_trained = True
        
        return {
            "success": True,
            "samples_used": len(historical_data),
            "learned_factors": self.churn_factors,
            "baseline_churn_rate": self.baseline_churn_rate
        }
    
    def predict(
        self,
        customer_state: Dict[str, Any],
        revenue_per_customer: float = 1000.0
    ) -> ChurnPrediction:
        """
        Predict churn probability for current customer state.
        
        Args:
            customer_state: Current customer/business metrics
            revenue_per_customer: Average revenue per customer for risk calculation
            
        Returns:
            ChurnPrediction with probability and recommendations
        """
        # Calculate base probability
        log_odds = np.log(self.baseline_churn_rate / (1 - self.baseline_churn_rate))
        
        # Calculate contributing factors
        factors = {}
        
        for metric, weight in self.churn_factors.items():
            value = customer_state.get(metric)
            
            if value is not None:
                # Normalize value to contribution
                if metric in ["sentiment", "engagement_score", "nps_score", "product_usage"]:
                    # These are 0-100 scale, higher is better
                    normalized = (value - 50) / 50
                    contribution = weight * normalized
                elif metric in ["delivery_delay", "days_since_purchase", "support_tickets"]:
                    # These increase churn, normalize to typical ranges
                    if metric == "delivery_delay":
                        normalized = min(value / 10, 2)  # Cap at 2x
                    elif metric == "days_since_purchase":
                        normalized = min(value / 30, 3)  # Compare to 30 days
                    else:
                        normalized = min(value / 5, 2)
                    contribution = weight * normalized
                elif metric == "price_increase":
                    normalized = value / 10  # 10% increase = 1.0
                    contribution = weight * normalized
                else:
                    contribution = weight * (value / 100)
                
                factors[metric] = contribution
                log_odds += contribution
        
        # Convert log odds to probability
        churn_probability = 1 / (1 + np.exp(-log_odds))
        churn_probability = np.clip(churn_probability, 0.01, 0.99)
        
        # Determine risk level
        risk_level = self._get_risk_level(churn_probability)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(factors, customer_state)
        
        # Calculate revenue at risk
        customers = customer_state.get("customers", customer_state.get("total_customers", 100))
        revenue_at_risk = customers * churn_probability * revenue_per_customer
        
        return ChurnPrediction(
            churn_probability=churn_probability,
            risk_level=risk_level,
            contributing_factors=factors,
            retention_recommendations=recommendations,
            expected_revenue_at_risk=revenue_at_risk,
            confidence=0.85 if self.is_trained else 0.60
        )
    
    def predict_cohort(
        self,
        cohort_data: List[Dict[str, Any]],
        revenue_per_customer: float = 1000.0
    ) -> Dict[str, Any]:
        """
        Predict churn for a cohort of customers.
        
        Returns aggregate statistics and risk distribution.
        """
        predictions = [
            self.predict(customer, revenue_per_customer)
            for customer in cohort_data
        ]
        
        probabilities = [p.churn_probability for p in predictions]
        
        risk_distribution = {
            "low": sum(1 for p in predictions if p.risk_level == ChurnRisk.LOW),
            "medium": sum(1 for p in predictions if p.risk_level == ChurnRisk.MEDIUM),
            "high": sum(1 for p in predictions if p.risk_level == ChurnRisk.HIGH),
            "critical": sum(1 for p in predictions if p.risk_level == ChurnRisk.CRITICAL),
        }
        
        return {
            "cohort_size": len(cohort_data),
            "average_churn_probability": np.mean(probabilities),
            "max_churn_probability": np.max(probabilities),
            "min_churn_probability": np.min(probabilities),
            "risk_distribution": risk_distribution,
            "total_revenue_at_risk": sum(p.expected_revenue_at_risk for p in predictions),
            "high_risk_count": risk_distribution["high"] + risk_distribution["critical"]
        }
    
    def segment_by_risk(
        self,
        customer_states: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Segment customers by churn risk level"""
        segments = {
            "low": [],
            "medium": [],
            "high": [],
            "critical": []
        }
        
        for customer in customer_states:
            prediction = self.predict(customer)
            segments[prediction.risk_level.value].append({
                **customer,
                "churn_probability": prediction.churn_probability
            })
        
        return segments
    
    # ============================================
    # PRIVATE METHODS
    # ============================================
    
    def _get_risk_level(self, probability: float) -> ChurnRisk:
        """Convert probability to risk level"""
        if probability >= self.thresholds["critical"]:
            return ChurnRisk.CRITICAL
        elif probability >= self.thresholds["high"]:
            return ChurnRisk.HIGH
        elif probability >= self.thresholds["medium"]:
            return ChurnRisk.MEDIUM
        return ChurnRisk.LOW
    
    def _generate_recommendations(
        self,
        factors: Dict[str, float],
        state: Dict[str, Any]
    ) -> List[str]:
        """Generate retention recommendations based on risk factors"""
        recommendations = []
        
        # Sort factors by contribution to churn
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        
        for metric, contribution in sorted_factors[:3]:  # Top 3 factors
            if contribution > 0:  # Contributing to churn
                if metric == "sentiment":
                    recommendations.append("Improve customer satisfaction through proactive outreach")
                elif metric == "delivery_delay":
                    recommendations.append(f"Address delivery delays (current: {state.get('delivery_delay', 'N/A')} days)")
                elif metric == "price_increase":
                    recommendations.append("Consider loyalty discounts or value-add services")
                elif metric == "support_tickets":
                    recommendations.append("Investigate root cause of support tickets")
                elif metric == "days_since_purchase":
                    recommendations.append("Launch re-engagement campaign for inactive customers")
                elif metric == "engagement_score":
                    recommendations.append("Implement engagement programs and incentives")
                elif metric == "nps_score":
                    recommendations.append("Address detractor concerns with feedback loop")
                elif metric == "product_usage":
                    recommendations.append("Provide training and onboarding support")
                elif metric == "competitor_mentions":
                    recommendations.append("Highlight unique value propositions vs competitors")
        
        if not recommendations:
            recommendations.append("Maintain current engagement strategies")
        
        return recommendations
    
    def _learn_from_labels(
        self,
        data: List[Dict[str, Any]],
        labels: List[bool]
    ):
        """Learn factor weights from labeled data"""
        churned = [d for d, l in zip(data, labels) if l]
        retained = [d for d, l in zip(data, labels) if not l]
        
        if not churned or not retained:
            return
        
        # Update baseline
        self.baseline_churn_rate = len(churned) / len(data)
        
        # Learn factor impacts
        for metric in self.churn_factors.keys():
            churned_values = [d.get(metric, 50) for d in churned]
            retained_values = [d.get(metric, 50) for d in retained]
            
            if churned_values and retained_values:
                diff = np.mean(churned_values) - np.mean(retained_values)
                std = np.std(churned_values + retained_values) + 1e-6
                
                # Update weight based on difference
                learned = diff / std * 0.1
                self.churn_factors[metric] = 0.7 * self.churn_factors[metric] + 0.3 * learned
    
    def _learn_from_churn_rate(self, data: List[Dict[str, Any]]):
        """Learn from aggregate churn_rate metric"""
        churn_rates = [d.get("churn_rate", 5) for d in data]
        self.baseline_churn_rate = np.mean(churn_rates) / 100  # Convert to probability


# ============================================
# CONVENIENCE FUNCTION
# ============================================

def create_churn_predictor(
    historical_data: Optional[List[Dict[str, Any]]] = None
) -> ChurnPredictor:
    """Create and optionally train a churn predictor"""
    predictor = ChurnPredictor()
    
    if historical_data and len(historical_data) >= 10:
        predictor.train(historical_data)
    
    return predictor
