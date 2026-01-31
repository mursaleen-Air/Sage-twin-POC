"""
Revenue Forecasting Model
Predicts future revenue based on historical patterns and business metrics.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class RevenueForecast:
    """Revenue forecast result"""
    predicted_revenue: float
    confidence_interval_low: float
    confidence_interval_high: float
    confidence_level: float
    horizon_days: int
    factors: Dict[str, float]  # Contributing factors
    trend: str  # "up", "down", "stable"
    seasonality_adjustment: float


class RevenueForecaster:
    """
    Multi-factor revenue forecasting model.
    Combines time-series analysis with causal business factors.
    """
    
    def __init__(self):
        self.is_trained = False
        self.coefficients: Dict[str, float] = {}
        self.baseline_revenue = 0.0
        self.trend_coefficient = 0.0
        self.seasonality_factors: Dict[int, float] = {}  # month -> factor
        self.historical_data: List[Dict[str, Any]] = []
        
        # Default causal relationships (can be learned from data)
        self.causal_weights = {
            "customers": 0.35,
            "marketing_spend": 0.20,
            "price": 0.15,
            "sentiment": 0.10,
            "churn_rate": -0.15,
            "delivery_delay": -0.05
        }
    
    def train(
        self,
        historical_data: List[Dict[str, Any]],
        min_samples: int = 10
    ) -> Dict[str, Any]:
        """
        Train the forecaster on historical data.
        
        Args:
            historical_data: List of dicts with 'revenue' and other metrics
            min_samples: Minimum samples required for training
            
        Returns:
            Training results including metrics
        """
        if len(historical_data) < min_samples:
            return {
                "success": False,
                "error": f"Need at least {min_samples} data points, got {len(historical_data)}"
            }
        
        self.historical_data = historical_data
        revenues = np.array([d.get("revenue", 0) for d in historical_data])
        
        # Calculate baseline (average revenue)
        self.baseline_revenue = np.mean(revenues)
        
        # Calculate trend
        x = np.arange(len(revenues))
        if len(revenues) > 1:
            slope, intercept = np.polyfit(x, revenues, 1)
            self.trend_coefficient = slope / self.baseline_revenue
        
        # Calculate seasonality factors by month
        self._calculate_seasonality(historical_data, revenues)
        
        # Learn causal coefficients if we have enough data
        self._learn_causal_relationships(historical_data)
        
        # Calculate training metrics
        predictions = [self._predict_single(d) for d in historical_data]
        predictions = np.array([p.predicted_revenue for p in predictions])
        
        mae = np.mean(np.abs(predictions - revenues))
        mape = np.mean(np.abs((predictions - revenues) / (revenues + 1e-6))) * 100
        rmse = np.sqrt(np.mean((predictions - revenues) ** 2))
        
        self.is_trained = True
        
        return {
            "success": True,
            "samples_used": len(historical_data),
            "baseline_revenue": self.baseline_revenue,
            "trend": "up" if self.trend_coefficient > 0.01 else "down" if self.trend_coefficient < -0.01 else "stable",
            "metrics": {
                "mae": mae,
                "mape": mape,
                "rmse": rmse
            },
            "learned_weights": self.causal_weights
        }
    
    def predict(
        self,
        current_state: Dict[str, Any],
        horizon_days: int = 30,
        scenario: Optional[Dict[str, float]] = None
    ) -> RevenueForecast:
        """
        Predict future revenue.
        
        Args:
            current_state: Current business metrics
            horizon_days: Days into the future to predict
            scenario: Optional adjustments to apply (what-if analysis)
            
        Returns:
            RevenueForecast with prediction and confidence intervals
        """
        if not self.is_trained:
            # Use simple heuristic if not trained
            return self._heuristic_forecast(current_state, horizon_days, scenario)
        
        # Apply scenario adjustments
        modified_state = current_state.copy()
        if scenario:
            for key, change_pct in scenario.items():
                if key in modified_state:
                    modified_state[key] *= (1 + change_pct / 100)
        
        return self._predict_single(modified_state, horizon_days)
    
    def predict_multi_horizon(
        self,
        current_state: Dict[str, Any],
        horizons: List[int] = [7, 30, 90, 180]
    ) -> Dict[int, RevenueForecast]:
        """Predict for multiple time horizons"""
        return {
            h: self.predict(current_state, horizon_days=h)
            for h in horizons
        }
    
    def monte_carlo_forecast(
        self,
        current_state: Dict[str, Any],
        horizon_days: int = 30,
        n_simulations: int = 1000,
        volatility: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for uncertainty quantification.
        """
        results = []
        
        for _ in range(n_simulations):
            # Add random noise to inputs
            noisy_state = current_state.copy()
            for key, value in noisy_state.items():
                if isinstance(value, (int, float)) and key != "timestamp":
                    noise = np.random.normal(0, volatility * abs(value))
                    noisy_state[key] = value + noise
            
            forecast = self.predict(noisy_state, horizon_days)
            results.append(forecast.predicted_revenue)
        
        results = np.array(results)
        
        return {
            "mean_revenue": np.mean(results),
            "std_revenue": np.std(results),
            "p10": np.percentile(results, 10),
            "p50": np.percentile(results, 50),
            "p90": np.percentile(results, 90),
            "confidence_interval_95": (
                np.percentile(results, 2.5),
                np.percentile(results, 97.5)
            ),
            "n_simulations": n_simulations,
            "distribution": results.tolist()
        }
    
    # ============================================
    # PRIVATE METHODS
    # ============================================
    
    def _predict_single(
        self,
        state: Dict[str, Any],
        horizon_days: int = 30
    ) -> RevenueForecast:
        """Generate a single forecast"""
        
        # Start with baseline
        base_prediction = self.baseline_revenue
        
        # Apply trend
        trend_adjustment = 1 + (self.trend_coefficient * horizon_days / 30)
        prediction = base_prediction * trend_adjustment
        
        # Calculate factor contributions
        factors = {}
        for metric, weight in self.causal_weights.items():
            if metric in state:
                value = state[metric]
                
                # Normalize value relative to expected baseline
                if metric == "customers" and self.baseline_revenue > 0:
                    # Customers: more = more revenue
                    factor = value / (self.baseline_revenue / 100)  # rough normalization
                    contribution = weight * (factor - 1)
                elif metric == "churn_rate":
                    # Churn: higher = less revenue
                    contribution = weight * (value / 100)
                elif metric == "sentiment":
                    # Sentiment: higher = more revenue
                    contribution = weight * ((value - 50) / 50)
                elif metric == "marketing_spend":
                    # Marketing has diminishing returns
                    contribution = weight * np.log1p(value / 10000)
                elif metric == "delivery_delay":
                    # Delay: higher = less revenue
                    contribution = weight * min(value / 10, 1)
                else:
                    contribution = weight * (value / 100 - 0.5)
                
                factors[metric] = contribution
                prediction *= (1 + contribution)
        
        # Apply seasonality
        seasonality_adj = self._get_seasonality_factor(horizon_days)
        prediction *= seasonality_adj
        
        # Calculate confidence interval
        # Wider confidence for longer horizons
        uncertainty = 0.05 + (horizon_days / 365) * 0.15
        
        return RevenueForecast(
            predicted_revenue=max(0, prediction),
            confidence_interval_low=max(0, prediction * (1 - uncertainty)),
            confidence_interval_high=prediction * (1 + uncertainty),
            confidence_level=0.95,
            horizon_days=horizon_days,
            factors=factors,
            trend="up" if self.trend_coefficient > 0.01 else "down" if self.trend_coefficient < -0.01 else "stable",
            seasonality_adjustment=seasonality_adj
        )
    
    def _heuristic_forecast(
        self,
        state: Dict[str, Any],
        horizon_days: int,
        scenario: Optional[Dict[str, float]]
    ) -> RevenueForecast:
        """Simple heuristic forecast when model is not trained"""
        
        base = state.get("revenue", 100000)
        
        # Apply scenario
        if scenario:
            for key, pct in scenario.items():
                multiplier = self.causal_weights.get(key, 0)
                base *= (1 + multiplier * pct / 100)
        
        # Simple growth assumption
        daily_growth = 0.001  # 0.1% daily
        prediction = base * (1 + daily_growth) ** horizon_days
        
        return RevenueForecast(
            predicted_revenue=prediction,
            confidence_interval_low=prediction * 0.85,
            confidence_interval_high=prediction * 1.15,
            confidence_level=0.80,
            horizon_days=horizon_days,
            factors={},
            trend="stable",
            seasonality_adjustment=1.0
        )
    
    def _calculate_seasonality(
        self,
        historical_data: List[Dict[str, Any]],
        revenues: np.ndarray
    ):
        """Calculate monthly seasonality factors"""
        monthly_revenues: Dict[int, List[float]] = {m: [] for m in range(1, 13)}
        
        for i, data in enumerate(historical_data):
            ts = data.get("timestamp")
            if ts:
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                month = ts.month
                monthly_revenues[month].append(revenues[i])
        
        overall_mean = np.mean(revenues)
        
        for month, values in monthly_revenues.items():
            if values:
                self.seasonality_factors[month] = np.mean(values) / overall_mean
            else:
                self.seasonality_factors[month] = 1.0
    
    def _get_seasonality_factor(self, horizon_days: int) -> float:
        """Get seasonality factor for forecast date"""
        future_date = datetime.now() + timedelta(days=horizon_days)
        return self.seasonality_factors.get(future_date.month, 1.0)
    
    def _learn_causal_relationships(self, historical_data: List[Dict[str, Any]]):
        """Learn causal weights from historical data using correlation"""
        if len(historical_data) < 20:
            return  # Keep defaults
        
        revenues = np.array([d.get("revenue", 0) for d in historical_data])
        
        for metric in self.causal_weights.keys():
            values = np.array([d.get(metric, 0) for d in historical_data])
            
            if np.std(values) > 0 and np.std(revenues) > 0:
                correlation = np.corrcoef(values, revenues)[0, 1]
                if not np.isnan(correlation):
                    # Blend learned correlation with prior
                    learned_weight = correlation * 0.3
                    prior_weight = self.causal_weights[metric]
                    self.causal_weights[metric] = 0.6 * prior_weight + 0.4 * learned_weight


# ============================================
# CONVENIENCE FUNCTION
# ============================================

def create_revenue_forecaster(
    historical_data: Optional[List[Dict[str, Any]]] = None
) -> RevenueForecaster:
    """Create and optionally train a revenue forecaster"""
    forecaster = RevenueForecaster()
    
    if historical_data and len(historical_data) >= 10:
        forecaster.train(historical_data)
    
    return forecaster
