"""
Feature Engineering Pipeline
Transforms raw business data into ML-ready features.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class FeatureConfig:
    """Configuration for feature engineering"""
    rolling_windows: List[int] = field(default_factory=lambda: [3, 7, 14, 30])
    lag_periods: List[int] = field(default_factory=lambda: [1, 7, 30])
    include_ratios: bool = True
    include_growth_rates: bool = True
    include_volatility: bool = True
    include_seasonality: bool = True


class FeatureEngineer:
    """
    Feature engineering pipeline for Digital Twin data.
    Transforms raw metrics into ML-ready features.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []
        self.feature_stats: Dict[str, Dict[str, float]] = {}
    
    def engineer_features(
        self,
        time_series_data: List[Dict[str, Any]],
        target_metric: Optional[str] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Transform time series data into feature matrix.
        
        Args:
            time_series_data: List of dictionaries with metric values over time
            target_metric: If specified, exclude this from features (for prediction)
            
        Returns:
            Feature matrix (numpy array) and list of feature names
        """
        if not time_series_data:
            return np.array([]), []
        
        features = []
        feature_names = []
        
        # Extract all numeric columns
        numeric_cols = self._get_numeric_columns(time_series_data[0])
        if target_metric and target_metric in numeric_cols:
            numeric_cols.remove(target_metric)
        
        # Convert to numpy arrays for efficient computation
        data_arrays = {
            col: np.array([row.get(col, np.nan) for row in time_series_data])
            for col in numeric_cols
        }
        
        for col, values in data_arrays.items():
            # Base features
            features.append(values)
            feature_names.append(col)
            
            # Rolling statistics
            for window in self.config.rolling_windows:
                if len(values) >= window:
                    rolling_mean = self._rolling_mean(values, window)
                    rolling_std = self._rolling_std(values, window)
                    
                    features.append(rolling_mean)
                    feature_names.append(f"{col}_rolling_mean_{window}")
                    
                    if self.config.include_volatility:
                        features.append(rolling_std)
                        feature_names.append(f"{col}_rolling_std_{window}")
            
            # Lag features
            for lag in self.config.lag_periods:
                if len(values) >= lag:
                    lagged = self._create_lag(values, lag)
                    features.append(lagged)
                    feature_names.append(f"{col}_lag_{lag}")
            
            # Growth rates
            if self.config.include_growth_rates:
                pct_change = self._pct_change(values)
                features.append(pct_change)
                feature_names.append(f"{col}_pct_change")
        
        # Ratio features
        if self.config.include_ratios:
            ratio_features, ratio_names = self._create_ratio_features(data_arrays)
            features.extend(ratio_features)
            feature_names.extend(ratio_names)
        
        # Seasonality features (if timestamps available)
        if "timestamp" in time_series_data[0]:
            seasonal_features, seasonal_names = self._create_seasonality_features(time_series_data)
            features.extend(seasonal_features)
            feature_names.extend(seasonal_names)
        
        # Stack features into matrix
        feature_matrix = np.column_stack(features) if features else np.array([])
        
        self.feature_names = feature_names
        return feature_matrix, feature_names
    
    def engineer_single_point(
        self,
        current_data: Dict[str, Any],
        historical_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Engineer features for a single data point using historical context.
        Used for real-time prediction.
        """
        features = {}
        
        # Current values
        numeric_cols = self._get_numeric_columns(current_data)
        for col in numeric_cols:
            features[col] = current_data.get(col, 0)
        
        if not historical_data:
            return features
        
        # Historical context
        for col in numeric_cols:
            hist_values = [row.get(col, np.nan) for row in historical_data[-30:]]
            hist_values = [v for v in hist_values if not np.isnan(v)]
            
            if hist_values:
                features[f"{col}_hist_mean"] = np.mean(hist_values)
                features[f"{col}_hist_std"] = np.std(hist_values) if len(hist_values) > 1 else 0
                
                # Current vs historical
                if features.get(col):
                    features[f"{col}_vs_mean"] = (features[col] - np.mean(hist_values)) / (np.std(hist_values) + 1e-6)
        
        return features
    
    def create_interaction_features(
        self,
        data: Dict[str, float]
    ) -> Dict[str, float]:
        """Create business-meaningful interaction features"""
        interactions = {}
        
        # Revenue per customer
        if data.get("revenue") and data.get("customers"):
            interactions["revenue_per_customer"] = data["revenue"] / max(data["customers"], 1)
        
        # Marketing efficiency
        if data.get("marketing_spend") and data.get("new_customers"):
            interactions["cac"] = data["marketing_spend"] / max(data["new_customers"], 1)
        
        if data.get("marketing_spend") and data.get("revenue"):
            interactions["marketing_roi"] = (data["revenue"] / max(data["marketing_spend"], 1)) - 1
        
        # Operational efficiency
        if data.get("revenue") and data.get("costs"):
            interactions["profit_margin"] = (data["revenue"] - data["costs"]) / max(data["revenue"], 1)
        
        # Customer health
        if data.get("sentiment") and data.get("churn_rate"):
            interactions["customer_health"] = data["sentiment"] * (100 - data["churn_rate"]) / 100
        
        # Delivery impact on sentiment
        if data.get("delivery_delay") and data.get("sentiment"):
            interactions["delivery_sentiment_ratio"] = data["sentiment"] / max(data["delivery_delay"], 0.1)
        
        return interactions
    
    def detect_anomalies(
        self,
        values: np.ndarray,
        threshold: float = 2.5
    ) -> np.ndarray:
        """Detect anomalies using Z-score method"""
        if len(values) < 3:
            return np.zeros(len(values), dtype=bool)
        
        mean = np.nanmean(values)
        std = np.nanstd(values)
        
        if std == 0:
            return np.zeros(len(values), dtype=bool)
        
        z_scores = np.abs((values - mean) / std)
        return z_scores > threshold
    
    def detect_trend(
        self,
        values: np.ndarray,
        window: int = 7
    ) -> str:
        """Detect trend direction: 'up', 'down', or 'stable'"""
        if len(values) < window:
            return "stable"
        
        recent = values[-window:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        
        # Normalize slope by value magnitude
        avg_value = np.mean(np.abs(recent)) + 1e-6
        normalized_slope = slope / avg_value
        
        if normalized_slope > 0.02:
            return "up"
        elif normalized_slope < -0.02:
            return "down"
        return "stable"
    
    # ============================================
    # PRIVATE HELPER METHODS
    # ============================================
    
    def _get_numeric_columns(self, data: Dict[str, Any]) -> List[str]:
        """Extract numeric column names from data dictionary"""
        return [
            k for k, v in data.items()
            if isinstance(v, (int, float)) and k not in ["timestamp", "id"]
        ]
    
    def _rolling_mean(self, values: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling mean with NaN handling"""
        result = np.full(len(values), np.nan)
        for i in range(window - 1, len(values)):
            result[i] = np.nanmean(values[i - window + 1:i + 1])
        return result
    
    def _rolling_std(self, values: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation"""
        result = np.full(len(values), np.nan)
        for i in range(window - 1, len(values)):
            result[i] = np.nanstd(values[i - window + 1:i + 1])
        return result
    
    def _create_lag(self, values: np.ndarray, lag: int) -> np.ndarray:
        """Create lagged feature"""
        result = np.full(len(values), np.nan)
        result[lag:] = values[:-lag]
        return result
    
    def _pct_change(self, values: np.ndarray) -> np.ndarray:
        """Calculate percentage change"""
        result = np.full(len(values), np.nan)
        result[1:] = (values[1:] - values[:-1]) / (np.abs(values[:-1]) + 1e-6) * 100
        return result
    
    def _create_ratio_features(
        self,
        data_arrays: Dict[str, np.ndarray]
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Create ratio features between related metrics"""
        features = []
        names = []
        
        # Define meaningful ratios
        ratio_pairs = [
            ("revenue", "customers", "revenue_per_customer"),
            ("profit", "revenue", "profit_margin"),
            ("marketing_spend", "revenue", "marketing_to_revenue"),
            ("costs", "revenue", "cost_ratio"),
        ]
        
        for num, denom, name in ratio_pairs:
            if num in data_arrays and denom in data_arrays:
                ratio = data_arrays[num] / (np.abs(data_arrays[denom]) + 1e-6)
                features.append(ratio)
                names.append(name)
        
        return features, names
    
    def _create_seasonality_features(
        self,
        time_series_data: List[Dict[str, Any]]
    ) -> Tuple[List[np.ndarray], List[str]]:
        """Create seasonality-based features"""
        features = []
        names = []
        
        try:
            timestamps = []
            for row in time_series_data:
                ts = row.get("timestamp")
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                elif isinstance(ts, datetime):
                    pass
                else:
                    continue
                timestamps.append(ts)
            
            if timestamps:
                # Day of week (0-6)
                dow = np.array([ts.weekday() for ts in timestamps])
                features.append(dow)
                names.append("day_of_week")
                
                # Month (1-12)
                month = np.array([ts.month for ts in timestamps])
                features.append(month)
                names.append("month")
                
                # Is weekend
                is_weekend = np.array([1 if ts.weekday() >= 5 else 0 for ts in timestamps])
                features.append(is_weekend)
                names.append("is_weekend")
                
                # Quarter
                quarter = np.array([(ts.month - 1) // 3 + 1 for ts in timestamps])
                features.append(quarter)
                names.append("quarter")
        
        except Exception:
            pass
        
        return features, names


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def engineer_features_from_state(
    current_state: Dict[str, Any],
    historical_states: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, float]:
    """
    Quick feature engineering from business state dictionaries.
    """
    engineer = FeatureEngineer()
    
    # Base features
    features = engineer.engineer_single_point(
        current_state,
        historical_states or []
    )
    
    # Interaction features
    interactions = engineer.create_interaction_features(current_state)
    features.update(interactions)
    
    return features


def calculate_feature_importance(
    feature_matrix: np.ndarray,
    target: np.ndarray,
    feature_names: List[str]
) -> Dict[str, float]:
    """
    Calculate simple feature importance using correlation.
    """
    importances = {}
    
    for i, name in enumerate(feature_names):
        col = feature_matrix[:, i]
        
        # Remove NaN values
        mask = ~(np.isnan(col) | np.isnan(target))
        if mask.sum() < 3:
            importances[name] = 0.0
            continue
        
        # Calculate correlation
        correlation = np.corrcoef(col[mask], target[mask])[0, 1]
        importances[name] = abs(correlation) if not np.isnan(correlation) else 0.0
    
    # Sort by importance
    return dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
