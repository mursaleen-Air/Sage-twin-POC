"""
Drift Detection Module
Monitors data and model drift for continuous learning.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque


class DriftSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DriftType(str, Enum):
    DATA_DRIFT = "data_drift"           # Input feature distribution change
    CONCEPT_DRIFT = "concept_drift"     # Relationship between features and target changed
    PERFORMANCE_DRIFT = "performance"   # Model performance degradation
    SUDDEN_DRIFT = "sudden"             # Abrupt change
    GRADUAL_DRIFT = "gradual"           # Slow change over time


@dataclass
class DriftAlert:
    """Alert for detected drift"""
    drift_type: DriftType
    severity: DriftSeverity
    metric_name: str
    current_value: float
    baseline_value: float
    deviation_percent: float
    detected_at: datetime
    message: str
    recommended_action: str


@dataclass
class DriftReport:
    """Comprehensive drift analysis report"""
    timestamp: datetime
    overall_drift_score: float  # 0-1
    severity: DriftSeverity
    alerts: List[DriftAlert]
    feature_drift_scores: Dict[str, float]
    model_performance_trend: str  # "improving", "stable", "degrading"
    requires_retraining: bool
    recommendations: List[str]


class DriftDetector:
    """
    Monitors for data and concept drift in the Digital Twin.
    Triggers alerts when significant changes are detected.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        baseline_window: int = 500,
        alert_threshold: float = 0.15
    ):
        self.window_size = window_size
        self.baseline_window = baseline_window
        self.alert_threshold = alert_threshold
        
        # Historical data storage
        self.baseline_data: Dict[str, deque] = {}
        self.recent_data: Dict[str, deque] = {}
        
        # Baseline statistics
        self.baseline_stats: Dict[str, Dict[str, float]] = {}
        
        # Performance tracking
        self.prediction_history: deque = deque(maxlen=1000)
        self.actual_history: deque = deque(maxlen=1000)
        
        # Alert history
        self.alert_history: List[DriftAlert] = []
    
    def add_observation(
        self,
        features: Dict[str, float],
        prediction: Optional[float] = None,
        actual: Optional[float] = None
    ):
        """
        Add a new observation for monitoring.
        
        Args:
            features: Feature values for this observation
            prediction: Model prediction (if available)
            actual: Actual outcome (if available)
        """
        for name, value in features.items():
            if not isinstance(value, (int, float)):
                continue
            
            if name not in self.baseline_data:
                self.baseline_data[name] = deque(maxlen=self.baseline_window)
                self.recent_data[name] = deque(maxlen=self.window_size)
            
            self.baseline_data[name].append(value)
            self.recent_data[name].append(value)
        
        if prediction is not None:
            self.prediction_history.append(prediction)
        if actual is not None:
            self.actual_history.append(actual)
        
        # Update baseline stats periodically
        if len(next(iter(self.baseline_data.values()), [])) >= self.baseline_window:
            self._update_baseline_stats()
    
    def check_drift(self) -> DriftReport:
        """
        Perform comprehensive drift analysis.
        
        Returns:
            DriftReport with all detected drift indicators
        """
        alerts = []
        feature_scores = {}
        
        # Check each feature for drift
        for name in self.baseline_data.keys():
            if name not in self.baseline_stats:
                continue
            
            score, alert = self._check_feature_drift(name)
            feature_scores[name] = score
            
            if alert:
                alerts.append(alert)
        
        # Check model performance drift
        perf_alert = self._check_performance_drift()
        if perf_alert:
            alerts.append(perf_alert)
        
        # Calculate overall drift score
        if feature_scores:
            overall_score = np.mean(list(feature_scores.values()))
        else:
            overall_score = 0.0
        
        # Determine severity
        severity = self._score_to_severity(overall_score)
        
        # Determine performance trend
        perf_trend = self._get_performance_trend()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(alerts, severity)
        
        # Determine if retraining needed
        requires_retraining = (
            severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL] or
            perf_trend == "degrading" or
            len([a for a in alerts if a.drift_type == DriftType.CONCEPT_DRIFT]) > 0
        )
        
        return DriftReport(
            timestamp=datetime.now(),
            overall_drift_score=overall_score,
            severity=severity,
            alerts=alerts,
            feature_drift_scores=feature_scores,
            model_performance_trend=perf_trend,
            requires_retraining=requires_retraining,
            recommendations=recommendations
        )
    
    def detect_sudden_change(
        self,
        metric_name: str,
        current_value: float,
        lookback_periods: int = 10
    ) -> Optional[DriftAlert]:
        """
        Detect sudden/abrupt changes in a metric.
        """
        if metric_name not in self.recent_data:
            return None
        
        recent = list(self.recent_data[metric_name])
        if len(recent) < lookback_periods:
            return None
        
        # Get recent values excluding current
        history = np.array(recent[-lookback_periods-1:-1])
        mean = np.mean(history)
        std = np.std(history) + 1e-6
        
        # Calculate z-score
        z_score = abs(current_value - mean) / std
        
        if z_score > 3.0:  # 3 standard deviations
            deviation = (current_value - mean) / (mean + 1e-6) * 100
            
            return DriftAlert(
                drift_type=DriftType.SUDDEN_DRIFT,
                severity=DriftSeverity.HIGH if z_score > 4 else DriftSeverity.MEDIUM,
                metric_name=metric_name,
                current_value=current_value,
                baseline_value=mean,
                deviation_percent=deviation,
                detected_at=datetime.now(),
                message=f"Sudden change detected in {metric_name}: {deviation:.1f}% deviation",
                recommended_action="Investigate root cause immediately"
            )
        
        return None
    
    def get_stability_score(self) -> float:
        """
        Get overall system stability score (0-1, higher = more stable).
        """
        if not self.baseline_stats:
            return 1.0
        
        instability_scores = []
        
        for name, stats in self.baseline_stats.items():
            if name in self.recent_data and len(self.recent_data[name]) > 10:
                recent = np.array(self.recent_data[name])
                current_std = np.std(recent)
                baseline_std = stats.get("std", 1.0)
                
                # Compare volatility
                if baseline_std > 0:
                    volatility_ratio = current_std / baseline_std
                    instability = abs(volatility_ratio - 1.0)
                    instability_scores.append(min(instability, 1.0))
        
        if instability_scores:
            return 1.0 - np.mean(instability_scores)
        return 1.0
    
    # ============================================
    # STATISTICAL TESTS
    # ============================================
    
    def kolmogorov_smirnov_test(
        self,
        metric_name: str
    ) -> Tuple[float, bool]:
        """
        Perform KS test to detect distribution change.
        
        Returns:
            (p_value, is_significant)
        """
        if metric_name not in self.baseline_data or metric_name not in self.recent_data:
            return (1.0, False)
        
        baseline = np.array(self.baseline_data[metric_name])
        recent = np.array(self.recent_data[metric_name])
        
        if len(baseline) < 30 or len(recent) < 10:
            return (1.0, False)
        
        # Simple KS statistic calculation
        n1, n2 = len(baseline), len(recent)
        combined = np.concatenate([baseline, recent])
        combined_sorted = np.sort(combined)
        
        cdf1 = np.searchsorted(np.sort(baseline), combined_sorted) / n1
        cdf2 = np.searchsorted(np.sort(recent), combined_sorted) / n2
        
        ks_stat = np.max(np.abs(cdf1 - cdf2))
        
        # Approximate p-value
        en = np.sqrt(n1 * n2 / (n1 + n2))
        p_value = 2 * np.exp(-2 * (en * ks_stat) ** 2)
        
        return (p_value, p_value < 0.05)
    
    def population_stability_index(
        self,
        metric_name: str,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        PSI < 0.1: No significant change
        PSI < 0.25: Small change
        PSI >= 0.25: Significant change
        """
        if metric_name not in self.baseline_data or metric_name not in self.recent_data:
            return 0.0
        
        baseline = np.array(self.baseline_data[metric_name])
        recent = np.array(self.recent_data[metric_name])
        
        if len(baseline) < 30 or len(recent) < 10:
            return 0.0
        
        # Create bins from baseline
        min_val, max_val = baseline.min(), baseline.max()
        bins = np.linspace(min_val, max_val + 1e-6, n_bins + 1)
        
        # Calculate proportions
        baseline_counts = np.histogram(baseline, bins=bins)[0]
        recent_counts = np.histogram(recent, bins=bins)[0]
        
        # Add small value to avoid division by zero
        baseline_props = (baseline_counts + 0.001) / len(baseline)
        recent_props = (recent_counts + 0.001) / len(recent)
        
        # Calculate PSI
        psi = np.sum((recent_props - baseline_props) * np.log(recent_props / baseline_props))
        
        return psi
    
    # ============================================
    # PRIVATE METHODS
    # ============================================
    
    def _update_baseline_stats(self):
        """Update baseline statistics for all features"""
        for name, values in self.baseline_data.items():
            if len(values) >= 30:
                arr = np.array(values)
                self.baseline_stats[name] = {
                    "mean": np.mean(arr),
                    "std": np.std(arr),
                    "min": np.min(arr),
                    "max": np.max(arr),
                    "median": np.median(arr),
                    "q25": np.percentile(arr, 25),
                    "q75": np.percentile(arr, 75)
                }
    
    def _check_feature_drift(
        self,
        name: str
    ) -> Tuple[float, Optional[DriftAlert]]:
        """Check drift for a single feature"""
        if name not in self.recent_data or len(self.recent_data[name]) < 10:
            return (0.0, None)
        
        stats = self.baseline_stats[name]
        recent = np.array(self.recent_data[name])
        
        recent_mean = np.mean(recent)
        baseline_mean = stats["mean"]
        baseline_std = stats["std"] + 1e-6
        
        # Calculate mean shift
        z_shift = abs(recent_mean - baseline_mean) / baseline_std
        
        # Calculate PSI
        psi = self.population_stability_index(name)
        
        # Combined drift score
        drift_score = (z_shift / 3.0 + psi) / 2.0
        drift_score = min(drift_score, 1.0)
        
        alert = None
        if drift_score > self.alert_threshold:
            deviation = (recent_mean - baseline_mean) / (baseline_mean + 1e-6) * 100
            
            alert = DriftAlert(
                drift_type=DriftType.DATA_DRIFT,
                severity=self._score_to_severity(drift_score),
                metric_name=name,
                current_value=recent_mean,
                baseline_value=baseline_mean,
                deviation_percent=deviation,
                detected_at=datetime.now(),
                message=f"Data drift detected in {name}: PSI={psi:.3f}",
                recommended_action="Review data source and consider model retraining"
            )
        
        return (drift_score, alert)
    
    def _check_performance_drift(self) -> Optional[DriftAlert]:
        """Check for model performance degradation"""
        if len(self.prediction_history) < 50 or len(self.actual_history) < 50:
            return None
        
        predictions = np.array(self.prediction_history)
        actuals = np.array(self.actual_history)
        
        n = min(len(predictions), len(actuals))
        predictions = predictions[:n]
        actuals = actuals[:n]
        
        # Calculate error over time
        errors = np.abs(predictions - actuals) / (np.abs(actuals) + 1e-6)
        
        # Compare recent to older errors
        mid = n // 2
        old_error = np.mean(errors[:mid])
        recent_error = np.mean(errors[mid:])
        
        error_increase = (recent_error - old_error) / (old_error + 1e-6)
        
        if error_increase > 0.2:  # 20% increase in error
            return DriftAlert(
                drift_type=DriftType.PERFORMANCE_DRIFT,
                severity=DriftSeverity.HIGH if error_increase > 0.5 else DriftSeverity.MEDIUM,
                metric_name="model_error",
                current_value=recent_error,
                baseline_value=old_error,
                deviation_percent=error_increase * 100,
                detected_at=datetime.now(),
                message=f"Model performance degraded by {error_increase*100:.1f}%",
                recommended_action="Trigger model retraining pipeline"
            )
        
        return None
    
    def _score_to_severity(self, score: float) -> DriftSeverity:
        """Convert drift score to severity level"""
        if score < 0.1:
            return DriftSeverity.NONE
        elif score < 0.2:
            return DriftSeverity.LOW
        elif score < 0.35:
            return DriftSeverity.MEDIUM
        elif score < 0.5:
            return DriftSeverity.HIGH
        return DriftSeverity.CRITICAL
    
    def _get_performance_trend(self) -> str:
        """Determine performance trend"""
        if len(self.prediction_history) < 30:
            return "stable"
        
        predictions = np.array(list(self.prediction_history)[-50:])
        actuals = np.array(list(self.actual_history)[-50:])
        
        n = min(len(predictions), len(actuals))
        if n < 20:
            return "stable"
        
        errors = np.abs(predictions[:n] - actuals[:n])
        
        # Fit linear trend
        x = np.arange(len(errors))
        slope = np.polyfit(x, errors, 1)[0]
        
        avg_error = np.mean(errors)
        normalized_slope = slope / (avg_error + 1e-6)
        
        if normalized_slope > 0.02:
            return "degrading"
        elif normalized_slope < -0.02:
            return "improving"
        return "stable"
    
    def _generate_recommendations(
        self,
        alerts: List[DriftAlert],
        severity: DriftSeverity
    ) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if severity == DriftSeverity.CRITICAL:
            recommendations.append("⚠️ CRITICAL: Immediate model retraining required")
            recommendations.append("Review recent data source changes")
        
        if severity == DriftSeverity.HIGH:
            recommendations.append("Schedule model retraining within 24 hours")
        
        data_drift_count = sum(1 for a in alerts if a.drift_type == DriftType.DATA_DRIFT)
        if data_drift_count > 3:
            recommendations.append("Multiple features drifting - investigate upstream data pipeline")
        
        if any(a.drift_type == DriftType.PERFORMANCE_DRIFT for a in alerts):
            recommendations.append("Model accuracy declining - consider ensemble approach")
        
        if any(a.drift_type == DriftType.SUDDEN_DRIFT for a in alerts):
            recommendations.append("Sudden change detected - check for data entry errors or external events")
        
        if not recommendations:
            recommendations.append("System operating within normal parameters")
        
        return recommendations


# ============================================
# GLOBAL INSTANCE
# ============================================

drift_detector = DriftDetector()
