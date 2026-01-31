"""
ML API Endpoints
Exposes machine learning capabilities through REST API.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

# ML imports
from ml.feature_engineering import FeatureEngineer, engineer_features_from_state
from ml.model_registry import model_registry, ModelStatus
from ml.models.revenue_forecaster import RevenueForecaster, create_revenue_forecaster
from ml.models.churn_predictor import ChurnPredictor, create_churn_predictor
from monitoring.drift_detector import drift_detector, DriftSeverity


# Import state for data access
from state_engine import business_state
from session_manager import session_manager


def get_user_components(session_id: Optional[str] = None):
    """Get session-specific components or fall back to globals"""
    if session_id:
        session = session_manager.get_session(session_id)
        if session:
            return (
                session.business_state,
                session.revenue_forecaster,
                session.churn_predictor,
                session.drift_detector
            )
    
    # Fallback to globals
    return (
        business_state, 
        revenue_forecaster, 
        churn_predictor, 
        drift_detector
    )

router = APIRouter(prefix="/ml", tags=["Machine Learning"])



# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class ForecastRequest(BaseModel):
    horizon_days: int = 30
    scenario: Optional[Dict[str, float]] = None  # What-if adjustments


class ChurnRequest(BaseModel):
    revenue_per_customer: float = 1000.0


class TrainRequest(BaseModel):
    model_type: str  # "revenue", "churn"
    use_uploaded_data: bool = True


class DriftCheckRequest(BaseModel):
    include_recommendations: bool = True


# ============================================
# GLOBAL MODEL INSTANCES
# ============================================

revenue_forecaster = RevenueForecaster()
churn_predictor = ChurnPredictor()


# ============================================
# FEATURE ENGINEERING ENDPOINTS
# ============================================

@router.get("/features")
def get_engineered_features():
    """Get engineered features from current business state"""
    if not business_state.initialized:
        raise HTTPException(status_code=400, detail="Digital Twin not initialized")
    
    features = engineer_features_from_state(business_state.current_state)
    
    return {
        "features": features,
        "feature_count": len(features),
        "source": "current_state"
    }


@router.get("/features/importance")
def get_feature_importance():
    """Get feature importance for revenue prediction"""
    if not business_state.initialized:
        raise HTTPException(status_code=400, detail="Digital Twin not initialized")
    
    # Use correlation-based importance
    features = engineer_features_from_state(business_state.current_state)
    
    # Simple importance based on causal relationships
    importance = {
        "customers": 0.35,
        "marketing_spend": 0.25,
        "sentiment": 0.20,
        "price": 0.15,
        "delivery_delay": -0.10,
        "churn_rate": -0.15
    }
    
    return {
        "importance": importance,
        "method": "causal_correlation",
        "top_features": sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    }


# ============================================
# FORECASTING ENDPOINTS
# ============================================

@router.post("/forecast/revenue")
def forecast_revenue(request: ForecastRequest):
    """Generate revenue forecast"""
    if not business_state.initialized:
        raise HTTPException(status_code=400, detail="Digital Twin not initialized")
    
    forecast = revenue_forecaster.predict(
        current_state=business_state.current_state,
        horizon_days=request.horizon_days,
        scenario=request.scenario
    )
    
    return {
        "forecast": {
            "predicted_revenue": forecast.predicted_revenue,
            "confidence_interval": {
                "low": forecast.confidence_interval_low,
                "high": forecast.confidence_interval_high
            },
            "confidence_level": forecast.confidence_level,
            "horizon_days": forecast.horizon_days,
            "trend": forecast.trend,
            "seasonality_adjustment": forecast.seasonality_adjustment
        },
        "contributing_factors": forecast.factors,
        "scenario_applied": request.scenario
    }


@router.get("/forecast/multi-horizon")
def forecast_multi_horizon(session_id: Optional[str] = Query(None)):
    """Generate forecasts for multiple time horizons"""
    user_state, user_forecaster, _, _ = get_user_components(session_id)
    
    if not user_state.initialized:
        raise HTTPException(status_code=400, detail="Digital Twin not initialized")
    
    forecasts = user_forecaster.predict_multi_horizon(
        current_state=user_state.current_state,
        horizons=[7, 30, 90, 180, 365]
    )
    
    return {
        "horizons": {
            str(h): {
                "predicted_revenue": f.predicted_revenue,
                "confidence_interval": [f.confidence_interval_low, f.confidence_interval_high],
                "trend": f.trend
            }
            for h, f in forecasts.items()
        },
        "current_revenue": user_state.current_state.get("revenue", 0)
    }


@router.post("/forecast/monte-carlo")
def monte_carlo_forecast(request: ForecastRequest):
    """Run Monte Carlo simulation for uncertainty quantification"""
    if not business_state.initialized:
        raise HTTPException(status_code=400, detail="Digital Twin not initialized")
    
    result = revenue_forecaster.monte_carlo_forecast(
        current_state=business_state.current_state,
        horizon_days=request.horizon_days,
        n_simulations=500,
        volatility=0.1
    )
    
    # Don't return full distribution to reduce payload size
    result.pop("distribution", None)
    
    return {
        "monte_carlo": result,
        "horizon_days": request.horizon_days,
        "interpretation": {
            "p10": f"10% chance revenue will be below ${result['p10']:,.0f}",
            "p50": f"Median expected revenue: ${result['p50']:,.0f}",
            "p90": f"10% chance revenue will exceed ${result['p90']:,.0f}"
        }
    }


# ============================================
# CHURN PREDICTION ENDPOINTS
# ============================================

@router.post("/predict/churn")
def predict_churn(request: ChurnRequest, session_id: Optional[str] = Query(None)):
    """Predict customer churn risk"""
    user_state, _, user_churn_predictor, _ = get_user_components(session_id)
    
    if not user_state.initialized:
        raise HTTPException(status_code=400, detail="Digital Twin not initialized")
    
    prediction = user_churn_predictor.predict(
        customer_state=user_state.current_state,
        revenue_per_customer=request.revenue_per_customer
    )
    
    return {
        "churn_prediction": {
            "probability": prediction.churn_probability,
            "risk_level": prediction.risk_level.value,
            "confidence": prediction.confidence
        },
        "contributing_factors": prediction.contributing_factors,
        "revenue_at_risk": prediction.expected_revenue_at_risk,
        "recommendations": prediction.retention_recommendations
    }


@router.get("/predict/churn/risk-segments")
def get_churn_risk_segments():
    """Get customer risk segmentation"""
    if not business_state.initialized:
        raise HTTPException(status_code=400, detail="Digital Twin not initialized")
    
    # Create synthetic segments based on current state
    base_state = business_state.current_state.copy()
    
    segments = {
        "high_engagement": {**base_state, "engagement_score": 80, "sentiment": 85},
        "medium_engagement": {**base_state, "engagement_score": 55, "sentiment": 60},
        "low_engagement": {**base_state, "engagement_score": 30, "sentiment": 40},
        "at_risk": {**base_state, "engagement_score": 20, "sentiment": 30, "support_tickets": 5}
    }
    
    results = {}
    for name, state in segments.items():
        pred = churn_predictor.predict(state)
        results[name] = {
            "churn_probability": pred.churn_probability,
            "risk_level": pred.risk_level.value,
            "revenue_at_risk": pred.expected_revenue_at_risk
        }
    
    return {
        "segments": results,
        "total_customers": base_state.get("customers", 0)
    }


# ============================================
# MODEL MANAGEMENT ENDPOINTS
# ============================================

@router.post("/models/train")
def train_model(request: TrainRequest):
    """Train a predictive model"""
    if not business_state.initialized:
        raise HTTPException(status_code=400, detail="Digital Twin not initialized")
    
    # Generate synthetic training data from current state
    historical_data = _generate_historical_data(business_state.current_state)
    
    if request.model_type == "revenue":
        result = revenue_forecaster.train(historical_data)
        return {
            "model_type": "revenue_forecaster",
            "training_result": result
        }
    
    elif request.model_type == "churn":
        result = churn_predictor.train(historical_data)
        return {
            "model_type": "churn_predictor",
            "training_result": result
        }
    
    raise HTTPException(status_code=400, detail=f"Unknown model type: {request.model_type}")


@router.get("/models")
def list_models():
    """List all registered models"""
    models = model_registry.list_models()
    
    return {
        "models": [
            {
                "id": m.model_id,
                "name": m.model_name,
                "type": m.model_type,
                "version": m.version,
                "status": m.status.value,
                "metrics": m.metrics,
                "trained_at": m.trained_at
            }
            for m in models
        ],
        "deployed": model_registry.deployed_models,
        "total_models": len(models)
    }


@router.get("/models/{model_id}")
def get_model_details(model_id: str):
    """Get details for a specific model"""
    metadata = model_registry.get_metadata(model_id)
    
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
    
    return {
        "model_id": metadata.model_id,
        "name": metadata.model_name,
        "type": metadata.model_type,
        "version": metadata.version,
        "status": metadata.status.value,
        "trained_at": metadata.trained_at,
        "training_samples": metadata.training_samples,
        "feature_names": metadata.feature_names,
        "target_metric": metadata.target_metric,
        "metrics": metadata.metrics,
        "hyperparameters": metadata.hyperparameters,
        "description": metadata.description
    }


# ============================================
# MONITORING ENDPOINTS
# ============================================

@router.get("/monitoring/drift")
def check_drift(session_id: Optional[str] = Query(None)):
    """Check for data and model drift"""
    _, _, _, user_drift_detector = get_user_components(session_id)
    
    report = user_drift_detector.check_drift()
    
    return {
        "drift_report": {
            "timestamp": report.timestamp.isoformat(),
            "overall_score": report.overall_drift_score,
            "severity": report.severity.value,
            "performance_trend": report.model_performance_trend,
            "requires_retraining": report.requires_retraining
        },
        "feature_scores": report.feature_drift_scores,
        "alerts": [
            {
                "type": a.drift_type.value,
                "severity": a.severity.value,
                "metric": a.metric_name,
                "message": a.message,
                "action": a.recommended_action
            }
            for a in report.alerts
        ],
        "recommendations": report.recommendations
    }


@router.get("/monitoring/stability")
def get_stability():
    """Get system stability score"""
    stability = drift_detector.get_stability_score()
    
    return {
        "stability_score": stability,
        "status": "stable" if stability > 0.8 else "unstable" if stability < 0.5 else "moderate",
        "interpretation": f"System is {stability*100:.0f}% stable"
    }


@router.post("/monitoring/record")
def record_observation():
    """Record current state for drift monitoring"""
    if not business_state.initialized:
        raise HTTPException(status_code=400, detail="Digital Twin not initialized")
    
    drift_detector.add_observation(business_state.current_state)
    
    return {
        "recorded": True,
        "current_state_metrics": len(business_state.current_state)
    }


# ============================================
# HELPER FUNCTIONS
# ============================================

def _generate_historical_data(current_state: Dict[str, Any], n_samples: int = 60) -> List[Dict[str, Any]]:
    """Generate synthetic historical data for training"""
    import numpy as np
    from datetime import datetime, timedelta
    
    data = []
    base_date = datetime.now() - timedelta(days=n_samples)
    
    for i in range(n_samples):
        # Add some variation
        noise_factor = 1 + np.random.normal(0, 0.1)
        trend_factor = 1 + (i / n_samples) * 0.05  # 5% growth over period
        
        row = {"timestamp": (base_date + timedelta(days=i)).isoformat()}
        
        for key, value in current_state.items():
            if isinstance(value, (int, float)):
                row[key] = value * noise_factor * trend_factor
        
        data.append(row)
    
    return data
