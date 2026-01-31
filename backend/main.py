# main.py
"""
SAGE-Twin Digital Twin POC
Complete implementation with Multi-Source Data Ingestion
"""

import os
import csv
import io
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

load_dotenv()

# Import Digital Twin components
from state_engine import business_state, BusinessState
from causal_graph import propagate_change, get_causal_graph_visualization, CAUSAL_MAP
from multi_agents import multi_agent_engine
from forecast import generate_forecast
from data_sources import data_source_manager, DATA_CATEGORIES, DataSourceManager
from session_manager import session_manager, UserSession

# Import ML API router (Phase 1-4 capabilities)
try:
    from ml_api import router as ml_router
    ML_ENABLED = True
except ImportError:
    ML_ENABLED = False

app = FastAPI(
    title="SAGE-Twin Digital Twin API",
    description="Multi-Agent Business Simulation System with ML, Forecasting & Continuous Learning",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include ML routes if available
if ML_ENABLED:
    app.include_router(ml_router)


# ============================================
# REQUEST MODELS
# ============================================

class SimulationRequest(BaseModel):
    price: Optional[float] = 0
    marketing_spend: Optional[float] = 0
    costs: Optional[float] = 0
    delivery_delay: Optional[float] = 0
    market_shock: Optional[bool] = False


class ImpactRequest(BaseModel):
    metric: str
    change_percent: float


# ============================================
# HEALTH CHECK
# ============================================

@app.get("/")
def root():
    return {
        "name": "SAGE-Twin Digital Twin",
        "version": "3.0",
        "status": "running",
        "initialized": business_state.initialized,
        "health_score": business_state.health_score if business_state.initialized else None,
        "data_sources": data_source_manager.get_combined_state()["total_files"],
        "ml_enabled": ML_ENABLED,
        "capabilities": {
            "simulation": True,
            "forecasting": ML_ENABLED,
            "churn_prediction": ML_ENABLED,
            "drift_monitoring": ML_ENABLED,
            "multi_agent": True
        }
    }


# ============================================
# SESSION MANAGEMENT (Multi-User Support)
# ============================================

@app.post("/session/create")
def create_session():
    """Create a new user session"""
    session_id = session_manager.create_session()
    return {
        "session_id": session_id,
        "message": "Session created. Include this ID in your requests."
    }


@app.get("/session/{session_id}")
def get_session_info(session_id: str):
    """Get session information"""
    info = session_manager.get_session_info(session_id)
    if not info:
        return {"error": "Session not found or expired"}
    return info


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    """Delete a session"""
    success = session_manager.delete_session(session_id)
    return {"success": success}


def get_user_state(session_id: Optional[str] = None):
    """Helper to get state for a session or fallback to global"""
    if session_id:
        session = session_manager.get_session(session_id)
        if session:
            return session.business_state, session.data_source_manager
    # Fallback to global (backwards compatible)
    return business_state, data_source_manager


# ============================================
# DATA SOURCE CATEGORIES
# ============================================

@app.get("/data-categories")
def get_data_categories():
    """Get available data source categories and their supported formats"""
    return {
        "categories": DATA_CATEGORIES,
        "total_categories": len(DATA_CATEGORIES)
    }


# ============================================
# MULTI-SOURCE FILE UPLOAD
# ============================================

@app.post("/upload/{category}")
async def upload_file(
    category: str,
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """
    Upload a file to a specific data category.
    
    Categories:
    - revenue: Revenue & Financial data (CSV)
    - customers: Customer data (CSV)
    - reviews: Customer reviews/feedback (CSV, DOCX, TXT)
    - marketing: Marketing campaign data (CSV)
    - operations: Operations & Delivery data (CSV)
    - general: Any other data
    """
    
    # Validate category
    if category not in DATA_CATEGORIES:
        return {
            "success": False,
            "error": f"Invalid category '{category}'. Valid categories: {list(DATA_CATEGORIES.keys())}"
        }
    
    # Get file extension
    filename = file.filename
    extension = filename.split(".")[-1].lower() if "." in filename else ""
    
    # Validate file format for category
    supported = DATA_CATEGORIES[category]["supported_formats"]
    if extension not in supported:
        return {
            "success": False,
            "error": f"Format '.{extension}' not supported for {category}. Supported: {supported}"
        }
    
    try:
        contents = await file.read()
        
        # Get user's state managers
        user_state, user_data_manager = get_user_state(session_id)
        
        # Parse based on file type
        if extension == "csv":
            parsed = user_data_manager.parse_csv(contents, category)
        elif extension == "docx":
            parsed = user_data_manager.parse_docx(contents, category)
        elif extension == "txt":
            parsed = user_data_manager.parse_txt(contents, category)
        else:
            return {"success": False, "error": f"Unsupported file format: {extension}"}
        
        if not parsed.get("success"):
            return parsed
        
        # Add to data source manager
        result = user_data_manager.add_source(category, filename, parsed)
        
        # Sync with business state
        combined = user_data_manager.get_combined_state()
        if combined["metrics"]:
            # Initialize business state from combined metrics if needed
            if not user_state.initialized:
                # Create a fake row for initialization
                fake_rows = [combined["metrics"]]
                user_state.initialize_from_csv(fake_rows)
            else:
                # Update existing state with new metrics
                for key, value in combined["metrics"].items():
                    user_state.current_state[key] = value
                user_state._calculate_health_score()
        
        return {
            "success": True,
            "message": f"File '{filename}' uploaded to {DATA_CATEGORIES[category]['name']}",
            "category": category,
            "category_info": DATA_CATEGORIES[category],
            "parsed_data": {
                "type": parsed.get("type"),
                "metrics": list(parsed.get("metrics", {}).keys()) if parsed.get("metrics") else None,
                "rows": parsed.get("rows_processed"),
                "paragraphs": parsed.get("paragraph_count"),
                "sentiment": parsed.get("sentiment_score"),
                "word_count": parsed.get("word_count")
            },
            "sources_summary": combined,
            "health_score": user_state.health_score,
            "session_id": session_id
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/sources")
def get_data_sources(session_id: Optional[str] = None):
    """Get all uploaded data sources and their status"""
    user_state, user_data_manager = get_user_state(session_id)
    return {
        "sources": user_data_manager.get_combined_state(),
        "history": user_data_manager.get_upload_history(),
        "categories": {
            cat: {
                **DATA_CATEGORIES[cat],
                "uploaded": cat in user_data_manager.sources,
                "file_count": len(user_data_manager.sources.get(cat, {}).get("files", []))
            }
            for cat in DATA_CATEGORIES
        },
        "session_id": session_id
    }


@app.delete("/sources/{category}")
def clear_category(category: str, session_id: Optional[str] = None):
    """Clear all data for a specific category"""
    if category not in DATA_CATEGORIES:
        return {"success": False, "error": f"Invalid category: {category}"}
    
    user_state, user_data_manager = get_user_state(session_id)
    success = user_data_manager.clear_category(category)
    return {
        "success": success,
        "message": f"Cleared all data for {category}" if success else f"No data found for {category}"
    }


@app.delete("/sources")
def clear_all_sources(session_id: Optional[str] = None):
    """Clear all data sources"""
    user_state, user_data_manager = get_user_state(session_id)
    user_data_manager.clear_all()
    user_state.__init__()  # Reset business state
    return {"success": True, "message": "All data sources cleared"}


# ============================================
# LEGACY CSV UPLOAD (Backward Compatibility)
# ============================================

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Legacy endpoint - uploads to 'general' category.
    Use /upload/{category} for categorized uploads.
    """
    try:
        contents = await file.read()
        decoded = contents.decode("utf-8")
        reader = csv.DictReader(io.StringIO(decoded))
        rows = list(reader)
        
        if not rows:
            return {"success": False, "error": "CSV file is empty"}
        
        # Initialize state engine
        result = business_state.initialize_from_csv(rows)
        
        # Also add to data sources
        parsed = data_source_manager.parse_csv(contents, "general")
        if parsed.get("success"):
            data_source_manager.add_source("general", file.filename, parsed)
        
        if result["success"]:
            return {
                "success": True,
                "message": f"Digital Twin initialized with {result['metrics_count']} metrics",
                "state": business_state.get_snapshot(),
                "health_score": business_state.health_score,
                "confidence_score": business_state.confidence_score
            }
        else:
            return result
            
    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================
# CAUSAL GRAPH
# ============================================

@app.get("/causal-graph")
def get_causal_graph():
    """Get the causal relationship map for visualization"""
    return get_causal_graph_visualization()


@app.post("/analyze-impact")
def analyze_impact(request: ImpactRequest):
    """Analyze how changing one metric propagates through the system"""
    if not business_state.initialized:
        return {"error": "Digital Twin not initialized. Please upload data first."}
    
    result = propagate_change(
        source_metric=request.metric,
        change_percent=request.change_percent,
        current_state=business_state.current_state
    )
    
    return {
        "source": request.metric,
        "change": request.change_percent,
        "propagation": result
    }


# ============================================
# SIMULATION
# ============================================

@app.post("/simulate")
def run_simulation(request: SimulationRequest, session_id: Optional[str] = Query(None)):
    """Run full simulation with multi-agent system"""
    # Get user session components
    user_state, _ = get_user_state(session_id)
    
    # Get user's multi-agent engine
    user_engine = multi_agent_engine
    if session_id:
        session = session_manager.get_session(session_id)
        if session:
            user_engine = session.multi_agent_engine
    
    if not user_state.initialized:
        return {"error": "Digital Twin not initialized. Please upload data first."}
    
    adjustments = {
        "price": request.price or 0,
        "marketing_spend": request.marketing_spend or 0,
        "costs": request.costs or 0,
        "delivery_delay": request.delivery_delay or 0,
    }
    
    if request.market_shock:
        adjustments["revenue"] = -15
        adjustments["customers"] = -10
        adjustments["sentiment"] = -20
    
    # Always simulate from BASELINE state (original data), not current state
    simulation_result = user_engine.run_simulation(
        current_state=user_state.baseline_state.copy(),
        adjustments=adjustments
    )
    
    propagation_results = {}
    for metric, change in adjustments.items():
        if change != 0 and metric in user_state.baseline_state:
            propagation_results[metric] = propagate_change(
                metric, change, user_state.baseline_state, max_depth=3
            )
    
    # Update current state for display
    user_state.current_state = simulation_result["projected_state"].copy()
    
    forecast = generate_forecast(
        simulation_result["projected_state"],
        adjustments,
        months=3
    )
    
    deltas = business_state.get_deltas()
    
    return {
        "success": True,
        "comparison": {
            "baseline": simulation_result["baseline_state"],
            "projected": simulation_result["projected_state"],
            "deltas": deltas
        },
        "health_score": simulation_result["health_score"],
        "confidence": simulation_result["confidence"],
        "propagation": propagation_results,
        "agent_outputs": simulation_result["agent_outputs"],
        "total_rules_triggered": simulation_result["total_rules_triggered"],
        "recommendations": simulation_result["recommendations"],
        "warnings": simulation_result["warnings"],
        "tradeoffs": simulation_result["tradeoffs"],
        "strategic_priority": simulation_result["strategic_priority"],
        "forecast": forecast,
        "adjustments": adjustments,
        "data_sources": data_source_manager.get_combined_state()
    }


# ============================================
# HEALTH & FORECAST
# ============================================

@app.get("/health")
def get_health():
    """Get unified Business Health Score"""
    if not business_state.initialized:
        return {"error": "Digital Twin not initialized"}
    
    return {
        "health_score": business_state.health_score,
        "confidence_score": business_state.confidence_score,
        "components": {
            "revenue_health": min(100, (business_state.current_state.get("revenue", 100000) / 100000) * 50 + 50),
            "sentiment_health": business_state.current_state.get("sentiment", 75),
            "risk_health": 100 - business_state.current_state.get("risk_score", 50),
            "stability_health": 100 - (sum(business_state.volatility_scores.values()) / max(1, len(business_state.volatility_scores)))
        },
        "status": "excellent" if business_state.health_score >= 80 else
                  "good" if business_state.health_score >= 60 else
                  "fair" if business_state.health_score >= 40 else
                  "poor"
    }


@app.get("/forecast")
def get_forecast():
    """Get 3-month forward projection"""
    if not business_state.initialized:
        return {"error": "Digital Twin not initialized"}
    
    return generate_forecast(business_state.current_state, {}, months=3)


# ============================================
# STATE MANAGEMENT
# ============================================

@app.get("/state")
def get_state(session_id: Optional[str] = None):
    """Get current Digital Twin state snapshot"""
    user_state, user_data_manager = get_user_state(session_id)
    
    if not user_state.initialized:
        return {"error": "Digital Twin not initialized. Please upload data first."}
    
    return {
        **user_state.get_snapshot(),
        "data_sources": user_data_manager.get_combined_state()
    }


@app.post("/reset")
def reset_state(session_id: Optional[str] = None):
    """Reset to baseline state"""
    user_state, _ = get_user_state(session_id)
    
    if not user_state.initialized:
        return {"error": "Digital Twin not initialized"}
    
    user_state.current_state = user_state.baseline_state.copy()
    user_state._calculate_health_score()
    
    return {
        "success": True,
        "message": "State reset to baseline",
        "state": user_state.get_snapshot()
    }


# ============================================
# AGENTS
# ============================================

@app.get("/agents")
def get_agents():
    """List all agents in the Digital Twin"""
    return {
        "agents": [
            {"name": "Revenue Agent", "role": "Adjusts revenue based on customers, pricing, marketing", "inputs": ["customers", "price", "marketing_spend"]},
            {"name": "Customer Agent", "role": "Manages customer count, churn, retention", "inputs": ["marketing_spend", "price", "sentiment", "delivery_delay"]},
            {"name": "Sentiment Agent", "role": "Tracks sentiment score and brand health", "inputs": ["price", "delivery_delay", "marketing_spend", "customer_satisfaction"]},
            {"name": "Operations Agent", "role": "Manages delivery delay and efficiency", "inputs": ["costs", "delivery_delay"]},
            {"name": "Risk Agent", "role": "Calculates financial, operational, and overall risk", "inputs": ["revenue", "sentiment", "churn_rate", "delivery_delay", "costs"]},
            {"name": "Strategy Agent", "role": "Executive brain - recommendations, warnings, tradeoffs", "inputs": ["all agent outputs"]}
        ],
        "total_agents": 6
    }
    