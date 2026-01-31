"""
ML Model Registry
Manages model versioning, storage, and deployment.
"""

import json
import pickle
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np


class ModelStatus(str, Enum):
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Metadata for a registered model"""
    model_id: str
    model_name: str
    model_type: str  # e.g., "revenue_forecaster", "churn_predictor"
    version: str
    status: ModelStatus
    
    # Training info
    trained_at: str
    training_data_hash: str
    training_samples: int
    feature_names: List[str]
    target_metric: str
    
    # Performance metrics
    metrics: Dict[str, float]  # e.g., {"mae": 0.05, "rmse": 0.08, "r2": 0.92}
    
    # Additional info
    hyperparameters: Dict[str, Any]
    description: str = ""
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelRegistry:
    """
    Central registry for all ML models in the Digital Twin.
    Handles versioning, storage, and model lifecycle.
    """
    
    def __init__(self, storage_path: str = "./ml_models"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, Dict[str, Any]] = {}  # model_id -> {model, metadata}
        self.deployed_models: Dict[str, str] = {}  # model_type -> model_id
        
        self._load_registry()
    
    def register(
        self,
        model: Any,
        model_name: str,
        model_type: str,
        feature_names: List[str],
        target_metric: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        training_data: Optional[np.ndarray] = None,
        description: str = "",
        tags: List[str] = None
    ) -> str:
        """
        Register a new model in the registry.
        
        Returns:
            model_id: Unique identifier for the registered model
        """
        # Generate unique ID
        timestamp = datetime.now().isoformat()
        version = self._get_next_version(model_type)
        model_id = f"{model_type}_v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate data hash
        data_hash = ""
        training_samples = 0
        if training_data is not None:
            data_hash = hashlib.md5(training_data.tobytes()).hexdigest()[:8]
            training_samples = len(training_data)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            version=version,
            status=ModelStatus.TRAINED,
            trained_at=timestamp,
            training_data_hash=data_hash,
            training_samples=training_samples,
            feature_names=feature_names,
            target_metric=target_metric,
            metrics=metrics,
            hyperparameters=hyperparameters,
            description=description,
            tags=tags or []
        )
        
        # Store model
        self.models[model_id] = {
            "model": model,
            "metadata": metadata
        }
        
        # Save to disk
        self._save_model(model_id, model, metadata)
        self._save_registry()
        
        return model_id
    
    def deploy(self, model_id: str) -> bool:
        """
        Mark a model as deployed and set it as primary for its type.
        """
        if model_id not in self.models:
            return False
        
        metadata = self.models[model_id]["metadata"]
        
        # Undeploy previous version
        model_type = metadata.model_type
        if model_type in self.deployed_models:
            old_id = self.deployed_models[model_type]
            if old_id in self.models:
                self.models[old_id]["metadata"].status = ModelStatus.DEPRECATED
        
        # Deploy new version
        metadata.status = ModelStatus.DEPLOYED
        self.deployed_models[model_type] = model_id
        
        self._save_registry()
        return True
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get a specific model by ID"""
        if model_id in self.models:
            return self.models[model_id]["model"]
        return self._load_model(model_id)
    
    def get_deployed_model(self, model_type: str) -> Optional[Any]:
        """Get the currently deployed model for a given type"""
        model_id = self.deployed_models.get(model_type)
        if model_id:
            return self.get_model(model_id)
        return None
    
    def get_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model"""
        if model_id in self.models:
            return self.models[model_id]["metadata"]
        return None
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        status: Optional[ModelStatus] = None
    ) -> List[ModelMetadata]:
        """List all models, optionally filtered"""
        results = []
        
        for model_id, data in self.models.items():
            metadata = data["metadata"]
            
            if model_type and metadata.model_type != model_type:
                continue
            if status and metadata.status != status:
                continue
            
            results.append(metadata)
        
        # Sort by version descending
        results.sort(key=lambda m: m.version, reverse=True)
        return results
    
    def compare_models(
        self,
        model_ids: List[str],
        metric: str = "rmse"
    ) -> Dict[str, float]:
        """Compare performance of multiple models on a specific metric"""
        comparison = {}
        
        for model_id in model_ids:
            metadata = self.get_metadata(model_id)
            if metadata and metric in metadata.metrics:
                comparison[model_id] = metadata.metrics[metric]
        
        return dict(sorted(comparison.items(), key=lambda x: x[1]))
    
    def predict(
        self,
        model_type: str,
        features: Dict[str, float]
    ) -> Optional[float]:
        """
        Make a prediction using the deployed model of a given type.
        """
        model = self.get_deployed_model(model_type)
        if model is None:
            return None
        
        # Get feature names from metadata
        model_id = self.deployed_models.get(model_type)
        metadata = self.get_metadata(model_id)
        
        if metadata is None:
            return None
        
        # Prepare feature vector
        feature_vector = np.array([
            features.get(name, 0.0)
            for name in metadata.feature_names
        ]).reshape(1, -1)
        
        # Predict
        if hasattr(model, "predict"):
            return float(model.predict(feature_vector)[0])
        elif callable(model):
            return float(model(feature_vector))
        
        return None
    
    # ============================================
    # PRIVATE METHODS
    # ============================================
    
    def _get_next_version(self, model_type: str) -> str:
        """Get next version number for a model type"""
        existing = self.list_models(model_type=model_type)
        if not existing:
            return "1.0"
        
        latest = existing[0].version
        major, minor = map(int, latest.split("."))
        return f"{major}.{minor + 1}"
    
    def _save_model(self, model_id: str, model: Any, metadata: ModelMetadata):
        """Save model and metadata to disk"""
        model_dir = self.storage_path / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
    
    def _load_model(self, model_id: str) -> Optional[Any]:
        """Load model from disk"""
        model_path = self.storage_path / model_id / "model.pkl"
        if not model_path.exists():
            return None
        
        with open(model_path, "rb") as f:
            return pickle.load(f)
    
    def _save_registry(self):
        """Save registry index to disk"""
        registry_path = self.storage_path / "registry.json"
        
        registry_data = {
            "deployed_models": self.deployed_models,
            "models": {
                model_id: asdict(data["metadata"])
                for model_id, data in self.models.items()
            }
        }
        
        with open(registry_path, "w") as f:
            json.dump(registry_data, f, indent=2, default=str)
    
    def _load_registry(self):
        """Load registry from disk"""
        registry_path = self.storage_path / "registry.json"
        
        if not registry_path.exists():
            return
        
        with open(registry_path, "r") as f:
            data = json.load(f)
        
        self.deployed_models = data.get("deployed_models", {})
        
        for model_id, metadata_dict in data.get("models", {}).items():
            # Convert status back to enum
            metadata_dict["status"] = ModelStatus(metadata_dict["status"])
            metadata = ModelMetadata(**metadata_dict)
            
            # Load model from disk
            model = self._load_model(model_id)
            
            if model is not None:
                self.models[model_id] = {
                    "model": model,
                    "metadata": metadata
                }


# ============================================
# GLOBAL REGISTRY INSTANCE
# ============================================

model_registry = ModelRegistry()
