# GeometricEnsemble-XACLE Source Package
# Exposes key functions for easier imports

from .config import (
    RAW_DATA_DIR, 
    FEATURE_DIR, 
    MODEL_DIR, 
    WEIGHT_XGBOOST, 
    WEIGHT_SVR
)

from .geometry import compute_geometric_features
from .fusion import construct_unified_vector
from .extract_features import process_dataset
from .train import train_ensemble_model

# Metadata
__version__ = "1.0.0"
__author__ = "XACLE Challenge Team"
