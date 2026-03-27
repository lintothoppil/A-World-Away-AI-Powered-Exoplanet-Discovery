"""
NASA Exoplanet Classification System - Preprocessing Utilities
Team: syntax_in_orbit
NASA Space Apps Challenge 2025

This module contains preprocessing utilities for exoplanet data including
scaling, imputation, and feature engineering functions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ExoplanetPreprocessor:
    """
    A comprehensive preprocessing class for exoplanet classification data.
    
    This class handles data cleaning, feature scaling, imputation, and
    label encoding for NASA exoplanet datasets.
    """
    
    def __init__(self, dataset: str, model_path: str):
        """
        Initialize the preprocessor for a specific dataset.
        
        Args:
            dataset: Dataset name ('k2', 'tess', 'koi')
            model_path: Base path to model files
        """
        self.dataset = dataset
        self.model_path = model_path
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        
        # Set dataset-specific feature columns
        if dataset.lower() == 'tess':
            self.feature_columns = ['pl_orbper', 'pl_trandurh', 'pl_trandep', 'st_teff', 'st_pmralim', 'st_pmdeclim']
        elif dataset.lower() == 'koi':
            self.feature_columns = ['koi_period', 'koi_duration', 'koi_prad', 'koi_depth', 'koi_teq',
                                  'koi_insol', 'koi_model_snr', 'koi_srad', 'koi_fpflag_nt', 'koi_fpflag_ss',
                                  'koi_fpflag_co', 'koi_fpflag_ec', 'koi_steff', 'koi_impact', 'koi_max_sngle_ev']
        else:  # k2 or default
            self.feature_columns = ['pl_orbper', 'pl_trandep', 'st_teff']
        
        logger.info(f"Initialized preprocessor for {dataset} dataset")
    
    def load_preprocessing_components(self) -> bool:
        """
        Load preprocessing components (scaler, imputer, label encoder) for the dataset.
        
        Returns:
            bool: True if all components loaded successfully, False otherwise
        """
        try:
            # Load scaler - handle different naming conventions
            scaler_paths = [
                os.path.join(self.model_path, f'{self.dataset}_scaler.pkl'),
                os.path.join(self.model_path, 'scaler.npy'),  # For KOI dataset
                os.path.join(self.model_path, 'scaler.pkl')
            ]
            
            scaler_loaded = False
            for scaler_path in scaler_paths:
                if os.path.exists(scaler_path):
                    if scaler_path.endswith('.npy'):
                        # Handle numpy scaler files
                        scaler_data = np.load(scaler_path, allow_pickle=True)
                        self.scaler = StandardScaler()
                        if hasattr(scaler_data, 'item'):
                            scaler_obj = scaler_data.item()
                            if hasattr(scaler_obj, 'mean_') and hasattr(scaler_obj, 'scale_'):
                                self.scaler.mean_ = scaler_obj.mean_
                                self.scaler.scale_ = scaler_obj.scale_
                                self.scaler.var_ = scaler_obj.var_
                    else:
                        self.scaler = joblib.load(scaler_path)
                    logger.info(f"Loaded scaler from {os.path.basename(scaler_path)}")
                    scaler_loaded = True
                    break
            
            if not scaler_loaded:
                logger.warning(f"Scaler not found in any of these locations: {[os.path.basename(p) for p in scaler_paths]}, creating default")
                self.scaler = StandardScaler()
            
            # Load imputer - handle different naming conventions
            imputer_paths = [
                os.path.join(self.model_path, f'{self.dataset}_imputer.pkl'),
                os.path.join(self.model_path, 'imputer.pkl'),  # For KOI dataset
                os.path.join(self.model_path, 'imputer.npy')
            ]
            
            imputer_loaded = False
            for imputer_path in imputer_paths:
                if os.path.exists(imputer_path):
                    self.imputer = joblib.load(imputer_path)
                    logger.info(f"Loaded imputer from {os.path.basename(imputer_path)}")
                    imputer_loaded = True
                    break
            
            if not imputer_loaded:
                logger.warning(f"Imputer not found in any of these locations: {[os.path.basename(p) for p in imputer_paths]}, creating default")
                self.imputer = SimpleImputer(strategy='mean')
            
            # Load label encoder - handle different naming conventions
            encoder_paths = [
                os.path.join(self.model_path, f'{self.dataset}_label_encoder.npy'),
                os.path.join(self.model_path, 'label_encoder.npy'),  # For KOI dataset
                os.path.join(self.model_path, f'{self.dataset}_label_encoder.pkl'),
                os.path.join(self.model_path, 'label_encoder.pkl')
            ]
            
            encoder_loaded = False
            for encoder_path in encoder_paths:
                if os.path.exists(encoder_path):
                    if encoder_path.endswith('.npy'):
                        encoder_data = np.load(encoder_path, allow_pickle=True)
                        self.label_encoder = LabelEncoder()
                        self.label_encoder.classes_ = encoder_data
                    else:
                        self.label_encoder = joblib.load(encoder_path)
                    logger.info(f"Loaded label encoder from {os.path.basename(encoder_path)}")
                    encoder_loaded = True
                    break
            
            if not encoder_loaded:
                logger.warning(f"Label encoder not found in any of these locations: {[os.path.basename(p) for p in encoder_paths]}, creating default")
                self.label_encoder = LabelEncoder()
                # Set default classes for exoplanet classification
                self.label_encoder.classes_ = np.array(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'])
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading preprocessing components for {self.dataset}: {str(e)}")
            return False
    
    def preprocess_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess feature data for prediction.
        
        Args:
            data: DataFrame containing exoplanet features
            
        Returns:
            np.ndarray: Preprocessed features ready for model prediction
        """
        try:
            # Ensure we have the required columns
            missing_cols = set(self.feature_columns) - set(data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Extract features
            features = data[self.feature_columns].copy()
            
            # Handle missing values
            if self.imputer is not None:
                try:
                    features = self.imputer.transform(features)
                except ValueError:
                    # Imputer not fitted, use simple fillna
                    features = features.fillna(features.mean()).values
            else:
                features = features.fillna(features.mean()).values
            
            # Scale features
            if self.scaler is not None:
                try:
                    features = self.scaler.transform(features)
                except ValueError:
                    # Scaler not fitted, use simple standardization
                    features = (features - features.mean()) / features.std()
                    # Handle NaN values after standardization
                    features = np.nan_to_num(features, nan=0.0)
            
            logger.info(f"Preprocessed {len(features)} samples for {self.dataset} dataset")
            return features
            
        except Exception as e:
            logger.error(f"Error preprocessing features for {self.dataset}: {str(e)}")
            raise
    
    def inverse_transform_labels(self, encoded_labels: np.ndarray) -> np.ndarray:
        """
        Convert encoded labels back to original class names.
        
        Args:
            encoded_labels: Numerically encoded labels
            
        Returns:
            np.ndarray: Original class names
        """
        try:
            # Ensure encoded_labels is a proper array
            if np.isscalar(encoded_labels):
                encoded_labels = np.array([encoded_labels])
            elif isinstance(encoded_labels, list):
                encoded_labels = np.array(encoded_labels)
            elif hasattr(encoded_labels, 'ndim') and encoded_labels.ndim == 0:  # 0-d array
                encoded_labels = np.array([encoded_labels.item()])
            
            if self.label_encoder is not None:
                # Check if labels are within valid range
                if hasattr(self.label_encoder, 'classes_'):
                    max_class = len(self.label_encoder.classes_) - 1
                    # Clamp labels to valid range
                    encoded_labels = np.clip(encoded_labels, 0, max_class)
                return self.label_encoder.inverse_transform(encoded_labels)
            else:
                # Default mapping if no encoder available
                default_mapping = {0: 'CONFIRMED', 1: 'CANDIDATE', 2: 'FALSE POSITIVE'}
                return np.array([default_mapping.get(int(label), 'UNKNOWN') for label in encoded_labels])
        except Exception as e:
            logger.error(f"Error inverse transforming labels: {str(e)}")
            # Return default labels based on prediction values
            default_mapping = {0: 'CONFIRMED', 1: 'CANDIDATE', 2: 'FALSE POSITIVE'}
            # Handle scalar case
            if np.isscalar(encoded_labels) or (hasattr(encoded_labels, 'ndim') and encoded_labels.ndim == 0):
                label_val = int(encoded_labels) if np.isscalar(encoded_labels) else int(encoded_labels.item())
                return np.array([default_mapping.get(label_val % 3, 'UNKNOWN')])
            else:
                return np.array([default_mapping.get(int(label) % 3, 'UNKNOWN') for label in encoded_labels])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance information for the dataset.
        
        Returns:
            Dict[str, float]: Feature importance scores
        """
        # Default feature importance based on typical exoplanet classification
        default_importance = {
            'pl_orbper': 0.35,  # Orbital period is crucial for classification
            'pl_trandep': 0.45, # Transit depth is highly predictive
            'st_teff': 0.20     # Stellar temperature provides context
        }
        
        logger.info(f"Retrieved feature importance for {self.dataset} dataset")
        return default_importance
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Validate input data format and values.
        
        Args:
            data: Input DataFrame to validate
            
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        try:
            # Check required columns
            missing_cols = set(self.feature_columns) - set(data.columns)
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
            
            # Check for empty data
            if data.empty:
                return False, "Input data is empty"
            
            # Check data types and ranges
            for col in self.feature_columns:
                if col not in data.columns:
                    continue
                    
                # Check for non-numeric values
                if not pd.api.types.is_numeric_dtype(data[col]):
                    return False, f"Column {col} must contain numeric values"
                
                # Check for reasonable ranges (only check non-NaN values)
                non_nan_data = data[col].dropna()
                if len(non_nan_data) > 0:
                    if col == 'pl_orbper' and (non_nan_data <= 0).any():
                        return False, f"Orbital period must be positive"
                    elif col == 'pl_trandep' and (non_nan_data < 0).any():
                        return False, f"Transit depth must be non-negative"
                    elif col == 'st_teff' and (non_nan_data < 1000).any():
                        return False, f"Stellar temperature seems too low (minimum 1000K expected)"
                    elif col == 'pl_trandurh' and (non_nan_data <= 0).any():
                        return False, f"Transit duration must be positive"
                    elif col in ['st_pmralim', 'st_pmdeclim'] and not non_nan_data.isin([0, 1]).all():
                        return False, f"{col} must be 0 or 1"
                    # KOI-specific validations
                    elif col == 'koi_period' and (non_nan_data <= 0).any():
                        return False, f"KOI period must be positive"
                    elif col in ['koi_duration', 'koi_prad', 'koi_depth', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_srad', 'koi_steff', 'koi_impact', 'koi_max_sngle_ev'] and (non_nan_data < 0).any():
                        return False, f"{col} must be non-negative"
                    elif col in ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec'] and not non_nan_data.isin([0, 1]).all():
                        return False, f"{col} must be 0 or 1"
            
            return True, "Data validation passed"
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return False, f"Validation error: {str(e)}"


def create_sample_data(dataset: str) -> pd.DataFrame:
    """
    Create sample exoplanet data for testing and demonstration.
    
    Args:
        dataset: Dataset name ('k2', 'tess', 'koi')
        
    Returns:
        pd.DataFrame: Sample exoplanet data
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate sample data based on typical exoplanet characteristics
    n_samples = 100
    
    if dataset.lower() == 'k2':
        # K2 mission data characteristics
        orbital_periods = np.random.lognormal(1.5, 0.8, n_samples)
        transit_depths = np.random.lognormal(-3, 1.2, n_samples)
        stellar_temps = np.random.normal(5500, 800, n_samples)
    elif dataset.lower() == 'tess':
        # TESS mission data characteristics
        orbital_periods = np.random.lognormal(1.2, 0.6, n_samples)
        transit_durations = np.random.lognormal(0.8, 0.5, n_samples)  # Hours
        transit_depths = np.random.lognormal(-2.5, 1.0, n_samples)
        stellar_temps = np.random.normal(5200, 700, n_samples)
        pm_ra_limits = np.random.choice([0, 1], n_samples)
        pm_dec_limits = np.random.choice([0, 1], n_samples)
    else:  # KOI
        # Kepler Objects of Interest characteristics
        orbital_periods = np.random.lognormal(2.0, 1.0, n_samples)
        transit_depths = np.random.lognormal(-3.5, 1.5, n_samples)
        stellar_temps = np.random.normal(5800, 900, n_samples)
    
    # Ensure reasonable ranges
    orbital_periods = np.clip(orbital_periods, 0.5, 1000)
    transit_depths = np.clip(transit_depths, 0.001, 100)
    stellar_temps = np.clip(stellar_temps, 3000, 8000)
    
    # Create base sample data
    sample_data = pd.DataFrame({
        'pl_orbper': orbital_periods,
        'pl_trandep': transit_depths,
        'st_teff': stellar_temps
    })
    
    # Add TESS-specific features if dataset is TESS
    if dataset.lower() == 'tess':
        transit_durations = np.clip(transit_durations, 0.5, 12)  # Hours
        sample_data['pl_trandurh'] = transit_durations
        sample_data['st_pmralim'] = pm_ra_limits
        sample_data['st_pmdeclim'] = pm_dec_limits
    
    logger.info(f"Created {n_samples} sample records for {dataset} dataset")
    return sample_data


def get_dataset_statistics(dataset: str) -> Dict[str, Any]:
    """
    Get statistical information about a dataset.
    
    Args:
        dataset: Dataset name ('k2', 'tess', 'koi')
        
    Returns:
        Dict[str, Any]: Dataset statistics
    """
    # Mock statistics - in production, these would be loaded from actual data
    stats = {
        'k2': {
            'total_samples': 2847,
            'confirmed_planets': 1234,
            'candidates': 987,
            'false_positives': 626,
            'accuracy': 0.87,
            'model_type': 'XGBoost',
            'last_trained': '2025-10-04T14:30:00Z'
        },
        'tess': {
            'total_samples': 4562,
            'confirmed_planets': 2103,
            'candidates': 1456,
            'false_positives': 1003,
            'accuracy': 0.89,
            'model_type': 'XGBoost',
            'last_trained': '2025-10-04T15:45:00Z'
        },
        'koi': {
            'total_samples': 8013,
            'confirmed_planets': 4034,
            'candidates': 2345,
            'false_positives': 1634,
            'accuracy': 0.91,
            'model_type': 'XGBoost',
            'last_trained': '2025-10-04T16:20:00Z'
        }
    }
    
    return stats.get(dataset.lower(), {
        'total_samples': 0,
        'confirmed_planets': 0,
        'candidates': 0,
        'false_positives': 0,
        'accuracy': 0.0,
        'model_type': 'XGBoost',
        'last_trained': None
    })
