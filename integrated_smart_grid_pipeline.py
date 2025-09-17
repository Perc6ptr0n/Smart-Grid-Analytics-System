#!/usr/bin/env python
"""
Integrated Smart Grid Pipeline System
=====================================

Complete pipeline system with optimized memory usage and GPU support.
Handles both training and deployment workflows.

Author: Laios Ioannis
Date: September 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
import joblib
import gc
import psutil
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
from pandas.tseries.holiday import USFederalHolidayCalendar
from smart_grid_data_generator import EnhancedDataGenerator
from schema_validator import SchemaValidator, safe_json_write, safe_json_read
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ===========================
# CONFIGURATION
# ===========================

@dataclass
class GridConfig:
    """Simplified configuration for the Smart Grid system."""
    
    # System settings
    max_ram_gb: int = 16
    use_gpu: bool = True
    checkpoint_frequency: int = 100
    
    # Data settings
    base_interval_minutes: int = 15
    simulation_years: int = 5
    train_years: int = 4
    # Canonical region IDs used across pipeline and dashboard
    regions: List[str] = field(default_factory=lambda: [
        'region_1', 'region_2', 'region_3', 'region_4',
        'region_5', 'region_6', 'region_7', 'region_8'
    ])
    
    # Model settings
    ensemble_algorithms: List[str] = field(default_factory=lambda: ['lightgbm', 'xgboost', 'catboost'])
    sliding_window_size: int = 96  # 24 hours of 15-min data
    max_features: int = 18
    hourly_window_size_hours: int = 72  # last 3 days for hourly models
    
    # Forecasting horizons
    horizons_15min: Dict[str, int] = field(default_factory=lambda: {
        '15min': 1, '1h': 4, '6h': 24, '12h': 48
    })
    horizons_hourly: Dict[str, int] = field(default_factory=lambda: {
        '24h': 24, '48h': 48
    })
    
    # Dashboard settings
    refresh_interval_minutes: int = 15
    max_historical_days: int = 30
    anomaly_threshold: float = 2.5
    forecast_anomaly_threshold: float = 2.0
    peer_anomaly_threshold: float = 2.0
    share_absolute_threshold: float = 0.40  # 40% of total
    share_anomaly_increase: float = 0.15    # +15% over recent mean share

# ===========================
# MEMORY MANAGEMENT
# ===========================

class MemoryManager:
    """Manages memory usage to stay within 16GB limit."""
    
    @staticmethod
    def check_memory():
        """Check current memory usage."""
        memory = psutil.virtual_memory()
        used_gb = memory.used / (1024**3)
        available_gb = memory.available / (1024**3)
        percent = memory.percent
        
        logger.info(f"Memory: {used_gb:.1f}GB used, {available_gb:.1f}GB available ({percent:.1f}%)")
        
        if percent > 85:
            logger.warning("High memory usage detected! Running garbage collection...")
            gc.collect()
            
        return percent < 90  # Return False if memory is critically high
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.float32)
        
        return df

# ===========================
# DATA GENERATOR
# ===========================

class OptimizedDataGenerator:
    """Memory-efficient data generator for Smart Grid."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.memory_manager = MemoryManager()
        # Adopt EnhancedDataGenerator as source of truth
        # We'll normalize region names to canonical IDs
        self.generator = EnhancedDataGenerator(simulation_years=config.simulation_years)

        # Mapping from enhanced generator region names to canonical region IDs
        # Preserves index order and cardinality (8 regions)
        self._region_map = {}

    def _normalize_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._region_map:
            # Build map by ordering columns and assigning region_1..region_8
            cols = list(df.columns)
            self._region_map = {cols[i]: f"region_{i+1}" for i in range(min(len(cols), len(self.config.regions)))}
        return df.rename(columns=self._region_map)

    def generate_training_data(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Generate multi-year training data using EnhancedDataGenerator."""
        logger.info("Generating training data with EnhancedDataGenerator...")
        data = self.generator.generate_complete_dataset()

        load = self._normalize_regions(data['load_data'])
        temp = self._normalize_regions(data['temperature_data'])
        humidity = self._normalize_regions(data['humidity_data'])
        solar = self._normalize_regions(data.get('solar_irradiance_data', pd.DataFrame(index=temp.index, columns=temp.columns, data=0.0)))
        solar = solar.astype(np.float32)

        # Return weather as dict of per-variable DataFrames (per-region columns)
        weather_frames = {
            'temperature': temp.astype(np.float32),
            'humidity': humidity.astype(np.float32),
            'solar_irradiance': solar.astype(np.float32)
        }

        # Optimize memory
        load = self.memory_manager.optimize_dataframe(load)
        for k in weather_frames:
            weather_frames[k] = self.memory_manager.optimize_dataframe(weather_frames[k])

        logger.info(f"Generated training dataset: {len(load):,} intervals")
        return load, weather_frames

    def generate_deployment_data(self, hours_back: int = 24) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Generate recent deployment data (default last 24 hours)."""
        logger.info("Generating deployment data with EnhancedDataGenerator...")
        batch = self.generator.generate_deployment_batch(hours_back=hours_back)

        load = self._normalize_regions(batch['load_data'])
        temp = self._normalize_regions(batch['temperature_data'])
        humidity = self._normalize_regions(batch['humidity_data'])
        solar = self._normalize_regions(batch.get('solar_irradiance_data', pd.DataFrame(index=temp.index, columns=temp.columns, data=0.0)))
        solar = solar.astype(np.float32)

        weather_frames = {
            'temperature': temp.astype(np.float32),
            'humidity': humidity.astype(np.float32),
            'solar_irradiance': solar.astype(np.float32)
        }

        # Occasionally inject special events to test anomaly detection
        if np.random.rand() < 0.2:
            region = np.random.choice(self.config.regions)
            factor = np.random.choice([0.5, 1.5, 2.0])
            load.iloc[-1:, load.columns.get_loc(region)] *= factor
            logger.info(f"Injected special event in {region} with factor {factor}")

        # Optimize memory
        load = self.memory_manager.optimize_dataframe(load)
        for k in weather_frames:
            weather_frames[k] = self.memory_manager.optimize_dataframe(weather_frames[k])

        return load, weather_frames

# ===========================
# FEATURE ENGINEERING
# ===========================

class SimpleFeatureEngineer:
    """Simplified feature engineering to avoid overfitting on generated data."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.feature_columns = []

    def _is_holiday_series(self, index: pd.DatetimeIndex) -> pd.Series:
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=index.min().date(), end=index.max().date())
        return pd.Series(index.normalize().isin(holidays).astype(int), index=index)

    def _build_weather_df_for_region(self, weather_frames: Dict[str, pd.DataFrame], region: str) -> pd.DataFrame:
        # Extract per-region columns into a single DataFrame with common column names
        df = pd.DataFrame(index=weather_frames['temperature'].index)
        df['temperature'] = weather_frames['temperature'][region]
        df['humidity'] = weather_frames['humidity'][region]
        df['solar_irradiance'] = weather_frames['solar_irradiance'][region]
        return df

    def create_features(self, load_data: pd.DataFrame, weather_frames: Dict[str, pd.DataFrame],
                        target_region: str) -> pd.DataFrame:
        """Create features at 15-min cadence as specified (temporal + lags + weather)."""
        features = pd.DataFrame(index=load_data.index)

        # Temporal features
        features['hour'] = load_data.index.hour
        features['day_of_week'] = load_data.index.dayofweek
        features['month'] = load_data.index.month
        features['is_weekend'] = (load_data.index.dayofweek >= 5).astype(int)
        features['is_holiday'] = self._is_holiday_series(load_data.index)

        # Lag features for target region
        target_load = load_data[target_region]
        features['lag_1h'] = target_load.shift(4)
        features['lag_6h'] = target_load.shift(24)
        features['lag_24h'] = target_load.shift(96)
        features['lag_48h'] = target_load.shift(192)
        features['lag_168h'] = target_load.shift(672)

        # Weather (per-region)
        weather_df = self._build_weather_df_for_region(weather_frames, target_region)
        features = features.join(weather_df)

        # Drop NaNs arising from lags
        features = features.dropna()
        self.feature_columns = features.columns.tolist()
        logger.info(f"Created {len(self.feature_columns)} features (15-min)")
        return features

    def create_hourly_features(self, load_data: pd.DataFrame, weather_frames: Dict[str, pd.DataFrame],
                               target_region: str) -> pd.DataFrame:
        """Create features at hourly cadence using mean upsampling for weather and load.
        Uses temporal, hourly lags, and weather (no rolling)."""
        # Resample to hourly
        load_hourly = load_data[target_region].resample('H').mean()
        weather_df_region = self._build_weather_df_for_region(weather_frames, target_region)
        weather_hourly = weather_df_region.resample('H').mean()

        features = pd.DataFrame(index=load_hourly.index)
        features['hour'] = features.index.hour
        features['day_of_week'] = features.index.dayofweek
        features['month'] = features.index.month
        features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)
        # Recompute holidays at hourly index
        features['is_holiday'] = self._is_holiday_series(features.index)

        # Hourly lags
        features['lag_1h'] = load_hourly.shift(1)
        features['lag_6h'] = load_hourly.shift(6)
        features['lag_24h'] = load_hourly.shift(24)
        features['lag_48h'] = load_hourly.shift(48)
        features['lag_168h'] = load_hourly.shift(168)

        # Join weather
        features = features.join(weather_hourly)
        features = features.dropna()
        self.feature_columns = features.columns.tolist()
        logger.info(f"Created {len(self.feature_columns)} features (hourly)")
        return features

# ===========================
# UNIFIED FORECASTER
# ===========================

class UnifiedForecaster:
    """GPU-optimized unified forecasting system."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.models = {}  # {region: {horizon: model}}
        self.scalers = {}
        self.is_trained = False
        # region -> horizon -> {'qhat': float, 'alpha': float}
        self.conformal = {}  # type: Dict[str, Dict[str, Dict[str, float]]]
        self.algorithms = self.config.ensemble_algorithms
        # region -> horizon -> {feature_name: importance}
        self.feature_importances: Dict[str, Dict[str, Dict[str, float]]] = {}
        
        # Check GPU availability
        self.gpu_available = self._check_gpu()
        
    def _check_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            if torch.cuda.is_available() and self.config.use_gpu:
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
                return True
        except ImportError:
            pass
        
        logger.info("Using CPU for training")
        return False
    
    def train_autoregressive(self, features_df: pd.DataFrame, target_series: pd.Series, region: str, window_size: int = 96, confidence_level: float = 0.95):
        """Train autoregressive next-step model for 15-min cadence."""
        # Prepare autoregressive training data: X = last window_size steps, y = next step
        X, y = [], []
        for i in range(window_size, len(features_df) - 1):
            window_features = features_df.iloc[i-window_size:i].values.flatten()
            target_value = target_series.iloc[i + 1]
            X.append(window_features)
            y.append(target_value)
        X = np.array(X)
        y = np.array(y)
        if len(X) < 200:
            logger.warning(f"Insufficient autoregressive data for {region}")
            return
        # Split train/val for conformal
        split_idx = int(len(X) * 0.9)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        self.scalers[region] = scaler
        # Train ensemble
        members = {}
        if 'lightgbm' in self.algorithms:
            members['lightgbm'] = self._train_lgbm(X_train_scaled, y_train, X_val_scaled, y_val)
        if 'xgboost' in self.algorithms:
            try:
                members['xgboost'] = self._train_xgb(X_train_scaled, y_train)
            except Exception as e:
                logger.warning(f"XGBoost training failed: {e}")
        if 'catboost' in self.algorithms:
            try:
                members['catboost'] = self._train_cat(X_train_scaled, y_train)
            except Exception as e:
                logger.warning(f"CatBoost training failed: {e}")
        if not members:
            logger.error("No models trained for autoregressive ensemble; aborting region")
            return
        self.models[region] = members
        # Conformal calibration
        preds_list = []
        for name, m in members.items():
            try:
                preds_list.append(self._predict_member(m, X_val_scaled, algo=name))
            except Exception as e:
                logger.warning(f"Validation prediction failed for {name}: {e}")
        preds_val = np.mean(np.vstack(preds_list), axis=0) if preds_list else np.zeros_like(y_val)
        residuals = np.abs(y_val - preds_val)
        alpha = 1.0 - confidence_level
        qhat = np.quantile(residuals, 1 - alpha)
        self.conformal[region] = {'qhat': float(qhat), 'alpha': float(alpha)}
        
    def _prepare_direct_forecast_data(self, features_df: pd.DataFrame,
                                     target_series: pd.Series,
                                     steps: int,
                                     window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for direct forecasting.
        If window_size is None, defaults to 15-min sliding_window_size."""
        if window_size is None:
            window_size = self.config.sliding_window_size
        
        X = []
        y = []
        
        for i in range(window_size, len(features_df) - steps):
            # Use sliding window of features
            window_features = features_df.iloc[i-window_size:i].values.flatten()
            # Target is 'steps' ahead
            target_value = target_series.iloc[i + steps]
            
            X.append(window_features)
            y.append(target_value)
        
        return np.array(X), np.array(y)
    
    def _train_lgbm(self, X: np.ndarray, y: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None):
        """Train LightGBM regressor."""
        import lightgbm as lgb
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }
        
        if self.gpu_available:
            params['device'] = 'gpu'
            params['gpu_platform_id'] = 0
            params['gpu_device_id'] = 0
        
        # Create dataset
        train_data = lgb.Dataset(X, label=y)
        valid_sets = [train_data]
        valid_names = ['train']
        callbacks = [lgb.log_evaluation(0)]
        if X_val is not None and y_val is not None and len(y_val) > 0:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=200,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        return model

    def _train_xgb(self, X: np.ndarray, y: np.ndarray):
        from xgboost import XGBRegressor
        params = dict(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='gpu_hist' if self.gpu_available else 'hist',
            predictor='gpu_predictor' if self.gpu_available else 'auto',
            reg_lambda=1.0,
        )
        model = XGBRegressor(**params)
        model.fit(X, y, verbose=False)
        return model

    def _train_cat(self, X: np.ndarray, y: np.ndarray):
        from catboost import CatBoostRegressor
        params = dict(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function='RMSE',
            task_type='GPU' if self.gpu_available else 'CPU',
            verbose=False,
        )
        model = CatBoostRegressor(**params)
        model.fit(X, y)
        return model

    def _predict_member(self, model: Any, X: np.ndarray, algo: str) -> np.ndarray:
        if hasattr(model, 'predict'):
            preds = model.predict(X)
            # lightgbm returns np.ndarray; xgboost returns np.ndarray; catboost returns np.ndarray
            return np.array(preds).ravel()
        # Fallback
        return np.zeros(X.shape[0])
    
    # Note: Hourly models are trained via train(..., is_hourly=True) with hourly features_df
    
    def predict(self, features_df: pd.DataFrame, region: str, horizon: str,
                is_hourly: bool = False) -> float:
        """Make point prediction for a specific horizon using prepared features_df.
        features_df cadence must match the model (15-min or hourly)."""
        if region not in self.models or horizon not in self.models[region]:
            logger.warning(f"No model found for {region}-{horizon}")
            return 0.0

        members = self.models[region][horizon]
        scaler = self.scalers[region].get(horizon)

        window_size = self.config.sliding_window_size if not is_hourly else self.config.hourly_window_size_hours
        base_n = features_df.shape[1]
        k = min(window_size, len(features_df))
        if k == 0 or base_n == 0:
            logger.error(f"Empty features for {region}-{horizon}; returning 0.0")
            return 0.0
        window = features_df.tail(k)
        flat = window.values.flatten()
        # Pad on the left (older part) if not enough rows
        if k < window_size:
            pad_len = (window_size - k) * base_n
            flat = np.concatenate([np.zeros(pad_len, dtype=flat.dtype), flat])
        X = flat.reshape(1, -1)
        if scaler is not None:
            X = scaler.transform(X)
        preds_list = []
        for name, m in members.items():
            try:
                preds_list.append(self._predict_member(m, X, algo=name))
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
        point = float(np.mean(np.vstack(preds_list), axis=0)[0]) if preds_list else 0.0
        return point

    def predict_with_interval(self, features_df: pd.DataFrame, region: str, horizon: str,
                              is_hourly: bool = False, confidence_level: float = 0.95) -> Dict[str, float]:
        point = self.predict(features_df, region, horizon, is_hourly=is_hourly)
        qhat = self.conformal.get(region, {}).get(horizon, {}).get('qhat', 0.0)
        lower = point - qhat
        upper = point + qhat
        return {'point': point, 'lower': lower, 'upper': upper}

# ===========================
# ANOMALY DETECTOR
# ===========================

class AnomalyDetector:
    """Detect anomalies in grid load patterns."""
    
    def __init__(self, config: GridConfig):
        self.config = config
        self.threshold = config.anomaly_threshold
        self.forecast_threshold = getattr(config, 'forecast_anomaly_threshold', self.threshold)
        self.peer_threshold = getattr(config, 'peer_anomaly_threshold', self.threshold)
        self.share_abs = getattr(config, 'share_absolute_threshold', 0.4)
        self.share_inc = getattr(config, 'share_anomaly_increase', 0.15)
        
    def detect_anomalies(self, load_data: pd.DataFrame, predictions: Dict,
                         forecast_paths: Optional[Dict[str, Dict[str, List[float]]]] = None) -> List[Dict]:
        """Detect anomalies based on statistical thresholds.
        - Uses recent window statistics (last 7 days if available)
        - Flags per-region and total-system anomalies for current and forecast values
        - If forecast_paths provided, checks per-step z-scores within horizons
        """
        anomalies = []
        # Use recent window for robust stats
        recent_len = min(len(load_data), 7 * int(24 * 60 / self.config.base_interval_minutes))
        recent = load_data.tail(recent_len) if recent_len > 0 else load_data

    # Per-region anomalies (time-wise against region's own history)
        for region in load_data.columns:
            # Calculate statistics
            series = recent[region]
            mean = series.mean()
            std = series.std(ddof=1)
            std = std if std > 0 else 1e-6
            current = load_data[region].iloc[-1]
            
            # Check for anomalies
            z_score = abs((current - mean) / std)
            
            if z_score > self.threshold:
                anomalies.append({
                    'region': region,
                    'severity': 'high' if z_score > 3 else 'medium',
                    'z_score': z_score,
                    'current_load': current,
                    'expected_range': (mean - 2*std, mean + 2*std),
                    'timestamp': load_data.index[-1]
                })
            
            # Check prediction anomalies (anchor per region against region's own history)
            if region in predictions:
                for horizon, pred_value in predictions[region].items():
                    pred_z_score = abs((pred_value - mean) / std)
                    if pred_z_score > self.forecast_threshold:
                        anomalies.append({
                            'region': region,
                            'severity': 'forecast',
                            'horizon': horizon,
                            'predicted_load': pred_value,
                            'z_score': pred_z_score,
                            'timestamp': load_data.index[-1]
                        })
        # Cross-sectional peer anomalies (current snapshot vs other regions)
        try:
            current_row = load_data.iloc[-1]
            peer_mean = current_row.mean()
            peer_std = current_row.std(ddof=1)
            peer_std = peer_std if peer_std > 0 else 1e-6
            for region, val in current_row.items():
                pz = abs((val - peer_mean) / peer_std)
                if pz > self.peer_threshold:
                    anomalies.append({
                        'region': region,
                        'severity': 'high' if pz > 3 else 'medium',
                        'z_score': float(pz),
                        'current_load': float(val),
                        'type': 'peer',
                        'timestamp': load_data.index[-1]
                    })
        except Exception as e:
            logger.warning(f"Peer anomaly check failed: {e}")

        # Share-of-total anomalies (level + change in share vs recent)
        try:
            total_recent = recent.sum(axis=1)
            share_recent = recent.divide(total_recent, axis=0).replace([np.inf, -np.inf], np.nan).dropna()
            current_total = current_row.sum()
            shares_now = (current_row / current_total) if current_total > 0 else current_row * 0
            
            # Robust share stats with fallback defaults
            if len(share_recent) > 0:
                share_stats = share_recent.agg(['mean', 'std'])  # per-region mean/std
            else:
                # Fallback: assume uniform distribution across regions
                n_regions = len(current_row)
                uniform_share = 1.0 / n_regions if n_regions > 0 else 0.125  # 8 regions default
                share_stats = pd.DataFrame({
                    'mean': {region: uniform_share for region in current_row.index},
                    'std': {region: uniform_share * 0.2 for region in current_row.index}  # 20% relative variation
                })
            
            for region, share_now in shares_now.items():
                # Safe access with fallbacks
                if isinstance(share_stats, pd.DataFrame):
                    mu = share_stats.get('mean', {}).get(region, 1.0/len(current_row))
                    sd = share_stats.get('std', {}).get(region, 0.025)  # 2.5% default std
                else:
                    mu = share_stats['mean'].get(region, 1.0/len(current_row))
                    sd = share_stats['std'].get(region, 0.025)
                
                # Handle NaN/invalid values
                if pd.isna(mu) or pd.isna(sd) or mu <= 0:
                    mu = 1.0 / len(current_row)  # uniform fallback
                if pd.isna(sd) or sd <= 0:
                    sd = mu * 0.2  # 20% of mean as fallback std
                
                sd = max(sd, 1e-6)  # minimum std to avoid division by zero
                dz = abs((share_now - mu) / sd)
                cond_abs = share_now >= self.share_abs
                cond_increase = (share_now - mu) >= self.share_inc
                
                if dz > self.peer_threshold or cond_abs or cond_increase:
                    anomalies.append({
                        'region': region,
                        'severity': 'high' if (cond_abs or dz > 3) else 'medium',
                        'z_score': float(dz),
                        'current_share': float(share_now),
                        'expected_share_range': (float(max(mu - 2*sd, 0.0)), float(min(mu + 2*sd, 1.0))),
                        'type': 'share',
                        'timestamp': load_data.index[-1]
                    })
        except Exception as e:
            logger.warning(f"Share-of-total check failed: {e}")

        # Total-system anomalies
        try:
            total_series = recent.sum(axis=1)
            t_mean = total_series.mean()
            t_std = total_series.std(ddof=1)
            t_std = t_std if t_std > 0 else 1e-6
            current_total = load_data.sum(axis=1).iloc[-1]
            t_z = abs((current_total - t_mean) / t_std)
            if t_z > self.threshold:
                anomalies.append({
                    'region': 'TOTAL',
                    'severity': 'high' if t_z > 3 else 'medium',
                    'z_score': t_z,
                    'current_load': current_total,
                    'expected_range': (t_mean - 2*t_std, t_mean + 2*t_std),
                    'timestamp': load_data.index[-1]
                })
            # Forecast totals by horizon (anchor)
            steps_map = {'15min': 1, '1h': 4, '6h': 24, '12h': 48, '24h': 96, '48h': 192}
            horizons = list(steps_map.keys())
            for h in horizons:
                total_pred = 0.0
                for region in load_data.columns:
                    total_pred += float(predictions.get(region, {}).get(h, 0.0))
                tz = abs((total_pred - t_mean) / t_std)
                if tz > self.forecast_threshold:
                    anomalies.append({
                        'region': 'TOTAL',
                        'severity': 'forecast',
                        'horizon': h,
                        'predicted_load': total_pred,
                        'z_score': tz,
                        'timestamp': load_data.index[-1]
                    })
            # If forecast_paths provided, evaluate per-step maxima within each horizon (TOTAL)
            if forecast_paths:
                for h, n in steps_map.items():
                    # Build total path for horizon h
                    total_path = None
                    for region in load_data.columns:
                        reg_path = (forecast_paths.get(region, {}) or {}).get(h)
                        if reg_path is None:
                            continue
                        arr = np.array(reg_path[:n], dtype=float)
                        if total_path is None:
                            total_path = arr
                        else:
                            total_path = total_path + arr
                    if total_path is not None and len(total_path) > 0:
                        # Max z-score across steps
                        step_z = np.max(np.abs((total_path - t_mean) / t_std))
                        if step_z > self.forecast_threshold:
                            anomalies.append({
                                'region': 'TOTAL',
                                'severity': 'forecast',
                                'horizon': h,
                                'predicted_load': float(total_path[np.argmax(np.abs((total_path - t_mean) / t_std))]),
                                'z_score': float(step_z),
                                'timestamp': load_data.index[-1]
                            })
            # Spike detection on first-step jump using recent diffs
            try:
                diffs = total_series.diff().dropna()
                if len(diffs) > 10 and forecast_paths:
                    # Use first step of the longest horizon path available
                    longest = max(horizons, key=lambda hk: steps_map[hk])
                    total_path = None
                    for region in load_data.columns:
                        reg_path = (forecast_paths.get(region, {}) or {}).get(longest)
                        if reg_path is None:
                            continue
                        arr = np.array(reg_path[:1], dtype=float)  # only first step
                        total_path = arr if total_path is None else total_path + arr
                    if total_path is not None and len(total_path) > 0:
                        current_total = load_data.sum(axis=1).iloc[-1]
                        first_jump = total_path[0] - current_total
                        d_mean = diffs.mean()
                        d_std = diffs.std(ddof=1)
                        d_std = d_std if d_std > 0 else 1e-6
                        dz = abs((first_jump - d_mean) / d_std)
                        if dz > max(3.0, self.forecast_threshold + 0.5):
                            anomalies.append({
                                'region': 'TOTAL',
                                'severity': 'forecast',
                                'horizon': '15min',
                                'predicted_load': float(total_path[0]),
                                'z_score': float(dz),
                                'timestamp': load_data.index[-1],
                                'note': 'Spike vs recent total diffs'
                            })
            except Exception as e:
                logger.warning(f"Spike detection failed: {e}")
        except Exception as e:
            logger.warning(f"Total-system anomaly check failed: {e}")

        # Per-region per-step forecast path anomalies
        try:
            if forecast_paths:
                steps_map = {'15min': 1, '1h': 4, '6h': 24, '12h': 48, '24h': 96, '48h': 192}
                for region in load_data.columns:
                    series = recent[region]
                    r_mean = series.mean()
                    r_std = series.std(ddof=1)
                    r_std = r_std if r_std > 0 else 1e-6
                    for h, n in steps_map.items():
                        reg_path = (forecast_paths.get(region, {}) or {}).get(h)
                        if not reg_path:
                            continue
                        arr = np.array(reg_path[:n], dtype=float)
                        max_z = float(np.max(np.abs((arr - r_mean) / r_std))) if len(arr) > 0 else 0.0
                        if max_z > self.forecast_threshold:
                            anomalies.append({
                                'region': region,
                                'severity': 'forecast',
                                'horizon': h,
                                'predicted_load': float(arr[np.argmax(np.abs((arr - r_mean) / r_std))]),
                                'z_score': max_z,
                                'type': 'per-step',
                                'timestamp': load_data.index[-1]
                            })
        except Exception as e:
            logger.warning(f"Per-region per-step forecast check failed: {e}")

        return anomalies

# ===========================
# MAIN PIPELINE
# ===========================

class IntegratedSmartGridPipeline:
    """Main integrated pipeline for Smart Grid system."""
    
    def __init__(self, config: Optional[GridConfig] = None):
        self.config = config or GridConfig()
        self.data_generator = OptimizedDataGenerator(self.config)
        self.feature_engineer = SimpleFeatureEngineer(self.config)
        self.forecaster = UnifiedForecaster(self.config)
        self.anomaly_detector = AnomalyDetector(self.config)
        self.schema_validator = SchemaValidator()
        
        # State tracking
        self.is_trained = False
        self.last_training_time = None
        self.last_inference_time = None
        
        # Setup directories
        self._setup_directories()
        
        # Check if models exist
        self._check_existing_models()
        
    def _setup_directories(self):
        """Create necessary directories."""
        dirs = ['data', 'models', 'outputs', 'logs']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)
    
    def _check_existing_models(self):
        """Check if trained models exist."""
        model_file = Path('models/forecaster.pkl')
        if model_file.exists():
            try:
                self.forecaster = joblib.load(model_file)
                self.is_trained = True
                logger.info("Loaded existing models")
            except Exception as e:
                logger.warning(f"Could not load models: {e}")
    
    def _avg_from_region_map(self, region_map: Dict[str, float]) -> float:
        """Compute average from region map, handling NaN values safely."""
        try:
            values = []
            for val in region_map.values():
                if val is not None and pd.notna(val):
                    # Ensure numeric conversion
                    numeric_val = pd.to_numeric(val, errors='coerce')
                    if pd.notna(numeric_val):
                        values.append(float(numeric_val))
            
            return np.mean(values) if values else 0.0
        except Exception as e:
            logger.warning(f"Failed to compute average from region map: {e}")
            return 0.0

    def _prepare_output_for_validation(self, output: Dict) -> Dict:
        """Prepare output for schema validation by converting non-serializable types."""
        import pandas as pd
        import numpy as np
        
        def convert_value(value):
            """Convert a value to JSON-serializable format."""
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                return convert_value(value.to_dict())
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                return float(value)
            elif isinstance(value, tuple):
                return list(value)  # Convert tuples to lists
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            else:
                return value
        
        return convert_value(output)
    
    def run_training_pipeline(self) -> bool:
        """Run the complete training pipeline."""
        logger.info("="*60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("="*60)
        
        try:
            # Step 1: Generate training data
            logger.info("Step 1: Generating training data...")
            load_data, weather_frames = self.data_generator.generate_training_data()
            
            # Split data
            train_end = int(len(load_data) * 0.8)
            val_end = int(len(load_data) * 0.9)
            
            train_load = load_data.iloc[:train_end]
            # weather_frames are dicts of aligned DataFrames; slice each
            train_weather_frames = {k: v.iloc[:train_end] for k, v in weather_frames.items()}
            
            # Step 2: Train models for each region and horizon
            logger.info("Step 2: Training models...")
            
            for region in self.config.regions:
                logger.info(f"Training autoregressive next-step model for {region}...")
                features = self.feature_engineer.create_features(train_load, train_weather_frames, region)
                target = train_load[region].loc[features.index]
                self.forecaster.train_autoregressive(features, target, region, window_size=self.config.sliding_window_size, confidence_level=0.95)
                # Memory cleanup
                gc.collect()
            
            # Step 3: Save models
            logger.info("Step 3: Saving models...")
            joblib.dump(self.forecaster, 'models/forecaster.pkl')
            joblib.dump(self.feature_engineer, 'models/feature_engineer.pkl')
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            logger.info("="*60)
            logger.info("TRAINING PIPELINE COMPLETE")
            logger.info("="*60)
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def run_deployment_pipeline(self) -> Dict:
        """Run the deployment pipeline."""
        if not self.is_trained:
            logger.error("Models must be trained first!")
            return {}
        
        logger.info("Running deployment pipeline...")
        
        try:
            # Compute required history to avoid NaNs and match window sizes
            # 15-min path: need max lag (168h => 672 steps) + window_size (96 steps)
            hours_needed_15 = int(((672 + self.config.sliding_window_size) * self.config.base_interval_minutes) / 60)
            # Hourly path: need max lag 168h + hourly window size (72h)
            hours_needed_hourly = self.config.hourly_window_size_hours + 168
            hours_back = max(24, hours_needed_15, hours_needed_hourly)
            # Generate deployment data with sufficient history
            load_data, weather_frames = self.data_generator.generate_deployment_data(hours_back=hours_back)
            
            # Make predictions for all regions
            steps_map = {'15min': 1, '1h': 4, '6h': 24, '12h': 48, '24h': 96, '48h': 192}
            max_steps = max(steps_map.values())
            forecast_paths: Dict[str, Dict[str, List[float]]] = {}
            interval_paths: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
            predictions = {}
            intervals = {}
            for region in self.config.regions:
                # Create features for this region
                features = self.feature_engineer.create_features(load_data, weather_frames, region)
                target = load_data[region].loc[features.index]
                # Prepare initial window for roll-forward
                window_size = self.config.sliding_window_size
                last_window = features.tail(window_size).values.flatten()
                # Roll forward autoregressively for max_steps
                scaler = self.forecaster.scalers.get(region)
                members = self.forecaster.models.get(region)
                qhat = self.forecaster.conformal.get(region, {}).get('qhat', 0.0)
                y_path = []
                lo_path = []
                hi_path = []
                ext_load = load_data.copy()
                ext_weather = {k: v.copy() for k, v in weather_frames.items()}
                last_ts = ext_load.index[-1]
                for step in range(max_steps):
                    # Build features for next step
                    feats_iter = self.feature_engineer.create_features(ext_load, ext_weather, region)
                    window = feats_iter.tail(window_size).values.flatten()
                    X = window.reshape(1, -1)
                    if scaler is not None:
                        X = scaler.transform(X)
                    preds_list = []
                    for name, m in members.items():
                        try:
                            preds_list.append(self.forecaster._predict_member(m, X, algo=name))
                        except Exception as e:
                            logger.warning(f"Prediction failed for {name}: {e}")
                    pi_step = float(np.mean(np.vstack(preds_list), axis=0)[0]) if preds_list else 0.0
                    y_path.append(pi_step)
                    lo_path.append(pi_step - qhat)
                    hi_path.append(pi_step + qhat)
                    # Append predicted load to future timestamp
                    next_ts = last_ts + pd.Timedelta(minutes=(step + 1) * self.config.base_interval_minutes)
                    new_row = ext_load.iloc[[-1]].copy()
                    new_row.index = [next_ts]
                    new_row.iloc[0] = ext_load.iloc[-1]
                    new_row.iloc[0][region] = pi_step
                    ext_load = pd.concat([ext_load, new_row])
                    for wf_key, wf_df in ext_weather.items():
                        new_w = wf_df.iloc[[-1]].copy()
                        new_w.index = [next_ts]
                        ext_weather[wf_key] = pd.concat([wf_df, new_w])
                # Slice paths per horizon
                region_paths = {}
                region_interval_paths = {}
                for h, n in steps_map.items():
                    region_paths[h] = y_path[:n]
                    region_interval_paths[h] = {
                        'lower': lo_path[:n],
                        'upper': hi_path[:n]
                    }
                forecast_paths[region] = region_paths
                interval_paths[region] = region_interval_paths
                # For dashboard compatibility, set anchor predictions and intervals
                predictions[region] = {h: region_paths[h][-1] for h in steps_map}
                intervals[region] = {h: {'lower': region_interval_paths[h]['lower'][-1], 'upper': region_interval_paths[h]['upper'][-1]} for h in steps_map}
            
            # Detect anomalies (include forecast paths for stronger checks)
            anomalies = self.anomaly_detector.detect_anomalies(load_data, predictions, forecast_paths=forecast_paths)
            
            # Prepare output with proper weather averaging
            weather_avg = {}
            for metric in ['temperature', 'humidity', 'solar_irradiance']:
                metric_avg = self._avg_from_region_map(
                    {r: weather_frames[metric][r].iloc[-1] for r in self.config.regions}
                )
                weather_avg[metric] = metric_avg
            
            output = {
                'current_load': load_data.iloc[-1].to_dict(),
                'predictions': predictions,
                'intervals': intervals,
                'forecast_paths': forecast_paths,
                'interval_paths': interval_paths,
                'feature_importances': self.forecaster.feature_importances,
                'anomalies': anomalies,
                'weather': weather_avg
            }
            
            # Save output with schema validation
            output_path = Path('outputs/latest_predictions.json')
            
            # Prepare output for validation (convert timestamps, etc.)
            validated_output = self._prepare_output_for_validation(output)
            
            if not safe_json_write(output_path, validated_output, self.schema_validator):
                logger.error("Failed to save output - fallback to basic write")
                with open(output_path, 'w') as f:
                    json.dump(validated_output, f, indent=2, default=str)
            
            self.last_inference_time = datetime.now()
            
            logger.info(f"Deployment complete: {len(predictions)} regions, {len(anomalies)} anomalies")
            
            return output
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return {}


# ===========================
# MAIN EXECUTION
# ===========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Smart Grid Pipeline')
    parser.add_argument('--mode', choices=['train', 'deploy', 'both'], 
                       default='both', help='Pipeline mode')
    args = parser.parse_args()
    
    pipeline = IntegratedSmartGridPipeline()
    
    if args.mode in ['train', 'both']:
        success = pipeline.run_training_pipeline()
        if not success:
            logger.error("Training failed!")
            exit(1)
    
    if args.mode in ['deploy', 'both']:
        output = pipeline.run_deployment_pipeline()
        if output:
            logger.info("Deployment successful!")
        else:
            logger.error("Deployment failed!")

            exit(1)
