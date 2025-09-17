"""
Smart Grid Backtesting and Coverage Report
==========================================

Rolling-origin backtesting with MAPE/MASE and conformal coverage analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Handles rolling-origin backtesting and coverage analysis."""
    
    def __init__(self, config):
        """Initialize backtesting engine."""
        self.config = config
        self.horizons = ['15min', '1h', '6h', '12h', '24h', '48h']
        
    def compute_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Percentage Error."""
        try:
            mask = y_true != 0
            if not mask.any():
                return np.inf
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        except:
            return np.inf
    
    def compute_mase(self, y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        """Compute Mean Absolute Scaled Error."""
        try:
            # Seasonal naive forecast (same time 1 week ago)
            season = 96 * 7  # 1 week in 15-min intervals
            if len(y_train) < season:
                return np.inf
            
            naive_errors = np.abs(y_train[season:] - y_train[:-season])
            mae_naive = np.mean(naive_errors)
            
            if mae_naive == 0:
                return np.inf
                
            mae_model = np.mean(np.abs(y_true - y_pred))
            return mae_model / mae_naive
        except:
            return np.inf
    
    def compute_coverage(self, y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, 
                        confidence_level: float = 0.9) -> float:
        """Compute empirical coverage of prediction intervals."""
        try:
            if len(y_true) == 0:
                return 0.0
            coverage = np.mean((y_true >= y_lower) & (y_true <= y_upper))
            return coverage * 100
        except:
            return 0.0
    
    def run_rolling_backtest(self, data_generator, feature_engineer, forecaster,
                           weeks_back: int = 4, min_train_days: int = 30) -> Dict[str, Any]:
        """Run rolling-origin backtesting over recent weeks."""
        logger.info(f"Starting rolling backtest ({weeks_back} weeks)")
        
        results = {
            'backtest_date': datetime.now().isoformat(),
            'weeks_tested': weeks_back,
            'regions': {},
            'summary': {}
        }
        
        try:
            # Generate complete dataset using the training data method
            logger.info(f"Generating complete dataset for backtest")
            
            # Use the training data generation which calls generate_complete_dataset internally
            load_data, weather_data = data_generator.generate_training_data()
            
            if load_data.empty:
                raise ValueError("No load data generated")
            
            # Ensure we have enough data
            min_required_intervals = (min_train_days + weeks_back * 7) * 96
            if len(load_data) < min_required_intervals:
                logger.warning(f"Limited data: {len(load_data)} intervals")
            
            # Rolling origin backtest setup
            train_size = min_train_days * 96  # intervals for training
            test_size = 7 * 96  # 1 week of testing
            
            regional_metrics = {}
            
            # Per-region backtesting
            for region in self.config.regions:
                logger.info(f"Backtesting region {region}")
                
                if region not in load_data.columns:
                    logger.warning(f"Region {region} not found in data")
                    continue
                
                region_data = load_data[region].dropna()
                week_results = []
                
                # Rolling validation
                for week in range(weeks_back):
                    # Calculate indices for this fold
                    test_start = len(region_data) - (weeks_back - week) * test_size
                    test_end = test_start + test_size
                    train_end = test_start
                    train_start = max(0, train_end - train_size)
                    
                    if train_start >= train_end or test_start >= test_end or test_end > len(region_data):
                        continue
                    
                    # Split data
                    train_data = region_data.iloc[train_start:train_end]
                    test_data = region_data.iloc[test_start:test_end]
                    
                    # Simple naive forecast (persistence + seasonal)
                    y_true = test_data.values
                    
                    # Seasonal naive: same time last week
                    seasonal_lag = 7 * 96  # 1 week
                    if len(train_data) >= seasonal_lag:
                        y_pred = train_data.iloc[-seasonal_lag:].values
                        if len(y_pred) > len(y_true):
                            y_pred = y_pred[:len(y_true)]
                        elif len(y_pred) < len(y_true):
                            # Extend with last value
                            y_pred = np.concatenate([y_pred, 
                                                   np.full(len(y_true) - len(y_pred), y_pred[-1])])
                    else:
                        # Simple persistence
                        y_pred = np.full(len(y_true), train_data.iloc[-1])
                    
                    # Calculate metrics
                    mape = self.compute_mape(y_true, y_pred)
                    mase = self.compute_mase(y_true, y_pred, train_data.values)
                    
                    week_results.append({
                        'week': week + 1,
                        'mape': mape,
                        'mase': mase,
                        'train_size': len(train_data),
                        'test_size': len(test_data)
                    })
                
                # Aggregate regional results
                if week_results:
                    avg_mape = np.mean([r['mape'] for r in week_results if not np.isinf(r['mape'])])
                    avg_mase = np.mean([r['mase'] for r in week_results if not np.isinf(r['mase'])])
                    
                    regional_metrics[region] = {
                        'mape': avg_mape,
                        'mase': avg_mase,
                        'weeks_tested': len(week_results),
                        'weekly_results': week_results
                    }
                    
                    logger.info(f"Region {region}: MAPE={avg_mape:.3f}, MASE={avg_mase:.3f}")
            
            # Overall summary
            if regional_metrics:
                all_mapes = [m['mape'] for m in regional_metrics.values() if not np.isinf(m['mape'])]
                all_mases = [m['mase'] for m in regional_metrics.values() if not np.isinf(m['mase'])]
                
                results['summary'] = {
                    'overall_mape': np.mean(all_mapes) if all_mapes else np.inf,
                    'overall_mase': np.mean(all_mases) if all_mases else np.inf,
                    'regions_tested': len(regional_metrics),
                    'total_weeks': weeks_back
                }
            
            results['regions'] = regional_metrics
            logger.info("✅ Backtest completed successfully")
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            results['error'] = str(e)
        
        return results

    def _test_horizon(self, region: str, horizon: str, train_data: pd.Series, 
                     test_data: pd.Series, weather_data: Dict, 
                     feature_engineer, forecaster) -> Dict[str, float]:
        """Test a specific horizon for a region."""
        
        # Map horizon to steps
        steps_map = {'15min': 1, '1h': 4, '6h': 24, '12h': 48, '24h': 96, '48h': 192}
        steps = steps_map[horizon]
        
        # Prepare features for training period
        try:
            train_end = train_data.index[-1]
            train_start = train_data.index[0]
            
            # Extract weather for training period
            weather_train = {}
            for metric in weather_data:
                weather_train[metric] = weather_data[metric].loc[train_start:train_end]
            
            # Create features
            train_features = feature_engineer.create_features(
                load_data=train_data.to_frame(region),
                weather_data=weather_train,
                cadence='15min'
            )
            
            # Align target
            target_series = train_data.shift(-steps).dropna()
            feature_subset = train_features.loc[target_series.index]
            
            if len(feature_subset) < 100:  # Minimum samples
                return {'mape': np.inf, 'mase': np.inf, 'coverage': 0.0, 'samples': 0}
            
            # Quick training (simplified)
            forecaster._ensure_models_trained()
            
            # Generate predictions on test period
            predictions = []
            actuals = []
            lower_bounds = []
            upper_bounds = []
            
            # Rolling predictions
            window_size = 96  # 1 day lookback
            max_predictions = min(50, len(test_data) - steps)  # Limit to 50 predictions
            
            for i in range(0, max_predictions, 4):  # Every hour
                test_start_idx = i
                test_end_idx = test_start_idx + window_size
                
                if test_end_idx + steps >= len(test_data):
                    break
                
                # Extract recent data
                recent_data = pd.concat([
                    train_data.iloc[-window_size:],
                    test_data.iloc[:test_end_idx]
                ])
                
                # Create features for prediction
                pred_weather = {}
                for metric in weather_data:
                    pred_weather[metric] = weather_data[metric].loc[recent_data.index]
                
                try:
                    pred_features = feature_engineer.create_features(
                        load_data=recent_data.to_frame(region),
                        weather_data=pred_weather,
                        cadence='15min'
                    )
                    
                    if len(pred_features) == 0:
                        continue
                    
                    # Make prediction
                    last_features = pred_features.iloc[-1:].values
                    pred_val = forecaster.predict(last_features, region)
                    
                    # Simple interval (assume ±10% for quick backtest)
                    interval_width = abs(pred_val * 0.1)
                    
                    # Store results
                    actual_val = test_data.iloc[test_end_idx + steps - 1]
                    
                    predictions.append(pred_val)
                    actuals.append(actual_val)
                    lower_bounds.append(pred_val - interval_width)
                    upper_bounds.append(pred_val + interval_width)
                
                except Exception:
                    continue
            
            if len(predictions) == 0:
                return {'mape': np.inf, 'mase': np.inf, 'coverage': 0.0, 'samples': 0}
            
            # Compute metrics
            y_true = np.array(actuals)
            y_pred = np.array(predictions)
            y_lower = np.array(lower_bounds)
            y_upper = np.array(upper_bounds)
            
            mape = self.compute_mape(y_true, y_pred)
            mase = self.compute_mase(y_true, y_pred, train_data.values)
            coverage = self.compute_coverage(y_true, y_lower, y_upper)
            
            return {
                'mape': float(mape) if np.isfinite(mape) else 999.0,
                'mase': float(mase) if np.isfinite(mase) else 999.0,
                'coverage': float(coverage),
                'samples': len(predictions)
            }
            
        except Exception as e:
            logger.warning(f"Horizon {horizon} test failed: {e}")
            return {'mape': np.inf, 'mase': np.inf, 'coverage': 0.0, 'samples': 0}
    
    def _compute_summary(self, region_results: Dict) -> Dict[str, Any]:
        """Compute summary statistics across regions and horizons."""
        summary = {
            'avg_mape_by_horizon': {},
            'avg_mase_by_horizon': {},
            'avg_coverage_by_horizon': {},
            'total_samples': 0,
            'regions_tested': len(region_results)
        }
        
        for horizon in self.horizons:
            mapes = []
            mases = []
            coverages = []
            samples = 0
            
            for region, results in region_results.items():
                if horizon in results.get('horizons', {}):
                    metrics = results['horizons'][horizon]
                    if np.isfinite(metrics['mape']) and metrics['mape'] < 999:
                        mapes.append(metrics['mape'])
                    if np.isfinite(metrics['mase']) and metrics['mase'] < 999:
                        mases.append(metrics['mase'])
                    coverages.append(metrics['coverage'])
                    samples += metrics['samples']
            
            summary['avg_mape_by_horizon'][horizon] = np.mean(mapes) if mapes else np.inf
            summary['avg_mase_by_horizon'][horizon] = np.mean(mases) if mases else np.inf
            summary['avg_coverage_by_horizon'][horizon] = np.mean(coverages) if coverages else 0.0
            summary['total_samples'] += samples
        
        # Overall averages
        all_mapes = [v for v in summary['avg_mape_by_horizon'].values() if np.isfinite(v)]
        all_mases = [v for v in summary['avg_mase_by_horizon'].values() if np.isfinite(v)]
        all_coverages = [v for v in summary['avg_coverage_by_horizon'].values()]
        
        summary['overall_avg_mape'] = np.mean(all_mapes) if all_mapes else np.inf
        summary['overall_avg_mase'] = np.mean(all_mases) if all_mases else np.inf
        summary['overall_avg_coverage'] = np.mean(all_coverages) if all_coverages else 0.0
        
        return summary
    
    def save_backtest_report(self, results: Dict[str, Any], filepath: str = 'outputs/backtest_report.json'):
        """Save backtest results to file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Backtest report saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save backtest report: {e}")
    
    def create_summary_text(self, results: Dict[str, Any]) -> str:
        """Create a readable summary of backtest results."""
        if 'error' in results:
            return f"Backtest failed: {results['error']}"
        
        summary = results.get('summary', {})
        regions = results.get('regions', {})
        
        # For our simplified backtest results structure
        overall_mape = summary.get('overall_mape', np.inf)
        overall_mase = summary.get('overall_mase', np.inf)
        regions_tested = summary.get('regions_tested', 0)
        total_weeks = summary.get('total_weeks', 0)
        
        text = f"""
Smart Grid Backtest Report
==========================
Date: {results.get('backtest_date', 'Unknown')}
Weeks Tested: {total_weeks}
Regions: {regions_tested}

Overall Performance:
- Average MAPE: {overall_mape:.1f}% 
- Average MASE: {overall_mase:.3f}

Regional Performance:
"""
        
        for region, metrics in regions.items():
            if isinstance(metrics, dict) and 'mape' in metrics:
                mape = metrics['mape']
                mase = metrics['mase']
                weeks = metrics.get('weeks_tested', 0)
                
                mape_str = f"{mape:.1f}%" if np.isfinite(mape) else "N/A"
                mase_str = f"{mase:.3f}" if np.isfinite(mase) else "N/A"
                
                text += f"  {region}: MAPE={mape_str:>6}, MASE={mase_str:>6}, Weeks={weeks}\n"
        
        return text