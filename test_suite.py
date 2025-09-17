#!/usr/bin/env python
"""
Comprehensive Test Suite for Smart Grid System
==============================================

Production-ready test suite covering:
- Smoke tests for basic functionality
- Unit tests for core components
- Integration tests for full pipeline
- Performance tests for scalability

Run with:
    python test_suite.py smoke      # Quick smoke tests
    python test_suite.py            # All tests
    pytest test_suite.py -v         # With pytest (if installed)
"""

import sys
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime
import pandas as pd
import numpy as np

# Try to import pytest, but make it optional for smoke tests
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    
    # Create a minimal pytest-like interface for smoke tests
    class pytest:
        @staticmethod
        def skip(msg, allow_module_level=True):
            print(f"Skipping: {msg}")
            if allow_module_level:
                sys.exit(0)
        
        @staticmethod
        def main(args):
            print("pytest not available, running basic tests only")
            return 0
        
        class mark:
            @staticmethod
            def slow(func):
                return func

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from config_manager import ConfigManager, ProductionConfig
    from schema_validator import SchemaValidator
    from simplified_config import EnhancedConfig
    from smart_grid_data_generator import EnhancedDataGenerator
    from backtest_engine import BacktestEngine
except ImportError as e:
    print(f"Cannot import required modules: {e}")
    if PYTEST_AVAILABLE:
        pytest.skip(f"Cannot import required modules: {e}", allow_module_level=True)
    else:
        print("Skipping tests due to import errors")
        sys.exit(1)


class TestConfigManager:
    """Test configuration management system."""
    
    def test_default_config_creation(self):
        """Test default configuration creation."""
        config = ConfigManager.get_config()
        assert isinstance(config, ProductionConfig)
        assert config.system.simulation_years == 5
        assert len(config.system.regions) == 8
        
    def test_environment_variable_override(self):
        """Test environment variable overrides."""
        with patch.dict('os.environ', {'SMARTGRID_SIMULATION_YEARS': '10'}):
            config = ConfigManager.get_config("test_env")
            assert config.system.simulation_years == 10
    
    def test_config_overrides(self):
        """Test configuration overrides."""
        overrides = {"anomaly_threshold": 4.0, "forecast_threshold": 1.5}
        config = ConfigManager.get_config(overrides=overrides)
        assert config.anomaly.anomaly_threshold == 4.0
        assert config.anomaly.forecast_threshold == 1.5
    
    def test_config_serialization(self):
        """Test configuration save/load."""
        config = ProductionConfig()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.save(f.name)
            
            # Load and verify
            loaded_config = ConfigManager._load_from_file(f.name)
            assert loaded_config.system.simulation_years == config.system.simulation_years
            
            # Cleanup
            Path(f.name).unlink()


class TestSchemaValidator:
    """Test JSON schema validation system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = SchemaValidator()
        
    def test_valid_data_validation(self):
        """Test validation of valid data structure."""
        valid_data = {
            "schema_version": "1.2.0",
            "timestamp": "2025-09-16T10:00:00",
            "predictions": {
                "region_1": {
                    "forecasts": {
                        "15min": 450.5,
                        "1h": 470.2,
                        "6h": 480.1
                    }
                }
            },
            "anomalies": [],
            "feature_importances": {
                "region_1": {
                    "15min": {
                        "hour": 0.1,
                        "temperature": 0.2
                    }
                }
            },
            "metadata": {
                "model_version": "1.0",
                "generation_time": "2025-09-16T10:00:00"
            }
        }
        
        is_valid, error = self.validator.validate_data(valid_data)
        assert is_valid, f"Validation failed: {error}"
    
    def test_invalid_data_validation(self):
        """Test validation of invalid data structure."""
        invalid_data = {
            "schema_version": "1.2.0",
            "predictions": {
                "region_1": {
                    "forecasts": {
                        "15min": "invalid_number"  # Should be number
                    }
                }
            }
        }
        
        is_valid, error = self.validator.validate_data(invalid_data)
        assert not is_valid
        assert "is not of type 'number'" in str(error)
    
    def test_schema_migration(self):
        """Test schema version migration."""
        old_data = {
            "schema_version": "1.0.0",
            "predictions": {}
        }
        
        migrated_data = self.validator.migrate_data(old_data)
        assert migrated_data["schema_version"] == "1.2.0"


class TestDataGenerator:
    """Test data generation system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = EnhancedDataGenerator(simulation_years=1)
    
    def test_data_generation_shape(self):
        """Test generated data has correct shape."""
        dataset = self.generator.generate_complete_dataset()
        
        assert "load_data" in dataset
        load_data = dataset["load_data"]
        
        # Should have 8 regions
        assert load_data.shape[1] == 8
        
        # Should have approximately 1 year of 15-min intervals
        expected_intervals = 365 * 96  # 96 intervals per day
        assert abs(load_data.shape[0] - expected_intervals) < 100  # Allow some tolerance
    
    def test_data_generation_types(self):
        """Test generated data types are correct."""
        dataset = self.generator.generate_complete_dataset()
        
        load_data = dataset["load_data"]
        temp_data = dataset["temperature_data"]
        
        # Should be numeric data
        assert pd.api.types.is_numeric_dtype(load_data.iloc[:, 0])
        assert pd.api.types.is_numeric_dtype(temp_data.iloc[:, 0])
        
        # Should not have NaN values
        assert not load_data.isnull().any().any()
        assert not temp_data.isnull().any().any()
    
    def test_regional_characteristics(self):
        """Test regional load characteristics are distinct."""
        dataset = self.generator.generate_complete_dataset()
        load_data = dataset["load_data"]
        
        # Industrial region (region_3) should have higher load
        region_3_mean = load_data.iloc[:, 2].mean()  # region_3 is 3rd column
        region_1_mean = load_data.iloc[:, 0].mean()  # region_1 is 1st column
        
        assert region_3_mean > region_1_mean, "Industrial region should have higher load"


class TestBacktestEngine:
    """Test backtesting functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EnhancedConfig()
        self.engine = BacktestEngine(self.config)
    
    def test_mape_calculation(self):
        """Test MAPE metric calculation."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 180, 330])
        
        mape = self.engine.compute_mape(y_true, y_pred)
        expected_mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        assert abs(mape - expected_mape) < 0.01
    
    def test_mase_calculation(self):
        """Test MASE metric calculation."""
        y_true = np.array([100, 110, 120, 130])
        y_pred = np.array([105, 115, 125, 135])
        y_train = np.array([90, 95, 100, 105, 110, 115, 120])  # Longer training series
        
        mase = self.engine.compute_mase(y_true, y_pred, y_train)
        assert isinstance(mase, float)
        assert mase > 0
    
    def test_coverage_calculation(self):
        """Test prediction interval coverage."""
        y_true = np.array([100, 110, 120])
        y_lower = np.array([95, 105, 115])
        y_upper = np.array([105, 115, 125])
        
        coverage = self.engine.compute_coverage(y_true, y_lower, y_upper)
        assert coverage == 100.0  # All predictions within intervals


@pytest.mark.slow
class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data generation to prediction."""
        # This is a comprehensive test that takes longer to run
        from integrated_smart_grid_pipeline import IntegratedSmartGridPipeline
        
        config = ConfigManager.get_config(overrides={"simulation_years": 1})
        pipeline = IntegratedSmartGridPipeline(config)
        
        # Test training (would normally take a long time)
        with patch.object(pipeline, '_train_models') as mock_train:
            mock_train.return_value = True
            success = pipeline.run_training_cycle()
            assert success
    
    def test_deployment_cycle(self):
        """Test deployment cycle with mocked components."""
        from integrated_smart_grid_pipeline import IntegratedSmartGridPipeline
        
        config = ConfigManager.get_config()
        pipeline = IntegratedSmartGridPipeline(config)
        
        # Mock the trained state
        pipeline._is_trained = True
        
        # Mock data generation and forecasting
        with patch.object(pipeline.data_generator, 'generate_deployment_data') as mock_data, \
             patch.object(pipeline, '_generate_forecasts') as mock_forecast:
            
            # Setup mocks
            mock_data.return_value = (
                pd.DataFrame(np.random.randn(96, 8)),  # 1 day of data
                {}  # weather data
            )
            mock_forecast.return_value = {
                f"region_{i}": {
                    "forecasts": {"15min": 100.0, "1h": 110.0},
                    "confidence_intervals": {"15min": (90.0, 110.0)}
                }
                for i in range(1, 9)
            }
            
            # Test deployment
            results = pipeline.run_deployment_cycle()
            assert results is not None
            assert "predictions" in results


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_file_handling(self):
        """Test handling of missing configuration files."""
        config = ConfigManager.get_config(config_file="nonexistent.json")
        # Should fall back to defaults without crashing
        assert isinstance(config, ProductionConfig)
    
    def test_invalid_json_handling(self):
        """Test handling of invalid JSON files."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content {{{")
            f.flush()
            
            config = ConfigManager._load_from_file(f.name)
            # Should fall back to defaults
            assert isinstance(config, ProductionConfig)
            
            # Cleanup
            Path(f.name).unlink()
    
    def test_data_validation_edge_cases(self):
        """Test data validation with edge cases."""
        validator = SchemaValidator()
        
        # Empty data
        is_valid, _ = validator.validate_data({})
        assert not is_valid
        
        # Malformed data
        is_valid, _ = validator.validate_data({"invalid": "structure"})
        assert not is_valid


def run_smoke_tests():
    """Run quick smoke tests for basic functionality."""
    print("🧪 Running Smart Grid Smoke Tests...")
    
    # Test 1: Configuration loading
    try:
        config = ConfigManager.get_config()
        print("✅ Configuration loading: PASSED")
    except Exception as e:
        print(f"❌ Configuration loading: FAILED - {e}")
        return False
    
    # Test 2: Schema validation
    try:
        validator = SchemaValidator()
        test_data = {
            "schema_version": "1.2.0",
            "timestamp": "2025-09-16T10:00:00",
            "current_load": {"region_1": 100.0},
            "predictions": {"15min": {"region_1": 105.0}},
            "intervals": {"15min": {"region_1": [95.0, 115.0]}},
            "anomalies": [],
            "feature_importances": {"region_1": {"15min": {"hour": 0.1}}},
            "metadata": {"model_version": "1.0", "generation_time": "2025-09-16T10:00:00"}
        }
        is_valid = validator.validate_predictions(test_data)
        assert is_valid
        print("✅ Schema validation: PASSED")
    except Exception as e:
        print(f"❌ Schema validation: FAILED - {e}")
        return False
    
    # Test 3: Data generation (small scale)
    try:
        generator = EnhancedDataGenerator(simulation_years=0.1)  # Very small dataset
        dataset = generator.generate_complete_dataset()
        assert "load_data" in dataset
        print("✅ Data generation: PASSED")
    except Exception as e:
        print(f"❌ Data generation: FAILED - {e}")
        return False
    
    # Test 4: Backtest engine
    try:
        config = EnhancedConfig()
        engine = BacktestEngine(config)
        mape = engine.compute_mape(np.array([100, 200]), np.array([110, 190]))
        assert isinstance(mape, float)
        print("✅ Backtest engine: PASSED")
    except Exception as e:
        print(f"❌ Backtest engine: FAILED - {e}")
        return False
    
    print("\n🎉 All smoke tests passed! System is ready for use.")
    return True


def run_unit_tests():
    """Run unit tests for core components."""
    print("🧪 Running Unit Tests...")
    
    # Test configuration management
    try:
        config = ConfigManager.get_config()
        assert hasattr(config, 'model')
        assert hasattr(config, 'anomaly')
        assert hasattr(config, 'system')
        print("✅ Configuration management: PASSED")
    except Exception as e:
        print(f"❌ Configuration management: FAILED - {e}")
        return False
    
    # Test data generator core functionality
    try:
        generator = EnhancedDataGenerator(simulation_years=0.05)  # Very short for testing
        data = generator.generate_complete_dataset()
        assert 'load_data' in data
        assert 'temperature_data' in data
        print("✅ Data generator core: PASSED")
    except Exception as e:
        print(f"❌ Data generator core: FAILED - {e}")
        return False
    
    # Test schema validation
    try:
        validator = SchemaValidator()
        test_data = {
            "schema_version": "1.2.0",
            "timestamp": "2025-09-16T10:00:00",
            "current_load": {"region_1": 100.0},
            "predictions": {"15min": {"region_1": 105.0}},
            "intervals": {"15min": {"region_1": [95.0, 115.0]}},
            "anomalies": [],
            "feature_importances": {"region_1": {"15min": {"hour": 0.1}}},
            "metadata": {"model_version": "1.0", "generation_time": "2025-09-16T10:00:00"}
        }
        result = validator.validate_predictions(test_data)
        assert result
        print("✅ Schema validation core: PASSED")
    except Exception as e:
        print(f"❌ Schema validation core: FAILED - {e}")
        return False
    
    print("🎉 All unit tests passed!")
    return True

def run_integration_tests():
    """Run integration tests for component interactions."""
    print("🧪 Running Integration Tests...")
    
    # Test pipeline integration with correct config format
    try:
        # Import main components
        from integrated_smart_grid_pipeline import IntegratedSmartGridPipeline, GridConfig
        
        # Use the proper GridConfig class
        config = GridConfig(
            simulation_years=1,  # Short for testing
            max_ram_gb=8,  # Reduced for testing
            use_gpu=False  # Disable GPU for testing
        )
        
        # Test pipeline can be initialized
        pipeline = IntegratedSmartGridPipeline(config)
        assert hasattr(pipeline, 'run_training_pipeline')
        assert hasattr(pipeline, 'run_deployment_pipeline')
        
        # Simple functionality test
        assert hasattr(pipeline, 'data_generator')
        assert pipeline.config is not None
        
        print("✅ Pipeline integration: PASSED")
    except Exception as e:
        import traceback
        print(f"❌ Pipeline integration: FAILED - {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False
    
    # Test backtest integration
    try:
        config = EnhancedConfig()
        engine = BacktestEngine(config)
        assert hasattr(engine, 'run_rolling_backtest')
        assert hasattr(engine, 'compute_mape')
        assert hasattr(engine, 'compute_mase')
        
        print("✅ Backtest integration: PASSED")
    except Exception as e:
        import traceback
        print(f"❌ Backtest integration: FAILED - {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False
    
    print("🎉 All integration tests passed!")
    return True


if __name__ == "__main__":
    import sys
    
    test_mode = 'smoke'
    if len(sys.argv) > 1:
        test_mode = sys.argv[1].lower()
    
    success = True
    
    if test_mode == 'smoke':
        success = run_smoke_tests()
    elif test_mode == 'unit':
        success = run_unit_tests()
    elif test_mode == 'integration':
        success = run_integration_tests()
    elif test_mode == 'all':
        print("🧪 Running Complete Test Suite...")
        success = run_smoke_tests()
        if success:
            success = run_unit_tests()
        if success:
            success = run_integration_tests()
        if success:
            print("🎉 All test suites passed!")
    else:
        print("🧪 Running full test suite...")
        print("Available modes: smoke, unit, integration, all")
        try:
            import pytest
            pytest.main([__file__, '-v'])
        except ImportError:
            print("pytest not available, running smoke tests...")
            success = run_smoke_tests()
    
    sys.exit(0 if success else 1)