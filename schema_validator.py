"""
Smart Grid JSON Schema and Validation
=====================================

Handles schema validation and versioning for outputs.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Current schema version
SCHEMA_VERSION = "1.2.0"

# JSON Schema for latest_predictions.json
PREDICTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "schema_version": {"type": "string"},
        "timestamp": {"type": "string"},
        "current_load": {
            "type": "object",
            "patternProperties": {
                "^region_[0-9]+$": {"type": "number"}
            }
        },
        "predictions": {
            "type": "object",
            "patternProperties": {
                "^(15min|1h|6h|12h|24h|48h)$": {
                    "type": "object",
                    "patternProperties": {
                        "^region_[0-9]+$": {"type": "number"}
                    }
                }
            }
        },
        "intervals": {
            "type": "object",
            "patternProperties": {
                "^(15min|1h|6h|12h|24h|48h)$": {
                    "type": "object",
                    "patternProperties": {
                        "^region_[0-9]+$": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        }
                    }
                }
            }
        },
        "forecast_paths": {
            "type": "object",
            "patternProperties": {
                "^region_[0-9]+$": {
                    "type": "object",
                    "patternProperties": {
                        "^(15min|1h|6h|12h|24h|48h)$": {
                            "type": "array",
                            "items": {"type": "number"}
                        }
                    }
                }
            }
        },
        "interval_paths": {
            "type": "object",
            "patternProperties": {
                "^region_[0-9]+$": {
                    "type": "object",
                    "patternProperties": {
                        "^(15min|1h|6h|12h|24h|48h)$": {
                            "type": "object",
                            "properties": {
                                "lower": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                },
                                "upper": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                }
                            },
                            "required": ["lower", "upper"]
                        }
                    }
                }
            }
        },
        "feature_importances": {
            "type": "object",
            "patternProperties": {
                "^region_[0-9]+$": {
                    "type": "object",
                    "patternProperties": {
                        "^(15min|1h|6h|12h|24h|48h)$": {
                            "type": "object",
                            "patternProperties": {
                                "^.+$": {"type": "number"}
                            }
                        }
                    }
                }
            }
        },
        "anomalies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "region": {"type": "string"},
                    "severity": {"type": "string"},
                    "z_score": {"type": "number"},
                    "timestamp": {"type": "string"},
                    "type": {"type": "string"},
                    "horizon": {"type": "string"},
                    "predicted_load": {"type": "number"},
                    "current_share": {"type": "number"},
                    "expected_share_range": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2
                    },
                    "note": {"type": "string"}
                },
                "required": ["region", "severity", "timestamp"]
            }
        },
        "weather": {
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "humidity": {"type": "number"},
                "solar_irradiance": {"type": "number"}
            }
        }
    },
    "required": [
        "schema_version", "timestamp", "current_load", 
        "predictions", "intervals", "anomalies"
    ]
}


class SchemaValidator:
    """Validates JSON outputs against schema."""
    
    def __init__(self):
        """Initialize validator."""
        try:
            import jsonschema
            self.validator = jsonschema.Draft7Validator(PREDICTIONS_SCHEMA)
            self.has_jsonschema = True
        except ImportError:
            logger.warning("jsonschema not available - skipping validation")
            self.has_jsonschema = False
    
    def validate_predictions(self, data: Dict[str, Any]) -> bool:
        """Validate predictions JSON against schema."""
        if not self.has_jsonschema:
            return True
        
        try:
            self.validator.validate(data)
            return True
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            return False
    
    def add_schema_metadata(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Add schema version and timestamp to data."""
        data = data.copy()
        data["schema_version"] = SCHEMA_VERSION
        data["timestamp"] = datetime.now().isoformat()
        return data
    
    def migrate_schema(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate old schema versions to current."""
        schema_version = data.get("schema_version", "1.0.0")
        
        if schema_version == "1.0.0":
            # Add missing fields for v1.1.0
            if "forecast_paths" not in data:
                data["forecast_paths"] = {}
            if "interval_paths" not in data:
                data["interval_paths"] = {}
            schema_version = "1.1.0"
        
        if schema_version == "1.1.0":
            # Add weather snapshot for v1.2.0
            if "weather" not in data:
                data["weather"] = {
                    "temperature": 20.0,
                    "humidity": 0.5,
                    "solar_irradiance": 500.0
                }
            schema_version = "1.2.0"
        
        data["schema_version"] = schema_version
        return data


def safe_json_write(filepath: Path, data: Dict[str, Any], validator: Optional[SchemaValidator] = None) -> bool:
    """Safely write JSON with atomic operation and validation."""
    try:
        # Validate if validator provided
        if validator:
            data = validator.add_schema_metadata(data)
            if not validator.validate_predictions(data):
                logger.error("Data failed schema validation")
                return False
        
        # Write to temporary file first
        temp_path = filepath.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Atomic rename
        temp_path.replace(filepath)
        return True
        
    except Exception as e:
        logger.error(f"Failed to write JSON: {e}")
        return False


def safe_json_read(filepath: Path, validator: Optional[SchemaValidator] = None) -> Optional[Dict[str, Any]]:
    """Safely read JSON with validation and migration."""
    try:
        if not filepath.exists():
            return None
            
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if validator:
            # Migrate if needed
            data = validator.migrate_schema(data)
            
            # Validate
            if not validator.validate_predictions(data):
                logger.warning("Loaded data failed validation - using anyway")
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to read JSON: {e}")
        return None