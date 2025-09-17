# 🔌 Smart Grid Load Balancing Optimizer

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Accelerated](https://img.shields.io/badge/GPU-Accelerated-green.svg)](https://developer.nvidia.com/cuda-zone)

A production-ready, GPU-accelerated machine learning system for predicting energy demand peaks and optimizing load distribution across smart grid regions. Features real-time anomaly detection, interactive dashboards, and comprehensive backtesting capabilities.

## 🌟 Key Features

- **🤖 AI-Powered Forecasting**: Multi-horizon predictions (15min to 48h) using hybrid ensemble models
- **⚡ Real-Time Monitoring**: Live anomaly detection with configurable thresholds
- **📊 Interactive Dashboard**: Web-based visualization with real-time updates
- **🔄 Continuous Operation**: Automated deployment pipeline with 15-minute intervals
- **📈 Performance Analytics**: Built-in backtesting with MAPE/MASE metrics
- **🔧 Production Ready**: Schema validation, error handling, and robust configuration
- **🚀 GPU Accelerated**: Optimized for NVIDIA GPUs with automatic fallback to CPU

## 🏗️ Architecture

```
Smart Grid System
├── Data Generation     → Realistic 8-region simulation (5 years)
├── Feature Engineering → 13 time-series features + weather integration
├── ML Pipeline        → CatBoost + LSTM hybrid ensemble
├── Anomaly Detection  → Multi-layer statistical analysis
├── Real-time API      → RESTful endpoints for live predictions
└── Monitoring        → Streamlit dashboard + JSON outputs
```

### Regional Profiles
- **Urban Metro**: High density residential/commercial (400MW base)
- **Suburban Mixed**: Balanced residential load (300MW base)
- **Industrial Heavy**: Manufacturing/data centers (650MW base) 
- **Rural Agricultural**: Farming operations (200MW base)
- **Coastal Renewable**: Solar/wind integration (320MW base)
- **Mountain Hydro**: Hydroelectric region (125MW base)
- **Desert Solar**: Large-scale solar farms (250MW base)
- **Mixed Commercial**: Business districts (260MW base)

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **8GB+ RAM** (16GB recommended)
- **NVIDIA GPU** (optional, auto-detects)
- **2GB+ disk space**

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/smart-grid-optimizer.git
cd smart-grid-optimizer

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### First-Time Setup

```bash
# Complete setup: training + dashboard (recommended)
python main.py full
```

**What this does:**
1. 🧠 Train ML models on 5 years of data (~20-30 minutes)
2. 🔄 Run initial deployment pipeline
3. 📊 Launch interactive dashboard
4. ⚡ Start continuous monitoring

### Individual Commands

```bash
# Training only (first time)
python main.py train

# Single deployment
python main.py deploy

# Continuous monitoring (every 15 min)
python main.py continuous

# Dashboard only
python main.py dashboard

# Performance analysis
python main.py backtest
```

## ⚙️ Configuration & Tuning

### CLI Threshold Overrides

Fine-tune anomaly detection sensitivity:

```bash
# More sensitive forecast detection
python main.py deploy --forecast-threshold 1.5

# Comprehensive threshold tuning
python main.py deploy \
  --anomaly-threshold 3.0 \
  --forecast-threshold 1.8 \
  --peer-threshold 2.2 \
  --share-absolute 0.35 \
  --share-increase 0.20
```

### Dashboard Controls

Access real-time threshold adjustment via the sidebar:
- **Anomaly Threshold**: Base statistical deviation limit
- **Forecast Threshold**: Prediction error sensitivity  
- **Peer Threshold**: Cross-region comparison sensitivity
- **Share Controls**: Regional load distribution limits

## 📊 Performance Metrics

### Backtesting Results

Latest 4-week rolling analysis:

```
Overall Performance:
- Average MAPE: 9.1%
- Average MASE: 1.008
- Regions Tested: 8

Regional Performance:
  region_1 (Urban):      MAPE=8.0%, MASE=1.021
  region_2 (Suburban):   MAPE=7.9%, MASE=1.008  
  region_3 (Industrial): MAPE=5.9%, MASE=1.001
  region_4 (Rural):      MAPE=10.0%, MASE=1.030
```

### Real-Time Monitoring

The system continuously monitors:
- **Load Predictions**: 6 forecast horizons (15min-48h)
- **Anomaly Detection**: 4-layer validation system
- **Feature Importance**: Dynamic model interpretability
- **Schema Validation**: Data integrity checks

## 🔧 Advanced Usage

### Custom Data Integration

```python
from integrated_smart_grid_pipeline import IntegratedSmartGridPipeline

# Initialize with custom config
pipeline = IntegratedSmartGridPipeline(custom_config)

# Run prediction cycle
results = pipeline.run_deployment_cycle()
```

### API Integration

```python
# Access latest predictions
import json
with open('outputs/latest_predictions.json') as f:
    predictions = json.load(f)
    
# Extract specific forecast
forecast_24h = predictions['predictions']['region_1']['forecasts']['24h']
```

### Schema Validation

All outputs follow JSON Schema v1.2.0 specification:
- ✅ Type validation for all fields
- ✅ Timestamp format standardization  
- ✅ Anomaly structure verification
- ✅ Feature importance validation

## 📁 Project Structure

```
smart-grid-optimizer/
├── main.py                              # CLI entry point
├── integrated_smart_grid_pipeline.py    # Core ML pipeline
├── smart_grid_data_generator.py         # Realistic data simulation
├── smart_grid_dashboard.py              # Streamlit dashboard
├── backtest_engine.py                   # Performance analysis
├── schema_validator.py                  # Data validation
├── simplified_config.py                 # Configuration management
├── requirements.txt                     # Python dependencies
├── outputs/                             # Generated predictions & reports
│   ├── latest_predictions.json          # Real-time forecasts
│   └── backtest_report.json            # Performance metrics
├── models/                              # Trained ML models (auto-generated)
├── logs/                                # System logs
└── data/                                # Generated datasets (auto-generated)
```

## 🛠️ Development

### Code Quality

The codebase follows production standards:
- **Type Hints**: Full typing support
- **Error Handling**: Comprehensive exception management
- **Logging**: Structured logging with multiple levels
- **Documentation**: Inline docstrings and comments
- **Testing**: Built-in smoke tests and validation

### Testing

```bash
# Run smoke tests
python smoke_test.py

# Validate all CLI commands
python main.py deploy --forecast-threshold 1.5
python main.py backtest
```

### GPU Requirements

**Supported GPUs:**
- NVIDIA RTX 20/30/40 series
- Tesla V100/A100
- Quadro RTX series

**Memory Requirements:**
- Minimum: 4GB VRAM
- Recommended: 8GB+ VRAM
- Falls back to CPU automatically

## 📈 Use Cases

### Utility Companies
- **Peak Load Prediction**: Avoid brownouts during high demand
- **Resource Planning**: Optimize generation capacity allocation
- **Grid Stability**: Early warning for potential system stress

### Energy Traders
- **Market Forecasting**: Predict price movements from demand patterns
- **Risk Management**: Identify abnormal consumption events
- **Portfolio Optimization**: Balance renewable vs. traditional sources

### Smart City Planning
- **Infrastructure Design**: Size electrical systems appropriately
- **Sustainability Goals**: Optimize renewable energy integration
- **Emergency Response**: Predict and prepare for grid anomalies

## 🔍 Troubleshooting

### Common Issues

**"CUDA not available"**
```bash
# Check GPU detection
python -c "import torch; print(torch.cuda.is_available())"
# System falls back to CPU automatically
```

**"Models not found"**
```bash
# Ensure training completed
python main.py train
# Check models/ directory exists
```

**"Schema validation failed"**
```bash
# Check outputs/latest_predictions.json format
# System falls back to basic JSON write
```

### Performance Optimization

**Slow training (>1 hour):**
- Verify GPU usage in logs
- Reduce `simulation_years` in config
- Check available RAM (16GB+ recommended)

**Dashboard loading issues:**
- Ensure port 8501 is available
- Check firewall settings
- Try different browser

## 📝 API Reference

### Core Classes

#### `IntegratedSmartGridPipeline`
Main pipeline orchestrator with full ML workflow.

#### `SmartGridController` 
CLI interface with threshold override support.

#### `BacktestEngine`
Rolling-origin performance analysis with MAPE/MASE metrics.

#### `SchemaValidator`
JSON Schema validation with versioning support.

### Key Methods

```python
# Training
pipeline.run_training_cycle()

# Deployment  
results = pipeline.run_deployment_cycle()

# Backtesting
engine = BacktestEngine()
report = engine.run_rolling_backtest(...)
```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/smart-grid-optimizer.git

# Install development dependencies
pip install -r requirements.txt

# Run tests
python smoke_test.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **CatBoost**: Gradient boosting framework
- **PyTorch**: Deep learning platform
- **Streamlit**: Dashboard framework
- **NumPy/Pandas**: Data processing
- **Scikit-learn**: ML utilities

## 📞 Support

- **Documentation**: See inline code comments and docstrings
- **Issues**: [GitHub Issues](https://github.com/your-username/smart-grid-optimizer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/smart-grid-optimizer/discussions)

---

**Made with ⚡ for the future of smart energy**