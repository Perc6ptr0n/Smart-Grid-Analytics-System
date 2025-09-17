# 🚀 How to Run the Smart Grid Analytics System

This guide provides step-by-step instructions to get the Smart Grid Analytics System running on your machine.

## 📋 Quick Prerequisites Check

Before starting, ensure you have:
- ✅ Python 3.8+ installed (`python --version`)
- ✅ At least 8GB RAM (16GB recommended)
- ✅ 2GB+ free disk space
- ✅ Internet connection for package installation
- 🎯 NVIDIA GPU (optional, system auto-detects and falls back to CPU)

## 🛠️ Step 1: Environment Setup

### Clone the Repository
```bash
git clone https://github.com/Perc6ptr0n/Smart-Grid-Analytics-System.git
cd Smart-Grid-Analytics-System
```

### Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes as it includes ML frameworks like PyTorch, XGBoost, etc.

## 🪟 Windows Users

### Option 1: Use Windows Batch Script
```cmd
REM Run the automated Windows deployment
deploy.bat --env development

REM Or for production
deploy.bat --env production --skip-tests
```

### Option 2: Manual Windows Setup
```cmd
REM Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

REM Install dependencies
pip install -r requirements.txt

REM Run the system
python main.py full
```

## 🚀 Step 2: Choose Your Running Mode

### Option A: Complete Setup (Recommended for First Time)
```bash
python main.py full
```
**What this does:**
1. 🧠 Trains ML models on 5 years of simulated data (~20-30 minutes)
2. 🔄 Runs initial deployment pipeline  
3. 📊 Launches interactive web dashboard at http://localhost:8501
4. ⚡ Starts continuous monitoring in background

**Best for:** New users who want to see everything working

---

### Option B: Step-by-Step Execution

#### 1. Train Models First (Required)
```bash
python main.py train
```
- **Duration:** ~20-30 minutes
- **Output:** Creates `models/` directory with trained ML models
- **GPU:** Will use GPU if available, falls back to CPU

#### 2. Run Single Deployment
```bash
python main.py deploy
```
- **Duration:** ~2-3 minutes
- **Output:** Creates `outputs/latest_predictions.json` with forecasts

#### 3. Launch Dashboard
```bash
python main.py dashboard
```
- **Access:** http://localhost:8501
- **Features:** Real-time visualizations, threshold controls, system metrics

#### 4. Run Continuous Monitoring (Optional)
```bash
python main.py continuous
```
- **Behavior:** Runs deployment every 15 minutes
- **Use:** For production-like monitoring

---

### Option C: Analysis & Testing

#### Performance Analysis
```bash
python main.py backtest
```
- **Purpose:** Evaluates model performance over historical data
- **Output:** Creates `outputs/backtest_report.json` with metrics

#### Quick System Test
```bash
python test_suite.py smoke
```
- **Purpose:** Verifies all components work correctly
- **Duration:** ~30 seconds

## 🎯 Step 3: Understanding the Output

### Generated Files & Directories
After running, you'll see these created:
```
Smart-Grid-Analytics-System/
├── data/                    # Generated simulation data
├── models/                  # Trained ML models
├── logs/                    # System logs
└── outputs/                 # Predictions & reports
    ├── latest_predictions.json
    └── backtest_report.json
```

### Dashboard Features
When you access http://localhost:8501, you'll see:
- 📈 **Real-time load predictions** for 8 grid regions
- 🚨 **Anomaly detection alerts** with configurable thresholds
- 📊 **Performance metrics** (MAPE, MASE, accuracy)
- ⚙️ **Threshold controls** in the sidebar
- 🗺️ **Regional load distribution** maps

## 🔧 Step 4: Customization Options

### Threshold Tuning
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

### Different Time Horizons
The system predicts at multiple horizons:
- **15 minutes:** Immediate operational decisions
- **1 hour:** Short-term grid balancing
- **6 hours:** Medium-term planning
- **24 hours:** Day-ahead scheduling
- **48 hours:** Extended planning horizon

## 🐛 Troubleshooting

### Common Issues & Solutions

#### "ModuleNotFoundError"
```bash
# Make sure virtual environment is activated and requirements installed
pip install -r requirements.txt
```

#### "CUDA not available"
```bash
# Check GPU detection (optional)
python -c "import torch; print(torch.cuda.is_available())"
# System automatically falls back to CPU - no action needed
```

#### "Models not found"
```bash
# Ensure training completed successfully
python main.py train
# Check that models/ directory was created
```

#### "Port 8501 already in use"
```bash
# Kill existing streamlit processes
pkill -f streamlit
# Or use a different port (modify dashboard code)
```

#### Training Takes Too Long
- **Expected:** 20-30 minutes on modern CPU
- **With GPU:** 5-10 minutes
- **Check:** Monitor logs for progress updates

### Performance Optimization Tips

1. **For faster training:**
   - Use a machine with GPU support
   - Ensure adequate RAM (16GB+ recommended)
   - Close unnecessary applications

2. **For production deployment:**
   - Use the continuous mode: `python main.py continuous`
   - Monitor system resources with dashboard
   - Set up logging rotation for long-running instances

## 🔄 Different Usage Scenarios

### Scenario 1: Development & Testing
```bash
python test_suite.py smoke          # Quick verification
python main.py train                # Train models
python main.py backtest             # Analyze performance
python main.py dashboard            # Visualize results
```

### Scenario 2: Production Monitoring
```bash
python main.py full                 # Initial setup
# Let it run - dashboard shows real-time monitoring
# System automatically runs deployment every 15 minutes
```

### Scenario 3: One-time Analysis
```bash
python main.py train                # Train models
python main.py deploy               # Generate predictions
# Check outputs/latest_predictions.json for results
```

### Scenario 4: Threshold Experimentation
```bash
# Test different sensitivity levels
python main.py deploy --forecast-threshold 1.0  # Very sensitive
python main.py deploy --forecast-threshold 2.0  # Balanced
python main.py deploy --forecast-threshold 3.0  # Conservative
```

## 📊 Understanding the Results

### Prediction Output Format
The `outputs/latest_predictions.json` contains:
```json
{
  "timestamp": "2025-01-01T10:00:00",
  "current_load": {"region_1": 450.2, ...},
  "predictions": {
    "15min": {"region_1": 465.1, ...},
    "1hour": {"region_1": 480.3, ...},
    "6hour": {"region_1": 520.7, ...}
  },
  "confidence_intervals": {...},
  "anomalies": [...],
  "recommendations": [...]
}
```

### Performance Metrics
- **MAPE (Mean Absolute Percentage Error):** < 5% is excellent
- **MASE (Mean Absolute Scaled Error):** < 1.0 means better than naive forecast
- **Anomaly Detection Accuracy:** % of correctly identified anomalies

## 🎯 Next Steps

1. **Explore the Dashboard:** Access http://localhost:8501 and experiment with controls
2. **Review Predictions:** Check `outputs/latest_predictions.json` for detailed forecasts
3. **Analyze Performance:** Run `python main.py backtest` to see model accuracy
4. **Customize Thresholds:** Use CLI parameters to fine-tune anomaly detection
5. **Monitor Continuously:** Use `python main.py continuous` for ongoing monitoring

## 🆘 Need Help?

- **Check logs:** Look in the `logs/` directory for detailed execution logs
- **Run smoke tests:** `python test_suite.py smoke` to verify system health
- **Review outputs:** Check `outputs/` directory for generated files
- **Monitor resources:** Use system monitor to check CPU/RAM usage during training

## 🐳 Docker Deployment (Alternative)

If you prefer containerized deployment:

### Option 1: Quick Docker Start
```bash
# Build and run dashboard
docker-compose up smart-grid

# Access dashboard at http://localhost:8050
```

### Option 2: Development Mode
```bash
# Run in development mode with live code changes
docker-compose up smart-grid-dev

# Access at http://localhost:8051
```

### Option 3: Production Deployment Script
```bash
# Use the comprehensive deployment script
chmod +x deploy.sh
./deploy.sh --env production
```

### Docker Services Available:
- **smart-grid:** Production dashboard (port 8050)
- **smart-grid-dev:** Development dashboard (port 8051) 
- **smart-grid-pipeline:** Background processing
- **smart-grid-backtest:** Performance analysis

## 🔗 Additional Resources

- **README.md:** Complete feature overview and architecture
- **PROJECT_SUMMARY.md:** Technical implementation details
- **DEPLOYMENT_CHECKLIST.md:** Production deployment guide
- **docker-compose.yml:** Container orchestration setup
- **deploy.sh:** Production deployment script
- **requirements.txt:** Complete dependency list

---

**Happy forecasting! 🔮⚡**