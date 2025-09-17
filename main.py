#!/usr/bin/env python
"""
Main Entry Point for Smart Grid System
=======================================

Simple command-line interface to run the Smart Grid Load Balancing Optimizer.

Usage:
    python main.py train        # Train models (one-time)
    python main.py deploy       # Run deployment pipeline (once)
    python main.py continuous   # Run continuous deployment (every 15 min)
    python main.py dashboard    # Launch dashboard
    python main.py full         # Train then launch dashboard

Author: Laios Ioannis 
Date: September 2025
"""

import sys
import time
import subprocess
import threading
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
import schedule
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartGridController:
    """Main controller for the Smart Grid system."""
    
    def __init__(self, threshold_overrides: Optional[Dict[str, float]] = None) -> None:
        """Initialize the controller."""
        self.project_root: Path = Path.cwd()
        self.is_running: bool = False
        self.threshold_overrides: Dict[str, float] = threshold_overrides or {}
        
    def check_dependencies(self) -> bool:
        """Check if all required packages are installed."""
        required: List[str] = [
            'pandas', 'numpy', 'sklearn',
            'lightgbm', 'xgboost', 'catboost',
            'streamlit', 'plotly', 'joblib',
            'schedule', 'psutil'
        ]
        
        missing = []
        for package in required:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            logger.error(f"Missing packages: {missing}")
            logger.info("Install with: pip install " + " ".join(missing))
            return False
        
        logger.info("✅ All dependencies satisfied")
        return True
    
    def train_models(self):
        """Run the training pipeline."""
        logger.info("="*60)
        logger.info("STARTING TRAINING PIPELINE")
        logger.info("="*60)
        
        if not self.check_dependencies():
            return False
        
        try:
            from integrated_smart_grid_pipeline import IntegratedSmartGridPipeline, GridConfig
            
            # Apply threshold overrides if provided
            config = GridConfig()
            if self.threshold_overrides:
                for key, value in self.threshold_overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        logger.info(f"Override: {key} = {value}")
            
            pipeline = IntegratedSmartGridPipeline(config)
            
            # Check if already trained
            if pipeline.is_trained:
                response = input("\n⚠️ Models already trained. Retrain? (y/n): ")
                if response.lower() != 'y':
                    logger.info("Training skipped")
                    return True
            
            # Run training
            logger.info("\n🚀 Training models...")
            logger.info("⏱️ This will take approximately 20-30 minutes...")
            
            success = pipeline.run_training_pipeline()
            
            if success:
                logger.info("🎉 Training complete!")
                return True
            else:
                logger.error("Training failed!")
                return False
                
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
    
    def run_deployment(self):
        """Run a single deployment cycle."""
        logger.info("Running deployment pipeline...")
        
        try:
            from integrated_smart_grid_pipeline import IntegratedSmartGridPipeline, GridConfig
            
            # Apply threshold overrides if provided
            config = GridConfig()
            if self.threshold_overrides:
                for key, value in self.threshold_overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                        logger.info(f"Override: {key} = {value}")
            
            pipeline = IntegratedSmartGridPipeline(config)
            
            if not pipeline.is_trained:
                logger.error("Models must be trained first!")
                logger.info("Run: python main.py train")
                return False
            
            output = pipeline.run_deployment_pipeline()
            
            if output:
                logger.info(f"✅ Deployment complete")
                logger.info(f"   Regions: {len(output.get('predictions', {}))}")
                logger.info(f"   Anomalies: {len(output.get('anomalies', []))}")
                return True
            else:
                logger.error("Deployment failed!")
                return False
                
        except Exception as e:
            logger.error(f"Deployment error: {e}")
            return False
    
    def run_continuous(self):
        """Run continuous deployment every 15 minutes."""
        logger.info("="*60)
        logger.info("STARTING CONTINUOUS DEPLOYMENT")
        logger.info("="*60)
        
        def job():
            logger.info(f"Running scheduled deployment at {datetime.now()}")
            self.run_deployment()
        
        # Schedule job every 15 minutes
        schedule.every(15).minutes.do(job)
        
        # Run first deployment immediately
        job()
        
        logger.info("⏰ Scheduled deployment every 15 minutes")
        logger.info("Press Ctrl+C to stop")
        
        self.is_running = True
        
        try:
            while self.is_running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("\n👋 Stopping continuous deployment")
            self.is_running = False
    
    def launch_dashboard(self):
        """Launch the Streamlit dashboard."""
        logger.info("="*60)
        logger.info("LAUNCHING DASHBOARD")
        logger.info("="*60)
        
        # Check if models are trained
        try:
            from integrated_smart_grid_pipeline import IntegratedSmartGridPipeline
            pipeline = IntegratedSmartGridPipeline()
            
            if not pipeline.is_trained:
                logger.warning("⚠️ Models not trained - Dashboard will run in demo mode")
                response = input("Continue? (y/n): ")
                if response.lower() != 'y':
                    return
        except:
            logger.warning("Pipeline not available - Dashboard will run in demo mode")
        
        # Launch dashboard
        dashboard_file = self.project_root / 'smart_grid_dashboard.py'
        
        if not dashboard_file.exists():
            logger.error(f"Dashboard file not found: {dashboard_file}")
            return
        
        logger.info("🚀 Starting dashboard...")
        logger.info("🌐 Dashboard URL: http://localhost:8501")
        logger.info("Press Ctrl+C to stop")
        
        try:
            subprocess.run([
                sys.executable, '-m', 'streamlit', 'run',
                str(dashboard_file),
                '--server.port', '8501',
                '--server.address', 'localhost'
            ])
        except KeyboardInterrupt:
            logger.info("\n👋 Dashboard stopped")
    
    def run_backtest(self, weeks_back: int = 4):
        """Run backtesting analysis."""
        logger.info("="*60)
        logger.info("STARTING BACKTEST ANALYSIS")
        logger.info("="*60)
        
        try:
            from integrated_smart_grid_pipeline import IntegratedSmartGridPipeline, GridConfig
            from backtest_engine import BacktestEngine
            
            # Apply threshold overrides if provided
            config = GridConfig()
            if self.threshold_overrides:
                for key, value in self.threshold_overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            pipeline = IntegratedSmartGridPipeline(config)
            
            if not pipeline.is_trained:
                logger.error("Models must be trained first!")
                logger.info("Run: python main.py train")
                return False
            
            # Run backtest
            engine = BacktestEngine(config)
            results = engine.run_rolling_backtest(
                pipeline.data_generator,
                pipeline.feature_engineer, 
                pipeline.forecaster,
                weeks_back=weeks_back
            )
            
            # Save results
            engine.save_backtest_report(results)
            
            # Print summary
            summary_text = engine.create_summary_text(results)
            print(summary_text)
            logger.info("✅ Backtest complete")
            
            return True
                
        except Exception as e:
            logger.error(f"Backtest error: {e}")
            return False

    def run_full_pipeline(self):
        logger.info("="*60)
        
        # Step 1: Train models
        logger.info("\n📚 Step 1: Training models...")
        if not self.train_models():
            logger.error("Training failed! Aborting.")
            return
        
        # Step 2: Run initial deployment
        logger.info("\n🚀 Step 2: Running initial deployment...")
        if not self.run_deployment():
            logger.error("Initial deployment failed! Continuing anyway...")
        
        # Step 3: Start continuous deployment in background
        logger.info("\n⏰ Step 3: Starting background deployment...")
        deployment_thread = threading.Thread(target=self.run_continuous, daemon=True)
        deployment_thread.start()
        
        # Step 4: Launch dashboard
        logger.info("\n🌐 Step 4: Launching dashboard...")
        time.sleep(2)  # Give deployment thread time to start
        self.launch_dashboard()


def print_usage():
    """Print usage instructions."""
    print("""
Smart Grid Load Balancing Optimizer
====================================

Usage:
    python main.py <command> [options]

Commands:
    train        Train models (one-time, ~20-30 minutes)
    deploy       Run deployment pipeline once
    continuous   Run continuous deployment (every 15 min)
    dashboard    Launch the dashboard
    full         Complete setup (train + dashboard)
    backtest     Run backtesting analysis

Threshold Options:
    --anomaly-threshold FLOAT    Base anomaly detection threshold (default: 2.5)
    --forecast-threshold FLOAT   Forecast anomaly threshold (default: 2.0)
    --peer-threshold FLOAT       Peer anomaly threshold (default: 2.5)
    --share-absolute FLOAT       Share absolute threshold (default: 0.40)
    --share-increase FLOAT       Share increase threshold (default: 0.15)
    
Examples:
    python main.py train                              # First-time setup
    python main.py dashboard                          # View dashboard
    python main.py deploy --forecast-threshold 1.5   # More sensitive forecast detection
    python main.py backtest                           # Run performance analysis
    python main.py full                               # Complete pipeline
    """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Smart Grid Load Balancing Optimizer')
    parser.add_argument('command', choices=['train', 'deploy', 'continuous', 'dashboard', 'launch_dashboard', 'full', 'backtest'],
                      help='Command to execute')
    
    # Threshold parameters
    parser.add_argument('--anomaly-threshold', type=float, help='Base anomaly detection threshold (default: 2.5)')
    parser.add_argument('--forecast-threshold', type=float, help='Forecast anomaly threshold (default: 2.0)')
    parser.add_argument('--peer-threshold', type=float, help='Peer anomaly threshold (default: 2.5)')
    parser.add_argument('--share-absolute', type=float, help='Share absolute threshold (default: 0.40)')
    parser.add_argument('--share-increase', type=float, help='Share increase threshold (default: 0.15)')
    
    args = parser.parse_args()
    
    # Build threshold overrides
    threshold_overrides = {}
    if args.anomaly_threshold is not None:
        threshold_overrides['threshold'] = args.anomaly_threshold
    if args.forecast_threshold is not None:
        threshold_overrides['forecast_threshold'] = args.forecast_threshold
    if args.peer_threshold is not None:
        threshold_overrides['peer_threshold'] = args.peer_threshold
    if args.share_absolute is not None:
        threshold_overrides['share_abs'] = args.share_absolute
    if args.share_increase is not None:
        threshold_overrides['share_inc'] = args.share_increase
    
    controller = SmartGridController(threshold_overrides)
    
    if args.command == 'train':
        success = controller.train_models()
        sys.exit(0 if success else 1)
    
    elif args.command == 'deploy':
        success = controller.run_deployment()
        sys.exit(0 if success else 1)
    
    elif args.command == 'continuous':
        controller.run_continuous()
        sys.exit(0)
    
    elif args.command in ('dashboard', 'launch_dashboard'):
        controller.launch_dashboard()
        sys.exit(0)
    
    elif args.command == 'backtest':
        success = controller.run_backtest()
        sys.exit(0 if success else 1)
    
    elif args.command == 'full':
        controller.run_full_pipeline()
        sys.exit(0)
    
    else:
        print(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":

    main()
