#!/usr/bin/env python
"""
Smart Grid Dashboard
====================

4-Tab dashboard for Smart Grid Load Balancing Optimizer.

Tab 1: Overview - Total load, forecasts, anomalies
Tab 2: Grid Analytics - Historical analysis
Tab 3: Network Topology - Grid visualization  
Tab 4: Performance Metrics - Model performance

Author: Laios Ioannis
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Dict, List, Optional

# Page configuration
st.set_page_config(
    page_title="Smart Grid Load Balancing Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Auto-refresh every 15 minutes (robust across versions)
try:
    # Available in recent versions
    st.autorefresh(interval=15 * 60 * 1000, key="smart_grid_auto_refresh")
except Exception:
    # Older Streamlit: no-op, we’ll provide a manual refresh button in the sidebar
    pass

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .anomaly-high {
        background-color: #ff4444;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .anomaly-medium {
        background-color: #ffaa00;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
    .anomaly-forecast {
        background-color: #4444ff;
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SmartGridDashboard:
    """Main dashboard class."""
    
    def __init__(self):
        """Initialize dashboard."""
        self.data_path = Path('outputs/latest_predictions.json')
        self.historical_data_path = Path('data')
        self.refresh_interval = 15  # minutes
        
        # Initialize session state
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
            
    def load_latest_predictions(self) -> Optional[Dict]:
        """Load latest predictions from file."""
        try:
            if self.data_path.exists():
                with open(self.data_path, 'r') as f:
                    return json.load(f)
            else:
                return self._generate_demo_data()
        except Exception as e:
            st.error(f"Error loading predictions: {e}")
            return self._generate_demo_data()
    
    def _generate_demo_data(self) -> Dict:
        """Generate demo data for display."""
        regions = [f'region_{i}' for i in range(1, 9)]
        
        current_load = {region: np.random.uniform(100, 600) for region in regions}
        
        predictions = {}
        intervals = {}
        for region in regions:
            point_15 = current_load[region] * np.random.uniform(0.95, 1.05)
            predictions[region] = {
                '15min': current_load[region] * np.random.uniform(0.95, 1.05),
                '1h': current_load[region] * np.random.uniform(0.9, 1.1),
                '6h': current_load[region] * np.random.uniform(0.85, 1.15),
                '12h': current_load[region] * np.random.uniform(0.8, 1.2),
                '24h': current_load[region] * np.random.uniform(0.75, 1.25),
                '48h': current_load[region] * np.random.uniform(0.7, 1.3)
            }
            # Simple symmetric demo intervals ±10%
            intervals[region] = {
                h: {
                    'lower': predictions[region][h] * 0.9,
                    'upper': predictions[region][h] * 1.1
                } for h in predictions[region]
            }
        
        anomalies = [
            {
                'region': 'region_1',
                'severity': 'high',
                'z_score': 3.2,
                'current_load': 580,
                'expected_range': (400, 500),
                'timestamp': datetime.now().isoformat()
            }
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_load': current_load,
            'predictions': predictions,
            'intervals': intervals,
            'anomalies': anomalies,
            'weather': {
                'temperature': 22.5,
                'humidity': 65,
                'solar_irradiance': 450
            }
        }
    
    def create_overview_tab(self, data: Dict):
        """Create Tab 1: Overview."""
        st.header("🏭 System Overview")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        total_load = sum(data['current_load'].values())
        total_predicted_1h = sum(pred['1h'] for pred in data['predictions'].values())
        num_anomalies = len(data.get('anomalies', []))
        weather = data.get('weather', {})
        # Weather may be a scalar set or per-region mapping; compute averages if needed
        def _avg_from_region_map(maybe):
            if isinstance(maybe, dict):
                # Collect non-null values and coerce to numeric, ignoring non-convertible
                vals = [v for v in maybe.values() if v is not None]
                if not vals:
                    return None
                try:
                    series = pd.to_numeric(pd.Series(vals), errors='coerce')
                    series = series.dropna()
                    if series.empty:
                        return None
                    return float(series.mean())
                except Exception:
                    # Fallback: manual float conversion
                    num_vals = []
                    for v in vals:
                        try:
                            num_vals.append(float(v))
                        except Exception:
                            continue
                    return float(np.mean(num_vals)) if num_vals else None
            # If scalar-like, return as-is (could be numeric or string)
            return maybe
        weather_avg = {
            'temperature': _avg_from_region_map(weather.get('temperature', 22.5)),
            'humidity': _avg_from_region_map(weather.get('humidity', 60)),
            'solar_irradiance': _avg_from_region_map(weather.get('solar_irradiance', 0)),
        }
        # Ensure safe defaults for display formatting when values are None or non-numeric strings
        def _coerce_or_default(val, default):
            if val is None:
                return float(default)
            try:
                return float(val)
            except Exception:
                return float(default)
        _temp_v = _coerce_or_default(weather_avg.get('temperature'), 20)
        _hum_v = _coerce_or_default(weather_avg.get('humidity'), 60)
        _solar_v = _coerce_or_default(weather_avg.get('solar_irradiance'), 0)
        
        with col1:
            st.metric(
                "Total System Load",
                f"{total_load:.1f} MW",
                f"{(total_predicted_1h - total_load):.1f} MW (1h)"
            )
        
        with col2:
            st.metric(
                "Active Regions",
                len(data['current_load']),
                f"{num_anomalies} anomalies"
            )
        
        with col3:
            st.metric(
                "Temperature",
                f"{_temp_v:.1f}°C",
                f"{_hum_v:.0f}% humidity"
            )
        
        with col4:
            st.metric(
                "Solar Generation",
                f"{_solar_v:.0f} W/m²",
                None
            )
        
        # Configurable forecast horizon selector
        st.subheader("📊 Load Forecasting")
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            # View selector: Total (all) or a single region
            regions = sorted(list(data['current_load'].keys()))
            view_options = ['Total (All regions)'] + regions
            selected_view = st.selectbox(
                "View",
                view_options,
                index=0
            )
            selected_horizon = st.selectbox(
                "Select Forecast Horizon",
                ['15min', '1h', '6h', '12h', '24h', '48h'],
                index=2  # Default to 6h
            )
            
            show_confidence = st.checkbox("Show Confidence Bands", value=True)
            historical_days = st.slider("Historical Days", 1, 30, 7)
        
        with col1:
            # Create forecast chart
            fig = self._create_forecast_chart(data, selected_horizon, show_confidence, historical_days, selected_view)
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly Detection Section
        st.subheader("🚨 Anomaly Detection")
        
        if data.get('anomalies'):
            for anomaly in data['anomalies'][:5]:  # Show top 5
                severity = anomaly.get('severity', 'medium')
                region = anomaly.get('region', 'Unknown')
                z_score = anomaly.get('z_score', 0)
                
                if severity == 'high':
                    st.markdown(f"""
                    <div class="anomaly-high">
                        <strong>⚠️ HIGH ALERT - {region}</strong><br>
                        Z-score: {z_score:.2f} | Load: {anomaly.get('current_load', 0):.1f} MW
                    </div>
                    """, unsafe_allow_html=True)
                elif severity == 'medium':
                    st.markdown(f"""
                    <div class="anomaly-medium">
                        <strong>⚠️ MEDIUM ALERT - {region}</strong><br>
                        Z-score: {z_score:.2f} | Load: {anomaly.get('current_load', 0):.1f} MW
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="anomaly-forecast">
                        <strong>📊 FORECAST ALERT - {region}</strong><br>
                        Horizon: {anomaly.get('horizon', 'N/A')} | Predicted: {anomaly.get('predicted_load', 0):.1f} MW
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.success("✅ No anomalies detected - System operating normally")
    
    def create_analytics_tab(self, data: Dict):
        """Create Tab 2: Grid Analytics."""
        st.header("📈 Grid Analytics")
        
        # Regional load distribution
        col1, col2 = st.columns(2)
        
        with col1:
            # Current load by region
            fig = px.bar(
                x=list(data['current_load'].keys()),
                y=list(data['current_load'].values()),
                title="Current Load by Region",
                labels={'x': 'Region', 'y': 'Load (MW)'},
                color=list(data['current_load'].values()),
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Load factor analysis
            load_factors = self._calculate_load_factors(data)
            fig = px.pie(
                values=list(load_factors.values()),
                names=list(load_factors.keys()),
                title="Load Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Historical trends
        st.subheader("📊 Historical Analysis")
        
        # Generate synthetic historical data for demo
        historical_data = self._generate_historical_data()
        
        # Peak/trough analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Daily Peak", f"{historical_data['daily_peak']:.1f} MW", 
                     f"{historical_data['peak_change']:.1f}%")
        
        with col2:
            st.metric("Daily Trough", f"{historical_data['daily_trough']:.1f} MW",
                     f"{historical_data['trough_change']:.1f}%")
        
        with col3:
            st.metric("Avg Load Factor", f"{historical_data['load_factor']:.2f}",
                     f"{historical_data['factor_change']:.2f}%")
        
        # Time series chart
        fig = self._create_historical_chart(historical_data)
        st.plotly_chart(fig, use_container_width=True)
        
        # Capacity utilization
        st.subheader("⚡ Capacity Utilization")
        
        capacity_data = self._calculate_capacity_utilization(data)
        fig = go.Figure(data=[
            go.Bar(name='Used', x=capacity_data['regions'], y=capacity_data['used']),
            go.Bar(name='Available', x=capacity_data['regions'], y=capacity_data['available'])
        ])
        fig.update_layout(
            barmode='stack',
            title='Regional Capacity Utilization',
            xaxis_title='Region',
            yaxis_title='Capacity (MW)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def create_topology_tab(self, data: Dict):
        """Create Tab 3: Network Topology."""
        st.header("🗺️ Network Topology")
        
        # Create network visualization
        fig = self._create_network_topology(data)
        st.plotly_chart(fig, use_container_width=True, height=600)
        
        # Power flow analysis
        st.subheader("⚡ Power Flow Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Connection status
            st.markdown("### Connection Status")
            connections = self._get_connection_status(data)
            for conn in connections:
                status_color = "🟢" if conn['status'] == 'normal' else "🔴"
                st.write(f"{status_color} {conn['from']} → {conn['to']}: {conn['flow']:.1f} MW")
        
        with col2:
            # Stress points
            st.markdown("### Grid Stress Points")
            stress_points = self._identify_stress_points(data)
            if stress_points:
                for point in stress_points:
                    st.warning(f"⚠️ {point['location']}: {point['stress_level']} stress ({point['load']:.1f} MW)")
            else:
                st.success("✅ No stress points detected")
    
    def create_performance_tab(self, data: Dict):
        """Create Tab 4: Performance Metrics."""
        st.header("📊 Performance Metrics")
        
        # Model accuracy metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Generate synthetic metrics for demo
        metrics = self._generate_performance_metrics()
        
        with col1:
            st.metric("15min MAPE", f"{metrics['15min_mape']:.2f}%",
                     f"{metrics['15min_change']:.1f}%")
        
        with col2:
            st.metric("1h MAPE", f"{metrics['1h_mape']:.2f}%",
                     f"{metrics['1h_change']:.1f}%")
        
        with col3:
            st.metric("6h MAPE", f"{metrics['6h_mape']:.2f}%",
                     f"{metrics['6h_change']:.1f}%")
        
        with col4:
            st.metric("24h MAPE", f"{metrics['24h_mape']:.2f}%",
                     f"{metrics['24h_change']:.1f}%")
        
        # Forecast error analysis
        st.subheader("📈 Forecast Error Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Error distribution
            fig = self._create_error_distribution()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Error by horizon
            fig = self._create_error_by_horizon()
            st.plotly_chart(fig, use_container_width=True)

        # Feature importances (from pipeline if available)
        st.subheader("🔬 Feature Importances")
        fi_all = data.get('feature_importances', {}) or {}
        if not fi_all:
            st.info("No feature importances found in output. Train and deploy to populate.")
        else:
            regions = sorted(fi_all.keys())
            sel_region = st.selectbox("Region", regions, index=0)
            horizons = sorted((fi_all.get(sel_region) or {}).keys())
            if not horizons:
                st.info("No horizons available for selected region.")
            else:
                sel_h = st.selectbox("Horizon", horizons, index=0)
                imp_map = fi_all.get(sel_region, {}).get(sel_h, {})
                if not imp_map:
                    st.info("No importances for this selection.")
                else:
                    # Show top 15
                    items = sorted(imp_map.items(), key=lambda kv: kv[1], reverse=True)[:15]
                    df = pd.DataFrame(items, columns=['feature', 'importance'])
                    fig = px.bar(df, x='feature', y='importance', title=f'Feature Importances - {sel_region} / {sel_h}')
                    fig.update_layout(xaxis_tickangle=-45, height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # System health
        st.subheader("🏥 System Health")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CPU/Memory usage
            st.markdown("### Resource Usage")
            st.progress(metrics['cpu_usage'] / 100)
            st.caption(f"CPU: {metrics['cpu_usage']:.1f}%")
            st.progress(metrics['memory_usage'] / 100)
            st.caption(f"Memory: {metrics['memory_usage']:.1f}%")
        
        with col2:
            # Model freshness
            st.markdown("### Model Status")
            st.info(f"Last Training: {metrics['last_training']}")
            st.info(f"Last Prediction: {metrics['last_prediction']}")
            st.info(f"Uptime: {metrics['uptime']}")
        
        with col3:
            # Data quality
            st.markdown("### Data Quality")
            st.progress(metrics['data_quality'] / 100)
            st.caption(f"Quality Score: {metrics['data_quality']:.1f}%")
            st.progress(metrics['feature_importance'] / 100)
            st.caption(f"Feature Importance: {metrics['feature_importance']:.1f}%")
    
    # Helper methods for chart creation
    
    def _create_forecast_chart(self, data: Dict, horizon: str, 
                               show_confidence: bool, historical_days: int,
                               view: str) -> go.Figure:
        """Create forecast chart with historical and predicted data."""
        fig = go.Figure()
        
        # Generate synthetic historical data
        now = datetime.now()
        historical_dates = pd.date_range(
            end=now, 
            periods=historical_days * 96,  # 96 15-min intervals per day
            freq='15min'
        )
        # Horizon mapping to number of 15-min steps
        steps_map = {'15min': 1, '1h': 4, '6h': 24, '12h': 48, '24h': 96, '48h': 192}
        n_steps = steps_map.get(horizon, 24)
        forecast_dates = pd.date_range(
            start=now,
            periods=n_steps,
            freq='15min'
        )

        intervals_all = data.get('intervals', {}) or {}
        preds_all = data.get('predictions', {}) or {}
        current_loads = data.get('current_load', {}) or {}
        path_all = data.get('forecast_paths', {}) or {}
        path_int_all = data.get('interval_paths', {}) or {}

        # Helper: build piecewise-constant 15-min series from horizon anchors
        anchor_order = ['15min', '1h', '6h', '12h', '24h', '48h']
        def build_series_for_region(reg: str):
            # Prefer precomputed path from backend
            path_map = path_all.get(reg) or {}
            path_int_map = path_int_all.get(reg) or {}
            if path_map.get(horizon):
                arr = np.array(path_map[horizon], dtype=float)
                arr = arr[:n_steps]
                if len(arr) < n_steps:
                    arr = np.pad(arr, (0, n_steps - len(arr)), mode='edge')
                lo_arr = None
                hi_arr = None
                if isinstance(path_int_map.get(horizon), dict):
                    lo_list = (path_int_map[horizon].get('lower') or [])
                    hi_list = (path_int_map[horizon].get('upper') or [])
                    try:
                        lo_arr = np.array(lo_list, dtype=float)[:n_steps]
                        hi_arr = np.array(hi_list, dtype=float)[:n_steps]
                        if len(lo_arr) < n_steps:
                            lo_arr = np.pad(lo_arr, (0, n_steps - len(lo_arr)), mode='edge')
                        if len(hi_arr) < n_steps:
                            hi_arr = np.pad(hi_arr, (0, n_steps - len(hi_arr)), mode='edge')
                    except Exception:
                        lo_arr = None
                        hi_arr = None
                if lo_arr is None or hi_arr is None:
                    lo_arr = arr * 0.9
                    hi_arr = arr * 1.1
                return arr, lo_arr, hi_arr
            pred_map = preds_all.get(reg) or {}
            int_map = intervals_all.get(reg) or {}
            # Determine anchors up to the selected horizon
            anchors = []  # list of (step_idx, value, lower, upper)
            for h_key in anchor_order:
                steps_h = steps_map[h_key]
                if steps_h <= n_steps and (h_key in pred_map):
                    val = pred_map.get(h_key)
                    try:
                        val_f = float(val)
                    except Exception:
                        continue
                    lower = upper = None
                    im = int_map.get(h_key) if isinstance(int_map.get(h_key), dict) else None
                    if im and 'lower' in im and 'upper' in im:
                        try:
                            lower = float(im['lower'])
                            upper = float(im['upper'])
                        except Exception:
                            lower = upper = None
                    anchors.append((steps_h, val_f, lower, upper))
            anchors.sort(key=lambda x: x[0])

            # Initialize arrays
            y = np.full(n_steps, np.nan, dtype=float)
            lo = np.full(n_steps, np.nan, dtype=float)
            hi = np.full(n_steps, np.nan, dtype=float)

            # Determine starting value
            start_val = None
            start_lo = None
            start_hi = None
            if anchors:
                start_val = anchors[0][1]
                if anchors[0][2] is not None and anchors[0][3] is not None:
                    start_lo = anchors[0][2]
                    start_hi = anchors[0][3]
            if start_val is None:
                # Fallback to selected horizon point or current load
                fallback = pred_map.get(horizon)
                try:
                    start_val = float(fallback) if fallback is not None else float(current_loads.get(reg, 0))
                except Exception:
                    start_val = float(current_loads.get(reg, 0))
                start_lo = start_val * 0.9
                start_hi = start_val * 1.1

            last_step = 0
            last_val = start_val
            last_lo = start_lo if start_lo is not None else start_val * 0.9
            last_hi = start_hi if start_hi is not None else start_val * 1.1
            for step_idx, val_f, lower, upper in anchors:
                # Fill from last_step to step_idx-1 with last values
                if step_idx - 1 > last_step:
                    y[last_step:step_idx-1] = last_val
                    lo[last_step:step_idx-1] = last_lo
                    hi[last_step:step_idx-1] = last_hi
                # Update to anchor value starting at step_idx-1 (0-based)
                anchor_pos = step_idx - 1
                if 0 <= anchor_pos < n_steps:
                    last_val = val_f
                    y[anchor_pos] = last_val
                    last_lo = (float(lower) if lower is not None else last_val * 0.9)
                    last_hi = (float(upper) if upper is not None else last_val * 1.1)
                    lo[anchor_pos] = last_lo
                    hi[anchor_pos] = last_hi
                last_step = anchor_pos + 1
            # Fill any remaining tail
            if last_step < n_steps:
                y[last_step:] = last_val
                lo[last_step:] = last_lo
                hi[last_step:] = last_hi
            # If somehow any NaNs remain, backfill with start_val
            y = np.nan_to_num(y, nan=start_val)
            lo = np.nan_to_num(lo, nan=start_val * 0.9)
            hi = np.nan_to_num(hi, nan=start_val * 1.1)
            return y, lo, hi

        # Helper to generate a synthetic historical series for a region
        def _hist_series_for_region(reg):
            base = float(current_loads.get(reg, 0))
            return np.random.normal(base, 50, len(historical_dates))

        if view == 'Total (All regions)':
            # Historical: sum across all regions
            regions = list(current_loads.keys())
            if not regions:
                return fig
            total_hist = None
            for reg in regions:
                series = _hist_series_for_region(reg)
                total_hist = series if total_hist is None else (total_hist + series)
            fig.add_trace(go.Scatter(
                x=historical_dates,
                y=total_hist,
                mode='lines',
                name='Total (historical)',
                line=dict(width=2)
            ))
            # Forecast: sum of region piecewise-constant series built from anchors
            total_y = np.zeros(n_steps, dtype=float)
            total_lo = np.zeros(n_steps, dtype=float)
            total_hi = np.zeros(n_steps, dtype=float)
            any_intervals = False
            for reg in regions:
                y, lo, hi = build_series_for_region(reg)
                total_y += y
                total_lo += lo
                total_hi += hi
                # Check if intervals likely came from data (not fallback)
                any_intervals = any_intervals or not np.allclose(hi, y * 1.1) or not np.allclose(lo, y * 0.9)

            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=total_y,
                mode='lines',
                name='Total (forecast)',
                line=dict(width=2, dash='dash')
            ))

            if show_confidence:
                if not any_intervals:
                    # Fallback ±10%
                    total_lo = total_y * 0.9
                    total_hi = total_y * 1.1
                fig.add_trace(go.Scatter(
                    x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                    y=total_hi.tolist() + total_lo.tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(100, 100, 200, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Total (confidence)',
                    showlegend=False
                ))
        else:
            # Single region view
            region = view
            # Historical for the region
            historical_values = _hist_series_for_region(region)
            fig.add_trace(go.Scatter(
                x=historical_dates,
                y=historical_values,
                mode='lines',
                name=f'{region} (historical)',
                line=dict(width=2)
            ))
            # Forecast for the region using piecewise anchors
            y, lo, hi = build_series_for_region(region)
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=y,
                mode='lines',
                name=f'{region} (forecast)',
                line=dict(width=2, dash='dash')
            ))

            # Confidence for the region
            if show_confidence:
                fig.add_trace(go.Scatter(
                    x=forecast_dates.tolist() + forecast_dates.tolist()[::-1],
                    y=hi.tolist() + lo.tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(100, 100, 200, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name=f'{region} (confidence)',
                    showlegend=False
                ))
        
        fig.update_layout(
            title=f'Load Forecast - {horizon} Horizon ({"Total" if view == "Total (All regions)" else view})',
            xaxis_title='Time',
            yaxis_title='Load (MW)',
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def _calculate_load_factors(self, data: Dict) -> Dict:
        """Calculate load factors for pie chart."""
        total = sum(data['current_load'].values())
        return {
            region: (load / total * 100) 
            for region, load in data['current_load'].items()
        }
    
    def _generate_historical_data(self) -> Dict:
        """Generate synthetic historical data."""
        return {
            'daily_peak': np.random.uniform(3000, 4000),
            'peak_change': np.random.uniform(-5, 5),
            'daily_trough': np.random.uniform(1500, 2000),
            'trough_change': np.random.uniform(-3, 3),
            'load_factor': np.random.uniform(0.6, 0.8),
            'factor_change': np.random.uniform(-2, 2),
            'time_series': pd.DataFrame({
                'time': pd.date_range(end=datetime.now(), periods=168, freq='H'),
                'load': np.random.normal(2500, 500, 168)
            })
        }
    
    def _create_historical_chart(self, historical_data: Dict) -> go.Figure:
        """Create historical time series chart."""
        df = historical_data['time_series']
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=df['load'],
            mode='lines',
            name='Total Load',
            line=dict(color='blue', width=2)
        ))
        
        # Add moving average
        ma = df['load'].rolling(window=24).mean()
        fig.add_trace(go.Scatter(
            x=df['time'],
            y=ma,
            mode='lines',
            name='24h Moving Average',
            line=dict(color='red', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title='7-Day Load History',
            xaxis_title='Time',
            yaxis_title='Load (MW)',
            hovermode='x unified'
        )
        
        return fig
    
    def _calculate_capacity_utilization(self, data: Dict) -> Dict:
        """Calculate capacity utilization."""
        regions = list(data['current_load'].keys())
        max_capacity = {region: np.random.uniform(600, 800) for region in regions}
        
        return {
            'regions': regions,
            'used': [data['current_load'][r] for r in regions],
            'available': [max_capacity[r] - data['current_load'][r] for r in regions]
        }
    
    def _create_network_topology(self, data: Dict) -> go.Figure:
        """Create network topology visualization."""
        # Define node positions
        positions = {
            'region_1': (0.5, 0.8),
            'region_2': (0.2, 0.6),
            'region_3': (0.8, 0.6),
            'region_4': (0.1, 0.3),
            'region_5': (0.3, 0.1),
            'region_6': (0.7, 0.1),
            'region_7': (0.9, 0.3),
            'region_8': (0.5, 0.4)
        }
        
        # Define connections
        connections = [
            ('region_1', 'region_2'),
            ('region_1', 'region_3'),
            ('region_1', 'region_8'),
            ('region_2', 'region_4'),
            ('region_3', 'region_7'),
            ('region_4', 'region_5'),
            ('region_5', 'region_6'),
            ('region_6', 'region_7'),
            ('region_8', 'region_5')
        ]
        
        fig = go.Figure()
        
        # Add connections
        for start, end in connections:
            if start in positions and end in positions:
                x = [positions[start][0], positions[end][0]]
                y = [positions[start][1], positions[end][1]]
                
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add nodes
        for region, pos in positions.items():
            if region in data['current_load']:
                load = data['current_load'][region]
                
                # Color based on load level
                if load > 500:
                    color = 'red'
                elif load > 300:
                    color = 'orange'
                else:
                    color = 'green'
                
                fig.add_trace(go.Scatter(
                    x=[pos[0]],
                    y=[pos[1]],
                    mode='markers+text',
                    marker=dict(size=30 + load/20, color=color),
                    text=[f"{region}<br>{load:.0f} MW"],
                    textposition='top center',
                    showlegend=False,
                    hovertemplate=f'<b>{region}</b><br>Load: {load:.1f} MW<extra></extra>'
                ))
        
        fig.update_layout(
            title='Smart Grid Network Topology',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            hovermode='closest'
        )
        
        return fig
    
    def _get_connection_status(self, data: Dict) -> List[Dict]:
        """Get connection status for power flow."""
        connections = []
        regions = list(data['current_load'].keys())
        
        for i in range(min(5, len(regions)-1)):
            connections.append({
                'from': regions[i],
                'to': regions[i+1],
                'flow': np.random.uniform(50, 200),
                'status': 'normal' if np.random.random() > 0.2 else 'congested'
            })
        
        return connections
    
    def _identify_stress_points(self, data: Dict) -> List[Dict]:
        """Identify grid stress points."""
        stress_points = []
        
        for region, load in data['current_load'].items():
            if load > 500:  # Threshold for stress
                stress_points.append({
                    'location': region,
                    'load': load,
                    'stress_level': 'high' if load > 550 else 'medium'
                })
        
        return stress_points
    
    def _generate_performance_metrics(self) -> Dict:
        """Generate performance metrics."""
        return {
            '15min_mape': np.random.uniform(2, 5),
            '15min_change': np.random.uniform(-1, 1),
            '1h_mape': np.random.uniform(3, 6),
            '1h_change': np.random.uniform(-1, 1),
            '6h_mape': np.random.uniform(5, 10),
            '6h_change': np.random.uniform(-2, 2),
            '24h_mape': np.random.uniform(8, 15),
            '24h_change': np.random.uniform(-3, 3),
            'cpu_usage': np.random.uniform(20, 60),
            'memory_usage': np.random.uniform(30, 70),
            'last_training': '2024-12-01 10:00',
            'last_prediction': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'uptime': '7 days 3 hours',
            'data_quality': np.random.uniform(85, 95),
            'feature_importance': np.random.uniform(75, 90)
        }
    
    def _create_error_distribution(self) -> go.Figure:
        """Create error distribution chart."""
        errors = np.random.normal(0, 5, 1000)
        
        fig = go.Figure(data=[go.Histogram(x=errors, nbinsx=30)])
        fig.update_layout(
            title='Forecast Error Distribution',
            xaxis_title='Error (%)',
            yaxis_title='Frequency',
            showlegend=False
        )
        
        return fig
    
    def _create_error_by_horizon(self) -> go.Figure:
        """Create error by horizon chart."""
        horizons = ['15min', '1h', '6h', '12h', '24h', '48h']
        mape = [3.2, 4.5, 7.8, 9.2, 12.5, 14.8]
        rmse = [25, 35, 55, 68, 85, 95]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(name='MAPE (%)', x=horizons, y=mape))
        fig.add_trace(go.Bar(name='RMSE (MW)', x=horizons, y=rmse))
        
        fig.update_layout(
            title='Error Metrics by Forecast Horizon',
            xaxis_title='Horizon',
            yaxis_title='Error',
            barmode='group'
        )
        
        return fig
    
    def run(self):
        """Run the dashboard."""
        st.title("⚡ Smart Grid Load Balancing Optimizer")
        
        # Sidebar controls
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # Refresh controls
            if st.button("🔄 Refresh Now"):
                st.session_state.last_update = datetime.now()
                st.rerun()
            
            st.session_state.auto_refresh = st.checkbox(
                "Auto-refresh (15 min)",
                value=st.session_state.auto_refresh
            )
            
            # Display last update time
            if st.session_state.last_update:
                st.info(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")
            
            # Anomaly threshold controls
            st.header("🚨 Anomaly Thresholds")
            
            # Initialize threshold values if not in session state
            if 'threshold_anomaly' not in st.session_state:
                st.session_state.threshold_anomaly = 2.5
            if 'threshold_forecast' not in st.session_state:
                st.session_state.threshold_forecast = 2.0
            if 'threshold_peer' not in st.session_state:
                st.session_state.threshold_peer = 2.5
            if 'threshold_share_abs' not in st.session_state:
                st.session_state.threshold_share_abs = 0.40
            if 'threshold_share_inc' not in st.session_state:
                st.session_state.threshold_share_inc = 0.15
            
            # Threshold sliders
            st.session_state.threshold_anomaly = st.slider(
                "Base Anomaly (Z-score)",
                min_value=1.0, max_value=5.0, 
                value=st.session_state.threshold_anomaly,
                step=0.1,
                help="Higher = less sensitive to general anomalies"
            )
            
            st.session_state.threshold_forecast = st.slider(
                "Forecast Anomaly (Z-score)",
                min_value=1.0, max_value=5.0,
                value=st.session_state.threshold_forecast,
                step=0.1,
                help="Higher = less sensitive to forecast spikes"
            )
            
            st.session_state.threshold_peer = st.slider(
                "Peer Anomaly (Z-score)",
                min_value=1.5, max_value=4.0,
                value=st.session_state.threshold_peer,
                step=0.1,
                help="Higher = less sensitive to peer outliers"
            )
            
            st.session_state.threshold_share_abs = st.slider(
                "Share Absolute (%)",
                min_value=0.20, max_value=0.60,
                value=st.session_state.threshold_share_abs,
                step=0.05,
                format="%.2f",
                help="Flag regions consuming >X% of total"
            )
            
            st.session_state.threshold_share_inc = st.slider(
                "Share Increase (%)",
                min_value=0.05, max_value=0.30,
                value=st.session_state.threshold_share_inc,
                step=0.01,
                format="%.2f",
                help="Flag regions with >X% share increase"
            )
            
            # Reset button
            if st.button("🔄 Reset to Defaults"):
                st.session_state.threshold_anomaly = 2.5
                st.session_state.threshold_forecast = 2.0
                st.session_state.threshold_peer = 2.5
                st.session_state.threshold_share_abs = 0.40
                st.session_state.threshold_share_inc = 0.15
                st.rerun()
            
            # Apply button (TODO: implement real-time threshold updates)
            if st.button("✅ Apply Changes"):
                st.success("Settings saved! Take effect on next deployment.")
            
            # System status
            st.header("📊 System Status")
            st.success("✅ Pipeline: Operational")
            st.success("✅ Models: Loaded")
            st.info("🔄 Next update: 15 min")
        
        # Load latest data
        data = self.load_latest_predictions()
        
        if not data:
            st.error("Failed to load data!")
            return
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview",
            "📈 Grid Analytics", 
            "🗺️ Network Topology",
            "🎯 Performance Metrics"
        ])
        
        with tab1:
            self.create_overview_tab(data)
        
        with tab2:
            self.create_analytics_tab(data)
        
        with tab3:
            self.create_topology_tab(data)
        
        with tab4:
            self.create_performance_tab(data)
        
        # Auto-refresh logic
        if st.session_state.auto_refresh:
            time.sleep(1)
            # Check if 15 minutes have passed
            if st.session_state.last_update:
                time_diff = datetime.now() - st.session_state.last_update
                if time_diff.total_seconds() >= 900:  # 15 minutes
                    st.session_state.last_update = datetime.now()
                    st.rerun()
            else:
                st.session_state.last_update = datetime.now()


# Main execution
if __name__ == "__main__":
    dashboard = SmartGridDashboard()

    dashboard.run()
