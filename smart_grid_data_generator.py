# enhanced_data_generator.py
"""
Enhanced Smart Grid Data Generator
==================================

Ensures consistent 15-minute intervals and realistic smart grid behavior
for multi-horizon forecasting training and deployment.

Author: Smart Grid Team
Date: September 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class EnhancedDataGenerator:
    """Enhanced data generator with proper 15-minute interval handling."""
    
    def __init__(self, simulation_years: int = 5, regions: List[str] = None):
        """Initialize the enhanced data generator."""
        self.simulation_years = simulation_years
        
        # Load region configuration
        if regions is None:
            from simplified_config import enhanced_config
            self.regions = list(enhanced_config.regions.keys())
        else:
            self.regions = regions
        
        # Time configuration - ensure 15-minute intervals
        self.base_interval_minutes = 15
        self.intervals_per_hour = 60 // self.base_interval_minutes  # 4
        self.intervals_per_day = 24 * self.intervals_per_hour      # 96
        
        # Regional characteristics
        self.region_characteristics = self._define_region_characteristics()
        
        print("⚡ Enhanced Data Generator initialized")
        print(f"📊 Simulation years: {simulation_years}")
        print(f"🏙️ Regions: {len(self.regions)}")
        print(f"⏰ Base interval: {self.base_interval_minutes} minutes")
        print(f"📈 Intervals per day: {self.intervals_per_day}")
    
    def _define_region_characteristics(self) -> Dict:
        """Define unique characteristics for each region type."""
        return {
            'region_1_urban_metro': {
                'base_load': 400,
                'peak_hours': [8, 9, 18, 19, 20],
                'seasonal_variation': 0.3,
                'daily_variation': 0.4,
                'noise_level': 25,
                'temp_sensitivity': 0.8
            },
            'region_2_suburban_mixed': {
                'base_load': 300,
                'peak_hours': [7, 8, 17, 18, 19],
                'seasonal_variation': 0.25,
                'daily_variation': 0.35,
                'noise_level': 20,
                'temp_sensitivity': 0.6
            },
            'region_3_industrial_heavy': {
                'base_load': 650,
                'peak_hours': [6, 7, 8, 14, 15, 22, 23],
                'seasonal_variation': 0.15,
                'daily_variation': 0.25,
                'noise_level': 35,
                'temp_sensitivity': 0.3
            },
            'region_4_rural_agricultural': {
                'base_load': 200,
                'peak_hours': [5, 6, 18, 19],
                'seasonal_variation': 0.4,
                'daily_variation': 0.3,
                'noise_level': 15,
                'temp_sensitivity': 0.4
            },
            'region_5_coastal_renewable': {
                'base_load': 320,
                'peak_hours': [10, 11, 12, 13, 14],  # Solar peak
                'seasonal_variation': 0.35,
                'daily_variation': 0.45,
                'noise_level': 30,
                'temp_sensitivity': 0.5
            },
            'region_6_mountain_hydro': {
                'base_load': 125,
                'peak_hours': [11, 12, 13, 17, 18],
                'seasonal_variation': 0.5,  # High seasonal due to water availability
                'daily_variation': 0.2,
                'noise_level': 10,
                'temp_sensitivity': 0.2
            },
            'region_7_desert_solar': {
                'base_load': 250,
                'peak_hours': [10, 11, 12, 13, 14, 15],  # Solar peak
                'seasonal_variation': 0.3,
                'daily_variation': 0.6,  # High variation due to solar
                'noise_level': 20,
                'temp_sensitivity': 0.7
            },
            'region_8_mixed_commercial': {
                'base_load': 260,
                'peak_hours': [9, 10, 11, 14, 15, 16],
                'seasonal_variation': 0.2,
                'daily_variation': 0.5,
                'noise_level': 22,
                'temp_sensitivity': 0.6
            }
        }
    
    def generate_complete_dataset(self) -> Dict[str, pd.DataFrame]:
        """Generate complete dataset with proper 15-minute intervals."""
        print("\n" + "="*70)
        print("🚀 ENHANCED GRID DATA GENERATION STARTING")
        print("="*70)
        
        # Create time index with exact 15-minute intervals
        start_date = datetime(2022, 1, 1, 0, 0, 0)  # Start at exact hour
        total_intervals = self.simulation_years * 365 * self.intervals_per_day
        
        # Create time index
        time_index = pd.date_range(
            start=start_date,
            periods=total_intervals,
            freq='15min'  # Exact 15-minute frequency
        )
        
        print(f"📅 Time range: {time_index[0]} to {time_index[-1]}")
        print(f"📊 Total intervals: {len(time_index):,}")
        print(f"⏰ Frequency: Every {self.base_interval_minutes} minutes")
        
        # Validate time index consistency
        self._validate_time_index(time_index)
        
        # Generate load data
        print("\n⚡ Generating load data...")
        load_data = self._generate_enhanced_load_data(time_index)
        
        # Generate weather data
        print("🌡️ Generating temperature data...")
        temperature_data = self._generate_enhanced_temperature_data(time_index)
        
        print("💧 Generating humidity data...")
        humidity_data = self._generate_enhanced_humidity_data(time_index, temperature_data)
        
        print("💨 Generating wind speed data...")
        wind_speed_data = self._generate_enhanced_wind_speed_data(time_index)

        print("☀️ Generating solar irradiance data...")
        solar_irradiance_data = self._generate_enhanced_solar_irradiance_data(time_index)
        
        # Final validation
        self._validate_dataset_consistency(load_data, temperature_data, humidity_data, wind_speed_data)
        
        print("\n" + "="*70)
        print("✅ ENHANCED GRID DATA GENERATION COMPLETE!")
        print("="*70)
        print(f"📊 Load data shape: {load_data.shape}")
        print(f"🌡️ Temperature data shape: {temperature_data.shape}")
        print(f"💧 Humidity data shape: {humidity_data.shape}")
        print(f"💨 Wind speed data shape: {wind_speed_data.shape}")
        print(f"☀️ Solar irradiance data shape: {solar_irradiance_data.shape}")
        
        return {
            'load_data': load_data,
            'temperature_data': temperature_data,
            'humidity_data': humidity_data,
            'wind_speed_data': wind_speed_data,
            'solar_irradiance_data': solar_irradiance_data
        }
    
    def _validate_time_index(self, time_index: pd.DatetimeIndex):
        """Validate that time index has proper 15-minute intervals."""
        # Check frequency
        if not all(time_index.to_series().diff()[1:] == timedelta(minutes=15)):
            raise ValueError("Time index does not have consistent 15-minute intervals")
        
        # Check minute values are multiples of 15
        minutes = time_index.minute
        if not all(minute % 15 == 0 for minute in minutes):
            raise ValueError("Time index minutes are not multiples of 15")
        
        # Check no seconds or microseconds
        if any(time_index.second) or any(time_index.microsecond):
            raise ValueError("Time index contains non-zero seconds or microseconds")
        
        print("✅ Time index validation passed")
    
    def _generate_enhanced_load_data(self, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate realistic load data for all regions."""
        load_data = pd.DataFrame(index=time_index, columns=self.regions)
        
        for region in self.regions:
            print(f"   ⚡ Generating load for {region}")
            
            # Get region characteristics
            char = self.region_characteristics[region]
            
            # Generate load profile
            load_profile = self._generate_region_load_profile(
                time_index, char
            )
            
            load_data[region] = load_profile
        
        return load_data
    
    def _generate_region_load_profile(self, time_index: pd.DatetimeIndex, 
                                    characteristics: Dict) -> np.ndarray:
        """Generate load profile for a specific region."""
        n_points = len(time_index)
        base_load = characteristics['base_load']
        
        # Initialize with base load
        load = np.full(n_points, base_load, dtype=float)
        
        # Add seasonal variation
        day_of_year = time_index.dayofyear
        seasonal_factor = characteristics['seasonal_variation'] * np.sin(
            2 * np.pi * day_of_year / 365.25
        )
        load += base_load * seasonal_factor
        
        # Add daily variation
        hour = time_index.hour
        minute = time_index.minute
        
        # Convert to fractional hour for smooth daily curve
        fractional_hour = hour + minute / 60.0
        
        # Daily pattern (morning and evening peaks)
        daily_factor = characteristics['daily_variation'] * (
            0.3 * np.sin(2 * np.pi * fractional_hour / 24) +  # Main daily cycle
            0.2 * np.sin(4 * np.pi * fractional_hour / 24) +  # Morning/evening peaks
            0.1 * np.sin(6 * np.pi * fractional_hour / 24)    # Fine variations
        )
        load += base_load * daily_factor
        
        # Add peak hour adjustments
        for peak_hour in characteristics['peak_hours']:
            # Create peak around specific hours
            hour_diff = np.abs(fractional_hour - peak_hour)
            # Handle day boundary
            hour_diff = np.minimum(hour_diff, 24 - hour_diff)
            
            # Peak effect (Gaussian-like)
            peak_effect = 0.15 * np.exp(-hour_diff**2 / 0.5)
            load += base_load * peak_effect
        
        # Add weekly pattern
        day_of_week = time_index.dayofweek
        weekend_factor = np.where(day_of_week >= 5, -0.1, 0.05)  # Lower on weekends
        load += base_load * weekend_factor
        
        # Add temperature correlation (simplified)
        temp_factor = characteristics['temp_sensitivity'] * np.sin(
            2 * np.pi * day_of_year / 365.25 + np.pi/4  # Phase shifted from seasonal
        ) * 0.2
        load += base_load * temp_factor
        
        # Add realistic noise
        noise = np.random.normal(0, characteristics['noise_level'], n_points)
        load += noise
        
        # Ensure minimum load
        load = np.maximum(load, base_load * 0.3)
        
        return load
    
    def _generate_enhanced_temperature_data(self, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate realistic temperature data for all regions."""
        temp_data = pd.DataFrame(index=time_index, columns=self.regions)
        
        # Base temperature parameters for different regions
        base_temps = {
            'region_1_urban_metro': 18,      # Urban heat island
            'region_2_suburban_mixed': 16,
            'region_3_industrial_heavy': 19,  # Industrial heat
            'region_4_rural_agricultural': 14,
            'region_5_coastal_renewable': 20,  # Coastal moderation
            'region_6_mountain_hydro': 8,     # High altitude
            'region_7_desert_solar': 25,      # Hot desert
            'region_8_mixed_commercial': 17
        }
        
        for region in self.regions:
            print(f"   🌡️ Generating temperature for {region}")
            
            base_temp = base_temps.get(region, 15)
            temp_profile = self._generate_temperature_profile(time_index, base_temp)
            temp_data[region] = temp_profile
        
        return temp_data
    
    def _generate_temperature_profile(self, time_index: pd.DatetimeIndex, 
                                    base_temp: float) -> np.ndarray:
        """Generate temperature profile."""
        n_points = len(time_index)
        
        # Start with base temperature
        temp = np.full(n_points, base_temp, dtype=float)
        
        # Seasonal variation
        day_of_year = time_index.dayofyear
        seasonal_variation = 12 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
        temp += seasonal_variation
        
        # Daily variation
        hour = time_index.hour
        minute = time_index.minute
        fractional_hour = hour + minute / 60.0
        
        daily_variation = 8 * np.sin(2 * np.pi * (fractional_hour - 6) / 24)
        temp += daily_variation
        
        # Random weather variations
        temp += np.random.normal(0, 2, n_points)
        
        return temp
    
    def _generate_enhanced_humidity_data(self, time_index: pd.DatetimeIndex, 
                                       temperature_data: pd.DataFrame) -> pd.DataFrame:
        """Generate humidity data correlated with temperature."""
        humidity_data = pd.DataFrame(index=time_index, columns=self.regions)
        
        for region in self.regions:
            print(f"   💧 Generating humidity for {region}")
            
            temp = temperature_data[region]
            
            # Base humidity (inverse correlation with temperature)
            base_humidity = 70 - 1.5 * (temp - 15)
            
            # Add daily variation
            hour = time_index.hour
            minute = time_index.minute
            fractional_hour = hour + minute / 60.0
            
            # Higher humidity at night
            daily_variation = 20 * np.sin(2 * np.pi * (fractional_hour + 6) / 24)
            
            humidity = base_humidity + daily_variation
            
            # Add noise
            humidity += np.random.normal(0, 5, len(time_index))
            
            # Constrain to realistic range
            humidity = np.clip(humidity, 20, 95)
            
            humidity_data[region] = humidity
        
        return humidity_data
    
    def _generate_enhanced_wind_speed_data(self, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate wind speed data for all regions."""
        wind_data = pd.DataFrame(index=time_index, columns=self.regions)
        
        # Base wind speeds for different regions
        base_winds = {
            'region_1_urban_metro': 5,       # Lower in urban areas
            'region_2_suburban_mixed': 7,
            'region_3_industrial_heavy': 6,
            'region_4_rural_agricultural': 10,
            'region_5_coastal_renewable': 15,  # Higher on coast
            'region_6_mountain_hydro': 12,    # Higher in mountains
            'region_7_desert_solar': 8,
            'region_8_mixed_commercial': 6
        }
        
        for region in self.regions:
            print(f"   💨 Generating wind speed for {region}")
            
            base_wind = base_winds.get(region, 8)
            
            # Generate wind profile
            wind_profile = self._generate_wind_profile(time_index, base_wind)
            wind_data[region] = wind_profile
        
        return wind_data
    
    def _generate_wind_profile(self, time_index: pd.DatetimeIndex, 
                              base_wind: float) -> np.ndarray:
        """Generate wind speed profile."""
        n_points = len(time_index)
        
        # Start with base wind
        wind = np.full(n_points, base_wind, dtype=float)
        
        # Seasonal variation (windier in winter)
        day_of_year = time_index.dayofyear
        seasonal_variation = 3 * np.sin(2 * np.pi * (day_of_year + 180) / 365.25)
        wind += seasonal_variation
        
        # Daily variation (windier during day)
        hour = time_index.hour
        minute = time_index.minute
        fractional_hour = hour + minute / 60.0
        
        daily_variation = 2 * np.sin(2 * np.pi * (fractional_hour - 3) / 24)
        wind += daily_variation
        
        # Add weather front effects (occasional high wind periods)
        wind_events = np.random.exponential(base_wind * 0.5, n_points)
        wind_events = np.where(np.random.random(n_points) < 0.05, wind_events, 0)
        wind += wind_events
        
        # Regular variation
        wind += np.random.normal(0, base_wind * 0.3, n_points)
        
        # Ensure non-negative
        wind = np.maximum(wind, 0.5)
        
        return wind

    def _generate_enhanced_solar_irradiance_data(self, time_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate explicit solar irradiance (W/m^2) based on solar geometry approximation.
        Uses simple declination/hour-angle model with per-region latitudes and stochastic cloud cover.
        """
        solar = pd.DataFrame(index=time_index, columns=self.regions, dtype=float)

        # Assign approximate latitudes per region (degrees)
        region_lats = {
            'region_1_urban_metro': 40.7,
            'region_2_suburban_mixed': 35.2,
            'region_3_industrial_heavy': 45.0,
            'region_4_rural_agricultural': 30.5,
            'region_5_coastal_renewable': 32.0,
            'region_6_mountain_hydro': 38.5,
            'region_7_desert_solar': 34.1,
            'region_8_mixed_commercial': 41.0
        }

        # Solar constant and scaling
        I_sc = 1000.0  # approximate clear-sky peak W/m^2

        # Pre-compute day of year and fractional hour
        day_of_year = time_index.dayofyear.values.astype(float)
        hour = time_index.hour.values.astype(float)
        minute = time_index.minute.values.astype(float)
        frac_hour = hour + minute / 60.0

        # Declination angle in radians (Cooper's formula)
        decl_deg = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365.25)
        decl = np.deg2rad(decl_deg)

        # Hour angle H in radians relative to solar noon (approx)
        H = np.deg2rad(15.0 * (frac_hour - 12.0))

        for region in self.regions:
            lat_deg = region_lats.get(region, 37.0)
            phi = np.deg2rad(lat_deg)

            # Solar elevation angle sin(alt)
            sin_alt = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(H)
            sin_alt = np.maximum(sin_alt, 0.0)  # no negative at night

            # Atmospheric/seasonal attenuation: simpler function of air mass proxy
            seasonal = 0.9 + 0.1 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)

            # Cloud cover stochastic factor (region-specific variability)
            rng = np.random.default_rng(abs(hash(region)) % (2**32))
            base_cloud = rng.uniform(0.85, 0.95)  # average clearness per region
            # Short-term variation
            cloud_variation = np.clip(np.random.normal(1.0, 0.05, len(time_index)), 0.7, 1.1)
            clearness = base_cloud * cloud_variation

            irradiance = I_sc * sin_alt * seasonal * clearness
            # Cap values to realistic range
            irradiance = np.clip(irradiance, 0.0, 1100.0)

            # Add small sensor noise
            irradiance += np.random.normal(0, 15.0, len(time_index))
            irradiance = np.clip(irradiance, 0.0, None)

            solar[region] = irradiance

        return solar
    
    def _validate_dataset_consistency(self, load_data: pd.DataFrame, 
                                    temperature_data: pd.DataFrame,
                                    humidity_data: pd.DataFrame, 
                                    wind_speed_data: pd.DataFrame):
        """Validate consistency across all datasets."""
        # Check shapes match
        shapes = [data.shape for data in [load_data, temperature_data, humidity_data, wind_speed_data]]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("Dataset shapes do not match")
        
        # Check indices match
        indices = [data.index for data in [load_data, temperature_data, humidity_data, wind_speed_data]]
        for i, index in enumerate(indices[1:]):
            if not index.equals(indices[0]):
                raise ValueError(f"Dataset {i+1} index does not match first dataset")
        
        # Check for NaN values
        for name, data in [('load', load_data), ('temperature', temperature_data), 
                          ('humidity', humidity_data), ('wind_speed', wind_speed_data)]:
            if data.isna().any().any():
                raise ValueError(f"{name} data contains NaN values")
        
        # Check realistic ranges
        if (load_data < 0).any().any():
            raise ValueError("Load data contains negative values")
        
        if (temperature_data < -50).any().any() or (temperature_data > 60).any().any():
            raise ValueError("Temperature data outside realistic range")
        
        if (humidity_data < 0).any().any() or (humidity_data > 100).any().any():
            raise ValueError("Humidity data outside realistic range")
        
        if (wind_speed_data < 0).any().any():
            raise ValueError("Wind speed data contains negative values")
        
        print("✅ Dataset consistency validation passed")
    
    def generate_deployment_batch(self, current_time: datetime = None, 
                                 hours_back: int = 24) -> Dict[str, pd.DataFrame]:
        """Generate a small batch of recent data for deployment/inference."""
        if current_time is None:
            current_time = datetime.now()
        
        # Round to nearest 15-minute interval
        minutes = (current_time.minute // 15) * 15
        current_time = current_time.replace(minute=minutes, second=0, microsecond=0)
        
        # Create time index for the last N hours
        start_time = current_time - timedelta(hours=hours_back)
        time_index = pd.date_range(
            start=start_time,
            end=current_time,
            freq='15min'
        )
        
        print(f"📊 Generating deployment batch: {len(time_index)} intervals")
        print(f"⏰ Time range: {time_index[0]} to {time_index[-1]}")
        
        # Generate data using same methods as full dataset
        load_data = self._generate_enhanced_load_data(time_index)
        temperature_data = self._generate_enhanced_temperature_data(time_index)
        humidity_data = self._generate_enhanced_humidity_data(time_index, temperature_data)
        wind_speed_data = self._generate_enhanced_wind_speed_data(time_index)
        solar_irradiance_data = self._generate_enhanced_solar_irradiance_data(time_index)
        
        return {
            'load_data': load_data,
            'temperature_data': temperature_data,
            'humidity_data': humidity_data,
            'wind_speed_data': wind_speed_data,
            'solar_irradiance_data': solar_irradiance_data
        }


# Test function
def test_enhanced_generator():
    """Test the enhanced data generator."""
    print("🧪 Testing Enhanced Data Generator...")
    
    # Test with small dataset
    generator = EnhancedDataGenerator(simulation_years=1)
    
    # Generate data
    data = generator.generate_complete_dataset()
    
    # Validate
    print(f"\n📊 Test Results:")
    print(f"   Load data shape: {data['load_data'].shape}")
    print(f"   Time span: {data['load_data'].index[0]} to {data['load_data'].index[-1]}")
    print(f"   Regions: {list(data['load_data'].columns)}")
    
    # Test deployment batch
    deployment_data = generator.generate_deployment_batch(hours_back=2)
    print(f"   Deployment batch shape: {deployment_data['load_data'].shape}")
    
    print("✅ Enhanced data generator test completed!")

if __name__ == "__main__":
    test_enhanced_generator()