import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load 2024 data as base
data_2024 = pd.read_csv('../utilities/datasets/load_data_2024_cleaned.csv')
data_2024['datetime'] = pd.to_datetime(data_2024['datetime'])

# Create 2025 dates
data_2025 = data_2024.copy()
data_2025['datetime'] = data_2025['datetime'] + pd.DateOffset(years=1)

# Add some variation to the load values (assuming 5% yearly growth plus some random variation)
data_2025['load'] = data_2025['load'] * 1.05 + np.random.normal(0, 50, len(data_2025))

# Ensure no negative values
data_2025['load'] = data_2025['load'].clip(lower=0)

# Save the generated data
data_2025.to_csv('../utilities/datasets/load_data_2025_cleaned.csv', index=False)
