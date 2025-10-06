import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import csv
import time
import os

def generate_synthetic_data(start_date, num_days=365):
    """Generate synthetic load data when actual data is not available"""
    data = []
    base_load = 3000  # Base load in MW
    daily_pattern = np.sin(np.linspace(0, 2*np.pi, 288)) * 500 + base_load
    
    for day in range(num_days):
        date = start_date + timedelta(days=day)
        for i, load in enumerate(daily_pattern):
            time = datetime.strptime('00:00:00', '%H:%M:%S') + timedelta(minutes=5*i)
            # Add some random variation
            varied_load = load * (1 + np.random.normal(0, 0.05))
            data.append([
                date.strftime('%d/%m/%Y') + ' ' + time.strftime('%H:%M:%S'),
                max(0, varied_load)  # Ensure no negative loads
            ])
    
    return data

def get_load_data(date, output_file):
    load = []
    url = 'http://www.delhisldc.org/Loaddata.aspx?mode='
    print(f'Scraping {date}', end=' ')
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    # Add delay between requests
    time.sleep(2)
    
    try:
        resp = requests.get(url + date, headers=headers, timeout=15)
        if resp.status_code != 200:
            print(f'Error: HTTP {resp.status_code}')
            return load
            
        soup = BeautifulSoup(resp.text, 'html.parser')
        table = soup.find('table', {'id': 'ContentPlaceHolder3_DGGridAv'})
        
        if table is None:
            print('Table not found')
            return load
            
        rows = table.find_all('tr')[1:]  # Skip header row
        if not rows:
            print('No data rows found')
            return load
            
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for row in rows:
                cells = row.find_all('font')
                if len(cells) >= 2:
                    time_cell, delhi_cell = cells[:2]
                    load.append(delhi_cell.text.strip())
                    writer.writerow([f"{date} {time_cell.text.strip()}", delhi_cell.text.strip()])
            
        if len(rows) != 288:
            print(f'Warning: Expected 288 values, got {len(rows)}')
        else:
            print('Done')
            
    except requests.RequestException as e:
        print(f'Network error: {str(e)}')
    except Exception as e:
        print(f'Error: {str(e)}')

    try:
        trs = table.findAll('tr')
        if len(trs[1:]):
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for tr in trs[1:]:
                    time, delhi = tr.findChildren('font')[:2]
                    load.append(delhi.text)
                    writer.writerow([date + ' ' + time.text, delhi.text])
        if len(trs[1:]) != 288:
            print('Some of the load values are missing..')
        else:
            print('Done')
    except Exception as e:
        print(f'Error occurred: {str(e)}')
    return load

def clean_and_save_data(input_file, output_file):
    # Data Cleaning
    data = pd.read_csv(input_file, header=None, names=['datetime', 'load'])
    data.dropna(inplace=True)
    data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True, format='mixed')
    data = data.sort_values('datetime')  # Sort by datetime
    
    # Remove duplicates if any
    data = data.drop_duplicates(subset=['datetime'])
    
    # Fill missing timestamps with interpolated values
    full_range = pd.date_range(start=data['datetime'].min(), 
                              end=data['datetime'].max(), 
                              freq='5min')
    data = data.set_index('datetime').reindex(full_range).interpolate()
    data = data.reset_index().rename(columns={'index': 'datetime'})
    
    # Save cleaned data
    data.to_csv(output_file, index=False)
    print(f"Data cleaned and saved to {output_file}")
    return data

# Function to generate dates between two dates
def generate_date_range(start_date, end_date):
    dates = []
    current_date = start_date
    while current_date <= end_date:
        dates.append(current_date)
        current_date += timedelta(days=1)
    return dates

# Setup date ranges
today = datetime.now()

# For 2024 data (full year)
start_date_2024 = datetime(2024, 1, 1)
end_date_2024 = datetime(2024, 12, 31)

# For 2025 data (up to today)
start_date_2025 = datetime(2025, 1, 1)
end_date_2025 = today

# Process 2024 data
print("Processing 2024 data...")
raw_file_2024 = 'temp_load_data_2024_raw.csv'
# Clear the file if exists
open(raw_file_2024, 'w').close()

for date in generate_date_range(start_date_2024, end_date_2024):
    get_load_data(date.strftime('%d/%m/%Y'), raw_file_2024)

clean_and_save_data(raw_file_2024, '../utilities/datasets/load_data_2024_cleaned.csv')

# Process 2025 data
print("\nProcessing 2025 data...")
raw_file_2025 = 'temp_load_data_2025_raw.csv'
# Clear the file if exists
open(raw_file_2025, 'w').close()

for date in generate_date_range(start_date_2025, end_date_2025):
    get_load_data(date.strftime('%d/%m/%Y'), raw_file_2025)

clean_and_save_data(raw_file_2025, '../utilities/datasets/load_data_2025_cleaned.csv')

# Clean up temporary files
import os
if os.path.exists(raw_file_2024):
    os.remove(raw_file_2024)
if os.path.exists(raw_file_2025):
    os.remove(raw_file_2025)

print("\nData processing completed!")
