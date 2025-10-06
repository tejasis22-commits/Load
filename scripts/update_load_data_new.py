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
    
    try:
        # Add delay between requests
        time.sleep(2)
        
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
        
    return load

def clean_and_save_data(input_file, output_file, year):
    try:
        if not os.path.exists(input_file) or os.path.getsize(input_file) == 0:
            print(f"No data found in {input_file}, generating synthetic data for {year}")
            # Generate synthetic data
            start_date = datetime(year, 1, 1)
            synthetic_data = generate_synthetic_data(start_date, 366 if year % 4 == 0 else 365)
            
            # Save synthetic data to input file
            with open(input_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(synthetic_data)
        
        # Read and clean data
        data = pd.read_csv(input_file, header=None, names=['datetime', 'load'])
        data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True, format='mixed')
        data = data.sort_values('datetime')
        data = data.drop_duplicates(subset=['datetime'])
        
        # Ensure all 5-minute intervals are present
        full_range = pd.date_range(
            start=datetime(year, 1, 1),
            end=datetime(year, 12, 31, 23, 59, 59) if year == 2024 else datetime.now(),
            freq='5min'
        )
        
        data = data.set_index('datetime').reindex(full_range).interpolate()
        data = data.reset_index().rename(columns={'index': 'datetime'})
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save cleaned data
        data.to_csv(output_file, index=False)
        print(f"Data saved to {output_file}")
        return data
        
    except Exception as e:
        print(f"Error in cleaning data: {str(e)}")
        return None

def main():
    # Create temp directory
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)

    # Process 2024 data
    print("\nProcessing 2024 data...")
    raw_file_2024 = os.path.join(temp_dir, 'load_data_2024_raw.csv')
    open(raw_file_2024, 'w').close()  # Clear/create file
    
    start_date_2024 = datetime(2024, 1, 1)
    end_date_2024 = datetime(2024, 12, 31)
    
    date_range = [start_date_2024 + timedelta(days=x) for x in range((end_date_2024 - start_date_2024).days + 1)]
    for date in date_range:
        get_load_data(date.strftime('%d/%m/%Y'), raw_file_2024)
    
    clean_and_save_data(raw_file_2024, '../utilities/datasets/load_data_2024_cleaned.csv', 2024)

    # Process 2025 data
    print("\nProcessing 2025 data...")
    raw_file_2025 = os.path.join(temp_dir, 'load_data_2025_raw.csv')
    open(raw_file_2025, 'w').close()  # Clear/create file
    
    start_date_2025 = datetime(2025, 1, 1)
    end_date_2025 = datetime.now()
    
    date_range = [start_date_2025 + timedelta(days=x) for x in range((end_date_2025 - start_date_2025).days + 1)]
    for date in date_range:
        get_load_data(date.strftime('%d/%m/%Y'), raw_file_2025)
    
    clean_and_save_data(raw_file_2025, '../utilities/datasets/load_data_2025_cleaned.csv', 2025)

    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    print("\nData processing completed!")

if __name__ == "__main__":
    main()
