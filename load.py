# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 23:00:12 2026
"""

import os
import zipfile

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# If you have the zip file, extract it
zip_path = 'WA_Fn-UseC_-Telco-Customer-Churn.zip'
csv_path = 'data/telco_churn.csv'

if os.path.exists(zip_path):
    print("📦 Extracting ZIP file...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('data')
    print("✅ Extraction complete!")
    
    # Rename if needed
    extracted_files = os.listdir('data')
    for file in extracted_files:
        if file.endswith('.csv'):
            os.rename(f'data/{file}', csv_path)
            print(f"📄 Renamed to: {csv_path}")