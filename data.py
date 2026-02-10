# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 23:07:02 2026

@author: Nishant
"""

import os
import zipfile

print("📁 Your current directory:", os.getcwd())
print("\n📦 Files in your directory:")
for file in os.listdir():
    print(f"  - {file}")

# Check if your dataset exists
dataset_path = r"C:\Users\Nishant\Downloads\archive.zip"
if os.path.exists(dataset_path):
    print(f"\n✅ Dataset found at: {dataset_path}")
    
    # Let's see what's inside
    try:
        with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
            print("📋 Files inside archive.zip:")
            for file in zip_ref.namelist():
                print(f"  - {file}")
    except:
        print("Could not read ZIP file")
else:
    print(f"\n❌ Dataset NOT found at: {dataset_path}")