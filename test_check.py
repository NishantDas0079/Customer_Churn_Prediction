# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 23:32:50 2026

@author: Nishant
"""

import os
import glob

# Check current directory
print(f"📂 Current directory: {os.getcwd()}")

# Check for reports folder
reports_path = 'reports'
if os.path.exists(reports_path):
    print(f"✅ Found reports folder")
    
    # List all files
    print("\n📋 Files in reports folder:")
    files = glob.glob(os.path.join(reports_path, '*'))
    for file in files:
        size_kb = os.path.getsize(file) / 1024
        print(f"  📄 {os.path.basename(file)} - {size_kb:.1f} KB")
    
    # Count by type
    png_files = glob.glob(os.path.join(reports_path, '*.png'))
    csv_files = glob.glob(os.path.join(reports_path, '*.csv'))
    txt_files = glob.glob(os.path.join(reports_path, '*.txt'))
    
    print(f"\n📊 Summary:")
    print(f"  📈 Images (.png): {len(png_files)} files")
    print(f"  📊 Data (.csv): {len(csv_files)} files")
    print(f"  📝 Reports (.txt): {len(txt_files)} files")
    
else:
    print(f"❌ Reports folder not found!")
    print("\n💡 Creating it now...")
    os.makedirs(reports_path, exist_ok=True)
    print("✅ Created reports folder")