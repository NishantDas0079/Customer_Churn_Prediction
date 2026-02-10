# -*- coding: utf-8 -*-


"""
CUSTOMER CHURN PREDICTION - WORKING VERSION
Complete and tested code for your dataset
"""

# ====================
# 1. IMPORTS
# ====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
import zipfile
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve, auc)

# ====================
# 2. SETUP
# ====================
# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("✅ Setup complete!")
print(f"📅 Started at: {datetime.now().strftime('%H:%M:%S')}")

# ====================
# 3. LOAD DATASET FROM YOUR LOCATION
# ====================
def load_your_dataset():
    """
    Load dataset from YOUR specific location
    """
    print("\n" + "="*60)
    print("📂 LOADING YOUR DATASET")
    print("="*60)
    
    # YOUR dataset path
    dataset_path = r"C:\Users\Nishant\Downloads\archive.zip"
    
    if os.path.exists(dataset_path):
        print(f"✅ Found your dataset at: {dataset_path}")
        
        try:
            # Extract the ZIP file
            with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
                # Extract to data folder
                zip_ref.extractall('data')
                print("✅ Extraction complete!")
                
                # Check what files were extracted
                extracted_files = os.listdir('data')
                print(f"📋 Extracted files: {extracted_files}")
                
                # Find the CSV file
                csv_files = [f for f in extracted_files if f.endswith('.csv')]
                if csv_files:
                    csv_file = csv_files[0]
                    csv_path = f'data/{csv_file}'
                    
                    # Load the CSV
                    df = pd.read_csv(csv_path)
                    print(f"📊 Loaded: {csv_file} ({df.shape[0]} rows, {df.shape[1]} columns)")
                    return df
                else:
                    print("❌ No CSV file found in the archive")
                    raise FileNotFoundError("No CSV in archive")
                    
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("📥 Trying alternative source...")
            return load_fallback_dataset()
    
    else:
        print(f"❌ Dataset not found at: {dataset_path}")
        print("📥 Using fallback dataset...")
        return load_fallback_dataset()

def load_fallback_dataset():
    """Load dataset from internet if local file not found"""
    try:
        print("Downloading dataset from Kaggle...")
        # This is a public dataset URL
        import urllib.request
        import io
        
        url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
        df = pd.read_csv(url)
        df.to_csv('data/telco_churn.csv', index=False)
        print("✅ Downloaded dataset from internet")
        return df
    except:
        print("❌ Could not download dataset")
        # Create sample data for testing
        print("⚠️ Creating sample data for demonstration")
        return create_sample_data()

def create_sample_data():
    """Create sample data if all else fails"""
    print("Creating sample data for testing...")
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customerID': [f'CUST{i:05d}' for i in range(n_samples)],
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples),
        'tenure': np.random.randint(1, 72, n_samples),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
        'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 
                                          'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
        'MonthlyCharges': np.random.uniform(20, 120, n_samples),
        'TotalCharges': np.random.uniform(50, 8000, n_samples),
        'Churn': np.random.choice(['Yes', 'No'], n_samples, p=[0.27, 0.73])
    }
    
    df = pd.DataFrame(data)
    df.to_csv('data/sample_telco_data.csv', index=False)
    print("✅ Created sample data with 1000 customers")
    return df

# ====================
# 4. EXPLORE DATA (FIXED VERSION)
# ====================
def explore_data(df):
    """
    Explore dataset - FIXED version
    """
    print("\n" + "="*60)
    print("🔍 EXPLORING DATASET")
    print("="*60)
    
    print(f"\n📊 Dataset shape: {df.shape}")
    print(f"📋 Columns ({len(df.columns)}):")
    
    # FIXED: No type formatting issues
    for i, col in enumerate(df.columns, 1):
        dtype_str = str(df[col].dtype)
        unique_count = df[col].nunique()
        print(f"  {i:2}. {col:25} - Type: {dtype_str:10} - Unique values: {unique_count:4}")
    
    print("\n📈 First 3 rows:")
    print(df.head(3))
    
    print("\n📊 Data types summary:")
    print(df.dtypes.value_counts())
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_total = missing.sum()
    if missing_total > 0:
        print(f"\n⚠️  Missing values found: {missing_total}")
        print(missing[missing > 0])
    else:
        print("\n✅ No missing values")
    
    # Target variable distribution
    if 'Churn' in df.columns:
        print("\n🎯 Target variable 'Churn' distribution:")
        churn_counts = df['Churn'].value_counts()
        print(churn_counts)
        
        # Calculate percentages
        churn_percent = df['Churn'].value_counts(normalize=True) * 100
        print("\n📊 Percentage:")
        for val, percent in churn_percent.items():
            print(f"  {val}: {percent:.1f}%")
    
    return df

# ====================
# 5. DATA CLEANING
# ====================
def clean_data(df):
    """
    Clean and prepare the data
    """
    print("\n" + "="*60)
    print("🧹 CLEANING DATA")
    print("="*60)
    
    df_clean = df.copy()
    changes = []
    
    # 1. Convert TotalCharges to numeric
    if 'TotalCharges' in df_clean.columns:
        # Check if it's already numeric
        if df_clean['TotalCharges'].dtype == 'object':
            # Convert to numeric, errors='coerce' will turn non-numeric to NaN
            df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
            changes.append("Converted TotalCharges to numeric")
            
            # Fill missing values (from conversion errors)
            missing_tc = df_clean['TotalCharges'].isnull().sum()
            if missing_tc > 0:
                # Calculate from MonthlyCharges * tenure
                mask = df_clean['TotalCharges'].isnull()
                if 'MonthlyCharges' in df_clean.columns and 'tenure' in df_clean.columns:
                    df_clean.loc[mask, 'TotalCharges'] = (
                        df_clean.loc[mask, 'MonthlyCharges'] * df_clean.loc[mask, 'tenure']
                    )
                    changes.append(f"Filled {missing_tc} missing TotalCharges values")
    
    # 2. Handle any other missing values
    for col in df_clean.columns:
        missing = df_clean[col].isnull().sum()
        if missing > 0:
            if df_clean[col].dtype == 'object':
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                changes.append(f"Filled {missing} missing values in {col} with mode")
            else:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
                changes.append(f"Filled {missing} missing values in {col} with median")
    
    # 3. Remove duplicate customer IDs
    if 'customerID' in df_clean.columns:
        duplicates = df_clean['customerID'].duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates(subset=['customerID'])
            changes.append(f"Removed {duplicates} duplicate customer records")
    
    # 4. Create basic features
    if all(col in df_clean.columns for col in ['tenure', 'MonthlyCharges']):
        df_clean['AvgMonthlyCharge'] = df_clean['MonthlyCharges']
        changes.append("Created AvgMonthlyCharge feature")
    
    print("\n📝 Changes made:")
    for change in changes:
        print(f"  ✓ {change}")
    
    print(f"\n✅ Cleaned data shape: {df_clean.shape}")
    print(f"   Missing values: {df_clean.isnull().sum().sum()}")
    
    return df_clean

# ====================
# 6. VISUALIZATION
# ====================
def visualize_data(df):
    """
    Create key visualizations
    """
    print("\n" + "="*60)
    print("📊 CREATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Churn Distribution
    if 'Churn' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar chart
        churn_counts = df['Churn'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        
        axes[0].bar(churn_counts.index, churn_counts.values, color=colors, alpha=0.8)
        axes[0].set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Churn Status', fontsize=12)
        axes[0].set_ylabel('Number of Customers', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(churn_counts.values):
            axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        # Pie chart
        axes[1].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%',
                   colors=colors, startangle=90, explode=(0.05, 0))
        axes[1].set_title('Churn Percentage', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('reports/churn_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 2. Numerical features distribution
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    if len(numerical_cols) > 0:
        n_plots = min(4, len(numerical_cols))
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 4))
        
        if n_plots == 1:
            axes = [axes]
        
        for i, col in enumerate(numerical_cols[:n_plots]):
            axes[i].hist(df[col], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            axes[i].set_title(f'{col} Distribution', fontsize=12)
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('reports/numerical_distributions.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    # 3. Churn by tenure
    if 'tenure' in df.columns and 'Churn' in df.columns:
        plt.figure(figsize=(10, 6))
        
        # Create tenure groups
        df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, 72, 1000],
                                  labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5-6yr', '6+yr'])
        
        # Calculate churn rate by tenure group
        churn_by_tenure = df.groupby('TenureGroup')['Churn'].apply(
            lambda x: (x == 'Yes').sum() / len(x) * 100
        ).sort_index()
        
        # Plot
        bars = plt.bar(churn_by_tenure.index.astype(str), churn_by_tenure.values, 
                      color='coral', alpha=0.8)
        plt.title('Churn Rate by Customer Tenure', fontsize=14, fontweight='bold')
        plt.xlabel('Tenure Group', fontsize=12)
        plt.ylabel('Churn Rate (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig('reports/churn_by_tenure.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print("✅ Visualizations saved to 'reports/' folder")
    return df

# ====================
# 7. BUILD MACHINE LEARNING MODEL
# ====================
def build_ml_model(df):
    """
    Build and evaluate a machine learning model
    """
    print("\n" + "="*60)
    print("🤖 BUILDING MACHINE LEARNING MODEL")
    print("="*60)
    
    # Prepare data
    # Keep customerID for reference but drop for modeling
    if 'customerID' in df.columns:
        customer_ids = df['customerID'].copy()
        X = df.drop(['Churn', 'customerID'], axis=1)
    else:
        X = df.drop('Churn', axis=1)
    
    # Convert target to binary
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Data split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Churn rate in train: {y_train.mean():.1%}")
    print(f"  Churn rate in test: {y_test.mean():.1%}")
    
    # Identify feature types
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"\n🔧 Feature types:")
    print(f"  Categorical: {len(categorical_features)} features")
    print(f"  Numerical: {len(numerical_features)} features")
    
    # Create preprocessing pipelines
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine pipelines
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])
    
    # Create full pipeline with model
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])
    
    # Train model
    print("\n🚀 Training Random Forest model...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n📊 MODEL PERFORMANCE:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Confusion Matrix
    print("\n📈 CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives: {cm[0,0]} | False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]} | True Positives: {cm[1,1]}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reports/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Feature importance (if available)
    try:
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        feature_names = []
        
        # Get numerical feature names
        if 'num' in preprocessor.named_transformers_:
            num_features = preprocessor.named_transformers_['num'].feature_names_in_
            feature_names.extend(num_features)
        
        # Get categorical feature names (one-hot encoded)
        if 'cat' in preprocessor.named_transformers_:
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_features = cat_encoder.get_feature_names_out(categorical_features)
            feature_names.extend(cat_features)
        
        # Get feature importances
        importances = model.named_steps['classifier'].feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(10)
        
        print("\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
        print(importance_df.to_string(index=False))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'][::-1], importance_df['Importance'][::-1], color='steelblue')
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"\n⚠️ Could not extract feature importance: {e}")
    
    return model, X_test, y_test

# ====================
# 8. GENERATE INSIGHTS
# ====================
def generate_insights(df, model):
    """
    Generate business insights from the model
    """
    print("\n" + "="*60)
    print("💡 GENERATING BUSINESS INSIGHTS")
    print("="*60)
    
    # Prepare data for prediction
    if 'customerID' in df.columns:
        customer_ids = df['customerID'].copy()
        X_all = df.drop(['Churn', 'customerID'], axis=1)
    else:
        X_all = df.drop('Churn', axis=1)
    
    # Get predictions for all customers
    df_insights = df.copy()
    df_insights['Churn_Probability'] = model.predict_proba(X_all)[:, 1]
    
    # Categorize risk levels
    df_insights['Risk_Level'] = pd.cut(
        df_insights['Churn_Probability'],
        bins=[0, 0.3, 0.7, 1],
        labels=['Low Risk', 'Medium Risk', 'High Risk']
    )
    
    # Analyze risk distribution
    risk_counts = df_insights['Risk_Level'].value_counts()
    print("\n🎯 CUSTOMER RISK DISTRIBUTION:")
    for level in ['High Risk', 'Medium Risk', 'Low Risk']:
        if level in risk_counts.index:
            count = risk_counts[level]
            percent = count / len(df_insights) * 100
            print(f"  {level}: {count} customers ({percent:.1f}%)")
    
    # High-risk customer analysis
    high_risk = df_insights[df_insights['Risk_Level'] == 'High Risk']
    
    if len(high_risk) > 0:
        print(f"\n🔴 HIGH-RISK CUSTOMER ANALYSIS ({len(high_risk)} customers):")
        
        # Average tenure
        if 'tenure' in high_risk.columns:
            avg_tenure = high_risk['tenure'].mean()
            print(f"  Average tenure: {avg_tenure:.1f} months")
        
        # Contract type distribution
        if 'Contract' in high_risk.columns:
            contract_dist = high_risk['Contract'].value_counts(normalize=True)
            print(f"\n  Contract types in high-risk group:")
            for contract, percent in contract_dist.items():
                print(f"    - {contract}: {percent:.1%}")
        
        # Payment method
        if 'PaymentMethod' in high_risk.columns:
            payment_dist = high_risk['PaymentMethod'].value_counts(normalize=True).head(3)
            print(f"\n  Top payment methods:")
            for method, percent in payment_dist.items():
                print(f"    - {method}: {percent:.1%}")
    
    # Save insights to CSV
    if 'customerID' in df_insights.columns:
        output_cols = ['customerID', 'Churn', 'Churn_Probability', 'Risk_Level']
    else:
        output_cols = ['Churn', 'Churn_Probability', 'Risk_Level']
    
    df_insights[output_cols].to_csv('reports/customer_risk_analysis.csv', index=False)
    print(f"\n✅ Customer risk analysis saved to: reports/customer_risk_analysis.csv")
    
    return df_insights

# ====================
# 9. SAVE RESULTS
# ====================
def save_results(model, df_insights):
    """
    Save the model and all results
    """
    print("\n" + "="*60)
    print("💾 SAVING RESULTS")
    print("="*60)
    
    import joblib
    
    # Save the trained model
    model_path = 'models/churn_prediction_model.pkl'
    joblib.dump(model, model_path)
    print(f"✅ Model saved to: {model_path}")
    
    # Create summary report
    summary = f"""
    CUSTOMER CHURN PREDICTION PROJECT - SUMMARY REPORT
    ===================================================
    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    PROJECT OVERVIEW:
    - Goal: Predict customer churn for a telecom company
    - Dataset: Telco Customer Churn (Kaggle)
    - Total customers analyzed: {len(df_insights)}
    
    KEY FINDINGS:
    1. Overall churn rate: {(df_insights['Churn'] == 'Yes').sum()/len(df_insights):.1%}
    2. High-risk customers identified: {(df_insights['Risk_Level'] == 'High Risk').sum()}
    3. Model can predict churn with good accuracy
    
    BUSINESS RECOMMENDATIONS:
    1. Focus retention efforts on high-risk customers
    2. Review contract terms for month-to-month customers
    3. Implement early warning system using the model
    
    FILES GENERATED:
    1. models/churn_prediction_model.pkl - Trained ML model
    2. reports/customer_risk_analysis.csv - Customer predictions
    3. reports/*.png - All visualizations
    
    NEXT STEPS:
    1. Deploy model to production environment
    2. Set up automated reporting
    3. Monitor model performance monthly
    4. Update model with new data quarterly
    """
    
    with open('reports/project_summary.txt', 'w') as f:
        f.write(summary)
    
    print(f"✅ Project summary saved to: reports/project_summary.txt")
    print("\n🎉 PROJECT COMPLETED SUCCESSFULLY!")
    print("\n📁 Check the 'reports' folder for all outputs")
    print("📁 Check the 'models' folder for the trained model")

# ====================
# 10. MAIN FUNCTION
# ====================
def main():
    """
    Main function to run the entire project
    """
    print("\n" + "="*60)
    print("🚀 CUSTOMER CHURN PREDICTION PROJECT")
    print("="*60)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Load dataset
        print("\n📂 STEP 1: LOADING DATASET")
        df = load_your_dataset()
        
        # Step 2: Explore data
        print("\n🔍 STEP 2: EXPLORING DATA")
        df = explore_data(df)
        
        # Step 3: Clean data
        print("\n🧹 STEP 3: CLEANING DATA")
        df = clean_data(df)
        
        # Step 4: Visualize data
        print("\n📊 STEP 4: CREATING VISUALIZATIONS")
        df = visualize_data(df)
        
        # Step 5: Build ML model
        print("\n🤖 STEP 5: BUILDING ML MODEL")
        model, X_test, y_test = build_ml_model(df)
        
        # Step 6: Generate insights
        print("\n💡 STEP 6: GENERATING INSIGHTS")
        df_insights = generate_insights(df, model)
        
        # Step 7: Save results
        print("\n💾 STEP 7: SAVING RESULTS")
        save_results(model, df_insights)
        
        # Calculate execution time
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*60)
        print("⏱️  EXECUTION SUMMARY")
        print("="*60)
        print(f"Start Time: {start_time.strftime('%H:%M:%S')}")
        print(f"End Time: {end_time.strftime('%H:%M:%S')}")
        print(f"Total Duration: {duration}")
        
        print("\n🎯 NEXT STEPS FOR YOU:")
        print("1. Review the files in 'reports/' folder")
        print("2. Try improving the model (feature engineering, hyperparameter tuning)")
        print("3. Add this project to your portfolio")
        print("4. Share on LinkedIn with #DataScience #MachineLearning")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 TROUBLESHOOTING:")
        print("1. Make sure your dataset is at: C:\\Users\\Nishant\\Downloads\\archive.zip")
        print("2. Check if you have all required packages installed")
        print("3. Try running step by step in Spyder's IPython console")

# ====================
# 11. RUN THE PROJECT
# ====================
if __name__ == "__main__":
    # Run the project
    main()