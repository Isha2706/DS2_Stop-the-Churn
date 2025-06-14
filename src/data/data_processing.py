import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

# Create directories if they don't exist
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def load_data(file_path='data/raw/telco_train.csv'):
    """
    Load the dataset from CSV file
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please make sure to place the telco_train.csv file in the data/raw directory.")
        return None

def explore_data(df):
    """
    Perform exploratory data analysis
    """
    if df is None:
        return
    
    # Display basic information
    print("\nBasic Information:")
    print(f"Dataset shape: {df.shape}")
    print("\nData types:")
    print(df.dtypes)
    
    # Check for missing values
    print("\nMissing values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    
    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe())
    
    # Check target variable distribution
    if 'Churn' in df.columns:
        print("\nTarget variable distribution:")
        churn_counts = df['Churn'].value_counts(normalize=True) * 100
        print(churn_counts)
        
        # Create a directory for plots
        os.makedirs('notebooks/plots', exist_ok=True)
        
        # Plot churn distribution
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Churn', data=df)
        plt.title('Churn Distribution')
        plt.savefig('notebooks/plots/churn_distribution.png')
        
        # Check correlation with numeric features
        numeric_df = df.select_dtypes(include=['int64', 'float64'])
        plt.figure(figsize=(12, 10))
        correlation = numeric_df.corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix')
        plt.savefig('notebooks/plots/correlation_matrix.png')
        
        # Explore relationship between categorical features and churn
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        for col in categorical_cols:
            if col != 'Churn' and col != 'customerID':
                plt.figure(figsize=(10, 6))
                churn_by_category = pd.crosstab(df[col], df['Churn'], normalize='index') * 100
                churn_by_category.plot(kind='bar', stacked=True)
                plt.title(f'Churn Rate by {col}')
                plt.ylabel('Percentage')
                plt.savefig(f'notebooks/plots/churn_by_{col}.png')
    
    return

def clean_data(df):
    """
    Clean the dataset by handling missing values, outliers, and data type conversions
    """
    if df is None:
        return None
    
    # Create a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()
    
    # Handle missing values
    print("\nHandling missing values...")
    for column in cleaned_df.columns:
        missing_count = cleaned_df[column].isnull().sum()
        if missing_count > 0:
            print(f"Column {column} has {missing_count} missing values.")
            
            # For numeric columns, fill with median
            if cleaned_df[column].dtype in ['int64', 'float64']:
                median_value = cleaned_df[column].median()
                cleaned_df[column].fillna(median_value, inplace=True)
                print(f"  - Filled with median value: {median_value}")
            
            # For categorical columns, fill with mode
            else:
                mode_value = cleaned_df[column].mode()[0]
                cleaned_df[column].fillna(mode_value, inplace=True)
                print(f"  - Filled with mode value: {mode_value}")
    
    # Convert data types
    print("\nConverting data types...")
    
    # Convert 'TotalCharges' to numeric if it's not already
    if 'TotalCharges' in cleaned_df.columns and cleaned_df['TotalCharges'].dtype == 'object':
        cleaned_df['TotalCharges'] = pd.to_numeric(cleaned_df['TotalCharges'], errors='coerce')
        # Fill any new NaN values created during conversion
        if cleaned_df['TotalCharges'].isnull().sum() > 0:
            cleaned_df['TotalCharges'].fillna(cleaned_df['TotalCharges'].median(), inplace=True)
    
    # Convert binary categorical variables to numeric
    binary_vars = ['Churn']
    for var in binary_vars:
        if var in cleaned_df.columns and cleaned_df[var].dtype == 'object':
            # Map 'Yes'/'No' to 1/0
            if set(cleaned_df[var].unique()) == {'Yes', 'No'}:
                cleaned_df[var] = cleaned_df[var].map({'Yes': 1, 'No': 0})
                print(f"  - Converted {var} to binary (1/0)")
    
    # Handle categorical variables
    categorical_cols = cleaned_df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if col != 'customerID':  # Skip ID columns
            # One-hot encode categorical variables
            print(f"  - One-hot encoding {col}")
            dummies = pd.get_dummies(cleaned_df[col], prefix=col, drop_first=True)
            cleaned_df = pd.concat([cleaned_df, dummies], axis=1)
            cleaned_df.drop(col, axis=1, inplace=True)
    
    # Handle outliers in numeric columns using IQR method
    print("\nHandling outliers...")
    numeric_cols = cleaned_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        if col not in ['Churn', 'customerID'] and 'ID' not in col:  # Skip target and ID variables
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers = ((cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)).sum()
            if outliers > 0:
                print(f"  - {col} has {outliers} outliers.")
                
                # Cap outliers instead of removing them
                cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                print(f"    - Capped outliers to range [{lower_bound:.2f}, {upper_bound:.2f}]")
    
    return cleaned_df

def split_data(df, target_col='Churn', test_size=0.2, random_state=42):
    """
    Split the dataset into training and validation sets
    """
    if df is None or target_col not in df.columns:
        print(f"Error: Cannot split data. Either dataframe is None or '{target_col}' column not found.")
        return None, None, None, None
    
    # Separate features and target
    X = df.drop([target_col, 'customerID'] if 'customerID' in df.columns else target_col, axis=1)
    y = df[target_col]
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    print(f"\nData split into training and validation sets:")
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Validation set: {X_val.shape[0]} samples")
    
    return X_train, X_val, y_train, y_val

def save_processed_data(X_train, X_val, y_train, y_val, output_dir='data/processed'):
    """
    Save the processed datasets to CSV files
    """
    if X_train is None or X_val is None or y_train is None or y_val is None:
        print("Error: Cannot save processed data. One or more datasets are None.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_val.to_csv(f"{output_dir}/y_val.csv", index=False)
    
    print(f"\nProcessed datasets saved to {output_dir}")

def main():
    # Load the dataset
    print("Step 1: Loading the dataset...")
    df = load_data()
    
    if df is not None:
        # Explore the data
        print("\nStep 2: Exploring the data...")
        explore_data(df)
        
        # Clean the data
        print("\nStep 3: Cleaning the data...")
        cleaned_df = clean_data(df)
        
        # Split the data
        print("\nStep 4: Splitting the data...")
        X_train, X_val, y_train, y_val = split_data(cleaned_df)
        
        # Save the processed data
        print("\nStep 5: Saving the processed data...")
        save_processed_data(X_train, X_val, y_train, y_val)
        
        print("\nData processing completed successfully!")
    else:
        print("\nData processing failed. Please check the file path and try again.")

if __name__ == "__main__":
    main()