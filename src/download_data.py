import pandas as pd
from sklearn.datasets import fetch_california_housing
import os

def download_data():
    """Download California housing dataset and save as CSV"""
    print("Downloading California housing dataset...")
    
    # Fetch the dataset
    housing = fetch_california_housing(as_frame=True)
    
    # Create DataFrame with all features and target
    df = housing.frame
    
    # Rename target column to match expected name
    df = df.rename(columns={'MedHouseVal': 'MedHouseVal'})
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/housing.csv'
    df.to_csv(output_path, index=False)
    
    print(f"Data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    download_data()
