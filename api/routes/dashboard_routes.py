from flask import Blueprint, jsonify
import pandas as pd
import os

dashboard_bp = Blueprint('dashboard', __name__)

# Initialize dataframe variable
df = None

# Correct the file path to match the actual file location
csv_path = 'data/features/crypto_features.csv'

# Check if the CSV file exists
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print(f"Data loaded successfully with {len(df)} rows.")
else:
    print(f"Error: {csv_path} not found.")



@dashboard_bp.route('/dashboard/historical', methods=['GET'])
def historical_data():
    try:
        if df is None:
            return jsonify({'error': 'Historical data file not found.'}), 404
        
        print(f"Data sample:\n{df.head()}")
        
        recent_data = df.tail(100)
        return jsonify(recent_data.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500



