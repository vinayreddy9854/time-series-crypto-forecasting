# 1. Data Preparation (Preprocessing)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load your data (replace this part with your actual data loading code)
data = [
    {"timestamp": "2025-04-16 00:00:00", "lag_1": 84523.4524914945, "price": 83656.49248858042, "rolling_mean_3": 83926.92169354012, "rolling_mean_7": 83249.09688042219, "rolling_std_3": 517.360221101683, "rolling_std_7": 1821.7444130688768},
    # Add other rows here...
]

# Convert to DataFrame
df = pd.read_csv('data/features/crypto_features.csv')

# Drop the timestamp column as it's not needed for prediction
df = df.drop(columns=['timestamp'])

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['price'])
y = df['price']

# Split the data into training and testing datasets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Data Preprocessing Complete.")

# 2. Model Training and Evaluation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")








