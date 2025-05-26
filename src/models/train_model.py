import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = [
    {"timestamp": "2025-04-16 00:00:00", "lag_1": 84523.4524914945, "price": 83656.49248858042, "rolling_mean_3": 83926.92169354012, "rolling_mean_7": 83249.09688042219, "rolling_std_3": 517.360221101683, "rolling_std_7": 1821.7444130688768},
    ]
df = pd.read_csv('data/features/crypto_features.csv')
df = df.drop(columns=['timestamp'])
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Data Preprocessing Complete.")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")








