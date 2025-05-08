import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 2: Load the dataset (assuming the dataset is named 'indian_house_prices.csv')
df = pd.read_csv('indian_house_prices.csv')


# Step 3: Explore the dataset
print(df.head())

# Step 4: Data Preprocessing
# Handle missing values if any
df = df.dropna()

# Encode categorical columns (e.g., 'Location') using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Check the columns after encoding
print(df.head())

# Step 5: Split the dataset into features (X) and target (y)
X = df.drop('Price', axis=1)
y = df['Price']

# Step 6: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Standardize the data (optional but recommended for models like Linear Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 8: Initialize the Linear Regression model
model = LinearRegression()

# Step 9: Train the model
model.fit(X_train_scaled, y_train)

# Step 10: Make predictions on the test set
y_pred = model.predict(X_test_scaled)

# Step 11: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Output the evaluation results
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Step 12: Plot the actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices (India)')
plt.show()
