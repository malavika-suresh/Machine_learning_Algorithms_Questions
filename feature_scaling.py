from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
X = np.array([[1, 2], [3, 4], [5, 6]])

# Creating the scaler
scaler = StandardScaler()

# Fitting and transforming the data
X_scaled = scaler.fit_transform(X)

print("Original Data:\n", X)
print("Scaled Data:\n", X_scaled)
