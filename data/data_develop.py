import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)
# Create random data for regression model
n_samples = 100
# Features
X1 = np.random.rand(n_samples) * 10  # Random values between 0 and 10
X2 = np.random.rand(n_samples) * 5   # Random values between 0 and 5

# Response variable with some noise
y = 3.5 * X1 + 2.2 * X2 + np.random.randn(n_samples) * 2  # Linear combination with noise

# Create a DataFrame
df = pd.DataFrame({
    'Feature_1': X1,
    'Feature_2': X2,
    'Response': y
})

df.to_csv('/Users/kc/Desktop/Basic_zemi/data/sampledata.csv', index=False)