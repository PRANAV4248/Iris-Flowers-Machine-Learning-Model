from sklearn.datasets import load_iris
import pandas as pd

# Load iris as DataFrame
X, y = load_iris(return_X_y=True, as_frame=True)

# Calculate min and max for each column
min_max = pd.DataFrame({
    "min": X.min(),
    "max": X.max()
})

print(min_max)