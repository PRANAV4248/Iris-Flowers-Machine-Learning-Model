from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
import joblib

# Loading the dataset
X, y = load_iris(return_X_y=True, as_frame=True)

# Training the model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Saving the model
joblib.dump(model, "model.pkl")
print("Model is successfully trained and saved.")