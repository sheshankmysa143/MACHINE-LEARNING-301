import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = {
    "Weight": [79, 69, 73, 95, 82, 55, 69, 71, 64, 69],
    "Height": [1.80, 1.68, 1.82, 1.70, 1.87, 1.55, 1.50, 1.78, 1.67, 1.64],
    "Age": [35, 39, 25, 60, 27, 18, 89, 42, 16, 52],
    "Gender": ["Male", "Male", "Male", "Male", "Male", "Female", "Female", "Female", "Female", "Female"]
}

df = pd.DataFrame(data)

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])

X = df[["Height", "Age", "Gender"]]
y = df["Weight"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Actual Weights:")
print(y_test.values)

print("Predicted Weights:")
print(y_pred)

print("MAE =", mean_absolute_error(y_test, y_pred))
print("MSE =", mean_squared_error(y_test, y_pred))
print("R2 Score =", r2_score(y_test, y_pred))

new_data = pd.DataFrame({
    "Height": [1.75],
    "Age": [30],
    "Gender": [1]
})

prediction = model.predict(new_data)

print("Predicted Weight =", prediction[0])