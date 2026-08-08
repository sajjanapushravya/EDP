import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load Iris dataset
df = pd.read_csv("Iris.csv")

print("========== First 5 Rows ==========")
print(df.head())


# Feature (Input)
X = df[["SepalLengthCm"]]

# Label (Output)
y = df["PetalLengthCm"]


print("\n========== Features (X) ==========")
print(X.head())

print("\n========== Labels (y) ==========")
print(y.head())


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n========== Train-Test Split ==========")
print("Training data:", len(X_train))
print("Testing data:", len(X_test))


# Create Linear Regression model
model = LinearRegression()


# Train the model
model.fit(X_train, y_train)


# Predict values
y_pred = model.predict(X_test)


print("\n========== Predictions ==========")

for actual, predicted in zip(y_test, y_pred):
    print("Actual:", actual, "Predicted:", round(predicted, 2))


# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print("\n========== Evaluation Metrics ==========")

print("MAE:", mae)
print("MSE:", mse)
print("R2 Score:", r2)


# Linear Regression equation
print("\n========== Linear Regression Equation ==========")

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

print(
    "Equation: PetalLengthCm =",
    round(model.coef_[0], 2),
    "* SepalLengthCm +",
    round(model.intercept_, 2)
)