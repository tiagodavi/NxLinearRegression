# NxLinearRegression

**Linear Regression example with Nx Library**

## How to run?

Try to predict the likely fuel consumption efficiency

DataSet: https://www.kaggle.com/vinicius150987/regression-fuel-consumption

```elixir
# Get Dependencies
mix deps.get

# Run Elixir
iex -S mix

# Load Data
data = NxLinearRegression.FuelEconomy.load_data()

# Split into 80% for training and 20% for testing
{train, test} = NxLinearRegression.train_test_split(data, 0.8)

# Train the model and obtain the params
params = NxLinearRegression.FuelEconomy.train(train)

# Calculate MSE
# https://en.wikipedia.org/wiki/Mean_squared_error
mse = NxLinearRegression.FuelEconomy.mse(params, test)

# Get the test data
{x_test, _y_test} = Enum.unzip(test)

# Predict some values
# https://findanyanswer.com/what-is-the-difference-between-y-and-y-hat
y_hat = NxLinearRegression.FuelEconomy.predict(params, x_test)
```

Learn more about [Nx](https://github.com/elixir-nx/nx)