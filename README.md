# NxLinearRegression

**Linear Regression example with Nx Library**

## How to run?

Try to predict the likely numbers of pizzas needed based on the number of reservations.

```elixir
# Train to obtain the params
params = NxLinearRegression.Pizzaria.train()

# Try to predict the number of pizzas for 12 reservations
NxLinearRegression.Pizzaria.predict params, 12
```

Learn more about [Nx](https://github.com/elixir-nx/nx)