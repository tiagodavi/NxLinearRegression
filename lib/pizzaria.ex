defmodule NxLinearRegression.Pizzaria do
  @moduledoc """
  Try to predict the likely numbers of pizzas needed based on the number of reservations.

  Dataset:

    Reservations  Pizzas
    13            33
    2             16
    14            32
    23            51
    13            27
    1             16
    18            34
    10            17
    26            29
    3             15
    3             15
    21            32
    7             22
    22            37
    2             13
    27            44
    6             16
    10            21
    18            37
    15            30

    9             26
    26            34
    8             23
    15            39
    10            27
    21            37
    5             17
    6             18
    13            25
    13            23
  """

  @epochs 500
  @learning_rate 0.01

  @spec train :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def train do
    x_train = Nx.tensor([13, 2, 14, 23, 13, 1, 18, 10, 26, 3, 3, 21, 7, 22, 2, 27, 6, 10, 18, 15])

    y_train =
      Nx.tensor([33, 16, 32, 51, 27, 16, 34, 17, 29, 15, 15, 32, 22, 37, 13, 44, 16, 21, 37, 30])

    x_test = Nx.tensor([9, 26, 8, 15, 10, 21, 5, 6, 13, 13])
    y_test = Nx.tensor([26, 34, 23, 39, 27, 37, 17, 18, 25, 23])

    params = NxLinearRegression.train({x_train, y_train}, @learning_rate, @epochs)

    mse =
      NxLinearRegression.loss(params, x_test, y_test)
      |> Nx.to_number()
      |> Float.floor(2)

    # Log the MSE
    IO.inspect("MSE: #{mse}")

    params
  end

  @spec predict(params :: {Nx.Tensor.t(), Nx.Tensor.t()}, reservations :: integer()) :: integer()
  def predict(params, reservations) do
    params
    |> NxLinearRegression.predict(reservations)
    |> Nx.to_number()
    |> abs()
    |> round()
  end
end
