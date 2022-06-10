defmodule NxLinearRegression.FuelEconomy do
  @moduledoc """
  Try to predict the likely fuel consumption efficiency
  https://www.kaggle.com/vinicius150987/regression-fuel-consumption
  """

  # Functions to handle csv files
  # https://github.com/dashbitco/nimble_csv
  alias NimbleCSV.RFC4180, as: CSV

  # Set of useful machine learning functions
  # https://github.com/elixir-nx/scholar
  alias Scholar.Preprocessing

  # Let's define some defaults epochs and learning rate.
  @epochs 2000
  @learning_rate 0.1

  # This will call our internal training function with epochs and learning rate we defined above.
  @spec train(data :: tuple) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def train(data) do
    NxLinearRegression.train(data, @learning_rate, @epochs)
  end

  # This is going to predict based on the params previously learned.
  @spec predict(params :: tuple(), data :: list()) :: Nx.Tensor.t()
  def predict(params, data) do
    x =
      data
      |> Nx.tensor()
      |> Preprocessing.standard_scaler()

    NxLinearRegression.predict(params, x)
  end

  # This is going to calculate the MSE based on the params previously learned.
  @spec mse(params :: tuple(), data :: tuple()) :: Nx.Tensor.t()
  def mse(params, data) do
    {x, y} = Enum.unzip(data)

    x = Preprocessing.standard_scaler(Nx.tensor(x))
    y = Nx.tensor(y)

    NxLinearRegression.loss(params, x, y)
  end

  # This is going to load the data as streams.
  @spec load_data :: Stream.t()
  def load_data do
    "FuelEconomy.csv"
    |> File.stream!()
    |> CSV.parse_stream()
    |> Stream.map(fn [horse_power, fuel_economy] ->
      {
        Float.parse(horse_power) |> elem(0),
        Float.parse(fuel_economy) |> elem(0)
      }
    end)
  end
end
