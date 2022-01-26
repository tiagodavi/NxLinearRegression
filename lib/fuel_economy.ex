defmodule NxLinearRegression.FuelEconomy do
  @moduledoc """
  Try to predict the likely fuel consumption efficiency
  https://www.kaggle.com/vinicius150987/regression-fuel-consumption
  """
  alias NimbleCSV.RFC4180, as: CSV

  @epochs 500
  @learning_rate 0.01

  @spec train(data :: tuple) :: {Nx.Tensor.t(), Nx.Tensor.t()}
  def train(data) do
    NxLinearRegression.train(data, @learning_rate, @epochs)
  end

  @spec predict(params :: tuple(), data :: list()) :: Nx.Tensor.t()
  def predict(params, data) do
    x =
      data
      |> Nx.tensor()
      |> NxLinearRegression.normalize()

    NxLinearRegression.predict(params, x)
  end

  @spec mse(params :: tuple(), data :: tuple()) :: Nx.Tensor.t()
  def mse(params, data) do
    {x, y} = Enum.unzip(data)

    x = Nx.tensor(x) |> NxLinearRegression.normalize()
    y = Nx.tensor(y) |> NxLinearRegression.normalize()

    NxLinearRegression.loss(params, x, y)
  end

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
