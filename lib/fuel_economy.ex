defmodule NxLinearRegression.FuelEconomy do
  @moduledoc """
  Predict Fuel Economy with Linear Regression
  https://www.kaggle.com/vinicius150987/regression-fuel-consumption
  """
  alias NimbleCSV.RFC4180, as: CSV

  @epochs 100
  @learning_rate 0.01

  def train do
    data = load_data()

    {train, test} = train_test_split(data, 0.8)

    params = NxLinearRegression.train(train, @learning_rate, @epochs)

    # IO.inspect NxLinearRegression.loss(params, x_test, y_test)

    # params
  end

  defp load_data do
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

  defp train_test_split(data, size) do
    num_examples = Enum.count(data)
    num_train = floor(size * num_examples)
    Enum.split(data, num_train)
  end
end
