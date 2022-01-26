defmodule NxLinearRegression do
  @moduledoc """
  `NxLinearRegression`.
  """
  import Nx.Defn

  defn predict({w, b}, x) do
    w * x + b
  end

  defn loss(params, x, y) do
    y_hat = predict(params, x)

    (y - y_hat)
    |> Nx.power(2)
    |> Nx.mean()
  end

  defn update({w, b} = params, x, y, lr) do
    {grad_w, grad_b} = grad(params, &loss(&1, x, y))

    {
      w - grad_w * lr,
      b - grad_b * lr
    }
  end

  defn init_random_params do
    w = Nx.random_normal({}, 0.0, 0.1)
    b = Nx.random_normal({}, 0.0, 0.1)
    {w, b}
  end

  defn normalize(data) do
    total = elem(data.shape, 0)

    mean = Nx.mean(data)

    std =
      (data - mean)
      |> Nx.power(2)
      |> Nx.sum()
      |> Nx.divide(total)
      |> Nx.sqrt()

    data
    |> Nx.subtract(mean)
    |> Nx.divide(std)
  end

  @spec train(data :: tuple(), lr :: float(), epochs :: integer()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def train(data, lr, epochs) do
    init_params = init_random_params()

    for _ <- 1..epochs, reduce: init_params do
      acc ->
        {x, y} = Enum.unzip(data)

        x = Nx.tensor(x) |> normalize()
        y = Nx.tensor(y) |> normalize()

        update(acc, x, y, lr)
    end
  end

  @spec train_test_split(data :: list(), train_size :: float()) :: tuple()
  def train_test_split(data, train_size) do
    num_examples = Enum.count(data)
    num_train = floor(train_size * num_examples)
    Enum.split(data, num_train)
  end
end
