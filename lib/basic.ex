defmodule Basic do
  import Nx.Defn

  defn predict({m, b}, x) do
    m * x + b
  end

  defn loss(params, x, y) do
    y_pred = predict(params, x)
    Nx.mean(Nx.power(y - y_pred, 2))
  end

  defn update({m, b} = params, inp, tar) do
    {grad_m, grad_b} = grad(params, &loss(&1, inp, tar))

    {
      m - grad_m * 0.01,
      b - grad_b * 0.01
    }
  end

  defn init_random_params do
    m = Nx.random_normal({}, 0.0, 0.1)
    b = Nx.random_normal({}, 0.0, 0.1)
    {m, b}
  end

  def build_data do
    target_m = :rand.normal(0.0, 10.0)
    target_b = :rand.normal(0.0, 5.0)
    target_fn = fn x -> target_m * x + target_b end

    Stream.repeatedly(fn -> for _ <- 1..100, do: :rand.uniform() * 10 end)
    |> Stream.map(fn x -> Enum.zip(x, Enum.map(x, target_fn)) end)
  end

  def train(epochs, data) do
    init_params = init_random_params()

    for _ <- 1..epochs, reduce: init_params do
      acc ->
        data
        |> Enum.take(1)
        |> Enum.reduce(
          acc,
          fn batch, cur_params ->
            {inp, tar} = Enum.unzip(batch)
            x = Nx.tensor(inp)
            y = Nx.tensor(tar)

            IO.inspect(batch, label: "batch")
            IO.inspect(x, label: "X")
            IO.inspect(y, label: "Y")

            update(cur_params, x, y)
          end
        )
    end
  end
end
