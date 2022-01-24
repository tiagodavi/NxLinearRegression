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
    clipped_grad_w = Nx.clip(grad_w, -2.0, 2.0)
    clipped_grad_b = Nx.clip(grad_b, -2.0, 2.0)

    {
      w - clipped_grad_w * lr,
      b - clipped_grad_b * lr
    }
  end

  defn init_random_params do
    w = Nx.random_normal({}, 0.0, 0.1)
    b = Nx.random_normal({}, 0.0, 0.1)
    {w, b}
  end

  @spec train(data :: tuple(), lr :: float(), epochs :: integer(), params :: tuple()) ::
          {Nx.Tensor.t(), Nx.Tensor.t()}
  def train({x, y} = data, lr, epochs, params \\ init_random_params()) do
    if epochs > 0 do
      updated_params = update(params, x, y, lr)

      train(data, lr, epochs - 1, updated_params)
    else
      params
    end
  end
end
