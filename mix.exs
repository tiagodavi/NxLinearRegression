defmodule NxLinearRegression.MixProject do
  use Mix.Project

  def project do
    [
      app: :nx_linear_regression,
      version: "0.1.0",
      elixir: "~> 1.13",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.2.1"},
      {:nimble_csv, "~> 1.1"},
      {:scholar, "~> 0.1.0", github: "elixir-nx/scholar"}
    ]
  end
end
