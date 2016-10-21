module TLLR

using Convex
using Logging

export TLLRegression, fit, predict

# Two Levels Linear Regression
type TLLRegression
  α::Variable
  β::Variable
  P::Array{Int,2}
  model::AbstractExpr
end

TLLRegression(α,β,P) = TLLRegression(α,β,P,(P .* (α * ones(1,size(P,2)))) * β)

function fit(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2})
  # row normalization
  X = X ./ sum(X,2)

  M = size(X,2)
  K = size(P,2)

  tll = TLLRegression(Variable(M, Positive()), Variable(K), P)

  # initialization of β to a random vector
  tll.β.value = rand(K)

  # termination criteria is guided by epsilon... when two iterations optvalue
  # differ for a quantity that is less than epslon, we stop
  ϵ = 0.01
  δ = 1.0

  previous_optval = 100

  i=0
  maxiterations = 5

  problem = minimize( norm(X * tll.model - y) )

  Logging.configure(level=INFO)

  while δ > ϵ && i < maxiterations
    info("Starting iteration n. $i")
    # First solve fix beta and solve for alpha
    fix!(tll.β)
    solve!(problem)
    free!(tll.β)

    # Then fix alpha and solve for beta
    fix!(tll.α)
    solve!(problem)
    free!(tll.α)

    δ = abs(previous_optval - problem.optval)
    previous_optval = problem.optval
    i+=1
  end

  return tll
end

function predict(tll::TLLRegression, X::Array{Float64,2})
  X = X ./ sum(X,2)
  return evaluate(X * tll.model)
end

end
