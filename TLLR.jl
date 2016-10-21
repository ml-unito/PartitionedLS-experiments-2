module TLLR

using Convex
using Logging

import Base.size
export TLLRegression, fit, predict, alpha, beta, optval

"""
    type TLLRegression

The type of Two Levels Linear Regression models.

A TLLR model is a linear model has the form:
  \$h(x) = \sum_k \beta_k f_{k}(x)\$

where \$f_\beta(x)\$ are pseudo features built on the original \$x\$
features:
  \$f_{\beta_k}(x) = \sum_{m: P_{m,k} = 1} \alpha_m x_m\$
"""
type TLLRegression
  α::Variable
  β::Variable
  P::Array{Int,2}
  model::AbstractExpr
  problem::Problem

  function TLLRegression(P)
    M, K = size(P)
    tll = new()
    tll.α = Variable(M, Positive())
    tll.β = Variable(K)
    tll.P = P
    tll.model = (tll.P .* (tll.α * ones(1,size(tll.P,2)))) * tll.β

    tll
  end
end

# tll.α.value is a matrix,
function alpha(tll::TLLRegression)
  tll.α.value[:,1]
end

function beta(tll::TLLRegression)
  tll.β.value[:,1]
end

function size(tll::TLLRegression)
  size(tll.P)
end

optval(tll::TLLRegression) = tll.problem.optval

function solve_for(tll::TLLRegression, var::Symbol)
  info("solving for $(var)")
  info("before learning $(var): $(getfield(tll,var).value)")

  fix!(getfield(tll,var))
  solve!(tll.problem)
  free!(getfield(tll, var))

  info("after learning (status: $(tll.problem.status))$(var): $(getfield(tll,var).value)")
end

"""
    fit(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; beta=randomvalues)

Fits a TLLRegression model to the given data and resturns the
learnt model (`::TLLRegression`).

# Arguments

* `X`: NxM matrix describing the examples
* `y`: N vector with the output values for each examples
* `P`: MxK matrix specifying how to partition the M attributes into
    K subsets. P(m,k) should be 1 if attribute number m belongs to
    partition k.
* `beta`: Initial value for betas

"""
function fit(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; beta = rand(size(P,2)))

  Logging.configure(output=open("tllr.log", "a"), level=INFO)
  info("------------ FIT ---------------")

  # row normalization
  X = X ./ sum(X,2)

  tll = TLLRegression(P)
  tll.β.value = beta
  M,K = size(tll)

  # termination criteria is guided by epsilon... when two iterations optvalue
  # differ for a quantity δ that is less than epslon, we stop
  ϵ = 0.001
  δ = 1.0

  previous_optval = 100
  i=0
  maxiterations = 5

  tll.problem = minimize( norm(X * tll.model - y) )

  while δ > ϵ && i < maxiterations
    info("Starting iteration n. $i")
    solve_for(tll, :β)
    solve_for(tll, :α)

    δ = abs(previous_optval - optval(tll))
    previous_optval = optval(tll)
    i+=1
  end

  return tll
end

function predict(tll::TLLRegression, X::Array{Float64,2})
  X = X ./ sum(X,2)
  return evaluate(X * tll.model)
end

end
