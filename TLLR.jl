module TLLR

using Convex
using Logging
using SCS
using ECOS

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
  t::Variable

  model::AbstractExpr
  problem::Problem
  loss::AbstractExpr
  objective::AbstractExpr
  constraints::Array{Constraint,1}
  regularization::AbstractExpr

  P::Array{Int,2}
  X::Array{Float64,2}
  y::Array{Float64,1}

  verbose::Int

  function TLLRegression(P, X, y; verbose=0)
    M, K = size(P)
    tll = new()
    tll.α = Variable(M)
    tll.β = Variable(K)
    tll.t = Variable()
    tll.P = P
    tll.X = X
    tll.y = y
    tll.verbose = verbose

    setmodel!(tll, tll.α, tll.β)

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

function evalloss(tll::TLLRegression)
  evaluate(tll.loss)
end

function evalobjective(tll::TLLRegression)
  evaluate(tll.objective)
end

optval(tll::TLLRegression) = tll.problem.optval

function setmodel!(tll::TLLRegression, α, β)
  tll.regularization =  0.0 * vecnorm(β,1)
  if typeof(α) == Variable
    tll.constraints = [α >= 0, sum(α' * tll.P, 1) == 1]
  else
    tll.constraints = []
  end

  tll.model = (tll.P .* (α * ones(1,size(tll.P,2)))) * β
  tll.loss = norm(tll.X * tll.model + tll.t - tll.y)^2
  tll.objective = tll.loss + tll.regularization
  tll.problem = minimize(  tll.objective, tll.constraints )

  # tll.β.value = ones(size(tll.β))
  # tll.model = α
  # tll.loss = norm(tll.X * tll.model + tll.t - tll.y)
  # tll.objective = tll.loss
  # tll.problem = minimize( tll.objective )
end

function tllsolve!(tll::TLLRegression, freevarname; verbose=0)
  fixvar_symbol = setdiff([:α,:β],[freevarname])[1]
  fixvar = getfield(tll, fixvar_symbol)
  freevar = getfield(tll, freevarname)

  info("solving for $freevarname")
  info("before learning $freevarname: $(freevar.value)")
  info("$freevarname is free: $(!(:fixed in freevar.sets))")

  fix!(fixvar)
  Convex.solve!(tll.problem, SCSSolver(verbose=verbose))
  free!(fixvar)

  info("after learning (status: $(tll.problem.status)) $(freevarname): $(freevar)")
end

function tllsolve2!(tll::TLLRegression, freevarname; verbose=0)
  if(freevarname==:α)
    setmodel!(tll, tll.α, tll.β.value )
  else
    setmodel!(tll, tll.α.value, tll.β )
  end

  Convex.solve!(tll.problem, ECOSSolver(max_iters=1000000, verbose=verbose))
  println("solving for $freevarname problem status: $(tll.problem.status)")
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
function fit(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; beta = rand(size(P,2)), verbose=0)

  # Logging.configure(output=open("tllr.log", "a"), level=INFO)
  Logging.configure(level=INFO)
  info("------------ FIT ---------------")

  # row normalization
  X = X ./ sum(X,2)

  # FIXME: aggiungere termine noto

  tll = TLLRegression(P, X, y, verbose=verbose)

  M,K = size(tll)

  tll.α.value = rand(M)
  tll.β.value = beta

  maxiterations = 1000
  i = 0

  while i < maxiterations
    info("Starting iteration n. $i")
    # tllsolve2!(tll, :α, verbose=verbose)
    tllsolve2!(tll, :β, verbose=verbose)
    tllsolve2!(tll, :α, verbose=verbose)

    info("Loss at iteration $i: $(evalloss(tll))")
    info("Objective at iteration $i: $(evalobjective(tll))")
    info("β at iteration $i: $(tll.β.value)")
    i+=1
  end

  return tll
end

function predict(tll::TLLRegression, X::Array{Float64,2})
  X = X ./ sum(X,2)
  return evaluate(X * tll.model)
end

end
