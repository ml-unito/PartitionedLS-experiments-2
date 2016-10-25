module TLLR

using Convex
using Logging
using SCS

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
  verbose::Int
  X
  y

  function TLLRegression(P; verbose=0)
    M, K = size(P)
    tll = new()
    tll.α = Variable(M)
    tll.β = Variable(K)
    tll.P = P
    tll.model = (tll.P .* (tll.α * ones(1,size(tll.P,2)))) * tll.β
    tll.verbose = verbose

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

function tllsolve!(tll::TLLRegression, freevarname; verbose=0)
  fixvar_symbol = setdiff([:α,:β],[freevarname])[1]
  fixvar = getfield(tll, fixvar_symbol)
  freevar = getfield(tll, freevarname)

  info("solving for $freevarname")
  info("before learning $freevarname: $(freevar.value)")
  info("$freevarname is free: $(!(:fixed in freevar.sets))")

  fix!(fixvar)
  Convex.solve!(tll.problem, ECOSSolver(verbose=verbose))
  free!(fixvar)

  info("after learning (status: $(tll.problem.status)) $(freevarname): $(freevar)")
end

function tllsolvefor_alpha!(tll::TLLRegression; verbose=0)
  tll.model = (tll.P .* (tll.α.value * ones(1,size(tll.P,2)))) * tll.β
  tll.problem = minimize( norm(tll.X * tll.model - tll.y) )
  Convex.solve!(tll.problem, SCSSolver(verbose=verbose))
end

function tllsolvefor_beta!(tll::TLLRegression; verbose=0)
  tll.model = (tll.P .* (tll.α * ones(1,size(tll.P,2)))) * tll.β.value
  tll.problem = minimize( norm(tll.X * tll.model - tll.y) )
  Convex.solve!(tll.problem, SCSSolver(verbose=verbose))
end


function tllsolve2!(tll::TLLRegression, freevarname; verbose=0)
  if(freevarname==:α)
    tllsolvefor_alpha!(tll, verbose=verbose)
  else
    tllsolvefor_beta!(tll, verbose=verbose)
  end
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

  tll = TLLRegression(P, verbose=verbose)

  M,K = size(tll)

  tll.α.value = rand(M)
  tll.β.value = beta

  # termination criteria is guided by epsilon... when two iterations optvalue
  # differ for a quantity δ that is less than epslon, we stop
  ϵ = 0.001
  δ = 1.0

  previous_optval = 100
  i=0
  maxiterations = 5

  # FIXME: The following two assignment do not belong here
  tll.X = X
  tll.y = y
  tll.problem = minimize( norm(X * tll.model - y) )

  while i < maxiterations
    info("Starting iteration n. $i")
    tllsolve2!(tll, :α, verbose=verbose)
    tllsolve2!(tll, :β, verbose=verbose)

    δ = abs(previous_optval - optval(tll))
    previous_optval = optval(tll)
    info("Loss at iteration $i: $previous_optval")
    i+=1
  end

  println( evaluate(X * tll.model - y) )

  return tll
end

function predict(tll::TLLRegression, X::Array{Float64,2})
  X = X ./ sum(X,2)
  return evaluate(X * tll.model)
end

end
