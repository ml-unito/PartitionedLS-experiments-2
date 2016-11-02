module TLLR

using Convex
using ECOS

import Base.size
export fit, predict
"""
    type TLLRegression

The type of Two Levels Linear Regression models.

A TLLR model is a linear model has the form:
  \$h(x) = \sum_k \beta_k f_{\beta_k}(x)\$

where \$f_\beta(x)\$ are pseudo features built on the original \$x\$
features:
  \$f_{\beta_k}(x) = \sum_{m: P_{m,k} = 1} \alpha_m x_m\$

In this version \$\beta_k\$ are only used to maintain the sign associated
to the block and the \$\alpha_m\$ are not constrained to sum to one
blockwise.

"""

function indextobeta(b::Integer, K::Integer)
  result::Array{Int64,1} = []
  for k = 1:K
    push!(result, 2(b % 2)-1)
    b >>= 1
  end

  result
end

function indmin(mapfun, a)
  result = 1
  for i in 2:length(a)
    if mapfun(a[i]) < mapfun(a[result])
      result = i
    end
  end

  result
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
function fit(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; verbose=0)
  info("------------ FIT ---------------")

  # row normalization
  M,K = size(P)

  results = []

  for b in 0:(2^K-1)
    info("Starting iteration n. $b")

    α = Variable(M, Positive())
    t = Variable()
    β = indextobeta(b,K)

    loss = norm(X * (P .* (α * ones(1,K))) * β + t - y)^2
    regularization = 0.0 * norm(α,1)
    p = minimize(loss + regularization)
    Convex.solve!(p, ECOSSolver(verbose=verbose))

    println("optval: $(p.optval)")
    push!(results,(p.optval, α.value, β, t.value, P))
  end

  results[indmin(z -> z[1], results)]
end

function predict(model, X::Array{Float64,2})
  (_, α, β, t, P) = model
  X*(P .* α) * β + t
end

end
