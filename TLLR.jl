module TLLR

using Convex
using ECOS

import Base.size
export fit, predict

"""
  indextobeta(b::Integer, K::Integer)::Array{Int64,1}

  returns 2 * bin(b,K) - 1

  where bin(b) is a vector of K elements containing the binary
  representation of b.
"""
function indextobeta(b::Integer, K::Integer)
  result::Array{Int64,1} = []
  for k = 1:K
    push!(result, 2(b % 2)-1)
    b >>= 1
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
function fit(X::Array{Float64,2}, y::Array{Float64,1}, P::Array{Int,2}; verbose=0, η=1)
  # row normalization
  M,K = size(P)

  results = []

  for b in 0:(2^K-1)
    α = Variable(M, Positive())
    t = Variable()
    β = indextobeta(b,K)

    loss = norm(X * (P .* (α * ones(1,K))) * β + t - y)^2
    regularization = η * norm(α,2)
    p = minimize(loss + regularization)
    Convex.solve!(p, ECOSSolver(verbose=verbose))

    info("iteration $b optval: $(p.optval)")
    push!(results,(p.optval, α.value, β, t.value, P))
  end

  optindex = indmin((z -> z[1]).(results))
  opt,a,b,t,_ = results[optindex]


  A = sum(P .* a, 1)
  a = sum((P .* a) ./ A, 2)
  b = b .* A'

  (opt, a, b, t, P)
end


"""
  predict(model::Tuple, X::Array{Float64,2})

  returns the predictions of the given model on examples in X
"""
function predict(model, X::Array{Float64,2})
  (_, α, β, t, P) = model
  X * (P .* α) * β + t
end

end
