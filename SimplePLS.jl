using JSON
using CSV
using DataFrames
using Tables
using LinearAlgebra
using Random
using Statistics
using MLJBase, MLJModels


PLSRegressor = @load PLSRegressor pkg = PartialLeastSquaresRegressor

function sign(w)
    return [ifelse(w[i] >= 0, 1, -1) for i in 1:length(w)]
end

function indicesOfNonConstantCols(X)
    return vec(var(X, dims=1) .> 1e-10)
end

include(joinpath(@__DIR__,"PartitionedLS-expio.jl"))


if length(ARGS) < 1
    println("Usage: SimplePLS <datadir>")
    exit(1)
end

datadir = ARGS[1]

conf = read_train_conf(datadir)

results = DataFrame(
    Seed = Int[],
    TrainingError = Float64[],
    TestError = Float64[]
)

Random.seed!(1234)
seeds = rand(0:1000000, 100)

for i in 1:100 
    Xtr, Xte, ytr, yte, P, colNames = load_data(datadir, conf, shuffle = true, seed = seeds[i])
    @info "Iteration $i"

    clx = indicesOfNonConstantCols(Xtr)
    Xtr = Xtr[:, clx]
    Xte = Xte[:, clx]

    Xtr = MLJBase.table(Xtr)
    ytr = vec(ytr)
    Xte = MLJBase.table(Xte)
    yte = vec(yte)
  
    # Partial Least squares on Xtr, ytr
    
    pls = PLSRegressor(n_factors=size(P,2))
    mach = machine(pls, Xtr, ytr)
    fit!(mach)

    yhat_tr = predict(mach, Xtr)
    yhat_te = predict(mach, Xte)

    # Compute errors

    err_tr = sum((yhat_tr .- ytr).^2)
    err_te = sum((yhat_te .- yte).^2)

    push!(results, (seeds[i], err_tr, err_te))

    @info "Training/Test errors:" TrainingError = err_tr TestError = err_te
end

CSV.write("$datadir/PLSResults.csv", results)
