using JSON
using CSV
using DataFrames
using Tables
using LinearAlgebra
using Random

function sign(w)
    return [ifelse(w[i] >= 0, 1, -1) for i in 1:length(w)]
end

include(joinpath(@__DIR__,"PartitionedLS-expio.jl"))


if length(ARGS) < 1
    println("Usage: SimpleLeastSquares <datadir>")
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
    @info colNames

    # Least squares on Xtr, ytr

    Xtr = [ones(size(Xtr, 1)) Xtr]
    Xte = [ones(size(Xte, 1)) Xte]

    w = Xtr \ ytr
    # w = inv(Xtr' * Xtr) * Xtr' * ytr
    # w = pinv(Xtr, atol=100, rtol=100) * ytr

    # Predictions

    yhat_tr = Xtr * w
    yhat_te = Xte * w

    # Compute errors

    err_tr = sum((yhat_tr .- ytr).^2)
    err_te = sum((yhat_te .- yte).^2)

    push!(results, (seeds[i], err_tr, err_te))

    @info "Training/Test errors:" TrainingError = err_tr TestError = err_te


    rownames = setdiff(colNames, ["y"])
    signs = sign(w[2:end])

    df = DataFrame(
        Descriptor = rownames,
        Pos = [ifelse(signs[i] == 1, 1, 0) for i in 1:length(signs)],
        Neg = [ifelse(signs[i] == -1, 1, 0) for i in 1:length(signs)]
    )

    CSV.write("$datadir/LSBlocks-$i.csv", df)
end

CSV.write("$datadir/LSResults.csv", results)
