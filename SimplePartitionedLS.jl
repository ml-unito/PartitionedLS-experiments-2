# using Pkg
# Pkg.update()
# Pkg.instantiate()

using JSON
using CSV
using DataFrames
using Tables
using PartitionedLS
using Random
using Suppressor

include(joinpath(@__DIR__,"PartitionedLS-expio.jl"))


function performExperiments(datapath, blockUsage, blockname)
    results = DataFrame(
        Blocks=String[],
        Seed=Int[],
        TrainingError=Float64[],
        TestError=Float64[]
    )

    Random.seed!(1234)
    seeds = rand(0:1000000, 100)

    # Generating results for experiments with P matrices derived from the LS execution
    for i in 1:100
        @info "Iteration:" i seeds[i]

        conf = read_train_conf(datapath)
        Xtr, Xte, ytr, yte, P = load_data(datapath, conf, blocksfname=blockname(i), shuffle = true, seed = seeds[i]) 
        ytr = vec(ytr)

        # Least squares on Xtr, ytr
        _, a, b, t = 0,0,0,0
        @suppress begin
            _, a, b, t, P = PartitionedLS.fit(OptNNLS, Xtr, ytr, P, η = 0.0)
        end

        @info b

        preds_tr = PartitionedLS.predict(vec(a),vec(b),t,P, Xtr)
        preds_te = PartitionedLS.predict(vec(a),vec(b),t,P, Xte)

        # Compute errors
        err_tr = sum((preds_tr .- ytr).^2)
        err_te = sum((preds_te .- yte).^2)

        push!(results, (blockUsage, seeds[i], err_tr, err_te))

        @info "Train error: "  TrainingError = err_tr TestError = err_te
    end


    return results
end

if length(ARGS) < 1
    println("Usage: SimplePartitionedLS <datadir>")
    exit(1)
end

datadir = ARGS[1]
# datadir = "datasets/FCVD-test/"

resultsLS = performExperiments(datadir, "FromLS", seed -> "LSBlocks-$seed.csv")
CSV.write("$datadir/PartLSResults-PfromLS.csv", resultsLS)

resultsOrig = performExperiments(datadir, "Original", seed -> "blocks.csv")
CSV.write("$datadir/PartLSResults-POrig.csv", resultsOrig)