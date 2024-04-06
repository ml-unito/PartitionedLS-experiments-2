using JSON
using CSV
using DataFrames
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
        @info "PartitionedLS Iteration:" i seeds[i]

        conf = read_train_conf(datapath)
        Xtr, Xte, ytr, yte, P = load_data(datapath, conf, blocksfname=blockname(i), shuffle = true, seed = seeds[i]) 
        ytr = vec(ytr)

        # Least squares on Xtr, ytr
        model = PartLS(P=P, η=0.0)
        mach = machine(model, Xtr, ytr)
        @suppress begin
            fit!(mach)
        end

        preds_tr = predict(mach, Xtr)
        preds_te = predict(mach, Xte)

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
dircomponents = splitpath(datadir)
expdir = joinpath("experiments", "model-quality", dircomponents[end])

if !isdir(expdir)
    mkpath(expdir)
end


resultsLS = performExperiments(datadir, "FromLS", seed -> "$expdir/LSBlocks-$seed.csv")
CSV.write("$expdir/PartLSResults-PfromLS.csv", resultsLS)

resultsOrig = performExperiments(datadir, "Original", seed -> "$datadir/blocks.csv")
CSV.write("$expdir/PartLSResults-POrig.csv", resultsOrig)

resultsBestPart = performExperiments(datadir, "FromLSOpt", seed -> "$expdir/LSOptBlocks.csv")
CSV.write("$expdir/PartLSResults-PfromLSOpt.csv", resultsBestPart)
