using DataFrames
using CSV
using Printf
using LinearAlgebra
using JLD
using JSON
using ArgParse
using Logging

using PartitionedLS
# using Checkpoint
include("PartitionedLS-expio.jl")


function partlsopt_experiment_run(dir, conf, filename, algoName)
    Xtr, Xte, ytr, yte, P = load_data(dir, conf)
    
    @info size(P)
    
    df = DataFrame(
        Time = Float64[],
        TimeCumulative = Float64[],
        TrainObj = Float64[],
        TrainBest = Float64[],
        TestObj = Float64[],
        TestBest = Float64[]
    )

    algo = nothing
    if algoName == "Opt"
        algo = Opt
    elseif algoName == "BnB"
        algo = BnB
    else 
        @error "Unknown algorithm" algo=algo
        exit(1)
    end
    

    
    @info "Fitting the model"

    # Warming up julia environment (avoids counting the time julia needs to compile the function
    # when we time the algorithm execution on the next few lines) 
    @info "Warming up..."
    _, time, _ = @timed fit(algo, Xtr[1:10,:], vec(ytr[1:10]), P, η = conf["regularization"])
    @info "Warmup time: $(time) seconds"

    
    # Actual run
    result, time, _ = @timed  fit(algo, Xtr, vec(ytr), P, η = conf["regularization"])
    
    loss = (model, X, y) -> norm(predict(model, X) - y)^2
    
    train_objvalue = result[3].opt
    test_objvalue = loss(result[1], Xte, yte)
    
    push!(df, [time, time, train_objvalue, train_objvalue, test_objvalue, test_objvalue])
    
    @info "objvalue: $train_objvalue"
    @info "loss:" train = train_objvalue test = test_objvalue
    
    @info "Saving variables into file $filename.jld"
    save("$filename.jld", "objvalue", train_objvalue, "model", result[1])
    
    @info "Saving performances into $filename.csv"
    CSV.write("$filename.csv", df)
end

function partlsopt_experiment(datadir, conf, algo)
    try
        dircomponents = splitpath(datadir)
        expdir = joinpath("experiments", "time-vs-obj", dircomponents[end])
        filename = "$expdir/results-$algo"
    
        partlsopt_experiment_run(datadir, conf, filename, algo)
    catch error
        @error "Caught exception while executing experiment" conf=conf error=error
        for (exc, bt) in Base.catch_stack()
            showerror(global_logger().stream, exc, bt)
            println()
        end
        exit(1)
    end
end

# main


if length(ARGS)<2
    println("Usage: PartitionedLS-optimal <datadir> <algorithm>")
    exit(1)
end

dir = ARGS[1]
algo = ARGS[2]

conf = read_train_conf(dir)
# mkcheckpointpath(conf, path=dir)


partlsopt_experiment(dir, conf, algo)

