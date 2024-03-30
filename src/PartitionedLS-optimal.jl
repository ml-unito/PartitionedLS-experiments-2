using DataFrames
using CSV
using Printf
using LinearAlgebra
using JLD
using JSON
using ArgParse

# Decomment the following if you are actually planning to use
# these solvers and you have installed the proper sw on your system
# 
# using SCS
# using Gurobi
using Logging

using PartitionedLS
# using Checkpoint
include("PartitionedLS-expio.jl")

optimizers = Dict(
    "Gurobi" => (() -> GurobiSolver()),
    "ECOS" => (() -> ECOSSolver()),
    "SCS" => (() -> SCSSolver())
)

function partlsopt_experiment_run(dir, conf, filename)
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
    

    
    @info "Fitting the model"

    # Warming up julia environment (avoids counting the time julia needs to compile the function
    # when we time the algorithm execution on the next few lines) 
    @info "Warming up..."
    _, time, _ = @timed fit(Opt, Xtr, vec(ytr), P, η = conf["regularization"])
    @info "Warmup time: $(time) seconds"

    
    # Actual run
    result, time, _ = @timed  fit(Opt, Xtr, vec(ytr), P, η = conf["regularization"])
    
    loss = (model, X, y) -> norm(predict(model, X) - y)^2
    
    train_objvalue = result.opt
    test_objvalue = loss(result.model, Xte, yte)
    
    push!(df, [time, time, train_objvalue, train_objvalue, test_objvalue, test_objvalue])
    
    @info "objvalue: $train_objvalue"
    @info "loss:" train = train_objvalue test = test_objvalue
    
    @info "Saving variables into file $filename.jld"
    save("$filename.jld", "objvalue", train_objvalue, "model", result.model)
    
    @info "Saving performances into $filename.csv"
    CSV.write("$filename.csv", df)
end

function partlsopt_experiment(datadir, conf)
    try
        dircomponents = splitpath(datadir)
        expdir = joinpath("experiments", "time-vs-obj", dircomponents[end])
        filename = "$expdir/results-OPT"
    
        partlsopt_experiment_run(datadir, conf, filename)
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

s = ArgParseSettings()
@add_arg_table s begin
    "-s", "--silent"
        help = "Redirect all log messages to a log file in the results dir"
        action = :store_true
    "dir"
        required = true
end
opts = parse_args(s)


dir = opts["dir"]
conf = read_train_conf(dir)
# mkcheckpointpath(conf, path=dir)

if opts["silent"]
    filename = "$dir/results-OPT"
    std_logger = global_logger()
    
    io = open("$filename.log", "w+")
    logger = SimpleLogger(io)
    global_logger(logger)
end

partlsopt_experiment(dir, conf)

