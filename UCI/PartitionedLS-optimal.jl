using DataFrames
using CSV
using Printf
using LinearAlgebra
using JLD
using JSON
using ECOS
# Decomment the following if you are actually planning to use
# these solvers and you have installed the proper sw on your system
# 
# using SCS
# using Gurobi
using Logging

using PartitionedLS
using Checkpoint
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
    
    # Warming up julia environment (avoids counting the time julia needs to compile the function
    # when we time the algorithm execution on the next few lines) 
    # @info "Warming up..."
    # _ = fit(Xtr, ytr, P, verbose=1, η=1.0)
    
    @info "Fitting the model"

    if conf["use nnls"] == true
        tll, time, _ = @timed  fit(OptNNLS, Xtr, ytr, P, 
                                get_solver = optimizers[conf["optimizer"]],
                                checkpoint = (data) -> checkpoint(conf, data=data, nick="Opt", path=dir),
                                resume = (initvals) -> resume(conf, init=initvals, nick="Opt", path=dir))
    else
        tll, time, _ = @timed  fit(Opt, Xtr, ytr, P, η = conf["regularization"],
                               get_solver = optimizers[conf["optimizer"]],
                               checkpoint = (data) -> checkpoint(conf, data=data, nick="Opt", path=dir),
                               resume = (initvals) -> resume(conf, init=initvals, nick="Opt", path=dir))
    end
    
    loss = (model, X, y) -> norm(predict(model, X) - y)^2
    
    train_objvalue, α, β, t, _ = tll
    test_objvalue = loss(tll, Xte, yte)
    
    push!(df, [time, time, train_objvalue, train_objvalue, test_objvalue, test_objvalue])
    
    @info "objvalue: $train_objvalue"
    @info "loss:" train = train_objvalue test = test_objvalue
    
    @info "Saving variables into file $filename.jld" α β t
    save("$filename.jld", "objvalue", train_objvalue, "α", α, "β", β, "t", t)
    
    @info "Saving performances into $filename.csv"
    CSV.write("$filename.csv", df)
end

function partlsopt_experiment(dir, conf)
    mkcheckpointpath(conf, path=dir)
    exppath = checkpointpath(conf, path=dir)
    filename = "$exppath/results-OPT"
    std_logger = global_logger()
    
    io = open("$filename.log", "w+")
    logger = SimpleLogger(io)
    global_logger(logger)

    try
        partlsopt_experiment_run(dir, conf, filename)
    catch error
        @error "Caught exception while executing experiment" conf=conf error=error
        exit(1)
    end

    global_logger(std_logger)
end

# main

dir = ARGS[1]
conf = read_train_conf(dir)
partlsopt_experiment(dir, conf)

