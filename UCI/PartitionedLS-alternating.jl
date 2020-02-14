# using Gadfly
using DataFrames
using CSV
using LinearAlgebra
using JLD
using JSON
using Logging
# Decomment the following if you are actually planning to use
# these solvers and you have installed the proper sw on your system
# 
# using SCS
# using Gurobi
using ECOS
using PartitionedLS
using Checkpoint
include("PartitionedLS-expio.jl")

optimizers = Dict(
    "Gurobi" => (() -> GurobiSolver(BarHomogeneous=1)),
    "ECOS" => (() -> ECOSSolver()),
    "SCS" => (() -> SCSSolver())
)

# main


function partlsalt_experiment_run(dir, conf, filename)
    Xtr, Xte, ytr, yte, P = load_data(dir, conf)

    num_retrials = conf["Alt"]["num_retrials"]
    num_alternations = conf["Alt"]["num_alternations"]
    exp_name = conf["Alt"]["exp_name"]

    @info "Fitting the model"

    df = DataFrame(
        Time = Float64[],
        TimeCumulative = Float64[],
        TrainObj = Float64[],
        TrainBest = Float64[],
        TestObj = Float64[],
        TestBest = Float64[]
    )

    results = []
    best_objective = Inf64
    best_test_objective = Inf64
    cumulative_time = 0.0

    # Warming up julia environment (avoids counting the time julia needs to compile the function
    # when we time the algorithm execution in the next loop)
    # _ = fit_alternating(Xtr, ytr, P, verbose=0, η=1.0)

    i_start, cumulative_time, df = resume(conf, init=(0,0.0,df), nick="Alt-outer", path=dir)
    loss = (params, X, y) -> norm(predict(params, X) - y,2)^2

    for i in (i_start+1):num_retrials
        @info "Retrial $i/$num_retrials"

        if conf["use nnls"] == true
            fitted_params, time, _ = @timed fit(AltNNLS, Xtr, ytr, P, 
                                                get_solver = (() -> optimizers[conf["optimizer"]]()), 
                                                N=num_alternations,
                                                checkpoint = (data) -> checkpoint(conf, data=data, path=dir, nick="Alt-inner" ),
                                                resume = (init) -> resume(conf, init=init, path=dir, nick="Alt-inner"))
        else
            fitted_params, time, _ = @timed fit(Alt, Xtr, ytr, P, η = conf["regularization"],
                                                get_solver = (() -> optimizers[conf["optimizer"]]()), 
                                                N=num_alternations,
                                                checkpoint = (data) -> checkpoint(conf, data=data, path=dir, nick="Alt-inner" ),
                                                resume = (init) -> resume(conf, init=init, path=dir, nick="Alt-inner"))
        end


        removecheckpoint(conf, path=dir, nick="Alt-inner")
        train_objvalue, α, β, t, _ = fitted_params
        test_objvalue = loss(fitted_params, Xte, yte)

        cumulative_time += time

        if train_objvalue < best_objective
            best_objective = train_objvalue
            best_test_objective = test_objvalue
        end

        best_objective = min(best_objective, train_objvalue)
        @info "stats" time=time i=i objvalue=train_objvalue
        push!(df, [time,cumulative_time, train_objvalue, best_objective, test_objvalue, best_test_objective])
        push!(results, fitted_params)
        checkpoint(conf, data=(i, cumulative_time, df), path=dir, nick="Alt-outer")
    end

    function getobj(fitted_params)
        objvalue, α, β, t, _ = fitted_params
        objvalue
    end

    best_i = argmin(map( getobj, results))
    objvalue, α, β, t, _ = results[best_i]


    @info "Found variables" α β t

    @info "objvalue: $objvalue"
    @info "Losses:" train = loss(results[best_i], Xtr, ytr) test = loss(results[best_i], Xte, yte)

    @info "Saving optimal values of α β t and objvalue to $filename.jld"
    save("$filename.jld", "objvalue", objvalue, "α", α, "β", β, "t", t)

    @info "Saving performances to $filename.csv"
    CSV.write("$filename.csv", df)
end


function partlsalt_experiment(dir, conf)
    mkcheckpointpath(conf, path=dir)
    exppath = checkpointpath(conf, path=dir)
    filename = "$exppath/results-ALT"
    std_logger = global_logger()

    io = open("$filename.log", "w+")
    logger = SimpleLogger(io)    
    global_logger(logger)

    try
        partlsalt_experiment_run(dir, conf, filename)
    catch error
        @error "Caught exception while executing experiment" conf=conf error=error
        for (exc, bt) in Base.catch_stack()
            showerror(global_logger().stream, exc, bt)
            println()
        end
        exit(1)
    end

    global_logger(std_logger)
end


dir = ARGS[1]
conf = read_train_conf(dir)
partlsalt_experiment(dir, conf)
