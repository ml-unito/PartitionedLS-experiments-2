# using Gadfly
using DataFrames
using CSV
using LinearAlgebra
using JLD
using JSON
using Logging
using ArgParse 

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

function loss(params, X, y)
    norm(predict(params, X) - y,2)^2
end


# main

function fit_with_restarts(dir, conf, filename, Xtr, ytr, Xte, yte, P)
    algorithm = (conf["use nnls"] == true ? AltNNLS : Alt)

    df = DataFrame(
        Time = Float64[],
        TimeCumulative = Float64[],
        TrainObj = Float64[],
        TrainBest = Float64[],
        TestObj = Float64[],
        TestBest = Float64[]
    )

    i_start, cumulative_time, df, results = resume(conf, init=(0,0.0,df,[]), nick="Alt-outer", path=dir)

    num_retrials = conf["Alt"]["num_retrials"]
    num_alternations = conf["Alt"]["num_alternations"]

    best_objective = Inf64
    best_test_objective = Inf64
    cumulative_time = 0.0
    
    for i in (i_start+1):num_retrials
        if i == i_start+1
            # Warming up julia environment (avoids counting the time julia needs to compile the function
            # when we time the algorithm execution on the next few lines) 
            @info "Warming up..."
            _, time, _ = @timed fit(algorithm, Xtr, ytr, P, η = 0.0,
                    get_solver = (() -> optimizers[conf["optimizer"]]()), 
                    N=num_alternations)
            @info "Warmup time: $(time) seconds"
        end

        @info "Retrial $i/$num_retrials"
        fitted_params, time, _ = @timed fit(algorithm, Xtr, ytr, P, η = 0.0,
                                            get_solver = (() -> optimizers[conf["optimizer"]]()), 
                                            N=num_alternations,
                                            checkpoint = (data) -> checkpoint(conf, data=data, path=dir, nick="Alt-inner" ),
                                            resume = (init) -> resume(conf, init=init, path=dir, nick="Alt-inner"))


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
        checkpoint(conf, data=(i, cumulative_time, df, results), path=dir, nick="Alt-outer")

        @info "Saving (partial) performances to $filename.csv"
        CSV.write("$filename.csv", df)
    end

    return results, df
end


function partlsalt_experiment_run(dir, conf, filename)
    Xtr, Xte, ytr, yte, P = load_data(dir, conf)

    num_retrials = conf["Alt"]["num_retrials"]
    num_alternations = conf["Alt"]["num_alternations"]
    exp_name = conf["Alt"]["exp_name"]

    @info "Fitting the model"

    results, df = fit_with_restarts(dir, conf, filename, Xtr, ytr, Xte, yte, P)

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

    @info "Saving final performances to $filename.csv"
    CSV.write("$filename.csv", df)
end


function partlsalt_experiment(dir, conf)
    try
        exppath = checkpointpath(conf, path=dir)
        filename = "$exppath/results-ALT"

        partlsalt_experiment_run(dir, conf, filename)
    catch error
        @error "Caught exception while executing experiment" conf=conf error=error
        for (exc, bt) in Base.catch_stack()
            showerror(global_logger().stream, exc, bt)
            println()
        end
        exit(1)
    end
end


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
mkcheckpointpath(conf, path=dir)

if opts["silent"]
    exppath = checkpointpath(conf, path=dir)
    filename = "$exppath/results-ALT"
    std_logger = global_logger()
    
    io = open("$filename.log", "w+")
    logger = SimpleLogger(io)
    global_logger(logger)
end

partlsalt_experiment(dir, conf)
