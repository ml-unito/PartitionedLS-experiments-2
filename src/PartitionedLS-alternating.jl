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
using PartitionedLS
include("PartitionedLS-expio.jl")

function loss(params, X, y)
    norm(predict(params, X) - y,2)^2
end


# main

function fit_with_restarts(dir, conf, filename, Xtr, ytr, Xte, yte, P)
    df = DataFrame(
        Time = Float64[],
        TimeCumulative = Float64[],
        TrainObj = Float64[],
        TrainBest = Float64[],
        TestObj = Float64[],
        TestBest = Float64[]
    )

    @info "Starting experiment" conf
    num_retrials = conf["Alt"]["num_retrials"]
    num_alternations = conf["Alt"]["num_alternations"]

    best_objective = Inf64
    best_test_objective = Inf64
    cumulative_time = 0.0

    results = []

    for i in 1:num_retrials
        if i == 1
            # Warming up julia environment (avoids counting the time julia needs to compile the function
            # when we time the algorithm execution on the next few lines) 
            @info "Warming up..."
            _, time, _ = @timed fit(Alt, Xtr, vec(ytr), P, η = 0.0, T=num_alternations, nnlsalg=:nnls)
            @info "Warmup time: $(time) seconds"
        end

        @info "Retrial $i/$num_retrials"
        result, time, _ = @timed fit(Alt, Xtr, vec(ytr), P, η = 0.0, nnlsalg=:nnls, T=num_alternations)


        train_objvalue = result.opt
        test_objvalue = loss(result.model, Xte, yte)

        cumulative_time += time

        if train_objvalue < best_objective
            best_objective = train_objvalue
            best_test_objective = test_objvalue
        end

        best_objective = min(best_objective, train_objvalue)
        @info "stats" time=time i=i objvalue=train_objvalue
        push!(df, [time,cumulative_time, train_objvalue, best_objective, test_objvalue, best_test_objective])
        push!(results, result)

        @info "Saving (partial) performances to $filename.csv"
        CSV.write("$filename.csv", df)
    end

    return results, df
end


function partlsalt_experiment_run(dir, conf, filename, T)
    Xtr, Xte, ytr, yte, P = load_data(dir, conf)

    num_retrials = conf["Alt"]["num_retrials"]
    num_alternations = T

    @info "Fitting the model"

    results, df = fit_with_restarts(dir, conf, filename, Xtr, ytr, Xte, yte, P)

    function getobj(fitted_params)
        fitted_params.opt
    end

    best_i = argmin(map( getobj, results))
    best_result = results[best_i]


    @info "objvalue: $(best_result.opt)"
    @info "Losses:" train = loss(best_result.model, Xtr, ytr) test = loss(best_result.model, Xte, yte)

    @info "Saving optimal values of α β t and objvalue to $filename.jld"
    save("$filename.jld", "objvalue", best_result.opt, "model", best_result.model)

    @info "Saving final performances to $filename.csv"
    CSV.write("$filename.csv", df)
end


function partlsalt_experiment(datadir, conf, T)
    try
        dircomponents = splitpath(datadir)
        expdir = joinpath("experiments", "time-vs-obj", dircomponents[end])

        if !isdir(expdir)
            mkdir(expdir)
        end
    
        filename = "$expdir/results-ALT-T$T"

        partlsalt_experiment_run(datadir, conf, filename, T)
    catch error
        @error "Caught exception while executing experiment" conf=conf error=error
        for (exc, bt) in Base.catch_stack()
            showerror(global_logger().stream, exc, bt)
            println()
        end
        exit(1)
    end
end

if length(ARGS)<2
    println("Usage: PartitionedLS-alternating.jl <datadir> <num alternations>")
    exit(1)
end

dir = ARGS[1]
T = parse(Int64, ARGS[2])
conf = read_train_conf(dir)

partlsalt_experiment(dir, conf, T)
