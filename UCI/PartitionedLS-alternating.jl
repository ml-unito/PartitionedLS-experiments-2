# using Gadfly
using DataFrames
using CSV
using LinearAlgebra
using JLD
using JSON
using Gurobi
using SCS
using ECOS
using Logging

using PartitionedLS
using Checkpoint
include("PartitionedLS-expio.jl")

optimizers = Dict(
    "Gurobi" => (() -> GurobiSolver(BarHomogeneous=1)),
    "ECOS" => (() -> ECOSSolver()),
    "SCS" => (() -> SCSSolver())
)

# main

dir = ARGS[1]
conf = read_train_conf(dir)

mkcheckpointpath(conf, path=dir)
exppath = checkpointpath(conf, path=dir)
filename = "$exppath/results-ALT"

io = open("$filename.log", "w+")
logger = SimpleLogger(io)
global_logger(logger)

Xtr, Xte, ytr, yte, P = load_data(dir, conf)

num_retrials = conf["Alt"]["num_retrials"]
num_alternations = conf["Alt"]["num_alternations"]
exp_name = conf["Alt"]["exp_name"]

@info "Fitting the model"

df = DataFrame(
    Time = Float64[],
    TimeCumulative = Float64[],
    Objective = Float64[],
    Best = Float64[]
)

results = []
best_objective = Inf64
cumulative_time = 0.0

# Warming up julia environment (avoids counting the time julia needs to compile the function
# when we time the algorithm execution in the next loop)
# _ = fit_alternating(Xtr, ytr, P, verbose=0, η=1.0)

i_start, cumulative_time, df = resume(conf, init=(0,0.0,df), nick="Alt-outer", path=dir)

for i in (i_start+1):num_retrials
    @info "Retrial $i/$num_retrials"

    global best_objective, cumulative_time
    fitted_params, time, _ = @timed fit(AltNNLS, Xtr, ytr, P, 
                                        get_solver = (() -> optimizers[conf["optimizer"]]()), 
                                        N=num_alternations,
                                        checkpoint = (data) -> checkpoint(conf, data=data, path=dir, nick="Alt-inner" ),
                                        resume = (init) -> resume(conf, init=init, path=dir, nick="Alt-inner"))
    removecheckpoint(conf, path=dir, nick="Alt-inner")
    objvalue, α, β, t, _ = fitted_params

    cumulative_time += time

    best_objective = min(best_objective, objvalue)
    @info time=time i=i objvalue=objvalue
    push!(df, [time,cumulative_time, objvalue, best_objective])
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
@info "loss:" norm(predict(results[best_i], Xte) - yte)^2

@info "Saving optimal values of α β t and objvalue to $filename.jld"
save("$filename.jld", "objvalue", objvalue, "α", α, "β", β, "t", t)

@info "Saving performances to $filename.csv"
CSV.write("$filename.csv", df)
