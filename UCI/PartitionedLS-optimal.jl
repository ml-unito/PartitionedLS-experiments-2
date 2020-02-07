using DataFrames
using CSV
using Printf
using LinearAlgebra
using JLD
using JSON
using Gurobi
using ECOS
using Logging

using PartitionedLS
using Checkpoint
# include("../../PartitionedLS/src/PartitionedLS.jl")
include("PartitionedLS-expio.jl")

optimizers = Dict(
    "Gurobi" => (() -> GurobiSolver()),
    "ECOS" => (() -> ECOSSolver()),
    "SCS" => (() -> SCSSolver())
)

# main

dir = ARGS[1]
conf = read_train_conf(dir)

mkcheckpointpath(conf, path=dir)
exppath = checkpointpath(conf, path=dir)
filename = "$exppath/results-OPT"

io = open("$filename.log", "w+")
logger = SimpleLogger(io)
global_logger(logger)


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
    @assert conf["regularization"] == 0.0
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