using DataFrames
using CSV
using Printf
using LinearAlgebra
using JLD
using JSON
using Gurobi
using ECOS

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


Xtr, Xte, ytr, yte, P = load_data(dir, conf)

@info size(P)

df = DataFrame(
    Time = Float64[],
    TimeCumulative = Float64[],
    Objective = Float64[],
    Best = Float64[]
)

# Warming up julia environment (avoids counting the time julia needs to compile the function
# when we time the algorithm execution on the next few lines) 
# @info "Warming up..."
# _ = fit(Xtr, ytr, P, verbose=1, η=1.0)

@info "Fitting the model"
tll, time, _ = @timed  fit(Opt, Xtr, ytr, P, 
                           get_solver = optimizers[conf["optimizer"]],
                           checkpoint = (data) -> checkpoint(conf, data=data, nick="Opt", path=dir),
                           resume = (initvals) -> resume(conf, init=initvals, nick="Opt", path=dir))
objvalue, α, β, t, _ = tll

push!(df, [time, time, objvalue, objvalue])

@info "objvalue: $objvalue"
@info "loss:" norm(predict(tll, Xte) - yte)^2

exppath = checkpointpath(conf, path=dir)
filename = "$exppath/results-OPT"

@info "Saving variables into file $filename.jld" α β t
save("$filename.jld", "objvalue", objvalue, "α", α, "β", β, "t", t)

@info "Saving performances into $filename.csv"
CSV.write("$filename.csv", df)