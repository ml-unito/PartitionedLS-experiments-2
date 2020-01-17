using PartitionedLS: fit, predict
using DataFrames
using CSV
using Printf
using LinearAlgebra
using JLD
using JSON

include("PartitionedLS-expio.jl")

# main

dir = ARGS[1]
conf = read_train_conf(dir)

Xtr, Xte, ytr, yte, P = load_data(dir, conf)

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
tll, time, _ = @timed  fit(Xtr, ytr, P, verbose=1, η=1.0)
objvalue, α, β, t, _ = tll

push!(df, [time, time, objvalue, objvalue])

@info "objvalue: $objvalue"
@info "loss:" norm(predict(tll, Xte) - yte)^2

@info "Saving variables into file ./PartitionedLS-optimal-vars.jld" α β t
save("$dir/PartitionedLS-OPT.jld", "objvalue", objvalue, "α", α, "β", β, "t", t)

@info "Saving performances into ./PartitionedLS-optimal-results.csv"
CSV.write("$dir/PartitionedLS-OPT.csv", df)