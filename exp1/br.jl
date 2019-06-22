using PartitionedLS: fit, predict
using DataFrames
using CSV
using Printf
using LinearAlgebra
using JLD

# main

@info "Reading data..."
data = CSV.read("exp1/LogPTol_vsPlusDescr.csv", delim=';')
blocks = CSV.read("exp1/LogPTol_vsPlusDescr_blocks.csv", delim=';')

@info "Converting matrices..."
Xtr = convert(Matrix, data[1:30, 2:83])
Xte = convert(Matrix, data[31:end, 2:83])
ytr = convert(Array, data[1:30, :log_Ptol])
yte = convert(Array, data[31:end, :log_Ptol])
P = convert(Matrix, blocks[:, 2:7])

df = DataFrame(
    Time = Float64[],
    TimeCumulative = Float64[],
    Objective = Float64[],
    Best = Float64[]
)


# Warming up julia environment
_ = fit(Xtr, ytr, P, verbose=0, η=1.0)

@info "Fitting the model"
tll, time, _ = @timed  fit(Xtr, ytr, P, verbose=0, η=1.0)
objvalue, α, β, t, _ = tll

push!(df, [time, time, objvalue, objvalue])

@info "objvalue: $objvalue"
@info "loss:" norm(predict(tll, Xte) - yte)^2

@info "Saving variables into file ./br_vars.jld" α β t
save("./br_vars.jld", "objvalue", objvalue, "α", α, "β", β, "t", t)

@info "Saving performances into ./pls-b-results.csv"
CSV.write("./pls-b-results.csv", df)
