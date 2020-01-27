# using Gadfly
using DataFrames
using CSV
using LinearAlgebra
using JLD
using JSON
using Gurobi
using SCS
using ECOS

# using PartitionedLS: fit_alternating, fit_alternating_slow, predict
include("../../PartitionedLS/src/PartitionedLS.jl")
include("PartitionedLS-expio.jl")

optimizers = Dict(
    "Gurobi" => (() -> GurobiSolver(BarHomogeneous=1)),
    "ECOS" => (() -> ECOSSolver()),
    "SCS" => (() -> SCSSolver())
)

# main

dir = ARGS[1]
conf = read_train_conf(dir)
Xtr, Xte, ytr, yte, P = load_data(dir, conf)

num_retrials = conf["ALT"]["num_retrials"]
num_alternations = conf["ALT"]["num_alternations"]
exp_name = conf["ALT"]["exp_name"]

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

for i in 1:num_retrials
    global best_objective, cumulative_time
    fitted_params, time, _ = @timed fit(Alt, Xtr, ytr, P, η=conf["regularization"]; 
                                        get_solver = (() -> optimizers[conf["optimizer"]]()), 
                                        N=num_alternations)
    objvalue, α, β, t, _ = fitted_params

    cumulative_time += time

    best_objective = min(best_objective, objvalue)
    @info time=time i=i objvalue=objvalue
    push!(df, [time,cumulative_time, objvalue, best_objective])
    push!(results, fitted_params)
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
filename = "$dir/PartitionedLS-$(conf["optimizer"])-ALT-$exp_name"

@info "Saving optimal values of α β t and objvalue to $filename.jld"
save("$filename.jld", "objvalue", objvalue, "α", α, "β", β, "t", t)

print(df)

@info "Saving performances to $filename.csv"
CSV.write("$filename.csv", df)
