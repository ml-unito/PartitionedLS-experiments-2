# using Gadfly
using PartitionedLS: fit_iterative, fit_iterative_slow, predict
using DataFrames
using CSV
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

# this is needed to avoid measuring time needed by julia to setup things for this
# function
_ = fit_iterative(Xtr, ytr, P, verbose=0, η=1.0)

for i in 1:100
    global best_objective, cumulative_time
    fitted_params, time, _ = @timed fit_iterative(Xtr, ytr, P, verbose=0, η=1.0; N=100)
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

@info "Saving optimal values of α β t and objvalue to ./PartialLS-alternating-vars.jld"
save("./PartialLS-alternating-vars.jld", "objvalue", objvalue, "α", α, "β", β, "t", t)

print(df)

@info "Saving performances to ./PartialLS-alternating-results.csv"
CSV.write("./PartialLS-alternating-results.csv", df)
