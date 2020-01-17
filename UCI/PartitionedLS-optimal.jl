using PartitionedLS: fit, predict
using DataFrames
using CSV
using Printf
using LinearAlgebra
using JLD

function load_data(dir)
    @info "Reading data..."
    data = CSV.read(string(dir, "/data.csv"))

    @info "Reading blocks"
    blocks = CSV.read(string(dir, "/blocks.csv"))

    train_len = trunc(Int, nrow(data) * 0.9)
    test_len = nrow(data) - train_len

    @info "Converting matrices...", "train/test set split is" train = train_len test = test_len
    
    Xtr = convert(Matrix, data[1:train_len, setdiff(names(data), [:y])])
    Xte = convert(Matrix, data[(train_len+1):end, setdiff(names(data), [:y])])
    ytr = convert(Array, data[1:train_len, :y])
    yte = convert(Array, data[(train_len+1):end, :y])
    P = convert(Matrix, blocks[:, 2:end])

    return Xtr, Xte, ytr, yte, P
end


# main


dir = ARGS[1]
Xtr, Xte, ytr, yte, P = load_data(dir)

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
save("$dir/PartitionedLS-optimal-vars.jld", "objvalue", objvalue, "α", α, "β", β, "t", t)

@info "Saving performances into ./PartitionedLS-optimal-results.csv"
CSV.write("$dir/PartitionedLS-optimal-results.csv", df)