using PartitionedLS: fit, predict
using DataFrames
using CSV
using Printf
using LinearAlgebra
using JLD
using JSON

function read_train_conf(dir, data)
    json_path = "$dir/train_conf.json"
    if isfile(json_path)
        json = JSON.parsefile(json_path)
        train_start = json["train_start"]
        train_end = json["train_end"]
        test_start = json["test_start"]
        test_end = json["test_end"]
    else
        @info "Configuration file not found reverting to defaults"
        train_start = 1
        train_end = trunc(Int, nrow(data) * 0.9)
        test_start = train_end+1
        test_end = nrow(data)
    end

    return train_start, train_end, test_start, test_end
end


function load_data(dir)
    @info "Reading data..."
    data = CSV.read(string(dir, "/data.csv"))

    @info "Reading blocks"
    blocks = CSV.read(string(dir, "/blocks.csv"))

    @info "Filtering dataset"
    train_start, train_end, test_start, test_end = read_train_conf(dir, data)

    train_len = train_end - train_start + 1
    test_len = test_end - test_start + 1

    @info "Converting matrices...", "train/test set split is" train = train_len test = test_len
    
    Xtr = convert(Matrix, data[train_start:train_end, setdiff(names(data), [:y])])
    Xte = convert(Matrix, data[test_start:test_end, setdiff(names(data), [:y])])
    ytr = convert(Array, data[train_start:train_end, :y])
    yte = convert(Array, data[test_start:test_end, :y])
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