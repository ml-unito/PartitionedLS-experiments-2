using JSON
using CSV
using DataFrames
using Tables
using PartitionedLS
using Random

include(joinpath(@__DIR__,"PartitionedLS-expio.jl"))

results = DataFrame(
    Seed = Int[],
    TrainingError = Float64[],
    TestError = Float64[]
)

for seed in 0:10 
    conf = read_train_conf("datasets/Limpet")
    Xtr, Xte, ytr, yte, P = load_data("datasets/Limpet", conf, blocksfname="LSBlocks-$seed.csv", shuffle = true, seed = seed) 
    ytr = vec(ytr)

    # Least squares on Xtr, ytr
    opt, a, b, t, P = PartitionedLS.fit(Opt, Xtr, ytr, P, η = 0.0)

    preds_tr = PartitionedLS.predict(vec(a),vec(b),t,P, Xtr)
    preds_te = PartitionedLS.predict(vec(a),vec(b),t,P, Xte)

    # Compute errors
    err_tr = sum((preds_tr .- ytr).^2)
    err_te = sum((preds_te .- yte).^2)

    push!(results, (seed, err_tr, err_te))

    @info "Train error: "  TrainingError = err_tr TestError = err_te
end

CSV.write("datasets/Limpet/PartLSResults.csv", results)
