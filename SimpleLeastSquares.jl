using JSON
using CSV
using DataFrames
using Tables
using LinearAlgebra
using Random

function sign(w)
    return [ifelse(w[i] >= 0, 1, -1) for i in 1:length(w)]
end

include(joinpath(@__DIR__,"PartitionedLS-expio.jl"))


datadir = "datasets/Limpet"

conf = read_train_conf(datadir)

results = DataFrame(
    Seed = Int[],
    TrainingError = Float64[],
    TestError = Float64[]
)

for seed in 0:10 
    Xtr, Xte, ytr, yte, P = load_data(datadir, conf, shuffle = true, seed = seed)

    # Least squares on Xtr, ytr

    Xtr = [ones(size(Xtr, 1)) Xtr]
    Xte = [ones(size(Xte, 1)) Xte]

    w = Xtr \ ytr
    # w = inv(Xtr' * Xtr) * Xtr' * ytr
    # w = pinv(Xtr, atol=100, rtol=100) * ytr

    # Predictions

    yhat_tr = Xtr * w
    yhat_te = Xte * w

    # Compute errors

    err_tr = sum((yhat_tr .- ytr).^2)
    err_te = sum((yhat_te .- yte).^2)

    push!(results, (seed, err_tr, err_te))

    @info "Training/Test errors:" TrainingError = err_tr TestError = err_te


    rownames = ["V", "S", "R", "G", "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "WO1", "WO2", "WO3", "WO4", "WO5", "WO6", "WN1", "WN2", "WN3", "WN4", "WN5", "WN6", "IW1", "IW2", "IW3", "IW4", "CW1", "CW2", "CW3", "CW4", "CW5", "CW6", "CW7", "CW8", "ID1", "ID2", "ID3", "ID4", "CD1", "CD2", "CD3", "CD4", "CD5", "CD6", "CD7", "CD8", "HL1", "HL2", "A", "CP", "PSA", "HSA", "PSAR", "PHSAR", "DRDRDR", "DRDRAC", "DRDRDO", "DRACAC", "DRACDO", "DRDODO", "ACACAC", "ACACDO", "ACDODO", "DODODO", "DD1", "DD2", "DD3", "DD4", "DD5", "DD6", "DD7", "DD8"]
    signs = sign(w[2:end])

    df = DataFrame(
        Descriptor = rownames,
        Pos = [ifelse(signs[i] == 1, 1, 0) for i in 1:length(signs)],
        Neg = [ifelse(signs[i] == -1, 1, 0) for i in 1:length(signs)]
    )

    CSV.write("$datadir/LSBlocks-$seed.csv", df)
end

CSV.write("$datadir/LSResults.csv", results)
