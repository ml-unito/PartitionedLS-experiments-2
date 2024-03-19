using JSON
using CSV
using DataFrames
using Tables

include(joinpath(@__DIR__,"PartitionedLS-expio.jl"))


conf = read_train_conf("datasets/Limpet")
Xtr, Xte, ytr, yte, P = load_data("datasets/Limpet", conf)

# Least squares on Xtr, ytr

Xtr = [ones(size(Xtr, 1)) Xtr]
Xte = [ones(size(Xte, 1)) Xte]

w = Xtr \ ytr

# Predictions

yhat_tr = Xtr * w
yhat_te = Xte * w

# Compute errors

err_tr = sum((yhat_tr .- ytr).^2)
err_te = sum((yhat_te .- yte).^2)

println("Train error: ", err_tr)
println("Test error: ", err_te)