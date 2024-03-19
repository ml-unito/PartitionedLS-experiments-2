using JSON
using CSV
using DataFrames
using Tables
using PartitionedLS

include(joinpath(@__DIR__,"PartitionedLS-expio.jl"))


conf = read_train_conf("datasets/Limpet")
Xtr, Xte, ytr, yte, P = load_data("datasets/Limpet", conf)

ytr = vec(ytr)

print(size(Xtr))
print(size(ytr))
print(size(P))

# Least squares on Xtr, ytr
opt, a, b, t, P = PartitionedLS.fit(OptNNLS, Xtr, ytr, P)

print(a)

preds_tr = PartitionedLS.predict(vec(a),vec(b),t,P, Xtr)
preds_te = PartitionedLS.predict(vec(a),vec(b),t,P, Xte)

# Compute errors
err_tr = sum((preds_tr .- ytr).^2)
err_te = sum((preds_te .- yte).^2)

println("Train error: ", err_tr)
println("Test error: ", err_te)
