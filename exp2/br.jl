using Gadfly
using TLLR: fit, predict
using DataFrames
using CSV
using LinearAlgebra

# main

@info "Reading data"
ds_train = CSV.read("exp2/esempio2_train.csv", delim=';')
ds_test = CSV.read("exp2/esempio2_test.csv", delim=';')
blocks = CSV.read("exp2/LogPTol_vsPlusDescr_blocks.csv", delim=';')
blocks_colnames = blocks[:Descriptor]
ds_colnames = string.(names(ds_train))
colindices = indexin(blocks_colnames, ds_colnames)

@info "Converting data"
Xtr = convert(Matrix{Float64}, ds_train[:, colindices])
Xte = convert(Matrix{Float64}, ds_test[:, colindices])
ytr = convert(Array{Float64}, ds_train[:, :Log_Ptol])
yte = convert(Array{Float64}, ds_test[:, :Log_Ptol])
P = convert(Matrix, blocks[:, 2:7])

@info "Fitting model"
tll = fit(Xtr, ytr, P, verbose=0, η=10)
objvalue, α, β, t, _ = tll

@info α, β, t
@info objvalue 

@info "Evaluating performances"
ypls =[4.04152, 2.1756, 2.99146, 4.15148, 3.4963, 1.01396, 3.86776, 0.836909, -0.226174, 0.864335, 1.17424, 0.373529, 2.08099, -0.78, 1.09163, 0.0058372, 1.00101, -0.511264, 3.36397, 0.229335, 0.549681, 1.05189, 0.781232, 2.67853, -1.73915, -1.34271, 1.32651, 0.685616, 2.11528, 0.326774, 0.968697, -0.935262, 3.08746, 3.34281, 1.91912, 2.6557, 2.9992, 6.15191, 5.52205, 2.13303, 4.1696, 2.44663, 1.10757, 1.42657, 0.782802, 0.379736, 2.58375, 2.15175, 3.65936, 3.50455, -0.28716, 1.28474, 0.318317, 2.14827, 0.270728, -0.715285, 0.913745, 3.52522, 2.02935, 2.20636, 3.35802, 3.11136, ]

ytr_ttlr = predict(tll, Xtr)
yte_ttlr = predict(tll, Xte)
test_loss = norm(yte_ttlr - yte)^2   # η:1 -> 46.1, η:20 -> 42.1, η:0 -> 52.8
pls_loss = norm(ypls -yte)^2

@info "loss:" test_loss

@info "Writing CSV files"
CSV.write("exp2/y_test_ttlr.csv", DataFrame(yte_ttlr))
CSV.write("exp2/y_train_ttlr.csv", DataFrame(ytr_ttlr))

@info "Creating SVGs"
if length(ARGS) == 1 && ARGS[1]=="-p"
for name in names(blocks)[2:7]
  α_k = α[find(blocks[Symbol(name)])]
  α_k_names = blocks[:Descriptor][find(blocks[Symbol(name)])]
  α_plot = plot( x=α_k_names, y=round(α_k,6), Geom.bar, Theme(minor_label_font_size=3pt) )
  draw( SVG(string("exp2/alpha_plot",name,".svg"), 14cm, 10cm), α_plot)
end

β_plot = plot( x=names(blocks)[2:7], y=β, Geom.bar )
draw( SVG("exp2/beta_plot.svg", 14cm, 10cm), β_plot)
end
