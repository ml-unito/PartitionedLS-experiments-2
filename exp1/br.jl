using TLLR: fit, predict
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

@info "Fitting the model"
tll = fit(Xtr, ytr, P, verbose=0, η=1.0)
objvalue, α, β, t, _ = tll

@info "objvalue: $objvalue"
@info "loss:" norm(predict(tll, Xte) - yte)^2

@info "Saving variables" α β t
save("exp1/br_vars.jld", "objvalue", objvalue, "α", α, "β", β, "t", t)



# for i=1:82
#   @printf("α: %2.4f\n",α[i])
# end
# 
# for i=1:6
#   @printf("β: %2.4f\n", β[i])
# end
# 
# println("t: $t")

# if length(ARGS) == 1 && ARGS[1]=="-p"
# @info "Wriging α plot"
# α_plot = plot( x=blocks[:Descriptor], y=round.(α,digits=6), Geom.bar, Theme(minor_label_font_size=3pt) )
# draw( SVG("alpha_plot.svg", 14cm, 10cm), α_plot)

# @info "Wrigin β plot" 
# β_plot = plot( x=names(blocks)[2:7], y=β, Geom.bar )
# draw( SVG("beta_plot.svg", 14cm, 10cm), β_plot)
# end
