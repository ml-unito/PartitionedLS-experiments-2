include("../TLLR.jl")

# using Gadfly
using TLLR: fit, predict
using Convex
using DataFrames

# main

data = readtable("LogPTol_vsPlusDescr.csv", separator=';')
blocks = readtable("LogPTol_vsPlusDescr_blocks.csv", separator=';')

Xtr = convert(Array, data[1:30, 2:83])
Xte = convert(Array, data[31:end, 2:83])
ytr = convert(Array, data[1:30, :log_Ptol])
yte = convert(Array, data[31:end, :log_Ptol])
P = convert(Array, blocks[:, 2:7])

tll = fit(Xtr, ytr, P, verbose=0)

objvalue, α, β, t, _ = tll
println("objvalue: $objvalue")
println("loss: $(norm(predict(tll, Xte) - yte)^2)")
for i=1:82
  @printf("α: %2.4f\n",α[i])
end

for i=1:6
  @printf("β: %2.4f\n", β[i])
end

println("t: $t")

# # if length(ARGS) == 1 && ARGS[1]=="-p"
# α_plot = plot( x=blocks[:Descriptor], y=round(α,6), Geom.bar, Theme(minor_label_font_size=3pt) )
#   # draw( SVG("alpha_plot.svg", 14cm, 10cm), α_plot)

# β_plot = plot( x=names(blocks)[2:7], y=β, Geom.bar )
#   # draw( SVG("beta_plot.svg", 14cm, 10cm), β_plot)
# # end
