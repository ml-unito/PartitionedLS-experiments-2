include("./TLLR.jl")


using DataFrames
using Gadfly
using TLLR: fit, predict, alpha, beta, optval
using Convex

# main

data = readtable("LogPTol_vsPlusDescr.csv", separator=';')
blocks = readtable("LogPTol_vsPlusDescr_blocks.csv", separator=';')

X = convert(Array, data[:, 2:83])
y = convert(Array, data[:, :log_Ptol])
P = convert(Array, blocks[:, 2:7])

tll = fit(X, y, P, beta=[1,-1,1,-1,1,1], verbose=0)
# tll = fit(X, y, P, verbose=0)

α = alpha(tll)
β = beta(tll)

println( "loss: $(norm(predict(tll, X)-y))")
println("α: $α")
println("β: $β")

println("sum cnstr: $(evaluate(sum(tll.α' * tll.P, 1)))")

if length(ARGS) == 1 && ARGS[1]=="-p"
  α_plot = plot( x=blocks[:Descriptor], y=round(α,6), Geom.bar, Theme(minor_label_font_size=3pt) )
  draw( SVG("alpha_plot.svg", 14cm, 10cm), α_plot)
  β_plot = plot( x=names(blocks)[2:7], y=β, Geom.bar )
  draw( SVG("beta_plot.svg", 14cm, 10cm), β_plot)
end
