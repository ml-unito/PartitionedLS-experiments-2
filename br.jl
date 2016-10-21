include("./TLLR.jl")


using DataFrames
using Gadfly
using TLLR: fit, predict, alpha, beta, optval


# main

data = readtable("LogPTol_vsPlusDescr.csv", separator=';')
blocks = readtable("LogPTol_vsPlusDescr_blocks.csv", separator=';')


X = convert(Array, data[:, 2:83])
y = convert(Array, data[:, :log_Ptol])
P = convert(Array, blocks[:, 2:7])

tll = fit(X, y, P, beta=[0.1,0.1,0.1,0.1,0.1,1.0])

α = alpha(tll)
β = beta(tll)

println( "loss: $(norm(predict(tll, X)-y))")

#println(barplot(convert(Array, blocks[:Descriptor]), abs(round(a[:,1],6)), title = "α"))
α_plot = plot( x=blocks[:Descriptor], y=round(α,6), Geom.bar, Theme(minor_label_font_size=3pt) )
draw( SVG("alpha_plot.svg", 14cm, 10cm), α_plot)
#
#println(barplot(names(blocks)[2:7], abs(b[:,1]), title = "β"))
β_plot = plot( x=names(blocks)[2:7], y=β, Geom.bar )
draw( SVG("beta_plot.svg", 14cm, 10cm), β_plot)
