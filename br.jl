push!(LOAD_PATH, ".")

using DataFrames
using Gadfly
using TLLR: fit, predict


# main

data = readtable("LogPTol_vsPlusDescr.csv", separator=';')
blocks = readtable("LogPTol_vsPlusDescr_blocks.csv", separator=';')


X = convert(Array, data[:, 2:83])
y = convert(Array, data[:, :log_Ptol])
P = convert(Array, blocks[:, 2:7])   # should be a matrix of M x K bits, P[m,k] = 1 if feature m is in P[k], 0 otherwise

tll = fit(X, y, P)

a = tll.α.value
b = tll.β.value

println( "loss: $(norm(predict(tll, X)-y))")

#println(barplot(convert(Array, blocks[:Descriptor]), abs(round(a[:,1],6)), title = "α"))
α_plot = plot( x=blocks[:Descriptor], y=round(a[:,1],6), Geom.bar, Theme(minor_label_font_size=3pt) )
# draw( PNG("alpha_plot.png"), α_plot)

#println(barplot(names(blocks)[2:7], abs(b[:,1]), title = "β"))
β_plot = plot( x=names(blocks)[2:7], y=b[:,1], Geom.bar )
# draw( PNG("beta_plot.png"), β_plot)
