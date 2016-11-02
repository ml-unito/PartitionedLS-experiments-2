using Convex
using DataFrames
using Gadfly
# main

data = readtable("LogPTol_vsPlusDescr.csv", separator=';')


X = convert(Array, data[:, 2:83])
y = convert(Array, data[:, :log_Ptol])

α = Variable(size(X,2))
b = Variable()
problem = minimize( norm( X * α - y), [α >= 0] )
solve!(problem)

println("optval: $(problem.optval)")
println("norm(X*α - y): $(evaluate(norm(X*α - y)))")
