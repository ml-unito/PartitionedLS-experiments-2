using Convex

using DataFrames

data = readtable("LogPTol_vsPlusDescr.csv", separator=';')
blocks = readtable("LogPTol_vsPlusDescr_blocks.csv", separator=';')


X = convert(Array, data[:, 2:83])
y = convert(Array, data[:, :log_Ptol])
P = convert(Array, blocks[:, 2:7])

M,K = size(P)
α = Variable(M)
α.value = ones(M)

β = Variable(K)
β.value = [1,-1,1,-1,1,1]

model = P .* (α * ones(1,K)) * β
loss = norm( X * model - y)

println(β.value)

fix!(α)
problem = minimize(loss)
solve!(problem)

println(β.value)

println(problem.optval)
