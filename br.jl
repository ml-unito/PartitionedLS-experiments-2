using Convex
using DataFrames
using Gadfly

data = readtable("LogPTol_vsPlusDescr.csv", separator=';')
blocks = readtable("LogPTol_vsPlusDescr_blocks.csv", separator=';')

X = convert(Array, data[:, 2:83])
y = convert(Array, data[:, :log_Ptol])
P = convert(Array, blocks[:, 2:7])   # should be a matrix of M x K bits, P[m,k] = 1 if feature m is in P[k], 0 otherwise

# row normalization
X = X ./ sum(X,2)

M = size(X,2)
K = size(P,2)

α = Variable(M, Positive())
β = Variable(K)

# initialization of β to a random unit vector
β.value = rand(K)
β.value = β.value / norm(β.value)

# termination criteria is guided by epsilon... when two iterations optvalue
# differ for a quantity that is less than epslon, we stop
ϵ = 0.01
δ = 1.0

previous_optval = 100

i=0
maxiterations = 5

while δ > ϵ && i < maxiterations
  # α problem: solve for best αs keeping β constant
  b = β.value
  # the following shuold be
  # prediction_α =  X * (P .* α) * b
  # the current form is a workaround for a bug in Convex.jl .* implementation
  prediction_α =  X * (P .* (α * ones(1,K))) * b
  problem_α = minimize( norm(prediction_α - y) )
  solve!(problem_α)

  # β problem: solve for best βs keeping α constant
  a = α.value
  prediction_β = X * (P .* a) * β
  problem_β = minimize( norm(prediction_β - y) )
  solve!(problem_β)

  δ = abs(previous_optval - problem_β.optval)
  previous_optval = problem_β.optval
  i+=1

  println("δ: $δ")
  println("α: $(α.value)")
  println("β: $(β.value)")
end

println("α: $(α.value)")
println("β: $(β.value)")

a = α.value
b = β.value
loss = norm( X * (P .* a) * b - y)

println("loss: $loss")


#println(barplot(convert(Array, blocks[:Descriptor]), abs(round(a[:,1],6)), title = "α"))
α_plot = plot( x=blocks[:Descriptor], y=round(a[:,1],6), Geom.bar, Theme(minor_label_font_size=3pt) )
draw( PNG("alpha_plot.png"), α_plot)

#println(barplot(names(blocks)[2:7], abs(b[:,1]), title = "β"))
β_plot = plot( x=names(blocks)[2:7], y=b[:,1], Geom.bar )
draw( PNG("beta_plot.png"), β_plot)
