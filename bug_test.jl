using Convex
using ECOS


function dump_problem(problem)
  c, A, b, cones, var_to_ranges, vartypes, conic_constraints = conic_problem(problem)

  println("c: $c")
  println("A: $A")
  println("b: $b")
  println("cones: $cones")
  println("var_to_ranges: $var_to_ranges")
  println("vartypes: $vartypes")
  println("conic_constraints: $conic_constraints")
end

# sample P matrix
P = [
  [1 0 0 0]
  [1 0 0 0]
  [1 0 0 0]
  [0 1 0 0]
  [0 1 0 0]
  [0 1 0 0]
  [0 0 1 0]
  [0 0 1 0]
  [0 0 0 1]
  [0 0 0 1]
  [0 0 0 1]
]

M,K = size(P)
N = 50

# generating parameters to compute the target model
a = rand(M)
b = rand(K)

# generating the data for the example
X = rand(N,M)
y = X * (P .* (a * ones(1,K))) * b

# formulating the problem
α = ones(M)
α = Variable(M)
α.value = ones(M)

β = Variable(K)
β.value = [1,-1,1,-1, ]

model = P .* (α * ones(1,K)) * β
loss = vecnorm( X * model - y)

# solving the problem and checking the results
println("Before running $(evaluate(loss))")
println(β.value)

fix!(α)
problem = minimize(loss)

dump_problem(problem)
solve!(problem, ECOSSolver())

println(β.value)
println(problem.optval)

println(α.value)
