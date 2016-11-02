using NLopt
using DataFrames
using Base.Test

type OptProblem
  X
  y
  P
  N
  M
  K
end

function test(optproblem::OptProblem)
  M = optproblem.M
  K = optproblem.K

  @testset "Objective_finit_diff test" begin
    x = rand(M+K)
    grad = zeros(M+K)
    p0 = objective(x, grad, optproblem)
    fd = objective_finite_diff(x,optproblem)
    p1_ = p0 + fd[1] * 0.001

    x[1] += 0.001
    p1 = objective(x, grad, optproblem)

    @test_approx_eq_eps(p1, p1_, 0.001)

  end

  @testset "Sanity checks" begin
    x = rand(M+K)
    grad = zeros(M+K)
    objective(x, grad, optproblem)
    fd = objective_finite_diff(x,optproblem)
    for i in 1:(M+K)
      println("testing $i")
      @test_approx_eq(grad[i],fd[i])
    end
  end
end

type OptProblem
  X
  y
  P
  N
  M
  K
end

function objective_finite_diff(x::Vector, optproblem::OptProblem)
  α = x[1:optproblem.M]
  β = x[(optproblem.M+1):(optproblem.M + optproblem.K)]
  M = optproblem.M
  K = optproblem.K
  grad = zeros(length(x))
  X = optproblem.X
  P = optproblem.P
  y = optproblem.y

  Δ = 0.001
  obj = (a,b) -> norm(X * (P .* a) * b - y)^2

  for m in 1:M
    Δα = zeros(M)
    Δα[m] = Δ

    fplusΔα = obj(α + Δα, β)
    fminusΔα = obj(α - Δα, β)
    grad[m] = (fplusΔα - fminusΔα) / 2Δ
  end

  for k in 1:K
    Δβ = zeros(K)
    Δβ[k] = Δ

    fplusΔβ = obj(α,β + Δβ)
    fminusΔβ = obj(α,β - Δβ)
    grad[M+k] = (fplusΔβ - fminusΔβ) / 2Δ
  end

  grad
end

# x: M + K vector
function objective(x::Vector, grad::Vector, optproblem::OptProblem)
  α = x[1:optproblem.M]
  β = x[(optproblem.M+1):(optproblem.M + optproblem.K)]

  ρ = optproblem.X * (optproblem.P .* α) * β - optproblem.y

  if length(grad) > 0
    for m = 1:optproblem.M
      k = find(optproblem.P[m,:])
      if(length(k)!=1)
        warn("Found more than one k index: $k $(length(k))")
        exit(1)
      else
        k = k[1]
      end

      grad[m] = (2β[k] * (ρ' * optproblem.X[:,m]))[1]
    end


    # Column k in sumk = sum_{m \in P_k} α_m * x_nm}
    sumk = optproblem.X * (optproblem.P .* α)
    for k = (optproblem.M + 1):(optproblem.M + optproblem.K)
      grad[k] = (2ρ' * sumk[:,k - optproblem.M])[1]
    end

    # println("Computent gradient: $(grad)")
    # println("Finite diff gradient: $(objective_finite_diff(x, optproblem))")
    # grad = objective_finite_diff(x,optproblem)
  end

  return norm(ρ)^2
end

# function alpha_gt_zero(c::Vector, x::Vector, grad::Vector)
#     if length(grad) > 0
#       grad[:,:] = zeros(size(grad))
#       for m = 0:length(c)
#         grad[m,m] = -1
#       end
#     end
#
#     for m = 0:length(c)
#       c[m] = -x[m]
#     end
# end

function alphak_sum_to_1(c::Vector, x::Vector, grad::Matrix, optproblem::OptProblem)
  if length(grad) > 0
    grad[:,:] = optproblem.P
  end

  c[:] = P' * α .- 1
end

function alphak_sum_test(c::Vector, x::Vector, grad::Matrix)
  grad
end

data = readtable("LogPTol_vsPlusDescr.csv", separator=';')
blocks = readtable("LogPTol_vsPlusDescr_blocks.csv", separator=';')

X = convert(Array, data[:, 2:83])
y = convert(Array, data[:, :log_Ptol])
P = convert(Array, blocks[:, 2:7])

N,M = size(X)
K = size(P,2)

optproblem = OptProblem(X,y,P,N,M,K)

# test(optproblem)


opt = Opt(:LD_MMA, M+K)
lower_bounds!(opt, [zeros(M) ; (zeros(K) - Inf)])
xtol_rel!(opt,1e-8)

min_objective!(opt, (x,g) -> objective(x,g,optproblem) )
# equality_constraint!(opt, (c,x,g) -> alphak_sum_to_1(c,x,g,optproblem) ,zeros(K) + 1e-8)
equality_constraint!(opt, alphak_sum_test ,zeros(K) + 1e-8)

(optf,optx,ret) = optimize(opt, [rand(M); [1,-1,1,-1,1,1]])

print("optimization done")

α = optx[1:M]
β = optx[M+1:M+K]

println("α: $α")
println("β: $β")
println("optf: $optf")
println("ret: $ret")
