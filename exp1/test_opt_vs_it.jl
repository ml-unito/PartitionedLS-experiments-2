using JLD
using CSV
using LinearAlgebra
using Gadfly

function loss(X, y, α, β, t, P; η=1.0)
    _, K = size(P)
    obj = X * (P .* (α * ones(1,K))) * β .+ t - y
    reg = η * norm(β,2)^2
    return norm(obj,2)^2 + reg
end

function indexes(array::BitArray) 
    result = []
    for (index,elem) in enumerate(array)
        if elem
            push!(result, index)
        end
    end

    result
end

function fix_and_normalize(a::Array{Float64,2}, P::Array{Int64,2})
    inds = indexes(a .< 0)
    if inds == []
        return a
    end

    fixes = zeros(length(inds))
    a[inds] = fixes

    sums = (P' * a)'
    new_a = ((P .* a) ./ sums) * ones(size(P,2))
    
    return new_a
end

@info "Reading data..."
data = CSV.read("exp1/LogPTol_vsPlusDescr.csv", delim=';')
blocks = CSV.read("exp1/LogPTol_vsPlusDescr_blocks.csv", delim=';')

@info "Converting matrices..."
Xtr = convert(Matrix, data[1:30, 2:83])
Xte = convert(Matrix, data[31:end, 2:83])
ytr = convert(Array, data[1:30, :log_Ptol])
yte = convert(Array, data[31:end, :log_Ptol])
P = convert(Matrix, blocks[:, 2:7])


br_opt = load("exp1/br_vars.jld")
br_it = load("exp1/iterative__vars.jld")

α_opt = br_opt["α"]
β_opt = br_opt["β"]
t_opt = br_opt["t"]

α_it = br_it["α"]
β_it = br_it["β"]
t_it = br_it["t"]

α_opt = fix_and_normalize(α_opt, P)
α_it = fix_and_normalize(α_it, P)

indexes(α_it .< 0)
indexes(α_opt .< 0)

@info "iterative: " loss(Xtr, ytr, α_it, β_it, t_it, P, η=1.0)
@info "optimal: " loss(Xtr, ytr, α_opt, β_opt, t_opt, P, η=1.0)

for params in [("it", α_it, β_it), ("opt", α_opt, β_opt)]
    name, α, β = params

    @info "Writing α_$(name) plot"
    α_plot = plot( x=blocks[:Descriptor], y=round.(α,digits=6), Geom.bar, Theme(minor_label_font_size=3pt) )
    draw( SVG("alpha_$(name)_plot.svg", 14cm, 10cm), α_plot)

    @info "Writing β_$(name) plot" 
    β_plot = plot( x=names(blocks)[2:7], y=β, Geom.bar )
    draw( SVG("beta_$(name)_plot.svg", 14cm, 10cm), β_plot)
end
