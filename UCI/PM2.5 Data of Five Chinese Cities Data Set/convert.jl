# YearPrediction MSD

using DataFrames
using CSV
using DelimitedFiles

function add_block(blocks, cols, pos)
    for col in cols
        new_row = vcat([string(col)], [0 for i in 1:(pos-1)], 1, [0 for i in (pos+1):(ncol(blocks)-1)])
        push!(blocks, new_row)
    end

    return pos + 1
end

function load_X()
    # TODO
    df = CSV.read("", header=cols)
    
    return df, blocks
end

df, blocks = load_X()

@info "Saving blocks..."
CSV.write("blocks.csv", blocks)

@info("Saving data...\n")
open("data.csv", "w") do io
    writedlm(io, [[string(s) for s in names(df)]], ",")
    writedlm(io, convert(Matrix, df[:,:]),",")
end