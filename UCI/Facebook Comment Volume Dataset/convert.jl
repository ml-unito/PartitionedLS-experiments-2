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
    df = CSV.read("dataset_orig.csv")
    df = df[:, [1:37; 39:54]] # removing column #38 since it's the zero vector
    cols = [Symbol("col$i") for i in 1:53]


    @info "Reading train.csv\n"
    blocks = DataFrame(Descriptor = String[],  BLK1=Int8[], BLK2=Int8[], BLK3=Int8[],BLK4=Int8[], BLK5=Int8[])

    add_block(blocks, cols[1:10], 1)
    add_block(blocks, cols[11:20], 2)
    add_block(blocks, cols[21:30], 3)
    add_block(blocks, cols[31:40], 4)
    add_block(blocks, cols[41:52], 5)

    rename!(df, [cols[1:52]; :y] )

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