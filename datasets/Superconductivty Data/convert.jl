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
    df = CSV.read("train.csv", DataFrame)
    cols = names(df)


    @info "Reading train.csv\n"
    blocks = DataFrame(Descriptor = String[],  BLK1=Int8[], BLK2=Int8[], BLK3=Int8[],BLK4=Int8[], BLK5=Int8[],BLK6=Int8[], BLK7=Int8[], BLK8=Int8[])

    add_block(blocks, cols[1:10], 1)
    add_block(blocks, cols[11:20], 2)
    add_block(blocks, cols[21:30], 3)
    add_block(blocks, cols[31:40], 4)
    add_block(blocks, cols[41:50], 5)
    add_block(blocks, cols[51:60], 6)
    add_block(blocks, cols[61:70], 7)
    add_block(blocks, cols[71:81], 8)

    # rename!(df, [cols[1:81]; :y] )
    rename!(df, :critical_temp => :y)

    return df, blocks
end

df, blocks = load_X()

@info "Saving blocks..."
CSV.write("blocks.csv", blocks)

@info("Saving data...\n")
open("data.csv", "w") do io
    writedlm(io, [[string(s) for s in names(df)]], ",")
    writedlm(io, Matrix(df),",")
end
