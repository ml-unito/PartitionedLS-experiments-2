# YearPrediction MSD

using DataFrames
using CSV
using DelimitedFiles
using Tables

function add_block(blocks, cols, pos)
    for col in cols
        new_row = vcat([string(col)], [0 for i in 1:(pos-1)], 1, [0 for i in (pos+1):(ncol(blocks)-1)])
        push!(blocks, new_row)
    end

    return pos + 1
end

function load_X()
    avg_cols = [Symbol("AVG-$i") for i in 1:12]
    cov_cols = [Symbol("COV-$i") for i in 1:78]
    cols = vcat([:y], avg_cols, cov_cols)

    blocks = DataFrame(Descriptor = String[], AVG=Int8[],  COV1=Int8[], COV2=Int8[], COV3=Int8[],COV4=Int8[], COV5=Int8[],COV6=Int8[], COV7=Int8[], COV8=Int8[])

    add_block(blocks, avg_cols, 1)
    add_block(blocks, cov_cols[1:10], 2)
    add_block(blocks, cov_cols[11:20], 3)
    add_block(blocks, cov_cols[21:30], 4)
    add_block(blocks, cov_cols[31:40], 5)
    add_block(blocks, cov_cols[41:50], 6)
    add_block(blocks, cov_cols[51:60], 7)
    add_block(blocks, cov_cols[61:70], 8)
    add_block(blocks, cov_cols[71:78], 9)

    @info "Reading YearPredictionMSD.txt\n"
    df = CSV.read("YearPredictionMSD.txt", header=cols, DataFrame)
    

    @info "Moving y column to the end of the dataframe"
    df = hcat(df[:, 2:end], select(df, :y))

    return df, blocks
end

df, blocks = load_X()

@info "Saving blocks..."
CSV.write("blocks.csv", blocks)

@info("Saving data...\n")
open("data.csv", "w") do io
    writedlm(io, [[string(s) for s in names(df)]], ",")
    writedlm(io, Tables.matrix(df[:,:]),",")
end