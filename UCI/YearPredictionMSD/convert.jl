# YearPrediction MSD

using DataFrames
using CSV
using DelimitedFiles

function add_block(blocks, cols, pos)
    for col in cols
        new_row = vcat([string(col)], [0 for i in 1:(pos-1)], 1, [0 for i in (pos+1):(ncol(blocks)-1)])
        @info new_row
        push!(blocks, new_row)
    end

    return pos + 1
end

function load_X()
    avg_cols = [Symbol("AVG-$i") for i in 1:12]
    cov_cols = [Symbol("COV-$i") for i in 1:78]
    cols = vcat([:y], avg_cols, cov_cols)

    blocks = DataFrame(Descriptor = String[], AVG=Int8[],  COV=Int8[])

    add_block(blocks, avg_cols, 1)
    add_block(blocks, cov_cols, 2)

    @info "Reading YearPredictionMSD.txt\n"
    df = CSV.read("YearPredictionMSD.txt", header=cols)
    

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
    writedlm(io, convert(Matrix, df[:,:]),",")
end