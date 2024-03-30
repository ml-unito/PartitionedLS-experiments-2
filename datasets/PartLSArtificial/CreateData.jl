using DataFrames
using CSV
using Random


function createData(blocks, signs, dataError)
    w = []
    for (i, n) in enumerate(blocks)
        wblock = rand(n)
        w = vcat(w, signs[i] * wblock ./ sum(wblock))
    end

    t = rand()

    data = DataFrame()

    for i in 1:sum(blocks)
        data[!, "c$i"] = randn(numExamples)
    end

    y = Matrix(data) * w .+ t 

    # Adding random gaussian noise to the X matrix
    for i in 1:sum(blocks)
        data[!, "c$i"] += randn(numExamples) * dataError
    end

    data[!, :y] = y

    return data
end

function getBlock(blocks, n)
    w = []
    for i in 1:length(blocks)
        if i != n
            w = vcat(w, Int.(zeros(blocks[i])))
        else
            w = vcat(w, Int.(ones(blocks[i])))
        end
    end

    return w
end

function createBlockMatrix(blocks)
    df = DataFrame()
    df[!, :Descriptor] = ["c$i" for i in 1:sum(blocks)]

    for (i, b) in enumerate(blocks)
        df[!, "g$i"] = getBlock(blocks, i)
    end

    return df 
end


numExamples = 1000

# num features per block
blocks = [
    5, 10, 4, 12, 6
]

signs = [
    -11, 4, 2, -1, 3
]

dataError = 0.1

Random.seed!(123)

data = createData(blocks, signs, dataError)
blockmatrix = createBlockMatrix(blocks)

CSV.write("data.csv", data)
CSV.write("blocks.csv", blockmatrix)
