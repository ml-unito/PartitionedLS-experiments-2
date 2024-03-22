using DataFrames
using CSV
using Printf
using Statistics
using JSON

include(joinpath(@__DIR__,"PartitionedLS-expio.jl"))

function printStats(datapath)
    @info "Loading experiment config"
    config = read_train_conf(datapath)

    train_size = config["train_end"] - config["train_start"] + 1
    test_size = config["test_end"] - config["test_start"] + 1
    
    @info "Loading data frames"
    dfls = CSV.read("$datapath/LSResults.csv", DataFrame)
    dfpartls = CSV.read("$datapath/PartLSResults-PfromLS.csv", DataFrame)
    dfpartorig = CSV.read("$datapath/PartLSResults-POrig.csv", DataFrame)

    @info names(dfls)

    comb = DataFrame(
        Seed = dfls[!,"Seed"],
        Train_LS = dfls[!,"TrainingError"] / train_size,
        Train_Part_P_LS = dfpartls[!,"TrainingError"] / train_size,
        Train_Part_P_Orig = dfpartorig[!,"TrainingError"] / train_size,
        Test_LS = dfls[!,"TestError"] / test_size,
        Test_Part_P_LS = dfpartls[!, "TestError"] / test_size,
        Test_Part_P_Orig = dfpartorig[!,"TestError"] / test_size,
        Test_Diff_LS_vs_P_LS = (dfls[!, "TestError"] - dfpartls[!,"TestError"]) / test_size,
        Test_Diff_LS_vs_P_Orig = (dfls[!, "TestError"] - dfpartorig[!,"TestError"]) / test_size
    )

    comb = filter( row -> 
                all( x-> !(x isa Number && isnan(x)), row), 
                comb )

    println(comb)

    stats = describe(comb, :all)[!, [:mean, :std]]
    stats.field = names(comb)
     
    println(stats[!,[:field, :mean, :std]])

    return stats
end


@info "PartLSArtificial"
artificial = printStats("datasets/PartLSArtificial")

@info "Limpet"
limpet = printStats("datasets/Limpet")

@info "Facebook Comment Volume Dataset"
fb = printStats("datasets/Facebook Comment Volume Dataset")

@info "YearPredictionMSD"
ypred = printStats("datasets/YearPredictionMSD")

@info "Superconductivty Data"
supercond = printStats("datasets/Superconductivty Data")


