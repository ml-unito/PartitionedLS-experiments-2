using DataFrames
using CSV
using Printf
using Statistics
using JSON
using HypothesisTests

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
    dfpcls = CSV.read("$datapath/PCLSResults.csv", DataFrame)

    comb = DataFrame(
        Seed = dfls[!,"Seed"],
        Train_LS = dfls[!,"TrainingError"] / train_size,
        Train_Part_P_LS = dfpartls[!,"TrainingError"] / train_size,
        Train_Part_P_Orig = dfpartorig[!,"TrainingError"] / train_size,
        Train_PCLS = dfpcls[!,"TrainingError"] / train_size,
        Test_LS = dfls[!,"TestError"] / test_size,
        Test_Part_P_LS = dfpartls[!, "TestError"] / test_size,
        Test_Part_P_Orig = dfpartorig[!,"TestError"] / test_size,
        Test_PCLS = dfpcls[!,"TestError"] / test_size,
    )

    if any( row -> any( x-> x isa Number && isnan(x), row), eachrow(comb))
        @warn "Found NaNs in the data, removing them."
        comb = filter( row -> 
            all( x-> !(x isa Number && isnan(x)), row), 
            comb )
    end


    println(comb)

    stats = describe(comb, :all)[!, [:mean, :std]]
    stats.field = names(comb)

    ttest_vs_P_LS = OneSampleTTest(comb[!,"Test_LS"], comb[!,"Test_Part_P_LS"])
    ttest_vs_P_Orig = OneSampleTTest(comb[!,"Test_LS"], comb[!,"Test_Part_P_Orig"])
    ttest_vs_PCLS = OneSampleTTest(comb[!,"Test_LS"], comb[!,"Test_PCLS"])

    return (stats=stats, tt_P_LS = ttest_vs_P_LS, tt_P_Orig =ttest_vs_P_Orig, tt_PCLS = ttest_vs_PCLS)
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



datasets = [(artificial, "Artificial"), (limpet, "Limpet"), (fb, "Facebook"), (ypred, "Year Prediction"), (supercond, "Superconductivity")]
summary = DataFrame([[],[],[],[],[],[]], [:Dataset, :Method, :TrainMean, :TrainStd, :TestMean, :TestStd])
for dataset in datasets
    @info dataset[2] dataset[1].tt_P_LS dataset[1].tt_P_Orig

    for method in ["LS", "Part_P_LS", "Part_P_Orig", "PCLS"]
        push!(summary,
            [   dataset[2],
                replace(method, '_' => '-'),
                (@sprintf "%.4f" filter(row -> row.field == "Train_$method", dataset[1].stats)[1, :mean]),
                (@sprintf "%.2f" filter(row -> row.field == "Train_$method", dataset[1].stats)[1, :std]),
                (@sprintf "%.4f" filter(row -> row.field == "Test_$method",  dataset[1].stats)[1, :mean]),
                (@sprintf "%.2f" filter(row -> row.field == "Test_$method",  dataset[1].stats)[1, :std]),
            ]
        )
    end
end



@info summary

CSV.write("all-results.csv", summary)
