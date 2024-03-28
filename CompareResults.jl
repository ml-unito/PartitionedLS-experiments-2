using DataFrames
using CSV
using Printf
using Statistics
using JSON
using HypothesisTests

include(joinpath(@__DIR__,"PartitionedLS-expio.jl"))

function print_ttest_results(comb)
    columns = [" ", "LS", "Part_P_LS", "Part_P_Orig", "PCLS", "PLS", "Part_P_LS_Opt"]

    df = DataFrame([[],[],[],[],[],[], []] , [:Method, :LS, :Part_P_LS, :Part_P_Orig, :PCLS, :PLS, :Part_P_LS_Opt])

    for method1 in columns
        if method1 == " "
            continue
        end

        row = Vector{Any}()
        push!(row, method1)
        for method2 in columns
            if method2 == " "
                continue
            end

            if  method1 == method2
                push!(row, "-")
                continue
            end
            ttest = OneSampleTTest(comb[!, "Test_$method1"], comb[!, "Test_$method2"])

            if pvalue(ttest) < 0.01
                push!(row, @sprintf "≠ %.2f" pvalue(ttest))
            else
                push!(row, @sprintf "= %.2f" pvalue(ttest))
            end

            # @info "$method1 vs $method2" ttest
        end

        push!(df, row)
    end

    @info df
end


function getStats(datapath)
    @info "Loading experiment config"
    config = read_train_conf(datapath)

    train_size = config["train_end"] - config["train_start"] + 1
    test_size = config["test_end"] - config["test_start"] + 1
    
    @info "Loading data frames"
    dfls = CSV.read("$datapath/LSResults.csv", DataFrame)
    dfpartls = CSV.read("$datapath/PartLSResults-PfromLS.csv", DataFrame)
    dfpartorig = CSV.read("$datapath/PartLSResults-POrig.csv", DataFrame)
    dfpcls = CSV.read("$datapath/PCLSResults.csv", DataFrame)
    dfpartlsopt = CSV.read("$datapath/PartLSResults-PfromLSOpt.csv", DataFrame)
    dfpls = CSV.read("$datapath/PLSResults.csv", DataFrame)

    comb = DataFrame(
        Seed = dfls[!,"Seed"],
        Train_LS = dfls[!,"TrainingError"] / train_size,
        Train_Part_P_LS = dfpartls[!,"TrainingError"] / train_size,
        Train_Part_P_Orig = dfpartorig[!,"TrainingError"] / train_size,
        Train_Part_P_LS_Opt = dfpartlsopt[!,"TrainingError"] / train_size,
        Train_PCLS = dfpcls[!,"TrainingError"] / train_size,
        Train_PLS = dfpls[!,"TrainingError"] / train_size,
        Test_LS = dfls[!,"TestError"] / test_size,
        Test_Part_P_LS = dfpartls[!, "TestError"] / test_size,
        Test_Part_P_Orig = dfpartorig[!,"TestError"] / test_size,
        Test_Part_P_LS_Opt = dfpartlsopt[!,"TestError"] / test_size,
        Test_PCLS = dfpcls[!,"TestError"] / test_size,
        Test_PLS = dfpls[!,"TestError"] / test_size
    )

    if any( row -> any( x-> x isa Number && isnan(x), row), eachrow(comb))
        @warn "Found NaNs in the data, removing them."
        comb = filter( row -> 
            all( x-> !(x isa Number && isnan(x)), row), 
            comb )
    end

    # @info comb

    stats = describe(comb, :all)[!, [:mean, :std]]
    stats.field = names(comb)

    print_ttest_results(comb)

    return (stats=stats,)
end


@info "PartLSArtificial"
artificial = getStats("datasets/PartLSArtificial")

@info "Limpet"
limpet = getStats("datasets/Limpet")

@info "Facebook Comment Volume Dataset"
fb = getStats("datasets/Facebook Comment Volume Dataset")

@info "YearPredictionMSD"
ypred = getStats("datasets/YearPredictionMSD")

@info "Superconductivty Data"
supercond = getStats("datasets/Superconductivty Data")



datasets = [(artificial, "Artificial"), (limpet, "Limpet"), (fb, "Facebook"), (ypred, "Year Prediction"), (supercond, "Superconductivity")]

summary = DataFrame([[],[],[],[],[],[]], [:Dataset, :Method, :TrainMean, :TrainStd, :TestMean, :TestStd])
for dataset in datasets
    for method in ["LS", "Part_P_LS", "Part_P_Orig", "PCLS", "PLS", "Part_P_LS_Opt"]
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
