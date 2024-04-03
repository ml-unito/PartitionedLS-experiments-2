
# Returns the ordinal value for the given value name or missing if the value is not in the categories
# function apply_ordinal_mapping(value_name, categories)
#     feature_map = Dict(zip(categories, 0:length(categories)-1))
#     return get!(feature_map, value_name, missing)
# end

function ordinal_mapping(categories)
    return Dict(zip(categories, 0:length(categories)-1))
end

function onehot_features!(Xdf::DataFrame, feature)
    unique_values = unique(Xdf[!, feature])
    for value in unique_values
        new_col_name = Symbol("$(feature)_$(value)")
        Xdf[!, new_col_name] = ifelse.(Xdf[!, feature] .== value, 1, 0)
    end
    select!(Xdf, Not(Symbol(feature)))
end

# String to Floats
function parseStrings(stringType, stringList)
    NAIndexes = stringList .== "NA"
    notNAIndexes = stringList .!= "NA"
    stringList[NAIndexes] .= "0"
    result = parse.(Float64, stringList)
    meanValue = mean(result[notNAIndexes])
    result[NAIndexes] .= meanValue
    return result
end


function preprocess_categorical_features(Xdf::DataFrame)
    # Apply one-hot encoding for specified features
    onehot_features_list = [
        "LandContour", "LotConfig", "Neighborhood", "Condition1",
        "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl",
        "MasVnrType", "Heating", "Exterior1st", "Exterior2nd", "MasVnrType",
        "Electrical", "MiscFeature", "SaleType", "SaleCondition", "MSZoning", "MSSubClass"
    ]
    for feature in onehot_features_list
        if feature in names(Xdf)
            onehot_features!(Xdf, feature)
        end
    end
    return Xdf
end

function preprocess_ordinal_features!(Xdf::DataFrame)
    # Define ordinal feature mappings
    feature_mappings = Dict(
        "LotShape" => ordinal_mapping(["IR3", "IR2", "IR1", "Reg"]),
        "LandSlope" => ordinal_mapping(["Sev", "Mod", "Gtl"]),
        "Utilities" => ordinal_mapping(["ELO", "NoSeWa", "NoSewr", "AllPub"]),
        "ExterQual" => ordinal_mapping(["Po", "Fa", "TA", "Gd", "Ex"]),
        "ExterCond" => ordinal_mapping(["Po", "Fa", "TA", "Gd", "Ex"]),
        "Foundation" => ordinal_mapping(["Wood", "Stone", "Slab", "PConc", "CBlock", "BrkTil"]),
        "Functional" => ordinal_mapping(["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"]),
        "BsmtQual" => ordinal_mapping(["NA", "Po", "Fa", "TA", "Gd", "Ex"]),
        "BsmtCond" => ordinal_mapping(["NA", "Po", "Fa", "TA", "Gd", "Ex"]),
        "BsmtExposure" => ordinal_mapping(["NA", "No", "Mn", "Av", "Gd"]),
        "BsmtFinType1" => ordinal_mapping(["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]),
        "BsmtFinType2" => ordinal_mapping(["NA", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"]),
        "HeatingQC" => ordinal_mapping(["Po", "Fa", "TA", "Gd", "Ex"]),
        "CentralAir" => ordinal_mapping(["N", "Y"]),
        "KitchenQual" => ordinal_mapping(["Po", "Fa", "TA", "Gd", "Ex"]),
        "FireplaceQu" => ordinal_mapping(["NA", "Po", "Fa", "TA", "Gd", "Ex"]),
        "GarageType" => ordinal_mapping(["NA", "Detchd", "CarPort", "BuiltIn", "Basment", "Attchd", "2Types"]),
        "GarageFinish" => ordinal_mapping(["NA", "Unf", "RFn", "Fin"]),
        "GarageQual" => ordinal_mapping(["NA", "Po", "Fa", "TA", "Gd", "Ex"]),
        "GarageCond" => ordinal_mapping(["NA", "Po", "Fa", "TA", "Gd", "Ex"]),
        "PavedDrive" => ordinal_mapping(["N", "P", "Y"]),
        "PoolQC" => ordinal_mapping(["NA", "Fa", "TA", "Gd", "Ex"]),
        "Fence" => ordinal_mapping(["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"]),
        "Street" => ordinal_mapping(["Grvl", "Pave"]),
        "Alley" => ordinal_mapping(["NA", "Grvl", "Pave"])
    )

    for (feature, categories) in feature_mappings
        if feature in names(Xdf)
            Xdf[!, feature] = [categories[value] for value in Xdf[!, feature]]
        end
    end

    return Xdf

end


function mapTypesToColumns(dataframe)
    typesToColumns = Dict{DataType,Vector{Symbol}}()
    colNames = names(dataframe)
    for (index, column) in enumerate(eachcol(dataframe))
        columnsList = get!(typesToColumns, eltype(column), Symbol[])
        push!(columnsList, Symbol(colNames[index]))
        typesToColumns[eltype(column)] = columnsList
    end

    return typesToColumns
end

function vec1(n)
    result = zeros(1, n)
    result[n] = 1
    result
end

function homogeneousCoords(X, P::Array{Int,2})
    Xo = hcat(X, ones(size(X, 1), 1))
    Po::Matrix{Int} = vcat(hcat(P, zeros(size(P, 1))), vec1(size(P, 2) + 1))

    Xo, Po
end