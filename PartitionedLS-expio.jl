function read_train_conf(dir)
    json_path = "$dir/train_conf.json"
    if isfile(json_path)
        json = JSON.parsefile(json_path)
        return json
    else
        @error "Cannot find configuration file"
        exit(1)
    end

end


function load_data(dir, conf)
    @info "Reading data..."
    data = CSV.read(string(dir, "/data.csv"), DataFrame)

    @info "Reading blocks"
    blocks = CSV.read(string(dir, "/blocks.csv"), DataFrame)

    @info "Filtering dataset"
    train_start, train_end, test_start, test_end = conf["train_start"], conf["train_end"], conf["test_start"], conf["test_end"]


    train_len = train_end - train_start + 1
    test_len = test_end - test_start + 1

    @info "Converting matrices...", "train/test set split is" train = train_len test = test_len

    print(names(data[train_start:train_end, setdiff(names(data), [:y])]))
    print(names(data))
    print(setdiff(names(data), ["y"]))

    
    # Xtr = convert(Matrix, data[train_start:train_end, setdiff(names(data), [:y])])
    # Xte = convert(Matrix, data[test_start:test_end, setdiff(names(data), [:y])])
    # ytr = convert(Array, data[train_start:train_end, :y])
    # yte = convert(Array, data[test_start:test_end, :y])
    Xtr = Tables.matrix(data[train_start:train_end, setdiff(names(data), ["y"])])
    Xte = Tables.matrix(data[test_start:test_end, setdiff(names(data), ["y"])])
    ytr = Tables.matrix(data[train_start:train_end, [:y]])
    yte = Tables.matrix(data[test_start:test_end, [:y]])

    
    colsums = sum(Xtr, dims=1)
    Xtr = Xtr ./ colsums
    Xte = Xte ./ colsums

    # P = convert(Matrix, blocks[:, 2:end])
    P = Tables.matrix(blocks[:, 2:end])

    return Xtr, Xte, ytr, yte, P
end