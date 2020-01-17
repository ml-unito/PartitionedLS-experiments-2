using DataFrames
using CSV
using DelimitedFiles

const global MAX_COLS_PER_BLOCK = 30

function add_block(blocks, cols, pos)
    for col in cols
        push!(blocks, vcat([col], [0 for i in 1:(pos-1)], 1, [0 for i in (pos+1):17]))
    end

    return pos + 1
end

function load_X()
    # files = ["PS3", "TS3",
    #     "CP"  , "PS4", "TS4",
    #     "EPS1", "PS5", "VS1",
    #     "FS1" , "PS6",           
    #     "FS2" , "SE",            
    #     "PS1" , "TS1",           
    #     "PS2" , "TS2"]

    files = [
        "PS1", "PS2", "EPS1", "FS1", "FS2", "TS1", "TS2", "VS1", "CE", "CP", "SE"]


    @info "Reading CE.txt\n"
    blocks = DataFrame(Descriptor = String[], 
                    CE = Int8[],
                    PS3 = Int8[], 
                    TS3 = Int8[],
                    CP = Int8[]  , 
                    PS4 = Int8[], 
                    TS4 = Int8[],
                    EPS1 = Int8[], 
                    PS5 = Int8[], 
                    VS1 = Int8[],
                    FS1 = Int8[] , 
                    PS6 = Int8[],           
                    FS2 = Int8[] , 
                    SE = Int8[],            
                    PS1 = Int8[] , 
                    TS1 = Int8[],           
                    PS2 = Int8[] , 
                    TS2 = Int8[])

    df = CSV.read("CE.txt", header=map(x -> string("CE",x), 1:60))
    df = df[:, 1:min(ncol(df), MAX_COLS_PER_BLOCK)]
    cols = ["CE-$i" for i in 1:min(ncol(df), MAX_COLS_PER_BLOCK)]
    
    cur_block = add_block(blocks, cols, 1)

    for file in files
        @info "Reading $file.txt\n"
        tmpdf = CSV.read("$file.txt", header=0)
        num_cols = min(ncol(tmpdf), MAX_COLS_PER_BLOCK)

        all_cols = ["$file-$i" for i in 1:ncol(tmpdf)]
        filtered_cols = ["$file-$i" for i in 1:num_cols]

        rename!(tmpdf, [Symbol(c) for c in all_cols])
        df = hcat(df, tmpdf[:, 1:num_cols])

        cur_block = add_block(blocks, filtered_cols, cur_block)
    end

    @info "Reading profile.txt\n"
    dfy = CSV.read("profile.txt", header=["1", "2", "3", "y", "4"])
    df = hcat(df, select(dfy, :y))


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