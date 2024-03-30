using JSON
using ArgParse


include("PartitionedLS-expio.jl")

function minutes(n)
    n * 60
end

function execute_experiment_within_time_limits(algorithm, expdir, time_bound)
    @info "Starting experiment -- dir: $expdir algorithm: $algorithm"
    print("julia --project=. src/PartitionedLS-$algorithm.jl $expdir")
    ps = run(`julia PartitionedLS-$algorithm.jl -s $expdir`, wait=false)

    starttime = time()
    while time() - starttime < time_bound && process_running(ps) 
        sleep(1)
    end

    if process_running(ps)
        @info "Time is out: killing $algorithm algorithm for $expdir..."
        kill(ps)
    elseif !success(ps)
        @info "Process quit with exit status $(ps.exitcode)"
    end
end


# --- MAIN ---

s = ArgParseSettings()
@add_arg_table s begin
    "-t", "--time-limit"
        help = "Time limit per experiment (in minutes)"
        arg_type = Int
        default = 60
    "-f", "--focus-on"
        help = "Only run experiments in the provided directory"
        arg_type = String
        default = ""
    "-n", "--no-download"
        help = "Skip downloading and converting the datasets"
        arg_type = Bool
        default = false
end

opts = parse_args(s)
time_limit = minutes(opts["time-limit"])

ENV["JULIA_DEBUG"] = "all"

if opts["focus-on"] == ""
    dirs = [
        # "Condition monitoring of hydraulic systems/",
        "Facebook Comment Volume Dataset",
        "Limpet",
        # "PM2.5 Data of Five Chinese Cities Data Set",
        "Superconductivty Data",
        "YearPredictionMSD"
    ]
else
    dirs = [opts["focus-on"]]
end

startdir = pwd()

for dir in dirs
    @info "Processing dir: $dir"

    if !isdir("experiments/$dir")
        mkdir("experiments/$dir")
    end

    if !opts["no-download"]
        cp("datasets/$dir/download.sh", "experiments/$dir/download.sh", force=true)
        chmod("experiments/$dir/download.sh", 0o500)

        cp("datasets/$dir/convert.jl", "experiments/$dir/convert.jl", force=true)

        cd("experiments/$dir")
        @info "Changed to dir: $(pwd()). Downloading data"
        run(`./download.sh`)

        @info "converting data"
        run(`julia --project=../.. convert.jl`)
        cd(startdir)
    end

    @info "Executing experiment"

    expdir = "experiments/$dir"
    conf = read_train_conf("datasets/$dir")

    conf["use nnls"] = true
    for alternations in [20, 100]
        @info "Configuring for $alternations alternations"
        conf["Alt"]["exp_name"] = "N_100_T_$(alternations)"
        conf["Alt"]["num_alternations"] = alternations
        
        open("$expdir/train_conf.json", "w+") do file
            write(file, json(conf))
        end

        execute_experiment_within_time_limits("optimal", expdir, time_limit)
        execute_experiment_within_time_limits("alternating", expdir, time_limit)
    end
end
