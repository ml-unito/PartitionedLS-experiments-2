using JSON
include("PartitionedLS-expio.jl")

function minutes(n)
    n * 60
end

function execute_experiment_within_time_limits(algorithm, expdir, time_bound)
    @info "Starting experiment -- dir: $expdir algorithm: $algorithm"
    ps = run(`julia --project=. PartitionedLS-$algorithm.jl $expdir`, wait=false)

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

ENV["JULIA_DEBUG"] = "all"

dirs = [
    # "Condition monitoring of hydraulic systems/",
    "Facebook Comment Volume Dataset",
    "Limpet",
    "PM2.5 Data of Five Chinese Cities Data Set",
    "Superconductivty Data",
    "YearPredictionMSD"
]

min15 = 60 * 15
startdir = pwd()

for dir in dirs
    @info "Processing dir: $dir"

    if !isdir("experiments/$dir")
        mkdir("experiments/$dir")
    end

    cp("datasets/$dir/download.sh", "experiments/$dir/download.sh", force=true)
    chmod("experiments/$dir/download.sh", 0o500)

    cp("datasets/$dir/convert.jl", "experiments/$dir/convert.jl", force=true)

    cd("experiments/$dir")
    @info "downloading data"
    run(`download.sh`)

    @info "converting data"
    run(`julia --project=. convert.jl`)
    cd(startdir)

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

        execute_experiment_within_time_limits("optimal", expdir, minutes(15))
        execute_experiment_within_time_limits("alternating", expdir, minutes(15))
    end
end