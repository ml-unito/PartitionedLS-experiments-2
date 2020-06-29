import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import pandas
import os
import json

def plot_opt_at_path(plt, path):
    pd_opt = pandas.read_csv(os.path.join(path,"results-OPT.csv"))
    plt.plot(pd_opt["TimeCumulative"], pd_opt["TrainBest"], "o", label="PartitionedLS-Opt")

def plot_alt_at_path(plt, path, expname):
    pd_alt = pandas.read_csv(os.path.join(path, "results-ALT.csv"))
    plt.plot(pd_alt["TimeCumulative"], pd_alt["TrainBest"], "-o", label="PartitionedLS-Alt T{}".format(expname))

def read_configuration(path):
    with open(os.path.join(path, "conf.json")) as f:
        return json.load(f)

def plot_exp(name, exp20, exp100):
    plt.clf()
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor', linestyle="--")
    
    plot_alt_at_path(plt, exp20, '20')
    plot_alt_at_path(plt, exp100, '100')
    plot_opt_at_path(plt, exp20)

    plt.title(name + " dataset ")
    plt.xscale("log")
    plt.xlabel("Time (log scale)")
    plt.ylabel("Objective")
    plt.margins
    plt.legend()

    plt.savefig("{}.pdf".format(name), format="pdf")


matplotlib.rcParams['font.serif'] = "CMU serif"
matplotlib.rcParams['font.family'] = "serif"
configurations = {}
for dir in [d for d in os.listdir(".") if os.path.isdir(d) and d[0] != '.']:
    if dir == "images":
        continue
    configurations[dir] = {}

    for expdir in [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir,d)) and len(d) == 64]:
        configurations[os.path.join(dir)][expdir] = read_configuration(os.path.join(dir, expdir))


for exp in configurations:
    exps = {}
    for sub_exp in configurations[exp]:
        print(json.dumps(configurations[exp][sub_exp], indent=2))

        exps[configurations[exp][sub_exp]["Alt"]["num_alternations"]] = os.path.join(exp, sub_exp)

    plot_exp(exp, exps[20], exps[100])
    
