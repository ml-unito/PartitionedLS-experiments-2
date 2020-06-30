# Partitioned Least Squares

This repository contains the code to replicate experiments on the PartitioningLS model
as reported in the article published on [arXiv](https://arxiv.org/abs/2006.16202).

## Setup you environment

Clone the repository, move into the project directory and create the 
experiments directory inside it.

From inside the project directory update the Julia packages by running:

```bash
julia --project=. --color=yes -e 'using Pkg; Pkg.update()'
```

## Run the experiments

```bash
julia --project=. run_experiments.jl
```

## Accessing results

A python script that implements a simple web server is provided so to allow
to peruse the results once they are terminated.

To launch the web server **move into the experiments** folder and run:

```bash
    python3 ../http_serve_results.py
```

Point your browser to "http://localhost:8080" and you should see a web
page showing the names of the four datasets. Opening the links will show
the results for each dataset.