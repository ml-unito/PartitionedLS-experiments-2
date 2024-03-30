# Partitioned Least Squares

This repository contains the code to replicate experiments on the PartitioningLS model
as reported in the article published on ```TODO```.

## Setup you environment

Clone the repository and move into the project directory.

From inside the project directory update the Julia packages by running:

```bash
julia --project=. --color=yes -e 'using Pkg; Pkg.update()'
```

## Run the experiments

```bash
bash run_model_quality_experiments.sh
bash run_time_vs_obj_experiments.sh
```

## Accessing results

After running the experiments, the results will be stored in the `experiments` directory.

You can get a summary table of the results about model-quality by running:

```bash
julia --project=. src/CompareResults.jl
```

You can create a pdf file with a plot of the time-vs-obj experiments by running:

```bash
pdflatex src/time-vs-obj.tex
```

and access the results by opening the `time-vs-obj.pdf` file.

## Citing

If you use this code in your research, please cite the following article:

```
TODO
```