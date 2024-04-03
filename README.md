# Partitioned Least Squares

This repository contains the code to replicate experiments on the PartitioningLS model
as reported in the article published on ```TODO```.

## Setup you environment

Clone the repository and move into the project directory.

From inside the project directory update the Julia packages by running:

```bash
julia --project=. --color=yes -e 'using Pkg; Pkg.update()'
```

## Download the datasets

The datasets directory contains one subfolder for each available dataset.

In each folder you can find either:
- the actual data, in this case you will see files `data.csv` and `blocks.csv` already present in the directory
- or a script to generate the data. In this case you will find a CreateData.jl file, that you can launch as it follows:
  ```bash
    julia --project=../.. CreateData.jl
  ```
- or scripts two download and preprocess the data. In this case you can do these operations as it follows:
  ```bash
    bash download.sh
    julia --project=../.. convert.jl 
  ```

All five datasets need to be downloaded before the experiments can be run. If you fail to download them, or you
don't want to use some of them, you can modify the run_* scripts to avoid experimenting with them, but note that
the scripts that will allow you to format the results require all results to be present. Updating them is feasible,
but not as easy as changing the bash scripts.

## Run the experiments

```bash
bash run_model_quality_experiments.sh
bash run_time_vs_obj_experiments.sh
```

The analysis of the Housing dataset is included in a julia notebook that can be found in the `housing` directory.

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
