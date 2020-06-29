# Block Least Squares

This repository contains the code to replicate experiments on the PartitioningLS model. 

## Setup you environment

Update the Julia packages by running:

```bash
julia --project=. --color=yes -e 'using Pkg; Pkg.update()'
```

## Launch the experiments using the optimal algorithm


```bash
julia --project=. --color=yes exp1/PartitionedLS-optimal.jl
```


## Launch the experiments using the alternating LS algorithm


```bash
julia --project=. --color=yes exp1/PartitionedLS-alternating.jl
```
