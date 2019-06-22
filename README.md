# Block Least Squares

This repository contains the code to replicate experiments on the PartitioningLS model. 

## Setup you environment

Update the Julia packages by running:

```bash
julia --project=. --color=yes -e 'using Pkg; Pkg.update()'
```

## Launch the expderiments using the optimal algorithm


```bash
julia --project=. --color=yes exp1/PartitioningLS-optimal.jl
```


## Launch the expderiments using the alternating LS algorithm


```bash
julia --project=. --color=yes exp1/PartitioningLS-alternating.jl
```
