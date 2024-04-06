ALGORITHMS="SimpleLeastSquares.jl SimplePartitionedLS.jl SimplePCLS.jl SimplePLS.jl"
# ALGORITHMS="SimplePLS.jl"


for algorithm in $ALGORITHMS; do

    julia --project=. src/$algorithm datasets/Limpet
    julia --project=. src/$algorithm datasets/PartLSArtificial
    julia --project=. src/$algorithm datasets/Facebook\ Comment\ Volume\ Dataset
    julia --project=. src/$algorithm datasets/Superconductivty\ Data
    julia --project=. src/$algorithm datasets/YearPredictionMSD
done

