# ALGORITHMS="SimpleLeastSquares.jl SimplePartitionedLS.jl SimplePCLS.jl SimplePLS.jl"
ALGORITHMS="SimplePLS.jl"


for algorithm in $ALGORITHMS; do
    julia --project=. $algorithm datasets/PartLSArtificial
    julia --project=. $algorithm datasets/Limpet
    julia --project=. $algorithm datasets/Facebook\ Comment\ Volume\ Dataset
    julia --project=. $algorithm datasets/Superconductivty\ Data
    julia --project=. $algorithm datasets/YearPredictionMSD
done

