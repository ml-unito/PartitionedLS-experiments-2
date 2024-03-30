DATASETS=("Limpet" "Facebook Comment Volume Dataset" "YearPredictionMSD" "Superconductivty Data")

echo $DATASETS

for dataset in "${DATASETS[@]}"; do
  julia --project=. src/PartitionedLS-alternating.jl "datasets/$dataset" 20
  julia --project=. src/PartitionedLS-alternating.jl "datasets/$dataset" 100
  julia --project=. src/PartitionedLS-optimal.jl "datasets/$dataset"
done
