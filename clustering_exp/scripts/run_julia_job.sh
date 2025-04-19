#!/bin/bash
#SBATCH --job-name=my_julia_job_ayazdani
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=12
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=a.yazdani.esfidvajani@tue.nl
#SBATCH --error=slurm-%j.err
#SBATCH --output=slurm-%j.out

module load 2024
export PATH="$HOME/.juliaup/bin:$PATH"
export MOSEKHOME=$HOME/mosek
export PATH=$MOSEKHOME/bin:$PATH
export MOSEKLM_LICENSE_FILE=$MOSEKHOME/mosek.lic
export GRB_LICENSE_FILE=/home/ayazdani/gurobi.lic
export JULIA_PROJECT="/home/ayazdani/scratch-shared/ayazdani/MinimumVolumeNormClustering"

# Check if directories and files exist
echo "Checking paths..."

# Check project directory
if [ ! -d "/home/ayazdani/scratch-shared/ayazdani/MinimumVolumeNormClustering/scripts" ]; then
    echo "Error: Project directory does not exist: /home/ayazdani/scratch-shared/ayazdani/MinimumVolumeNormClustering/scripts" >&2
    exit 1
fi

# Check packages.jl file
if [ ! -f "/home/ayazdani/packages.jl" ]; then
    echo "Error: packages.jl file does not exist: /home/ayazdani/packages.jl" >&2
    exit 1
fi

# Check NW4.jl file
if [ ! -f "/home/ayazdani/scratch-shared/ayazdani/MinimumVolumeNormClustering/NW_for_clustering.jl" ]; then
    echo "Error: NW4.jl file does not exist: /home/ayazdani/scratch-shared/ayazdani/MinimumVolumeNormClustering/clustering.jl" >&2
    exit 1
fi

# Navigate to the project directory
cd /home/ayazdani/scratch-shared/ayazdani/MinimumVolumeNormClustering/scripts

# Ensure logs directory exists
mkdir -p logs500Visual


# Parallel execution for R=1..30, r=0.0:0.05:0.35, and each P in P_values
max_jobs=12

# for R in {1..30}; do
for R in 12; do
    for iter in 80; do
        (
        echo "Running Julia script for R=$R, number_of_iteration=$iter, P=$P"
        export JULIA_NUM_THREADS=16
        julia --project=$JULIA_PROJECT ../codes/clustering.jl $R $iter > logs500Visual/output_R=${R}_iter=${iter}.log 2>&1
        ) &

        # Limit the number of parallel tasks
        if (( $(jobs -r | wc -l) >= max_jobs )); then
            wait -n  # Wait for any one job to finish
        fi
    done
done

wait  # Ensure all background jobs finish before exiting