#!/bin/bash
#SBATCH --job-name=my_julia_job_ayazdani
#SBATCH -p genoa
#SBATCH -N 1
#SBATCH --ntasks=12  # 12 parallel tasks
#SBATCH --cpus-per-task=16  # 12 cores per Julia job
#SBATCH --time=24:00:00
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
if [ ! -d "$JULIA_PROJECT/scripts" ]; then echo "Error: Project directory does not exist." >&2; exit 1; fi
# if [ ! -f "$HOME/packages.jl" ]; then echo "Error: packages.jl file does not exist." >&2; exit 1; fi
# if [ ! -f "$JULIA_PROJECT/NW4.jl" ]; then echo "Error: NW4.jl file does not exist." >&2; exit 1; fi

cd "$JULIA_PROJECT/scripts"

# Install dependencies
# if ! julia "$HOME/packages.jl"; then echo "Error: Failed to install dependencies." >&2; exit 1; fi

mkdir -p logs50points
echo "Logs will be saved in:" 
echo "$JULIA_PROJECT/scripts/logs50points"

# Define p-norm values
P_values=("1.0" "2.0" "Inf" "1.0,2.0" "1.0,Inf" "2.0,Inf")
# P_values=("1.0,Inf")


# Parallel execution using SLURM tasks
index=0  # Job index for SLURM array
for R in {1..30}; do
# for R in 28; do
    for r in 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5; do
    # for r in 0.05; do
        for P in "${P_values[@]}"; do
            echo "Submitting job R=$R, r=$r, P=$P on task $index"
            srun --exclusive -n 1 --cpus-per-task=12 julia --project=$JULIA_PROJECT ../code/NW_small.jl $R $r "$P" > "logs50points/output_R=${R}_r=${r}_P=${P//,/}.log" 2>&1 &
            
            ((index++))
            if (( index >= 12 )); then
                wait  # Wait for tasks to finish before submitting new ones
                index=0
            fi
        done
    done
done

wait  # Ensure all jobs finish