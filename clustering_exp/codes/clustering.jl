using JuMP
using Gurobi
using NPZ, JSON
using ArgParse
using Dates  # For tracking time
using Plots  # For saving plots
using LinearAlgebra
using MosekTools
using LaTeXStrings
using DelimitedFiles

# Include required files
include("../../src/HeuristicMethod.jl")
include("../../src/BendersDecomposition.jl")
include("../../src/multi_dimensional_data_generation.jl")
include("../../src/LocalSearchFunctions.jl")

import .HeuristicMethod


# Set up directories
results_dir = "../results"

# Ensure parent directories exist
mkpath(results_dir)

# Check if directories and files exist
println("Checking paths...")

if !isdir(results_dir)
    println("Error: Results directory does not exist: $results_dir")
    exit(1)
end

# Parse arguments
println("ARGS: ", ARGS)
R = parse(Int, ARGS[1])  # Parse R as an integer
iter = parse(Int, ARGS[2])

println("R: ", R)
println("iter: ", iter)

d = 2  # Dimension
N = 500  # Number of data points
K = 3
noise_rate = 0.05

random_seed = R

X = MultiDimensionalDataGeneration.generate_clusterable_clouds(
        N,  # no_of_points
        K,  # no_of_clusters
        d,  # dimension
        noise_rate,  # noise_rate
        random_seed,  # batch_number
    )

# Define parameters
P = [1.0, 2.0, Inf]  # p-norm values
K_p = Dict(1.0 => 1, 2.0 => 1, Inf => 1)  # Number of clusters for each p-norm
r = 0.05  # Outlier rate

# kind = "GMM"

# t_start = time()
# l_star, T_star, t_star, obj_val, tot_vol_vec = normBasedUncertaintySet_with_GMM(X, P, K_p, r, iter)
# # l_star = BendersDecomposition.assign_clusters_GMM(X, P, K_p, r)
# # T_star, t_star, vols = LocalSearchFunctions.solve_sub(X, P, K_p, l_star)
# # obj_val =  sum(vols[p][k] for p in P for k in 1:K_p[p])
# elapsed_time = time() - t_start

# BendersDecomposition.plot_clusters_and_boundaries(X, l_star, T_star, t_star, P, K_p)
# plot!(xlabel = L"x_1", ylabel = L"x_2")
# savefig(joinpath(results_dir, "plot_$kind-iter_$iter-seed_$random_seed-N_500-noise_0.05-P_1.02.0Inf-r_0.05.png"))

# results = [l_star, obj_val, elapsed_time, tot_vol_vec]
# writedlm(joinpath(results_dir, "results_$kind-iter_$iter-seed_$random_seed-N_500-noise_0.05-P_1.02.0Inf-r_0.05.csv"), results, ',')

kind = "Kmeans"

t_start = time()
l_star, T_star, t_star, obj_val, tot_vol_vec = HeuristicMethod.normBasedUncertaintySet_with_Kmeans(X, P, K_p, r, iter)
# l_star = BendersDecomposition.assign_clusters_using_kmeans(X, P, K_p, r)
# T_star, t_star, vols = LocalSearchFunctions.solve_sub(X, P, K_p, l_star)
# obj_val =  sum(vols[p][k] for p in P for k in 1:K_p[p])
elapsed_time = time() - t_start

BendersDecomposition.plot_clusters_and_boundaries(X, l_star, T_star, t_star, P, K_p)
plot!(xlabel = L"x_1", ylabel = L"x_2")
savefig(joinpath(results_dir, "plot_$kind-iter_$iter-seed_$random_seed-N_500-noise_0.05-P_1.02.0Inf-r_0.05.png"))

results = [l_star, obj_val, elapsed_time, tot_vol_vec]
writedlm(joinpath(results_dir, "results_$kind-iter_$iter-seed_$random_seed-N_500-noise_0.05-P_1.02.0Inf-r_0.05.csv"), results, ',')

println("experiment for seed $random_seed and iteration $iter is done!")