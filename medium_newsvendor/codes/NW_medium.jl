using JuMP
using Gurobi
using NPZ, JSON
using ArgParse
using Dates  # For tracking time
using Plots  # For saving plots
using LinearAlgebra
using MosekTools

# Include required files
include("../../src/HeuristicMethod.jl")
include("../../src/BendersDecomposition.jl")

import .HeuristicMethod

# Set up directories
datasets_dir = "../data"
results_dir = "../results/results200points-AA"

isdir(results_dir) || mkdir(results_dir)  # Ensure results directory exists

# Check if directories and files exist
println("Checking paths...")

if !isdir(datasets_dir)
    println("Error: Datasets directory does not exist: $datasets_dir")
    exit(1)
end

if !isdir(results_dir)
    println("Error: Results directory does not exist: $results_dir")
    exit(1)
end

# # Parse arguments
# println("ARGS: ", ARGS)
# R = parse(Int, ARGS[1])  # Parse R as an integer
# r = parse(Float64, ARGS[2])  # Parse r as a float
# P_vals = parse.(Float64, split(ARGS[3], ","))  # Parse P as a tuple of floats
# P = Set(P_vals)  # Convert to a set
# K_p = Dict(p => 1 for p in P)  # Number of clusters per norm

# Load the full datasets (for R = 30)
data_path = joinpath(datasets_dir, "demand_data_clustered_R=30.npy")
data_eval_path = joinpath(datasets_dir, "demand_data_eval_clustered_R=30.npy")

if !isfile(data_path)
    println("Error: Data file does not exist: $data_path")
    exit(1)
end

if !isfile(data_eval_path)
    println("Error: Data evaluation file does not exist: $data_eval_path")
    exit(1)
end

# Parse arguments
println("ARGS: ", ARGS)
R = parse(Int, ARGS[1])  # Parse R as an integer
r = parse(Float64, ARGS[2])  # Parse r as a float
P_vals = parse.(Float64, split(ARGS[3], ","))  # Parse P as a tuple of floats
P = collect(P_vals)  # Convert to a vector
println("type of P: ", typeof(P))
println("P: ", P)
K_p = Dict{Float64, Int}(p => 1 for p in P)  # Number of clusters per norm


# Load the full matrix for R = 30
Data_full = npzread(data_path)
Data_eval_full = npzread(data_eval_path)

# Extract the relevant data for the current R
if R > size(Data_full, 2)
    println("R=$R exceeds the available data points. Exiting...")
    exit(1)
end

Data = Data_full[:, R, :]  # Training data for the current R
Data_eval = Data_eval_full[:, R, :]  # Evaluation data for the current R

X = Matrix(Data')  # Transpose to match expected shape
X_test = Matrix(Data_eval')

d, N = size(X)
# Track start time
start_time = now()

# Run Heuristic
println("Running Heuristic for R=$R, r=$r, P=$P...")
labels, T, t, obj = HeuristicMethod.normBasedUncertaintySet_with_GMM(X, P, K_p, r)
end_time = now()
elapsed_time = end_time - start_time

# Generate and save the plot
println("Generating and saving plot for R=$R, r=$r, P=$P...")
plot_file = joinpath(results_dir, "plot_R=$(R)_r=$(r)_P=$(join(P_vals, "_")).png")

BendersDecomposition.plot_clusters_and_boundaries(X, labels, T, t, P, K_p)
savefig(plot_file)
println("Plot saved to $plot_file")

# Save results
result_file = joinpath(results_dir, "results_R=$(R)_r=$(r)_P=$(join(P_vals, "_")).json")
results = Dict(
    "R" => R,
    "r" => r,
    "P" => P_vals,
    "objective_value" => obj,
    "elapsed_time" => elapsed_time,
    "labels" => labels,
)
open(result_file, "w") do io
    JSON.print(io, results, 4)
end

println("Results for R=$R, r=$r, P=$P saved to $result_file")


# Define and solve the robust optimization model
println("Solving robust optimization model...")

di, D = Dict(), Dict()
for p in P
	di[p] = [T[p][k] * [40, 40] - t[p][k] for k in 1:K_p[p]]
	D[p] = [I(d) for k in 1:K_p[p]]
end

# Define parameters for the model
h = [4.0, 5.0]    # Selling price of the product
c = [5.0, 6.5]    # Cost price of the product

function e_vector(d, i)
    e = zeros(d)
    e[i] = 1
    return e
end

# Initialize robust optimization model
NWmodel = Model(MosekTools.Optimizer)

@variable(NWmodel, x[1:d], lower_bound = 0) # Quantity of the product
@variable(NWmodel, expen) # Auxilary variable for robust profit
w_1, w_2 = Dict(), Dict()
for p in P
	for j in 1:3
		w_1[(j, p)] = [@variable(NWmodel, [1:d], base_name = "w_1[$j][$p][$k]") for k in 1:K_p[p]]
		w_2[(j, p)] = [@variable(NWmodel, [1:d], base_name = "w_2[$j][$p][$k]") for k in 1:K_p[p]]
	end
end
# @variable(NWmodel, s[1:d], lower_bound = 0) # Shortage of the product
# @variable(NWmodel, u[1:d], lower_bound = 0) # Unsatisfied demand of the product

@objective(NWmodel, Min, h' * x + expen)  # Minimize the total cost

@constraint(NWmodel, -c' * x <= expen)  # Demand constraint
for p in P
	for k in 1:K_p[p]
		@constraint(NWmodel, [expen + e_vector(d, 1)' * c * e_vector(d, 1)' * x + e_vector(d, 2)' * c * e_vector(d, 2)' * inv(T[p][k]) * t[p][k] - di[p][k]' * inv(D[p][k])' * w_1[(1, p)][k]; w_2[(1, p)][k]] in BendersDecomposition.select_cone(1 + 1/(p-1), d))
		@constraint(NWmodel, w_1[(1, p)][k] + w_2[(1, p)][k] == vec(- e_vector(d, 2)' * c * e_vector(d, 2)' * inv(T[p][k])))
		@constraint(NWmodel, inv(D[p][k])' * w_1[(1, p)][k] >= 0)

		@constraint(NWmodel, [expen + e_vector(d, 2)' * c * e_vector(d, 2)' * x + e_vector(d, 1)' * c * e_vector(d, 1)' * inv(T[p][k]) * t[p][k] - di[p][k]' * inv(D[p][k])' * w_1[(2, p)][k]; w_2[(2, p)][k]] in BendersDecomposition.select_cone(1 + 1/(p-1), d))
		@constraint(NWmodel, w_1[(2, p)][k] + w_2[(2, p)][k] == vec(- e_vector(d, 1)' * c * e_vector(d, 1)' * inv(T[p][k])))
		@constraint(NWmodel, inv(D[p][k])' * w_1[(2, p)][k] >= 0)

		@constraint(NWmodel, [expen + c' * inv(T[p][k]) * t[p][k] - di[p][k]' * inv(D[p][k])' * w_1[(3, p)][k]; w_2[(3, p)][k]] in BendersDecomposition.select_cone(1 + 1/(p-1), d))
		@constraint(NWmodel, w_1[(3, p)][k] + w_2[(3, p)][k] == vec(- c' * inv(T[p][k])))
		@constraint(NWmodel, inv(D[p][k])' * w_1[(3, p)][k] >= 0)
	end
end

optimize!(NWmodel)


function evaluate_solution(X_test, x, obj, c, h)
    infeasible_points = 0
    Evalval = 0
    for i in 1:size(X_test, 2)
        evalval = h' * value.(x) - c' * min.(X_test[:, i], value.(x))
        Evalval += evalval
        if evalval > obj
            infeasible_points += 1
        end
    end
    return infeasible_points/size(X_test, 2) , Evalval/size(X_test, 2)
end

# Evaluate the solution
println("Evaluating the solution for R=$R, r=$r, P=$P...")
infeasible_points, EvalAve = evaluate_solution(X_test, x, objective_value(NWmodel), c, h)

# Save results
result_file = joinpath(results_dir, "results_R=$(R)_r=$(r)_P=$(join(P_vals, "_")).json")
results["optimal_solutions"] = value.(x)
results["objective_value"] = objective_value(NWmodel)
results["infeasibility_range"] = infeasible_points
results["EvalAve"] = EvalAve
open(result_file, "w") do io
    JSON.print(io, results, 4)
end

println("Results for R=$R, r=$r, P=$P saved to $result_file")