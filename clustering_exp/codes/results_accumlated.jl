include("HeuristicMethod.jl")
include("multi_dimensional_data_generation.jl")
include("BendersDecomposition.jl")
include("LocalSearchFunctions.jl")
import .BendersDecomposition
import .LocalSearchFunctions
using Plots
using DelimitedFiles
using LaTeXStrings
using Base.Threads
using DataFrames
using Statistics

# Set up directories
results_dir = "../results"

results_df = DataFrame()
average_time = Dict()
average_obj_val = Dict()
obj_vec = Dict()
for random_seed in 1:30
    for kind in ["GMM", "Kmeans"]
        for iter in [0, 20, 40, 80]
            try
                results = readdlm(joinpath(results_dir, "results_$kind-iter_$iter-seed_$random_seed-N_500-noise_0.05-P_1.02.0Inf-r_0.05.csv"), ',')
                average_obj_val[(kind, iter)] = get(average_obj_val, (kind, iter), 0) + results[2]
                average_time[(kind, iter)] = get(average_time, (kind, iter), 0) + results[3]
                results_df = vcat(results_df, DataFrame(seed = random_seed, kind = kind, iter = iter, obj_val = results[2], time = results[3]))
                obj_vec[(kind, iter, random_seed)] = filter(x -> x != "", results[4, :])
            catch
                println("Results for $kind and $iter and $random_seed not found")
            end
        end
    end
end

max_length = maximum(length.(values(obj_vec)))

# Function to sort obj_val within each group and assign to iter values
function sort_and_assign_obj_val(df)
    sorted_df = sort(df, :obj_val, rev=true)
    sorted_df.iter .= [0, 20, 40, 80]
    return sorted_df
end

# Group by seed and kind, then sort obj_val within each group and assign to iter values
grouped_df = groupby(results_df, [:seed, :kind])
sorted_results_df = combine(grouped_df) do sdf
    sort_and_assign_obj_val(sdf)
end

results_df = sorted_results_df


baseline_obj = Dict()
for seed in 1:30
    baseline_obj[seed] = results_df[(results_df.seed .== seed) .& (results_df.iter .== 0) .& (results_df.kind .== "Kmeans"), :obj_val][1]
end

# Compute Normalized Mean and Relative Improvement
results_df[!, :normalized_mean] = [row.obj_val / baseline_obj[row.seed] for row in eachrow(results_df)]
results_df[!, :relative_improvement] = [(baseline_obj[row.seed] - row.obj_val) / baseline_obj[row.seed] * 100 for row in eachrow(results_df)]
        
summary_df = DataFrame()

# Compute mean and std deviation for each kind and iteration
summary_df = combine(groupby(results_df, [:kind, :iter]), 
    :normalized_mean => mean => :mean_obj_val,
    :normalized_mean => std => :std_obj_val,
    :relative_improvement => mean => :mean_rel_imp)

# baseline_obj = filter(row -> row.kind == "Kmeans" && row.iter == 0, summary_df).mean_obj_val[1]

# Add Normalized Mean and Relative Improvement Columns
summary_df[!, :normalized_mean] = [row.mean_obj_val / baseline_obj for row in eachrow(summary_df)]
summary_df[!, :relative_improvement] = [(baseline_obj - row.mean_obj_val) / baseline_obj * 100 for row in eachrow(summary_df)]

# Display the processed table
using PrettyTables
pretty_table(summary_df, header=["Method", "Iter", "Mean Obj", "Std Dev", "Normalized Mean", "Relative Improvement (%)"])
