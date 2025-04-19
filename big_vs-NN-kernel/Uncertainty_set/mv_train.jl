using CSV
using DelimitedFiles
using Printf  # Add this line


# Include the main.jl file
include("../../Uset_construction/train_mvnbc/main.jl")

# Define the file paths
input_dir = "Exp_2/Data/"


# Define the output file paths
output_dir = "Exp_2/Uncertainty_set/mvnbc"
output_files = [
    "Hs",
    "x_hats",
    "vols",
    "labels",
    "outliers",
    "time_elapsed"
]

# Define the range of values for e and n_clusters
e_range = 0.60:0.05:0.95
n_clusters_range = [1, 2, 3]

for A in [1, 2, 3]
    for C in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        data_file = joinpath(input_dir, "train-$A-$C.txt")
        X = copy(readdlm(data_file, ',')')

        # Iterate over the values of e and n_clusters
        for e in e_range
            for n_clusters in n_clusters_range

                # Initialize variables to store the minimum sum and corresponding clusters
                min_sum = Inf
                min_clusters = nothing
                min_outliers = nothing
                min_time_elapsed = nothing
                
                # Call the train_mvnbc function 10 times
                for _ in 1:10
                    clusters = nothing
                    outliers = nothing
                    time_elapsed = nothing
                    try
                        # Call the train_mvnbc function with the specified arguments
                        start = time()
                        clusters, outliers = norm2BasedUncertaintySet(X, n_clusters, 1 - e)
                        finish = time()
                        time_elapsed = finish - start

                    catch err
                        continue
                    end

                    # Calculate the sum of clusters.vols
                    sum_vols = sum(clusters.vols)
                
                    # Check if the current sum is smaller than the minimum sum
                    if sum_vols < min_sum
                        min_sum = sum_vols
                        min_clusters = clusters
                        min_outliers = outliers
                        min_time_elapsed = time_elapsed
                    end

                end
                
                # Save the clusters to the specified files
                for (output_file, output_data) in zip(output_files, [min_clusters.transformation_matrices, min_clusters.translation_vectors, min_clusters.vols, min_clusters.labels, min_outliers, min_time_elapsed])

                    output_path = joinpath(output_dir, "$output_file-$A-$C-$n_clusters-$e.txt")

                    writedlm(output_path, output_data, ',')
                end
            end
        end
    end
end
