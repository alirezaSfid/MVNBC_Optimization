using Plots
using Random
using LinearAlgebra

function generate_point_cloud(
    n;                 # number of points
    dimensions = 2,    # number of dimensions
    scaling_factors = ones(dimensions),  # scaling factors in each dimension
    rotation_matrix = Matrix(I, dimensions, dimensions),     # rotation matrix
    translation_vector = zeros(dimensions),
    random_seed = 1,
)
    rng = Random.MersenneTwister(random_seed)
    P = randn(rng, Float64, dimensions, n)
    S = rotation_matrix * P .* scaling_factors .+ translation_vector
    return S
end

function generate_clusterable_clouds(
    no_of_points,
    no_of_clusters,
    dimension,
    noise_rate,
    batch_number = 1)

    number_of_noise_points = Int(no_of_points * noise_rate)
    n = no_of_points - number_of_noise_points

    min_points = dimension + 1

    rng = Random.MersenneTwister(batch_number)  
    
    random_numbers = sort(rand(rng, 0:(n - no_of_clusters*min_points), no_of_clusters-1))
    append!(random_numbers, n - no_of_clusters*min_points)
    prepend!(random_numbers, 0)
    points_per_cluster =  Int.(diff(random_numbers) .+ min_points)

    S = zeros(Float64, dimension, 0)

    i = 0
    while i < no_of_clusters
        i += 1

        m = points_per_cluster[i]
        rndnmbr = i + batch_number
        rng1 = Random.MersenneTwister(rndnmbr)
        rng2 = Random.MersenneTwister(rndnmbr + 1)
        rng3 = Random.MersenneTwister(rndnmbr + 2)
        rng4 = Random.MersenneTwister(rndnmbr + 3)

        s = generate_point_cloud(
            m,
            dimensions=dimension,
            scaling_factors = 0.5 .+ randn(rng1, Float64, dimension),
            rotation_matrix = randn(rng2, Float64, dimension, dimension),
            translation_vector = 6 * randn(rng3, Float64, dimension),
            random_seed = i
        )
        S = hcat(S, s)
    end

    X = zeros(Float64, dimension, no_of_points)
    X[:, 1:n] = S

    for j = 1:dimension
        X[j, n + 1:end] = rand(rng, minimum(X[j, :]):maximum(X[j, :]), number_of_noise_points)
    end

    return X[shuffle(1:size(X, 1)), :]
end

# i=0
# i += 1
# X = generate_clusterable_clouds(500, 3, 2, 0.05, 1)

# plot(X[1, :], X[2, :], seriestype = :scatter, label = "Data points", legend = :bottomright)