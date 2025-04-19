module DMDFunctions2

include("MVCEs.jl")
import .MVCEs

using Clustering
using Random
using LinearAlgebra
using GaussianMixtures

export DMD_ellipsoid
export DMDgmm

const MAX_ITERATION_MULTIPLIER = 3
const TERMINATION_INDEX_THRESHOLD = 5
const TOLERANCE = 1e-5

function DMD_ellipsoid(X::Matrix, n_clusters::Int64, e::Float64; max_iteration::Int64=3)
    clusters, π_k = initialize_clusters(X, n_clusters)
    K = n_clusters
    n = size(X, 2)
    last_log_p = 0
    bestClusters = clusters
    bestVolTotal = Inf

    # Initialize an array to store the sum of volumes
    sum_vols = Float64[]
    
    for iteration in 1:max_iteration
        iter = 0
        termination_index = 0
    
        gamma_nk = zeros(n, K)
    
        while termination_index <= TERMINATION_INDEX_THRESHOLD && iter <= max_iteration * MAX_ITERATION_MULTIPLIER
            iter += 1
    
            gamma_nk = expectation_step(X, clusters, π_k)
    
            r = 1
    
            clusters, π_k = maximization_step(X, clusters, gamma_nk, e, r)
    
            log_p = sum(log.(sum(gamma_nk, dims=2)))
    
            if abs(log_p - last_log_p) <= TOLERANCE
                termination_index +=1
            else
                termination_index = 0
                last_log_p = log_p
            end
    
        end
    
    
        labels = falses(n, K)
        [labels[i, argmax(gamma_nk[i, :])] = true for i in 1:n]
        
        dist = dist_clusters(X, clusters)
        outliers = outlier_detect(dist, e)
        labels[outliers, :] .= false
    
        clusters = MVCEs.minimum_volume_ell_labeled(X, labels)

        # Append the sum of volumes to the array
        push!(sum_vols, sum(clusters.vols))
    
        if sum(clusters.vols)/sum(sum(clusters.labels, dims=2) .> 0) <= bestVolTotal/sum(sum(bestClusters.labels, dims=2) .> 0)
            bestVolTotal = sum(clusters.vols)
            bestClusters = clusters
        end
    end
    
    return bestClusters, sum_vols
end


function initialize_clusters(X, n_clusters; r=4)
    d, n = size(X)
    R = kmeans(X, n_clusters)
    # R = dbscan(X', .25, min_neighbors=30, min_cluster_size=10)

    Hs = Matrix{Float64}[]
    x_hats = Vector{Float64}[]
    vols = []
    L = falses(n, n_clusters)

    for k in 1:n_clusters
        # H, x_hat, vol = MVCEs.minimum_volume_ell(X[:, R.assignments .== k])
        x = X[:, R.assignments .== k]

        H, x_hat, vol = MVCEs.minimum_volume_ell(x)

        for i in 1:n
            L[i, k] = (X[:, i] - x_hat)' * H * (X[:, i] - x_hat) <= d + TOLERANCE
        end
    
        push!(Hs, H)
        push!(x_hats, x_hat)
        push!(vols, vol)
    end

    mixing_coeffs = sum(L, dims=1) ./ sum(L)
    
    return MVCEs.MVBCResult(Hs, x_hats, L, vols), mixing_coeffs
end


# This function calculates the Norm_2 distance from each data point to each cluster center.
function dist_clusters(X, clusters)
    # L is a matrix where each row corresponds to a data point and each column corresponds to a cluster.
    # The element at the i-th row and k-th column is true if the i-th data point is assigned to the k-th cluster, and false otherwise.
    L = clusters.labels

    # n is the number of data points, and K is the number of clusters.
    n, K = size(L)

    # dist is a matrix where each row corresponds to a data point and each column corresponds to a cluster.
    # The element at the i-th row and k-th column will be the Norm_2 distance from the i-th data point to the k-th cluster center.
    dist = zeros(n, K)

    # Loop over each cluster.
    for k in 1:K
        # H is the transformation matrix for the k-th cluster.
        H = clusters.transformation_matrices[k]

        # x_hat is the translation vector for the k-th cluster.
        x_hat = clusters.translation_vectors[k]

        # Loop over each data point.
        for i in 1:n
            # Calculate the Norm_2 distance from the i-th data point to the k-th cluster center.
            dist[i, k] = (X[:, i] - x_hat)' * H * (X[:, i] - x_hat)
        end
    end

    # Return the matrix of distances.
    return dist
end


function expectation_step(X, clusters, mixing_coeffs)
    d, n = size(X)
    L = clusters.labels
    K = size(L, 2)
    gamma_nk = zeros(n, K)

    dist = dist_clusters(X, clusters)
    
    for k in 1:K
        for i in 1:n
            gamma_nk[i, k] = mixing_coeffs[k] * exp(-0.5 * (dist[i, k]))    
        end
    end
    
    gamma_nk ./= sum(gamma_nk, dims=2)
    gamma_nk = [gamma = max(gamma, TOLERANCE) for gamma in gamma_nk]    
    
    return gamma_nk
end


# This function identifies the indices of the `e` proportion of points 
# that have the greatest minimum distance to any cluster.
function outlier_detect(dist, e::Float64)
    # n is the number of data points.
    n = size(dist, 1)
    
    # num_outliers is the number of outliers to detect, 
    # calculated as a proportion `e` of the total number of data points.
    num_outliers = Int(round(e * n))
    
    # dist_min is a vector that contains the minimum distance 
    # from each data point to any cluster.
    dist_min = [minimum(dist[row_idx, :]) for row_idx in 1:n]
    
    # outlier_indices is a vector that contains the indices of the `num_outliers` data points 
    # that have the greatest minimum distance to any cluster.
    outlier_indices = partialsortperm(dist_min, 1:num_outliers, rev=true)
    
    # Return the indices of the outliers.
    return outlier_indices
end


function outlier_gamma_update(gamma_nk, outliers)
    gamma_nk[outliers, :] .= TOLERANCE
    return gamma_nk
end


function maximization_step(X, clusters, gamma_nk, e, r=1)
    d, n =size(X)
    K = size(gamma_nk, 2)
    L =  falses(n, K)
    dist = dist_clusters(X, clusters)
        
    outliers = outlier_detect(dist, e)
    gamma_nk = outlier_gamma_update(gamma_nk, outliers)
    
    Hs = Matrix{Float64}[]
    x_hats = Vector{Float64}[]
    vols = []
    for k in 1:K
        H, x_hat, vol = MVCEs.minimum_volume_ell_weighted(X, gamma_nk[:, k])
        for i in 1:n
            L[i, k] = (X[:, i] - x_hat)' * H * (X[:, i] - x_hat) <= d + TOLERANCE
        end
    
        push!(Hs, H)
        push!(x_hats, x_hat)
        push!(vols, vol)
    end

    mixing_coeffs = sum(gamma_nk, dims=1) ./ sum(gamma_nk)
    
    return MVCEs.MVBCResult(Hs, x_hats, L, vols), mixing_coeffs
end

function DMDgmm(X::Matrix, n_clusters::Int64)

    X_transposed = Matrix(X')

    n, d = size(X_transposed)

    gmm = GMM(n_clusters, X_transposed; method = :kmeans, kind = :full)
    posteriors = gmmposterior(gmm, X_transposed)
    labels = falses(n, n_clusters)
    for i in 1:n
        labels[i, argmax(posteriors[1][i, :])] = true
    end

    clusters = MVCEs.minimum_volume_ell_labeled(X, labels)

    return clusters
    
end

end