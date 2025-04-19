module LocalSearchFunctions

include("BendersDecomposition.jl")

import .BendersDecomposition

using Clustering
using ProgressMeter
using LinearAlgebra
using JuMP
using MosekTools
using StatsBase
using Plots
using GaussianMixtures

export local_search_ell, solve_sub

const TOLERANCE = 1e-5



function local_search_ell(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, r::Float64, l::Dict{Float64, Vector{BitVector}}; max_iteration::Int64 = 100)

    d, n = size(X)

    K = sum(K_p[p] for p in P)

    best_l = l
    # T, t, vols = solve_sub(X, P, K_p, best_l)
    # best_total_vol = sum(vols[p][k] for p in P for k in 1:K_p[p])

    # volumes = []
    
    Δ = max_iteration / 10

    best_total_vol_Δ = Inf

    no_change_counter_Δ = 0

    total_vol_vector = []

    for Δ_iter in 1:Δ  

        println("Phase number $Δ_iter started...")

        _, _, vols = solve_sub(X, P, K_p, l)
        best_total_vol = sum(vols[p][k] for p in P for k in 1:K_p[p])

        δ = 1

        η_max = compute_dist_clusters(X, P, K_p, r, l)

        η_δ = η_max / 2 ^ (δ - 1)

        # while η_δ > K * max_iteration / Δ
        while η_δ > K * (d+1)
            no_change_counter = 0

            push!(total_vol_vector, best_total_vol)
            
            for iter in 1:max_iteration

                # push!(total_vol_vector, best_total_vol)
                η = Int(ceil((1 - (iter-1) % Δ / Δ) * η_δ))
            
                l, total_vol = manage_boundary_points(X, P, K_p, l, η, r)

                # T, t, vols = solve_sub(X, P, K_p, l)
                # BendersDecomposition.plot_clusters_and_boundaries(X, l, T, t, P, K_p)

                if total_vol < best_total_vol
                    best_l = l
                    best_total_vol = total_vol
                    no_change_counter = 0
                else
                    no_change_counter += 1
                    # println("no_change_counter = $no_change_counter")
                    if no_change_counter >= Δ
                        break
                    end
                end
                # T, t, vols = solve_sub(X, P, K_p, best_l)
                # BendersDecomposition.plot_clusters_and_boundaries(X, best_l, T, t, P, K_p)        
                
            end

            l, total_vol = manage_overlap_points(X, P, K_p, best_l, r)

            # T, t, vols = solve_sub(X, P, K_p, l)
            # BendersDecomposition.plot_clusters_and_boundaries(X, l, T, t, P, K_p)

            if total_vol < best_total_vol
                best_total_vol = total_vol
                best_l = l
            end

            # Try reassigning clusters to see if it improves the total volume
            l, total_vol = reassign_clusters(X, P, K_p, best_l, best_total_vol)

            if total_vol < best_total_vol
                best_total_vol = total_vol
                best_l = l
            end

            # push!(volumes, total_vol)

            # T, t, vols = solve_sub(X, P, K_p, best_l)
            # BendersDecomposition.plot_clusters_and_boundaries(X, best_l, T, t, P, K_p)
            
            println("Phase number: $Δ_iter")
            println("η_δ = $η_δ")
            println("Best solution:")
            println("Total_volume = $(best_total_vol)")
            println("=============================")
            
            δ += 1
            η_δ = η_max / 2 ^ (δ - 1)

        end

        if best_total_vol < best_total_vol_Δ
            best_total_vol_Δ = best_total_vol
            no_change_counter_Δ = 0
        else
            no_change_counter_Δ += 1
            if no_change_counter_Δ >= 3
                break
            end
        end

        # println("Phase number: $Δ_iter finished")

    end

    # plt = plot(volumes, label = "Total volume", xlabel = "Iteration", ylabel = "Total volume", title = "Total volume vs. Iteration")
    # display(plt)

    
    return best_l ,total_vol_vector
    
end

"""
    printStatus(total_vol, best_total_vol, η_δ, iter)

    Prints the current status of the local search algorithm.

    # Arguments
    - `total_vol::Float64`: The total volume of the current solution.
    - `best_total_vol::Float64`: The total volume of the best solution found so far.
    - `η_δ::Int`: The initial number of candidate points.
    - `iter::Int`: The current iteration number.
"""
function printStatus(total_vol, best_total_vol, η_δ, iter)
    println("***Number of popout points: $η_δ")
    println("iter=$iter")
    println("Current solution:")
    println("Total_volume = $(total_vol)")
    println("=============================")
    println("Best solution:")
    println("Total_volume = $(best_total_vol)")
    println("=============================")
end


function popin_candidates(distk::Vector{Float64}, L::Dict{Float64, Vector{BitVector}}, addpoint::Int, P::Vector{Float64}, K_p::Dict{Float64, Int})
    popin = []
    i = 0
    sorted_indices = sortperm(distk)
    while length(unique(popin)) < addpoint
        i += 1
        if sum(L[p][k][sorted_indices[i]] for p in P for k in 1:K_p[p]) == 0
            push!(popin, sorted_indices[i])
        end
    end
    return popin
end

"""
    solve_sub(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, l_val::Dict{Float64, Vector{BitVector}})

    Solve the subproblems for the given data points and cluster assignments.

    # Arguments
    - `X::AbstractArray`: The data points.
    - `P::Vector{Float64}`: The set of p-norms.
    - `K_p::Dict{Float64, Int}`: The number of clusters for each p-norm.
    - `l_val::Dict{Float64, Vector{BitVector}}`: The initial cluster assignments.

    # Returns
    - `T_star::Dict{Float64, Vector{Symmetric{Float64, Matrix{Float64}}}}`: The optimal T matrices for each cluster.
    - `t_star::Dict{Float64, Vector{Vector{Float64}}}`: The optimal t vectors for each cluster.
    - `vols::Dict{Float64, Vector{Float64}}`: The volumes of the clusters.
"""
function solve_sub(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, l_val::Dict{Float64, Vector{BitVector}})
    d, N = size(X)

    θ, τ, T, t, constraints, subproblems = Dict(), Dict(), Dict(), Dict(), Dict(), Dict()

    for p in P
        subproblems[p] = [Model(MosekTools.Optimizer) for _ in 1:K_p[p]]
        θ[p] = Vector{VariableRef}(undef, K_p[p])
        τ[p] = Vector{VariableRef}(undef, K_p[p])
        T[p] = Vector{Symmetric{VariableRef, Matrix{VariableRef}}}(undef, K_p[p])
        t[p] = Vector{Vector{VariableRef}}(undef, K_p[p])

        for k in 1:K_p[p]
            set_optimizer_attribute(subproblems[p][k], "MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-8)
            set_optimizer_attribute(subproblems[p][k], "MSK_IPAR_LOG", 0)

            θ[p][k] = @variable(subproblems[p][k], lower_bound = 0, base_name = "θ[$p][$k]")
            τ[p][k] = @variable(subproblems[p][k], base_name = "τ[$p][$k]")

            T[p][k] = @variable(subproblems[p][k], [1:d, 1:d], Symmetric, base_name = "T[$p][$k]")
            t[p][k] = @variable(subproblems[p][k], [1:d], base_name = "t[$p][$k]")

            @constraint(subproblems[p][k], T[p][k] in MOI.PositiveSemidefiniteConeSquare(d))

            @objective(subproblems[p][k], Min, BendersDecomposition.unit_ball_volume(p, d) * θ[p][k])

            @constraint(subproblems[p][k], [τ[p][k]; 1; θ[p][k]] in MOI.ExponentialCone())
            @constraint(subproblems[p][k], [-τ[p][k]; 1; vec(T[p][k])] in MOI.LogDetConeSquare(d))

            dis_mat = T[p][k] * X .- t[p][k]
            for i in 1:N
                if l_val[p][k][i] > 0.0001
                    constraints[(p, k, i)] = @constraint(subproblems[p][k], 
                        [1 / l_val[p][k][i]; dis_mat[:, i]] in BendersDecomposition.select_cone(p, d))
                end
            end

            optimize!(subproblems[p][k])
        end
    end

    T_star = Dict{Float64, Vector{Symmetric{Float64, Matrix{Float64}}}}()
    t_star = Dict{Float64, Vector{Vector{Float64}}}()
    vols = Dict{Float64, Vector{Float64}}()
    for p in P
        T_star[p] = [Symmetric(value.(T[p][k])) for k in 1:K_p[p]]
        t_star[p] = [value.(t[p][k]) for k in 1:K_p[p]]
        vols[p] = [objective_value(subproblems[p][k]) for k in 1:K_p[p]]
    end

    return T_star, t_star, vols
end

"""
    outlier_detect(dist::Dict{Float64, Vector{Vector{Float64}}}, r::Float64)

    Detects outliers based on the given distance matrix and outlier ratio.

    # Arguments
    - `dist::Dict{Float64, Vector{Vector{Float64}}}`: A dictionary containing distance matrices for each p-norm and cluster.
    - `r::Float64`: The ratio of outliers to be detected.

    # Returns
    - `Set{Int}`: A set of indices corresponding to the detected outliers.
"""
function outlier_detect(dist::Dict{Float64, Vector{Vector{Float64}}}, r::Float64)

    # Initialize an empty set to store the indices of the outliers.
    outliers_set = Set{Int}()

    P = collect(keys(dist))
    K_p = Dict{Float64, Int}()
    for p in P
        K_p[p] = length(dist[p])
    end

    n = length(dist[P[1]][1])

    num_outliers = Int(round(r * n))

    dist_min = Vector{Float64}(undef, n)
    
    # from each data point to any cluster.
    for i in 1:n
        dist_min[i] = minimum(dist[p][k][i] for p in P for k in 1:K_p[p])
    end

    outliers_set = partialsortperm(dist_min, 1:num_outliers, rev=true)
    
    # Return the indices of the outliers.
    return outliers_set
end

function calculate_dist(X::AbstractArray, T::Dict{Float64, Vector{Symmetric{Float64, Matrix{Float64}}}}, t::Dict{Float64, Vector{Vector{Float64}}})
    d, n = size(X)
    P = collect(keys(T))
    K_p = Dict{Float64, Int}()
    for p in P
        K_p[p] = length(T[p])
    end

    dist = Dict{Float64, Vector{Vector{Float64}}}()

    for p in P
        dist[p] = [zeros(n) for _ in 1:K_p[p]]
        for k in 1:K_p[p]
            dist[p][k] = [norm(T[p][k] * X[:, i] .- t[p][k], p) for i in 1:n]
        end
    end

    return dist
end


"""
    update_L(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, L_in::Dict{Float64, Vector{BitVector}}, r::Float64)

    Update the cluster assignments based on the given data points, p-norms, and outlier ratio.

    # Arguments
    - `X::AbstractArray`: The data points.
    - `P::Vector{Float64}`: The set of p-norms.
    - `K_p::Dict{Float64, Int}`: The number of clusters for each p-norm.
    - `L_in::Dict{Float64, Vector{BitVector}}`: The initial cluster assignments.
    - `r::Float64`: The ratio of outliers to be detected.

    # Returns
    - `L_out::Dict{Float64, Vector{BitVector}}`: The updated cluster assignments.
"""
function update_L(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, L_in::Dict{Float64, Vector{BitVector}}, r::Float64)
    
    # Get the number of data points and dimension
    d, n = size(X)

    # Solve the problem for L_in
    T, t, _ = solve_sub(X, P, K_p, L_in)  

    # Calculate the distances from each data point to each cluster center
    dist = calculate_dist(X, T, t)

    # L_out is a matrix where each row corresponds to a data point and each column corresponds to a cluster.
    # The element at the i-th row and k-th column will be true if the i-th data point is assigned to the k-th cluster, and false otherwise.
    L_out = Dict{Float64, Vector{BitVector}}()
    for p in P
        L_out[p] = [falses(n) for _ in 1:K_p[p]]
    end
    
    outliers_set = outlier_detect(dist, r)

    # plot!(X[1, outliers_set], X[2, outliers_set], seriestype = :scatter, label = "Outliers", color = :red)

    for p in P
        for k in 1:K_p[p]
            for i in 1:n
                if i ∉ outliers_set
                    L_out[p][k][i] = dist[p][k][i] <= 1 + TOLERANCE
                end
            end
        end
    end
    # Return the updated cluster assignments.
    return L_out
end

function reassign_clusters(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, l::Dict{Float64, Vector{BitVector}}, best_total_vol::Float64)
    d, n = size(X)
    best_l = deepcopy(l)
    # best_total_vol = Inf
    flag = 1
    
    while flag == 1
        flag = 0
        for p1 in P
            for k1 in 1:K_p[p1]
                for p2 in P
                    for k2 in 1:K_p[p2]
                        if p1 != p2
                            # Create a copy of the current cluster assignments
                            l_temp = deepcopy(best_l)
                            
                            # Reassign all points from cluster (p1, k1) to cluster (p2, k2)
                            l_temp[p2][k2] .= best_l[p1][k1]
                            l_temp[p1][k1] .= best_l[p2][k2]
                            
                            # Solve the subproblem with the new assignments
                            _, _, vols_temp = solve_sub(X, P, K_p, l_temp)
                            total_vol_temp = sum(vols_temp[p][k] for p in P for k in 1:K_p[p])
                            
                            # Check if the new assignment gives a better total volume
                            if total_vol_temp < best_total_vol
                                best_total_vol = total_vol_temp
                                best_l = deepcopy(l_temp)
                                flag = 1
                            end
                        end
                    end
                end
            end
        end
    end

    return best_l, best_total_vol
end

"""
    manage_overlap_points(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, l::Dict{Float64, Vector{BitVector}}, r::Float64)

TBW
"""
function manage_overlap_points(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, l::Dict{Float64, Vector{BitVector}}, r::Float64)
    d, n = size(X)
    K = sum(K_p[p] for p in P)

    l = update_L(X, P, K_p, l, r)
    clusters_dict = create_clusters_dict(P, K_p)
    Oval = Float64(Inf)

    for j in 1:K - 1
        for k in j+1:K
            l, Oval = handle_overlap(X, P, K_p, l, clusters_dict, j, k, d, n)
        end
    end

    return l, Oval
end

function create_clusters_dict(P::Vector{Float64}, K_p::Dict{Float64, Int})
    clusters_dict = Vector{Pair{Float64, Int}}()
    for p in P
        for k in 1:K_p[p]
            clusters_dict = push!(clusters_dict, p => k)
        end
    end

    return clusters_dict
end

function handle_overlap(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, l::Dict{Float64, Vector{BitVector}}, clusters_dict::Vector{Pair{Float64, Int}}, j::Int, k::Int, d::Int, n::Int)
    l_temp_1 = deepcopy(l)
    l_temp_2 = deepcopy(l)
    overlap_indices = find_overlap_indices(l, clusters_dict, j, k, n)

    l_temp_1[clusters_dict[j][1]][clusters_dict[j][2]][overlap_indices] .= false
    l_temp_2[clusters_dict[k][1]][clusters_dict[k][2]][overlap_indices] .= false

    for l_temp in (l_temp_1, l_temp_2)
        l_temp = ensure_minimum_points(X, P, K_p, l_temp, d, n)
    end

    _, _, vols_temp_1 = solve_sub(X, P, K_p, l_temp_1)
    Ovals1 = sum(vols_temp_1[p][k] for p in P for k in 1:K_p[p])

    _, _, vols_temp_2 = solve_sub(X, P, K_p, l_temp_2)
    Ovals2 = sum(vols_temp_2[p][k] for p in P for k in 1:K_p[p])

    l = Ovals1 <= Ovals2 ? l_temp_1 : l_temp_2
    Oval = Ovals1 <= Ovals2 ? Ovals1 : Ovals2

    return l, Oval
end

function find_overlap_indices(l::Dict{Float64, Vector{BitVector}}, clusters_dict::Vector{Pair{Float64, Int}}, j::Int, k::Int, n::Int)
    overlap_indices = []
    for i in 1:n
        if l[clusters_dict[j][1]][clusters_dict[j][2]][i] && l[clusters_dict[k][1]][clusters_dict[k][2]][i]
            push!(overlap_indices, i)
        end
    end
    return overlap_indices
end

function ensure_minimum_points(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, l_temp::Dict{Float64, Vector{BitVector}}, d::Int, n::Int)
    for p in P
        for k in 1:K_p[p]
            if sum(l_temp[p][k]) < d + 1
                max_count = -1
                max_index = (-1.0, -1)
                for i in P
                    for j in 1:K_p[i]
                        count = sum(l_temp[i][j])
                        if count > max_count
                            max_count = count
                            max_index = (i, j)
                        end
                    end
                end
                p_max = max_index[1]
                k_p_max = max_index[2]
                if sum(l_temp[p_max][k_p_max]) > 1
                    # R = kmeans(X[:, l_temp[p_max][k_p_max]], 2)
                    try
                        X_transposed = Matrix(X[:, l_temp[p_max][k_p_max]]')

                        gmm = GMM(2, X_transposed; method = :kmeans, kind = :full)
                        posteriors = gmmposterior(gmm, X_transposed)
                        
                        i = 0
                        for point = 1:n
                            if l_temp[p_max][k_p_max][point]
                                i += 1
                                if argmax(posteriors[1][i, :]) == 2
                                    l_temp[p_max][k_p_max][point] = false
                                    l_temp[p][k][point] = true
                                end
                            end
                        end
                    catch
                        println("Error in ensure_minimum_points, using kmeans instead")
                        R = kmeans(X[:, l_temp[p_max][k_p_max]], 2)
                        i=0
                        for point in 1:n
                            if l_temp[p_max][k_p_max][point]
                                i += 1
                                if R.assignments[i] == 2
                                    l_temp[p_max][k_p_max][point] = false
                                    l_temp[p][k][point] = true
                                end
                            end
                        end
                    end
                else
                    # If not enough points, handle appropriately (e.g., assign all points to the current cluster)
                    for point = 1:n
                        if l_temp[p_max][k_p_max][point]
                            l_temp[p_max][k_p_max][point] = false
                            l_temp[p][k][point] = true
                        end
                    end
                end
            end
        end
    end

    return l_temp
end

function compute_dist_clusters(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, r::Float64, l::Dict{Float64, Vector{BitVector}})
    # Get the number of points and clusters
    d, n = size(X)

    l = ensure_minimum_points(X, P, K_p, l, d, n)

    l_temp = update_L(X, P, K_p, l, r)

    T, t, _ = solve_sub(X, P, K_p, l_temp)

    dist = calculate_dist(X, T, t)

    # Initialize an array to store the specific distance values for each cluster
    specific_dists = Dict()

    for p in P
        specific_dists[p] = zeros(K_p[p])
        for k in 1:K_p[p]
            # Get the distances of the points from this cluster
            cluster_dists = dist[p][k]

            # Sort the distances
            sorted_dists = sort(cluster_dists)

            # Find the specific distance value such that only d+1 points have a smaller distance
            specific_dists[p][k] = sorted_dists[d+1]
        end
    end

    specific_dist = maximum(specific_dists[p][k] for p in P for k in 1:K_p[p])
    # specific_dist = minimum(specific_dists[p][k] for p in P for k in 1:K_p[p])

    for p in P
        for k in 1:K_p[p]
            [l_temp[p][k][i] = dist[p][k][i] <= specific_dist for i in 1:n]
        end
    end

    max_num_candidates = sum(sum(l_temp[p][k] for p in P for k in 1:K_p[p]) .== 0) - Int(round(r * n))

    return max_num_candidates
end

function find_closest_to_one_indices(dist::Dict{Float64, Vector{Vector{Float64}}}, η::Int, l::Dict{Float64, Vector{BitVector}})
    # Initialize arrays to store distances and corresponding indices
    distances = Float64[]
    indices = Int[]

    # Iterate through the dictionary to find elements less than 1 + TOLERANCE * 10 and calculate distances
    for (p, vectors) in dist
        for k_p in 1:length(vectors)
            for i in 1:length(vectors[k_p])
                val = vectors[k_p][i]
                if l[p][k_p][i]
                    # && val < 1 + TOLERANCE * 10
                    distance = abs(val - 1)
                    push!(distances, distance)
                    push!(indices, i)
                end
            end
        end
    end

    # Sort the distances and corresponding indices based on distance
    sorted_indices = sortperm(distances)

    # Limit the number of elements to return
    num_elements = minimum([η, length(sorted_indices)])

    # Return sorted indices
    # Return the indices of the closest points to one
    return indices[sorted_indices[1:num_elements]]

    # # Calculate weights inversely proportional to the distances
    # weights = (1.0 ./ distances) .^ 2

    # # Normalize weights to sum to 1
    # weights /= sum(weights)

    # # Sample indices based on weights
    # sampled_indices = sample(1:length(indices), Weights(weights), η; replace=false)

    # # Return the sampled indices
    # return [indices[i] for i in sampled_indices]
end

"""
    manage_boundary_points(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, l::Dict{Float64, Vector{BitVector}}, η::Int, r::Float64)

    Manage the boundary points of clusters to minimize the total volume.

    # Arguments
    - `X::AbstractArray`: The data points.
    - `P::Vector{Float64}`: The set of p-norms.
    - `K_p::Dict{Float64, Int}`: The number of clusters for each p-norm.
    - `l::Dict{Float64, Vector{BitVector}}`: The initial cluster assignments.
    - `η::Int`: The number of candidate points to consider.
    - `r::Float64`: The ratio of outliers to be detected.

    # Returns
    - `best_l::Dict{Float64, Vector{BitVector}}`: The updated cluster assignments.
    - `best_total_vol::Float64`: The total volume of the best solution found.
"""
function manage_boundary_points(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, l::Dict{Float64, Vector{BitVector}}, η::Int, r::Float64)
    d, n = size(X)
    l = update_L(X, P, K_p, l, r)

    T, t, _ = solve_sub(X, P, K_p, l)

    dist = calculate_dist(X, T, t)

    K = sum(K_p[p] for p in P)

    candidates = find_closest_to_one_indices(dist, η, l)
    # plt = scatter!(X[1, candidates], X[2, candidates], seriestype = :scatter, label = "Candidates", color = :red)
    # display(plt)

    l_shrunk = shrink_clusters(l, candidates, P, K_p)

    l_shrunk = ensure_minimum_points(X, P, K_p, l_shrunk, d, n)

    best_l, best_total_vol = find_best_clusters(X, P, K_p, l_shrunk, r, d, n, K)

    return best_l, best_total_vol
end

function shrink_clusters(l::Dict{Float64, Vector{BitVector}}, candidates::Vector{Int}, P::Vector{Float64}, K_p::Dict{Float64, Int})
    l_shrunk = deepcopy(l)
    for p in P
        for k in 1:K_p[p]
            l_shrunk[p][k][candidates] .= false
        end
    end
    return l_shrunk
end

function find_best_clusters(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, l_shrunk::Dict{Float64, Vector{BitVector}}, r::Float64, d::Int, n::Int, K::Int)
    best_l = deepcopy(l_shrunk)
    best_total_vol = Inf

    neededPoint = 1
    while neededPoint > 0
        T, t, vols = solve_sub(X, P, K_p, l_shrunk)
        dist = calculate_dist(X, T, t)

        neededPoint = sum(sum(l_shrunk[p][k][i] for p in P for k in 1:K_p[p]) .== 0 for i in 1:n) - Int(round(r * n))
        addpoint = Int(ceil(neededPoint / K))

        total_vol = Inf

        for p in P
            for k in 1:K_p[p]
                L_copy = deepcopy(l_shrunk)
                candidates = popin_candidates(dist[p][k], L_copy, addpoint, P, K_p)
                L_copy[p][k][candidates] .= true

                T, t, vols = solve_sub(X, P, K_p, L_copy)
                clusters_vols_sum = sum(vols[p][k] for p in P for k in 1:K_p[p])

                if clusters_vols_sum < total_vol
                    total_vol = clusters_vols_sum
                    best_l = L_copy
                end
            end
        end

        l_shrunk = deepcopy(best_l)
        best_total_vol = total_vol
    end

    return best_l, best_total_vol
end


end