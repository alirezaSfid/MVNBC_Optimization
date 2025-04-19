module BendersDecomposition

const THRESHOLD = 0.0001

using JuMP
using Gurobi
using MosekTools
using LinearAlgebra
using MathOptInterface
using Clustering
using SpecialFunctions
using Random
using Distributions
using Plots
using PlotThemes
using Distances
using NPZ
using GaussianMixtures

theme(:bright)

"""
    solve_subproblem(X, l_val, P, K_p)

    Solves the subproblem for given data points, cluster assignments, p-norm values, and number of clusters.

    # Arguments
    - `X`: A matrix of data points (d × N), where `d` is the dimension and `N` is the number of points.
    - `l_val`: Initial cluster assignment values.
    - `P`: Array of p-norm values (e.g., [1.0, 2.0, Inf]).
    - `K_p`: Dictionary specifying the number of clusters for each p-norm.

    # Returns
    - `subproblems`: Dictionary of subproblem models.
    - `constraints`: Dictionary of constraints for each subproblem.
    - `T`: Dictionary of transformation matrices for each subproblem.
    - `t`: Dictionary of translation vectors for each subproblem.
    - `θ`: Dictionary of θ variables for each subproblem.
    - `τ`: Dictionary of τ variables for each subproblem.
"""
function solve_subproblem(X, l_val, P, K_p)
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

            @objective(subproblems[p][k], Min, unit_ball_volume(p, d) * θ[p][k])

            @constraint(subproblems[p][k], [τ[p][k]; 1; θ[p][k]] in MOI.ExponentialCone())
            @constraint(subproblems[p][k], [-τ[p][k]; 1; vec(T[p][k])] in MOI.LogDetConeSquare(d))

            dis_mat = T[p][k] * X .- t[p][k]
            for i in 1:N
                if l_val[p][k][i] > 0.0001
                    constraints[(p, k, i)] = @constraint(subproblems[p][k], 
                        [1 / l_val[p][k][i]; dis_mat[:, i]] in select_cone(p, d))
                end
            end

            optimize!(subproblems[p][k])
        end
    end

    return subproblems, constraints, T, t
end

"""
    setup_master_problem(N, P, K_p, d, r)

    Sets up the master problem for optimization, configuring variables, constraints, and objective function.

    # Arguments
    - `N`: Number of data points.
    - `P`: Array of p-norm values (e.g., [1.0, 2.0, Inf]).
    - `K_p`: Dictionary specifying the number of clusters for each p-norm.
    - `d`: Dimension of the dataset (used to ensure non-zero volume).
    - `r`: Outlier rate, representing the proportion of data points allowed as outliers.

    # Returns
    - `master`: JuMP model representing the master problem.
    - `l`: Dictionary of binary cluster assignment variables.
"""
function setup_master_problem(N, P, K_p, d, r)
    # Initialize Gurobi model and set optimizer attributes
    # master = direct_model(Gurobi.Optimizer())
    master = Model(Gurobi.Optimizer)

    # Define objective variable and assignment dictionary
    Φ = Dict{Float64, Vector{VariableRef}}()

    l = Dict{Float64, Vector{Vector{VariableRef}}}()

    # Create cluster assignment variables and constraints
    for p in P   
        l[p] = [@variable(master, [1:N], Bin, base_name = "l[$p][$k]") for k in 1:K_p[p]]
        Φ[p] = [@variable(master, lower_bound = 0, base_name = "Φ[$p][$k]") for k in 1:K_p[p]]
        for k in 1:K_p[p]
            # Ensure each cluster has at least `d + 1` points to avoid zero-volume clusters
            @constraint(master, sum(l[p][k]) >= d + 1)
        end
    end

    # Define binary variables to mark outliers
    @variable(master, w[1:N] >= 0, Bin, base_name = "w")

    # Enforce constraints for outlier and assignment relationship
    for i in 1:N
        @constraint(master, w[i] <= sum(l[p][k][i] for p in P for k in 1:K_p[p]))
        @constraint(master, sum(l[p][k][i] for p in P for k in 1:K_p[p]) <= w[i])
    end

    # Set total allowed outliers constraint
    @constraint(master, sum(w) == Int(round((1 - r) * N)))

    @objective(master, Min, sum(Φ[p][k] for p in P for k in 1:K_p[p]))

    return master, l, Φ, w  
end

function initialize_subproblems(X, P, K_p, d)
    subproblems, θ, τ, T, t, constraints = Dict(), Dict(), Dict(), Dict(), Dict(), Dict()

    for p in P
        subproblems[p] = [Model(MosekTools.Optimizer) for _ in 1:K_p[p]]
        θ[p] = Vector{VariableRef}(undef, K_p[p])
        τ[p] = Vector{VariableRef}(undef, K_p[p])
        T[p] = Vector{Symmetric{VariableRef, Matrix{VariableRef}}}(undef, K_p[p])
        t[p] = Vector{Vector{VariableRef}}(undef, K_p[p])

        for k in 1:K_p[p]
            model = subproblems[p][k]
            set_optimizer_attribute(model, "MSK_DPAR_INTPNT_TOL_REL_GAP", 1e-8)
            set_optimizer_attribute(model, "MSK_IPAR_LOG", 0)

            θ[p][k] = @variable(model, lower_bound = 0, base_name = "θ[$p][$k]")
            τ[p][k] = @variable(model, base_name = "τ[$p][$k]")
            T[p][k] = @variable(model, [1:d, 1:d], Symmetric, base_name = "T[$p][$k]")
            t[p][k] = @variable(model, [1:d], base_name = "t[$p][$k]")

            @constraint(model, T[p][k] in MOI.PositiveSemidefiniteConeSquare(d))
            @constraint(model, [τ[p][k]; 1; θ[p][k]] in MOI.ExponentialCone())
            @constraint(model, [-τ[p][k]; 1; vec(T[p][k])] in MOI.LogDetConeSquare(d))

            @objective(subproblems[p][k], Min, unit_ball_volume(p, d) * θ[p][k])

            for i in 1:size(X, 2)
                constraints[(p, k, i)] = @constraint(model, [1; X[:, i]] in select_cone(p, d))
            end
        end
    end

    return subproblems, θ, τ, T, t, constraints
end

"""
    initialize_values!(l, Φ, w, X, P, K_p, r)

    Sets initial values for the variables in the master problem.

    # Arguments:
    - `l`: Dictionary of binary cluster assignment variables.
    - `Φ`: Objective variable in the master problem.
    - `w`: Binary outlier variables in the master problem.
    - `X`: A matrix of data points (d × N), where `d` is the dimension and `N` is the number of points.
    - `P`: Array of p-norm values (e.g., [1.0, 2.0, Inf]).
    - `K_p`: Dictionary specifying the number of clusters for each p-norm.
    - `r`: Outlier rate, representing the proportion of data points allowed as outliers.

    # Returns:
    - None (modifies the master problem in-place).
"""
function initialize_values!(l, Φ, w, X, P, K_p, r)

    N = size(X, 2)

    l_val = assign_clusters_GMM(X, K_p, P, r)

    subproblems, _, _, _ = solve_subproblem(X, l_val, P, K_p)


    # Set initial values for cluster assignment variables
    for p in P
        for k in 1:K_p[p]
            set_start_value.(l[p][k], l_val[p][k])
            set_start_value(Φ[p][k], objective_value(subproblems[p][k]))
        end
    end

    # Set initial values for outlier variables
    set_start_value.(w, [sum(l_val[p][k][i] for p in P for k in 1:K_p[p]) > 0.5 for i in 1:N])
end

"""
    benders_callback_cached(cb_data, cb_where::Cint, callback_data)

    Callback function for Benders decomposition with cached subproblems.

    # Arguments
    - `cb_data`: Callback data provided by the optimizer.
    - `cb_where`: Integer indicating where the callback is triggered.
    if cb_where == GRB_CB_MIPSOL || cb_where == GRB_CB_MIPNODE

    # Returns
    - None (modifies the master problem in-place).
"""
function benders_callback_cached(cb_data, cb_where::Cint, callback_data)
    # Unpack cached data
    l, Φ, P, K_p, X, master, subproblems, T, t, constraints = 
        callback_data.l, callback_data.Φ, callback_data.P, callback_data.K_p, 
        callback_data.X, callback_data.master, callback_data.subproblems, 
        callback_data.T, callback_data.t, callback_data.constraints

    # Check where the callback is triggered
    if cb_where != GRB_CB_MIPSOL && cb_where != GRB_CB_MIPNODE
        return
    end

    # Skip non-optimal nodes
    if cb_where == GRB_CB_MIPNODE
        resultP = Ref{Cint}()
        GRBcbget(cb_data, cb_where, GRB_CB_MIPNODE_STATUS, resultP)
        if resultP[] != GRB_OPTIMAL
            return
        end
    end

    # Load variable values from the master problem
    Gurobi.load_callback_variable_primal(cb_data, cb_where)
    l_val = Dict(p => [callback_value.(cb_data, l[p][k]) for k in 1:K_p[p]] for p in P)


    # Solve cached subproblems with updated constraints
    subproblems, constraints, T, t = update_cached_subproblems(X, l_val, P, K_p, subproblems, T, t, constraints)
    # println("Constraints:", constraints)

    # Compute subproblem objective value
    sub_obj_val = sum(objective_value(subproblems[p][k]) for p in P for k in 1:K_p[p])
    # println("Subproblem objective value: ", sub_obj_val)
    Φ_val = sum(callback_value.(cb_data, Φ[p][k]) for p in P for k in 1:K_p[p])
    # println("Master problem objective value: ", Φ_val)

    # Add lazy constraint if subproblem is violated
    if sub_obj_val > Φ_val + 1e-6
        benders_cut = create_benders_cut_2(subproblems, constraints, T, t, l, l_val, X, P, K_p)
        for p in P
            for k in 1:K_p[p]
                con = @build_constraint(Φ[p][k] >= benders_cut[p][k])
                # println("Adding lazy constraint: ", con)
                MOI.submit(master, MOI.LazyConstraint(cb_data), con)
            end
        end
        # con = @build_constraint(sum(Φ[p][k] for p in P for k in 1:K_p[p]) >= fat_cut)
        # MOI.submit(master, MOI.LazyConstraint(cb_data), con)
    end
end

"""
    update_cached_subproblems(X, l_val, P, K_p, subproblems, T, t, constraints)

    Solves the subproblem with cached subproblems, updating constraints based on new cluster assignments.

    # Arguments
    - `X`: A matrix of data points (d × N), where `d` is the dimension and `N` is the number of points.
    - `l_val`: Current cluster assignment values.
    - `P`: Array of p-norm values (e.g., [1.0, 2.0, Inf]).
    - `K_p`: Dictionary specifying the number of clusters for each p-norm.
    - `subproblems`: Dictionary of cached subproblem models.
    - `T`: Dictionary of transformation matrices for each subproblem.
    - `t`: Dictionary of translation vectors for each subproblem.
    - `constraints`: Dictionary of constraints for each subproblem.

    # Returns
    - `subproblems`: Updated dictionary of subproblem models.
    - `constraints`: Updated dictionary of constraints for each subproblem.
    - `T`: Updated dictionary of transformation matrices for each subproblem.
    - `t`: Updated dictionary of translation vectors for each subproblem.
"""
function update_cached_subproblems(X, l_val, P, K_p, subproblems, T, t, constraints)
    d, N = size(X)

    for p in P
        for k in 1:K_p[p]
            model = subproblems[p][k]

            # Add new constraints based on updated l_val
            dis_mat = T[p][k] * X .- t[p][k]
            for i in 1:N
                if haskey(constraints, (p, k, i))
                    constraint = constraints[(p, k, i)]
                    try
                        delete(model, constraint)
                        delete!(constraints, (p, k, i))
                    catch e
                        if isa(e, MOI.InvalidIndex)
                            println("Warning: Invalid index encountered while deleting constraint for (p=$p, k=$k, i=$i).")
                        else
                            println("Error: Exception encountered while deleting constraint for (p=$p, k=$k, i=$i): ", e)
                            rethrow(e)
                        end
                    end
                end
                if l_val[p][k][i] > 0.0001
                    constraints[(p, k, i)] = @constraint(model, [1 / l_val[p][k][i]; dis_mat[:, i]] in select_cone(p, d))
                end
            end

            # Optimize the reused model
            optimize!(model)
        end
    end

    return subproblems, constraints, T, t
end

"""
    benders_algorithm_cached(X, P, K_p, r)

    Executes the Benders decomposition algorithm with cached subproblems.

    # Arguments
    - `X`: A matrix of data points (d × N), where `d` is the dimension and `N` is the number of points.
    - `P`: Array of p-norm values (e.g., [1.0, 2.0, Inf]).
    - `K_p`: Dictionary specifying the number of clusters for each p-norm.
    - `r`: Outlier rate, representing the proportion of data points allowed as outliers.

    # Returns
    - `l_star`: Dictionary of optimal cluster assignments.
    - `T_star`: Dictionary of optimal transformation matrices for each subproblem.
    - `t_star`: Dictionary of optimal translation vectors for each subproblem.
    - `objective_value`: The optimal value of the master problem.
"""
function benders_algorithm_cached(X, P, K_p, r)
    d, N = size(X)

    # Initialize master problem
    master, l, Φ, w = setup_master_problem(N, P, K_p, d, r)

    # Initialize subproblems
    subproblems, θ, τ, T, t, constraints = initialize_subproblems(X, P, K_p, d)

    # Set initial values
    initialize_values!(l, Φ, w, X, P, K_p, r)

    callback_data = (
        l = l,
        Φ = Φ,
        P = P,
        K_p = K_p,
        X = X,
        master = master,
        subproblems = subproblems,
        θ = θ,
        τ = τ,
        T = T,
        t = t,
        constraints = constraints,
    )

    # Attach callback with cached subproblems
    MOI.set(master, MOI.RawOptimizerAttribute("LazyConstraints"), 1)
    MOI.set(master, Gurobi.CallbackFunction(), (cb_data, cb_where) -> benders_callback_cached(cb_data, cb_where, callback_data))
    optimize!(master)

    # Extract optimal values
    l_star = Dict{Float64, Vector{BitVector}}()
    for p in P
        l_star[p] = [BitVector(value.(l[p][k]) .> 0.5) for k in 1:K_p[p]]
    end

    # Solve final subproblems
    subproblems, constraints, T, t = update_cached_subproblems(X, l_star, P, K_p, subproblems, T, t, constraints)

    T_star = Dict{Float64, Vector{Matrix{Float64}}}()
    t_star = Dict{Float64, Vector{Vector{Float64}}}()
    for p in P
        T_star[p] = [value.(T[p][k]) for k in 1:K_p[p]]
        t_star[p] = [value.(t[p][k]) for k in 1:K_p[p]]
    end

    return l_star, T_star, t_star, objective_value(master)
end

"""
    plot_clusters_and_boundaries(X, labels, T, t, P, K_p)

    Visualizes clustered data points and the corresponding cluster boundaries for each specified p-norm.

    # Arguments
    - `X`: A 2D array (d × N matrix) containing the data points to be clustered, where `d` is the dimension, and `N` is the number of points.
    - `labels`: A dictionary mapping each p-norm (e.g., 1.0, 2.0, Inf) to a list of boolean arrays. Each array indicates the cluster assignment for the data points under the respective p-norm.
    - `T`: A dictionary of transformation matrices. For each p-norm and cluster, these matrices define the scaling and rotation transformations used to compute the cluster boundaries.
    - `t`: A dictionary of translation vectors. For each p-norm and cluster, these vectors define the offsets for the cluster boundaries.
    - `P`: A collection (e.g., vector or set) of p-norm values (e.g., [1.0, 2.0, Inf]) used for clustering.
    - `K_p`: A dictionary specifying the number of clusters (`K_p[p]`) for each p-norm value `p`.

    # Functionality
    - Plots the data points in a scatter plot.
    - Visualizes the boundaries of the clusters for each p-norm using the transformation matrices (`T`) and translation vectors (`t`).
    - Labels the clusters and boundaries for each p-norm.
    - Uses `plot_boundary` to plot the boundary for a cluster in a specified p-norm.

    # Output
    - Displays a plot showing the clustered data points and their corresponding boundaries.
"""
function plot_clusters_and_boundaries(X, labels, T, t, P, K_p)
    # Create a new plot
    plt = scatter(X[1, :], X[2, :], label = "Data points", markersize = 3, xlabel = "x", ylabel = "y")

    # Precompute origin shapes for each p-norm
    origin_shapes = Dict(
        1.0 => generate_l1_shape(),
        2.0 => generate_l2_shape(),
        Inf => generate_linf_shape()
    )

    # Prepare a batch for cluster points and boundaries
    cluster_points_x = []
    cluster_points_y = []
    cluster_labels = []
    
    boundary_x = []
    boundary_y = []
    boundary_labels = []

    # Process each p-norm and its clusters
    for p in P
        shape = origin_shapes[p]  # Precomputed shape for this p-norm
        for k in 1:K_p[p]
            # Extract cluster points
            cluster_indices = labels[p][k]
            push!(cluster_points_x, X[1, cluster_indices])
            push!(cluster_points_y, X[2, cluster_indices])
            push!(cluster_labels, "Cluster $k (p=$p)")

            # Transform the shape for the cluster boundary
            B = inv(T[p][k])
            x_bar = B * t[p][k]
            transformed_shape = B * reduce(hcat, shape) .+ x_bar

            # Add boundary points for plotting
            push!(boundary_x, transformed_shape[1, :])
            push!(boundary_y, transformed_shape[2, :])
            push!(boundary_labels, "Boundary $k (p=$p)")
        end
    end

    # Batch plot cluster points
    for i in 1:length(cluster_points_x)
        scatter!(plt, cluster_points_x[i], cluster_points_y[i], label = cluster_labels[i], markersize = 3)
    end

    # Batch plot boundaries
    for i in 1:length(boundary_x)
        plot!(plt, boundary_x[i], boundary_y[i], label = boundary_labels[i])
    end

    # Display the final plot
    # display(plt)

    return plt
end

# Function to compute the volume of a unit ball in a given p-norm and dimension d
function unit_ball_volume(p::Float64, d::Int)
    log_volume = d * log(2gamma(1 + 1 / p)) - lgamma(1 + d / p)
    volume = exp(log_volume)
    return volume
end

"""
    assign_clusters_using_kmeans(X, K_p, P, r)

    Assigns clusters to data points using k-means clustering.

    # Arguments
    - `X`: A matrix of data points (d × N), where `d` is the dimension and `N` is the number of points.
    - `K_p`: Dictionary specifying the number of clusters for each p-norm.
    - `P`: Array of p-norm values (e.g., [1.0, 2.0, Inf]).
    - `r`: Outlier rate, representing the proportion of data points allowed as outliers.

    # Returns
    - `l_val`: Dictionary of binary cluster assignment variables.
"""
function assign_clusters_using_kmeans(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, r::Float64)
    N = size(X, 2)
    K = sum(K_p[p] for p in P)  # Total number of clusters
    num_outliers = N - Int(round((1 - r) * N))

    # Run k-means clustering with the specified number of clusters
    k_means_results = kmeans(X, K; maxiter = 100, display = :none)
    
    # Initialize the output dictionary
    l_val = Dict{Float64, Vector{BitVector}}()

    # Assign clusters based on k-means assignments
    cluster_idx = 1
    for p in P
        l_val[p] = [falses(N) for _ in 1:K_p[p]]
        for k_p in 1:K_p[p]
            l_val[p][k_p] .= (k_means_results.assignments .== cluster_idx)
            cluster_idx += 1
        end
    end

    l_vectors = [l_val[p][k] for p in P for k in 1:K_p[p]]

    best_T, best_t, best_l_val = Dict(), Dict(), Dict()
    least_vol = Inf

    for k in 0:(K-1)
        arr = collect(1:K)
        rotated_arr = circshift(arr, -k)
        iter = 1
        for p in P
            for k_p in 1:K_p[p]
                l_val[p][k_p] = l_vectors[rotated_arr[iter]]
                iter += 1
            end
        end
        subproblems, _, T, t = solve_subproblem(X, l_val, P, K_p)
        vol = sum(objective_value(subproblems[p][k]) for p in P for k in 1:K_p[p])

        if vol < least_vol
            least_vol = vol
            best_T = T
            best_t = t
            best_l_val = l_val
        end
    end

    T, t, l_val = best_T, best_t, best_l_val
    
    dist = Dict{Float64, Vector{Vector{Float64}}}()

    for p in P
        dist[p] = [zeros(N) for _ in 1:K_p[p]]
        for k in 1:K_p[p]
            dis_mat = value.(T[p][k]) * X .- value.(t[p][k])
            dist[p][k] = [norm(dis_mat[:, i], p) for i in 1:N]
        end
    end

    lowest_dist = zeros(N)
    for i in 1:N
        lowest_dist[i] = minimum([dist[p][k][i] for p in P for k in 1:K_p[p]])
    end
    sorted_indices = sortperm(lowest_dist)
    outliers = sorted_indices[end - num_outliers + 1:end]

    for i in outliers
        for p in P
            for k in 1:K_p[p]
                l_val[p][k][i] = false
            end
        end
    end

    return l_val
    return l_val
end

function assign_clusters_GMM(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, r::Float64)
    N = size(X, 2)
    K = sum(K_p[p] for p in P)  # Total number of clusters
    num_outliers = N - Int(round((1 - r) * N))

    X_transposed = Matrix(X')

    gmm = GMM(K, X_transposed; method = :kmeans, kind = :full)
    posteriors = gmmposterior(gmm, X_transposed)

    # Initialize the output dictionary
    l_val = Dict{Float64, Vector{BitVector}}()

    # Assign clusters based on GMM assignments
    cluster_idx = 1
    for p in P
        l_val[p] = [falses(N) for _ in 1:K_p[p]]
        for k_p in 1:K_p[p]
            for i in 1:N
                if argmax(posteriors[1][i, :]) == cluster_idx
                    l_val[p][k_p][i] = true
                end
            end
            cluster_idx += 1
        end
    end

    l_vectors = [l_val[p][k] for p in P for k in 1:K_p[p]]

    best_T, best_t, best_l_val = Dict(), Dict(), Dict()
    least_vol = Inf

    for k in 0:(K-1)
        arr = collect(1:K)
        rotated_arr = circshift(arr, -k)
        iter = 1
        for p in P
            for k_p in 1:K_p[p]
                l_val[p][k_p] = l_vectors[rotated_arr[iter]]
                iter += 1
            end
        end
        subproblems, _, T, t = solve_subproblem(X, l_val, P, K_p)
        vol = sum(objective_value(subproblems[p][k]) for p in P for k in 1:K_p[p])

        if vol < least_vol
            least_vol = vol
            best_T = T
            best_t = t
            best_l_val = l_val
        end
    end

    T, t, l_val = best_T, best_t, best_l_val
    
    dist = Dict{Float64, Vector{Vector{Float64}}}()

    for p in P
        dist[p] = [zeros(N) for _ in 1:K_p[p]]
        for k in 1:K_p[p]
            dis_mat = value.(T[p][k]) * X .- value.(t[p][k])
            dist[p][k] = [norm(dis_mat[:, i], p) for i in 1:N]
        end
    end

    lowest_dist = zeros(N)
    for i in 1:N
        lowest_dist[i] = minimum([dist[p][k][i] for p in P for k in 1:K_p[p]])
    end
    sorted_indices = sortperm(lowest_dist)
    outliers = sorted_indices[end - num_outliers + 1:end]

    for i in outliers
        for p in P
            for k in 1:K_p[p]
                l_val[p][k][i] = false
            end
        end
    end

    return l_val
end

# Helper function to select the appropriate cone for each p-norm
function select_cone(p, d)
    if p == 1
        return MOI.NormOneCone(d + 1)
    elseif p == 2
        return MOI.SecondOrderCone(d + 1)
    elseif p == Inf
        return MOI.NormInfinityCone(d + 1)
    else
        error("Unsupported p-norm: $p")
    end
end

function create_benders_cut_2(subproblems, constraints, T, t, l, l_val, X, P, K_p)
    bender_cuts = Dict()

    
    # Pre-compute the number of points
    N = size(X, 2)
    
    for p in P
        bender_cuts[p] = Vector{AffExpr}(undef, K_p[p])  # Create a vector of constraints for this p
        for k in 1:K_p[p]
            # Initialize the Benders cut with the objective value
            bender_cuts[p][k] = objective_value(subproblems[p][k])
            
            # Precompute the distance matrix for this cluster
            dis_mat = value.(T[p][k]) * X .- value.(t[p][k])

            # Cache the relevant constraints for (p, k)
            relevant_constraints = Dict(i => constraints[(p, k, i)] for i in 1:N if haskey(constraints, (p, k, i)))
            
            for i in 1:N
                # Check if a constraint exists for (p, k, i)
                if haskey(relevant_constraints, i)
                    try
                        dual_value = dual(relevant_constraints[i])[1]  # Access dual value
                        # Update the Benders cut
                        add_to_expression!(
                            bender_cuts[p][k],
                            dual_value * norm(dis_mat[:, i], p) * (l[p][k][i] - l_val[p][k][i])
                        )
                    catch e
                        println("Error accessing dual value for constraint (p=$p, k=$k, i=$i): ", e)
                    end
                end
            end
            # fat_cut += bender_cuts[p][k]
        end
    end
    
    return bender_cuts
end

# Function to generate cluster data
function generate_clusters(n::Int, means::Vector{Vector{Float64}}, covariances::Vector{Matrix{Float64}})

    d = length(means[1])
    K = length(means)
    clusters = Vector{Matrix{Float64}}(undef, K)

    for i in 1:K
        dist = MvNormal(means[i], covariances[i])
        clusters[i] = rand(dist, n)
    end
	return clusters
end

# Functions for generating unit shapes for different norms
function generate_l1_shape()
    delta_range = 0:0.05:1.0
    vcat(
        [[0 + δ, 1 - δ] for δ in delta_range],
        [[1 - δ, 0 - δ] for δ in delta_range],
        [[0 - δ, -1 + δ] for δ in delta_range],
        [[-1 + δ, 0 + δ] for δ in delta_range]
    )
end

function generate_l2_shape(step_size = 0.05)
    theta_range = range(0, stop=2π, step=step_size)
    [[cos(θ), sin(θ)] for θ in theta_range]
end

function generate_linf_shape()
    delta_range = 0:0.05:1.0
    vcat(
        [[1, -1 + 2δ] for δ in delta_range],
        [[1 - 2δ, -1] for δ in delta_range],
        [[-1, 1 - 2δ] for δ in delta_range],
        [[-1 + 2δ, 1] for δ in delta_range]
    )
end

"""
    plot_boundary(boundary_points, p, k; linestyle=:solid, color=:blue)

    Plots the boundary for a cluster in a specified p-norm.

    # Arguments
    - `boundary_points`: A matrix or array where each column represents a point on the boundary.
    - `p`: The p-norm used to define the boundary.
    - `k`: The cluster index.
    - `linestyle`: (Optional) The style of the boundary line (default: `:solid`).
    - `color`: (Optional) The color of the boundary line (default: `:blue`).
"""
function plot_boundary(boundary_points, p, k; linestyle=:solid, color=:blue)
    # Ensure boundary_points is a 2D array
    if size(boundary_points, 1) != 2
        error("plot_boundary is only designed for 2D boundary points.")
    end

    # Extract x and y values directly from the matrix
    x_vals, y_vals = boundary_points[1, :], boundary_points[2, :]

    # Plot boundary with optional styling
    plot!(x_vals, y_vals, label="P = $p, k = $k", linestyle=linestyle, color=color)
end

end # module BendersDecomposition
