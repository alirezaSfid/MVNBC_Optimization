module HeuristicMethod
include("LocalSearchFunctions.jl") 
include("BendersDecomposition.jl")
include("multi_dimensional_data_generation.jl")

import .BendersDecomposition
import .LocalSearchFunctions
import .MultiDimensionalDataGeneration

const DMD_MAX_ITERATION = 100

function normBasedUncertaintySet_with_Kmeans(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, r::Float64, max_iteration::Int64 = DMD_MAX_ITERATION)

    l = BendersDecomposition.assign_clusters_GMM(X, P, K_p, r)    

    l, tot_vols_vec = LocalSearchFunctions.local_search_ell(X, P, K_p, r, l; max_iteration=max_iteration)


    T, t, vols = LocalSearchFunctions.solve_sub(X, P, K_p, l)

    total_vol = sum(vols[p][k] for p in P for k in 1:K_p[p])

    return l, T, t, total_vol, tot_vols_vec
end

function normBasedUncertaintySet_with_GMM(X::AbstractArray, P::Vector{Float64}, K_p::Dict{Float64, Int}, r::Float64, max_iteration::Int64 = DMD_MAX_ITERATION)

    l = BendersDecomposition.assign_clusters_GMM(X, P, K_p, r)    

    l, tot_vols_vec = LocalSearchFunctions.local_search_ell(X, P, K_p, r, l; max_iteration=max_iteration)


    T, t, vols = LocalSearchFunctions.solve_sub(X, P, K_p, l)

    total_vol = sum(vols[p][k] for p in P for k in 1:K_p[p])

    return l, T, t, total_vol, tot_vols_vec
end

end