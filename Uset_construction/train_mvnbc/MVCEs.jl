# This module provides functions for computing Minimum Volume Covering Ellipsoids (MVCEs)
module MVCEs

# Import necessary packages
using MinimumVolumeEllipsoids
using JuMP
using MosekTools
using LinearAlgebra

# Export the functions and types that should be accessible from outside the module
export MVCE_init, MVCE_weighted, MVCE_labled, MVBCResult

# Define a constant for labeling the points for each cluster
const TOLERANCE = 1e-5

# Define a struct to hold the result of MVBC
struct MVBCResult
    transformation_matrices::Vector{Matrix{Float64}}
    translation_vectors::Vector{Vector{Float64}}
    labels::Matrix{Bool}
    vols::Vector{Float64}
end

# Function to compute the minimum volume ellipsoid for a set of points
function minimum_volume_ell(X)

    try
        ϵ = minimum_volume_ellipsoid(X)

        return ϵ.H, ϵ.x, volume(ϵ)
    catch err
        @warn "An error occurred with the minimum_volume_ellipsoid package: $err. Falling back to the optimization solver..."
        d, n = size(X)
        γ = ones(n) .* d
        H, x_bar, vol = minimum_volume_ell_weighted(X, γ)

        return H, x_bar, vol
    end
    
end

# Function to compute the minimum volume ellipsoid for a set of points with weights
function minimum_volume_ell_weighted(X, γ; r=1.0)
    model = Model(MosekTools.Optimizer)
    # We need to use a tighter tolerance for this example, otherwise the bounding
    # ellipse won't actually be bounding...
    set_optimizer_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1e-7)
    set_silent(model)
    d, n = size(X)
    @variable(model, z[1:d])
    @variable(model, s)
    @variable(model, t)
    # The former @variable(model, Z[1:d, 1:d], PSD)
    @variable(model, Z[1:d, 1:d], Symmetric)
    @constraint(model, Z >= 0, PSDCone())
    @constraint(model, [s z'; z Z] >= 0, PSDCone())
    @constraint(
        model,
        [i in 1:n],
        X[:, i]' * Z * X[:, i] - 2 * X[:, i]' * z + s <= d * r * γ[i],
    )
    @constraint(model, [t; vec(Z)] in MOI.RootDetConeSquare(d))
    @objective(model, Max, 1.0 * t + 0.0)
    optimize!(model)
    
    
    H = value.(Z)
    
    x_bar = H \ value.(z)
    
    return H, x_bar, (d^(d / 2) * MinimumVolumeEllipsoids._unit_ball_volume(d)) / sqrt(det(H))

end


function minimum_volume_ell_labeled(X, L)
    n, K = size(L)
    d = size(X, 1)
    
    Hs = Matrix{Float64}[]
    x_hats = Vector{Float64}[]
    vols = []
    labels = falses(n, K)
    
    for k in 1:K
        H, x_hat, vol = minimum_volume_ell(X[:, L[:, k]])

        for i in 1:n
            if !labels[i, k]
                labels[i, k] = (X[:, i] - x_hat)' * H * (X[:, i] - x_hat) / d <= 1 + TOLERANCE
            end
        end
    
        push!(Hs, H)
        push!(x_hats, x_hat)
        push!(vols, vol)
    end
    
    return MVBCResult(Hs, x_hats, L, vols)

end

end