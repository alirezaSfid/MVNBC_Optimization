module MVPlots

using Plots

export plot_MVCE, plot_outliers

const THETA_STEP = 0.05
const THETA_MAX = 2pi + THETA_STEP

function plot_MVCE(X, clusters)
    num_clusters = size(clusters.vols, 1)
    p = scatter(X[1, :], X[2, :], label = "Datapoints", markershape = :x, color=:gray, markersize=3)
    d, n = size(X)
    
    for k in 1:num_clusters
        A = sqrt(clusters.transformation_matrices[k] / size(X, 1))
        b = -A * clusters.translation_vectors[k]
        num_steps = Int(round(THETA_MAX / THETA_STEP))
        data = Vector{Tuple{Float64, Float64}}(undef, num_steps)
        
        for (i, θ) in enumerate(0:THETA_STEP:THETA_MAX)
            data[i] = tuple(A \ [cos(θ) - b[1], sin(θ) - b[2]]...)
        end
        
        plot!(p, data; linewidth=2, color=k, label = "cluster $k-MVE", legend=:bottomright, aspect_ratio=:equal)
    end
    
    return p
end

function plot_outliers(X, outliers; markershape=:circle, markersize=4, label="Outliers")
    scatter!(X[1, outliers], X[2, outliers], color=:yellow, markershape=:o, markersize=markersize, label=label, legend=:bottomright, aspect_ratio=:equal)
end


end