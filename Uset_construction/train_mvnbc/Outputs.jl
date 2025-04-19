module Outputs

export printStatus

function printClusterVolumes(clusters, label)
    K = size(clusters.labels, 2)
    println("$label")
    for j = 1:K
        println("Volume of cluster_$j = $(clusters.vols[j])")
    end
    println("Total_volume = $(sum(clusters.vols))")
    println("=============================")
end

function printStatus(clusters, bestClusters, n_candidatesInit, iter)
    println("***Number of popout points: $n_candidatesInit")
    println("iter=$iter")
    printClusterVolumes(clusters, "Current solution:")
    printClusterVolumes(bestClusters, "Best solution:")
end

end