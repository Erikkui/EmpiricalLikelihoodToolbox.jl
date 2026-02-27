function chamfer_distance(
    set_A::AbstractMatrix{<:Real},
    set_B::AbstractMatrix{<:Real};
    k::Int=1
    )

    tree_A = KDTree(set_A)
    tree_B = KDTree(set_B)

    _, dists_A_to_B = knn(tree_B, set_A, k, true)
    _, dists_B_to_A = knn(tree_A, set_B, k, true)

    chamfer_AB = mean(dists_A_to_B)
    chamfer_BA = mean(dists_B_to_A)

    return chamfer_AB .+ chamfer_BA
end

function chamfer_distance!(
    view_out::AbstractVector{<:Real},
    set_A::AbstractMatrix{<:Real},
    set_B::AbstractMatrix{<:Real};
    k::Int=1
    )

    tree_A = KDTree(set_A)
    tree_B = KDTree(set_B)

    _, dists_A_to_B = knn(tree_B, set_A, k, true)
    _, dists_B_to_A = knn(tree_A, set_B, k, true)

    # mean!(view_out, dists_A_to_B)
    view_out .= mean(dists_A_to_B) .+ mean(dists_B_to_A)

    return nothing
end



function chamfer_distance(
    set_A::AbstractMatrix{<:Real},
    set_B::AbstractMatrix{<:Real},
    tree_B::KDTree;
    k::Int=1
    )

    metric = tree_B.metric
    tree_A = KDTree(set_A, metric)

    _, dists_A_to_B = knn(tree_B, set_A, k, true)
    _, dists_B_to_A = knn(tree_A, set_B, k, true)

    chamfer_AB = mean(dists_A_to_B)
    chamfer_BA = mean(dists_B_to_A)

    return chamfer_AB .+ chamfer_BA
end

function chamfer_distance!(
    view_out::AbstractVector{<:Real},
    set_A::AbstractMatrix{<:Real},
    set_B::AbstractMatrix{<:Real},
    tree_B::KDTree;
    k::Int=1
    )

    metric = tree_B.metric
    tree_A = KDTree(set_A, metric)

    _, dists_A_to_B = knn(tree_B, set_A, k, true)
    _, dists_B_to_A = knn(tree_A, set_B, k, true)

    view_out .= mean(dists_A_to_B) .+ mean(dists_B_to_A)

    return nothing
end
