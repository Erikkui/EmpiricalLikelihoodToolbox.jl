function calculate_chamfer_result!( result, kvals, dists_A_to_B, dists_B_to_A, n, m )
    @inbounds for ( ii, kk ) in enumerate( kvals )
        sum_AB = 0.0
        for distances in dists_A_to_B
            sum_AB += distances[kk]
        end

        sum_BA = 0.0
        for distances in dists_B_to_A
            sum_BA += distances[kk]
        end

        result[ii] = (sum_AB / n) + (sum_BA / m)
    end
    return nothing
end

function chamfer_distance(
    set_A::AbstractMatrix{<:Real},
    set_B::AbstractMatrix{<:Real},
    kvals
    )

    n = size( set_A, 2 )
    m = size( set_B, 2 )
    k_max = maximum( kvals )

    tree_A = KDTree( set_A )
    tree_B = KDTree( set_B )

    _, dists_A_to_B = knn( tree_B, set_A, k_max, true )
    _, dists_B_to_A = knn( tree_A, set_B, k_max, true )

    result = Vector{Float64}( undef, length(kvals) )

    calculate_chamfer_result!( result, kvals, dists_A_to_B, dists_B_to_A, n, m )

    return result
end

function chamfer_distance!(
    view_out::AbstractVector{<:Real},
    set_A::AbstractMatrix{<:Real},
    set_B::AbstractMatrix{<:Real},
    kvals
    )

    n = size( set_A, 2 )
    m = size( set_B, 2 )
    k_max = maximum( kvals )

    tree_A = KDTree( set_A )
    tree_B = KDTree( set_B )

    _, dists_A_to_B = knn( tree_B, set_A, k_max, true )
    _, dists_B_to_A = knn( tree_A, set_B, k_max, true )

    calculate_chamfer_result!( view_out, kvals, dists_A_to_B, dists_B_to_A, n, m )

    return nothing
end



function chamfer_distance(
    set_A::AbstractMatrix{<:Real},
    set_B::AbstractMatrix{<:Real},
    tree_B::KDTree,
    kvals
    )

    n = size( set_A, 2 )
    m = size( set_B, 2 )

    tree_A = KDTree( set_A, tree_B.metric )

    k_max = maximum( kvals )

    _, dists_A_to_B = knn( tree_B, set_A, k_max, true )
    _, dists_B_to_A = knn( tree_A, set_B, k_max, true )

    result = Vector{Float64}( undef, length(kvals) )

    calculate_chamfer_result!( result, kvals, dists_A_to_B, dists_B_to_A, n, m )

    return result
end


function chamfer_distance!(
    view_out::AbstractVector{<:Real},
    set_A::AbstractMatrix{<:Real},
    set_B::AbstractMatrix{<:Real},
    tree_B::KDTree,
    kvals
    )
    n = size( set_A, 2 )
    m = size( set_B, 2 )

    tree_A = KDTree( set_A, tree_B.metric )

    k_max = maximum( kvals )

    # knn still allocates the output vectors, which is unavoidable with NearestNeighbors.jl
    _, dists_A_to_B = knn( tree_B, set_A, k_max, true )
    _, dists_B_to_A = knn(  tree_A, set_B, k_max, true )


    calculate_chamfer_result!( view_out, kvals, dists_A_to_B, dists_B_to_A, n, m )

    return nothing
end
