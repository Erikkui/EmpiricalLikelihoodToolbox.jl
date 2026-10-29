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


# Allocation-free Chamfer distance. The functions above build two new k-d trees and allocate the
# knn output on every call; these reuse one workspace instead and give bit-identical results: the
# same k-th neighbour distances, summed over the points in the same order.

# Reusable storage: the two point sets as SVectors, their k-d trees (rebuilt with KDTree!, which
# reuses the old tree's storage but returns a new tree object, hence the mutable struct), the knn!
# output and the per-k distance sums. The point dimension D is a type parameter, so every call on
# a workspace is type-stable.
mutable struct ChamferWorkspace{D, TR<:KDTree}
    points_A::Vector{SVector{D, Float64}}
    points_B::Vector{SVector{D, Float64}}
    tree_A::TR
    tree_B::TR
    knn_idxs::Vector{Int}
    knn_dists::Vector{Float64}
    sums_AB::Vector{Float64}
    sums_BA::Vector{Float64}
end

# `max_points` is the largest point set expected (eg. effective_N_obs); larger sets still work but
# allocate when the point vectors grow.
function ChamferWorkspace( dimension::Int, max_points::Int, kvals )
    return _chamfer_workspace( Val( dimension ), max_points, kvals )
end

function _chamfer_workspace( ::Val{D}, max_points::Int, kvals ) where {D}
    # Single-point placeholder trees, only there to fix the tree type and hand over their storage
    placeholder_points = [ zero( SVector{D, Float64} ) ]
    tree_A = KDTree( placeholder_points )
    tree_B = KDTree( placeholder_points )

    points_A = Vector{SVector{D, Float64}}( undef, max_points )
    points_B = Vector{SVector{D, Float64}}( undef, max_points )

    k_max = maximum( kvals )
    n_kvals = length( kvals )
    knn_idxs = Vector{Int}( undef, k_max )
    knn_dists = Vector{Float64}( undef, k_max )

    return ChamferWorkspace( points_A, points_B, tree_A, tree_B, knn_idxs, knn_dists, zeros( n_kvals ), zeros( n_kvals ) )
end

# Column `col` of `data` as a point of the workspace's dimension
@inline function _column_point( ::ChamferWorkspace{D}, data::AbstractMatrix, col::Integer ) where {D}
    return SVector{D, Float64}( ntuple( row -> data[ row, col ], Val( D ) ) )
end

# Copy the columns of `set` into `points`, resizing within the existing capacity
function _copy_points!( workspace::ChamferWorkspace, points, set::AbstractMatrix )
    n_points = size( set, 2 )
    resize!( points, n_points )
    for col in 1:n_points
        points[ col ] = _column_point( workspace, set, col )
    end
    return points
end

# sums[ii] = sum over `points`, in order, of the distance to the kvals[ii]-th nearest neighbour in `tree`
function _add_knn_distances!( sums, workspace::ChamferWorkspace, tree::KDTree, points, kvals )
    knn_idxs = workspace.knn_idxs
    knn_dists = workspace.knn_dists
    k_max = length( knn_idxs )

    fill!( sums, 0.0 )
    for point in points
        knn!( knn_idxs, knn_dists, tree, point, k_max, true )
        for ( ii, kk ) in enumerate( kvals )
            sums[ ii ] += knn_dists[ kk ]
        end
    end
    return sums
end

function _combine_chamfer_sums!( view_out, workspace::ChamferWorkspace, n::Int, m::Int )
    for ii in eachindex( workspace.sums_AB )
        view_out[ ii ] = ( workspace.sums_AB[ ii ] / n ) + ( workspace.sums_BA[ ii ] / m )
    end
    return nothing
end

# Same as chamfer_distance!( view_out, set_A, set_B, kvals ), with both trees rebuilt in `workspace`
function chamfer_distance!(
    view_out::AbstractVector{<:Real},
    workspace::ChamferWorkspace,
    set_A::AbstractMatrix{<:Real},
    set_B::AbstractMatrix{<:Real},
    kvals
    )

    points_A = _copy_points!( workspace, workspace.points_A, set_A )
    points_B = _copy_points!( workspace, workspace.points_B, set_B )
    workspace.tree_A = KDTree!( workspace.tree_A, points_A )
    workspace.tree_B = KDTree!( workspace.tree_B, points_B )

    _add_knn_distances!( workspace.sums_AB, workspace, workspace.tree_B, points_A, kvals )
    _add_knn_distances!( workspace.sums_BA, workspace, workspace.tree_A, points_B, kvals )
    _combine_chamfer_sums!( view_out, workspace, length( points_A ), length( points_B ) )

    return nothing
end

# For ChamferECDF, where set_B is fixed over many resamples of set_A: load set_B into the workspace
# (points_B and tree_B) once
function _load_set_B!( workspace::ChamferWorkspace, set_B::AbstractMatrix{<:Real} )
    points_B = _copy_points!( workspace, workspace.points_B, set_B )
    workspace.tree_B = KDTree!( workspace.tree_B, points_B )
    return workspace
end

# With set_B loaded, the A->B distances do not depend on the resample of set_A: table[ii, col]
# is the distance from column `col` of `data_A` to its kvals[ii]-th nearest neighbour in set_B, for
# every column in `columns`.
function _knn_distance_table!( table, workspace::ChamferWorkspace, data_A::AbstractMatrix, columns, kvals )
    knn_idxs = workspace.knn_idxs
    knn_dists = workspace.knn_dists
    k_max = length( knn_idxs )

    for col in columns
        point = _column_point( workspace, data_A, col )
        knn!( knn_idxs, knn_dists, workspace.tree_B, point, k_max, true )
        for ( ii, kk ) in enumerate( kvals )
            table[ ii, col ] = knn_dists[ kk ]
        end
    end
    return table
end

# Chamfer distance between set_A = data_A[:, x_inds] and the set_B loaded in the workspace, with the
# A->B part summed from `table` (see _knn_distance_table!). Same value as chamfer_distance! on the
# two sets.
function _chamfer_distance_from_table!( view_out, workspace::ChamferWorkspace, table, data_A::AbstractMatrix,
                                        x_inds, kvals )
    sums_AB = workspace.sums_AB
    fill!( sums_AB, 0.0 )
    for col in x_inds
        for ii in eachindex( kvals )
            sums_AB[ ii ] += table[ ii, col ]
        end
    end

    set_A = @view data_A[ :, x_inds ]
    points_A = _copy_points!( workspace, workspace.points_A, set_A )
    workspace.tree_A = KDTree!( workspace.tree_A, points_A )
    _add_knn_distances!( workspace.sums_BA, workspace, workspace.tree_A, workspace.points_B, kvals )
    _combine_chamfer_sums!( view_out, workspace, length( points_A ), length( workspace.points_B ) )

    return nothing
end
