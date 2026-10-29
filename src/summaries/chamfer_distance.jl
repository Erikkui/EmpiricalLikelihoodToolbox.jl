# Chamfer distance between point sets A (n points) and B (m points), for each neighbour index k in
# kvals:
#
#     d_k(A, B) = (1/n) Σ_{a ∈ A} r_k(a, B)  +  (1/m) Σ_{b ∈ B} r_k(b, A)
#
# where r_k(p, S) is the distance from p to its k-th nearest neighbour in S. Points are the columns
# of the data matrices.
#
# Implementation: r_k comes from k-d trees (NearestNeighbors.jl). A ChamferWorkspace keeps the point
# sets, both trees and the knn output between calls, so the MCMC hot loop allocates only one small
# object per tree rebuild. Each sum runs over the points in column order, starting from 0.0.
#
# ChamferECDF evaluates d_k for many resamples A of the observed data against one fixed simulated
# set B. The first term then only picks which r_k(a, B) to add, so those are looked up once per
# observed point and stored in a table (_knn_distance_table!); per resample only the tree of A is
# rebuilt for the second term (_chamfer_distance_from_table!).

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

# sums[ii] = Σ_{p ∈ points} r_kk(p, tree's set) with kk = kvals[ii], summed in the order of `points`
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

# d_k(set_A, set_B) for each k in kvals, written into view_out. Both trees are rebuilt in `workspace`.
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

# Allocating version with a one-off workspace, for code outside the hot loop (eg. bin calculation)
function chamfer_distance(
    set_A::AbstractMatrix{<:Real},
    set_B::AbstractMatrix{<:Real},
    kvals
    )

    dimension = size( set_A, 1 )
    max_points = max( size( set_A, 2 ), size( set_B, 2 ) )
    workspace = ChamferWorkspace( dimension, max_points, kvals )

    result = Vector{Float64}( undef, length( kvals ) )
    chamfer_distance!( result, workspace, set_A, set_B, kvals )

    return result
end

# ChamferECDF: load the fixed set B into the workspace (points_B and tree_B) once
function _load_set_B!( workspace::ChamferWorkspace, set_B::AbstractMatrix{<:Real} )
    points_B = _copy_points!( workspace, workspace.points_B, set_B )
    workspace.tree_B = KDTree!( workspace.tree_B, points_B )
    return workspace
end

# ChamferECDF: with set B loaded, table[ii, col] = r_kk(data_A[:, col], B) with kk = kvals[ii], for
# every column in `columns`
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

# ChamferECDF: d_k(data_A[:, x_inds], B) with the loaded set B. The A->B sums are read from `table`
# in x_inds order, so the value equals chamfer_distance! on the two sets.
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
