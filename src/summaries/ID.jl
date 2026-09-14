
"""
    ID{B, T, BUF}

Empirical CDF of intrinsic-dimension nearest-neighbour distance ratios.

For each requested neighbour index `k`, this takes the ratio of the `k`-th to the `(k+1)`-th
nearest-neighbour distance and maps it to `[0, 1)` as `u = 1 - r_k / r_{k+1}`. Ratios are collected
in both directions (x to y and y to x) and pooled, then summarized by an ECDF over `nbin` bins.
The distribution of these ratios is the basis of two-NN style intrinsic dimension estimators, so the
summary is sensitive to the local geometry of the attractor rather than to the marginal
distribution of the data.

Requesting several neighbours concatenates one `nbin`-long ECDF block per neighbour, in the order
given, so `summary_length == length(neighbors) * nbin`.

This is a two-set summary: it consumes the "x" and "y" index sets produced by the configured
resampler (`MethodsOptions.resampling_type`), not the raw series. Which resampler you choose
therefore changes what this statistic measures.

!!! note "Degenerate on constant data"
    If all pairwise distances are equal, every ratio is `0/0` and the statistic is undefined.

# Fields
- `bins::B`: One vector of bin edges per requested neighbour. If `nothing`, calculated from the data.
- `nbin::Int`: The number of bins per neighbour.
- `neighbors::T`: The neighbour indices to use, always stored as a vector.
- `summary_length::Int`: Equal to `length(neighbors) * nbin`.
- `buffer::BUF`: Scratch space, filled in by `TargetData`. `nothing` until then.

# Examples
```julia
stat = ID(10, 1)        # ratio of the 1st to the 2nd nearest neighbour, summary_length == 10
stat = ID(10, [1, 3])   # two ratios concatenated, summary_length == 20
```

!!! warning "Supplied bin edges are currently discarded"
    `TargetData` recomputes bin edges from the data for every eCDF-based summary and overwrites
    whatever was passed to the constructor, so the `bins` constructor has no effect on a full
    pipeline run. It does take effect when the summary is evaluated directly.

See also [`IDDiff`](@ref), [`CIL`](@ref).
"""
struct ID{B, T, BUF} <: IDSummary
    bins::B
    nbin::Int
    neighbors::T
    summary_length::Int
    buffer::BUF
end

function ID( nbin::Int, neighbors::Int )
    summary_len = length( neighbors )*nbin
    return ID( nothing, nbin, [neighbors], summary_len, nothing )
end

function ID( nbin::Int, neighbors::AbstractVector{<:Int} )
    return ID( nothing, nbin, vec(neighbors), length(neighbors)*nbin, nothing )
end

function ID( bins::AbstractVector{<:Real}, neighbors::Int )
    bins_vec = collect( vec(bins) )
    nbin = length( bins_vec )
    return ID( [bins_vec], nbin, [neighbors], nbin, nothing )
end

function ID( bins::AbstractVector{<:AbstractVector{<:Real}}, neighbors::AbstractVector{<:Int} )
    length(bins) == length(neighbors) || throw( ArgumentError(
        "bins and neighbors must have the same length, got $(length(bins)) and $(length(neighbors))" ) )

    bins_vecs = [ collect(vec(b)) for b in bins ]
    nbin = length( bins_vecs[1] )
    all( length(b) == nbin for b in bins_vecs ) || throw( ArgumentError(
        "all bins vectors must have the same length" ) )

    return ID( bins_vecs, nbin, vec(neighbors), length(neighbors)*nbin, nothing )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::ID,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    empcdf! = data.options.ecdf_function

    nbin = summary_statistic.nbin
    bins = summary_statistic.bins
    neighbors = summary_statistic.neighbors

    stat_buffer = summary_statistic.buffer
    dist_buffer_xy = stat_buffer.dist_buffer
    dist_buffer_yx = stat_buffer.dist_buffer_aux
    ratio_buffer = stat_buffer.ratio_buffer

    sort_max = maximum( neighbors ) + 1
    n_rows = size( dist_buffer_xy, 1 )

    data_X = @view data.observations[ :, x_inds ]
    data_Y = @view data.observations[ :, y_inds ]

    pairwise!( dist_buffer_xy, Euclidean(), data_X, data_Y )
    copyto!( dist_buffer_yx, dist_buffer_xy )

    # Sort each row from x->y and each column from y->x of the distance buffers and keep only specified neighbors
    for row in eachrow( dist_buffer_xy )
        partialsort!( row, 1:sort_max )
    end
    for col in eachcol( dist_buffer_yx )
        partialsort!( col, 1:sort_max )
    end

    # Compute the eCDF summary statistic
    for ii in eachindex(neighbors)
        dist_ind = neighbors[ii]

        # Apply transformation u = 1 - 1/ratio, mapping values to [0, 1] instead of [1, Inf)
        # X --> Y
        @views ratio_buffer[ 1:n_rows ] .= 1.0 .-  dist_buffer_xy[:, dist_ind] ./ dist_buffer_xy[:, dist_ind+1]
        # Y --> X
        @views ratio_buffer[ n_rows+1:end ] .= 1.0 .-  dist_buffer_yx[ dist_ind, : ] ./ dist_buffer_yx[ dist_ind+1, :]

        bins_ii = bins[ii]
        cdf_view = @view view_out[ (ii-1)*nbin+1 : ii*nbin ]
        empcdf!( cdf_view, ratio_buffer, nbin, bins_ii )
    end

    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::ID,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    empcdf! = target.data.options.ecdf_function

    nbin = summary_statistic.nbin
    bins = summary_statistic.bins
    neighbors = summary_statistic.neighbors

    R0 = target.data.observations
    Rsim = sim_data_all.observations

    stat_buffer = summary_statistic.buffer
    dist_buffer_xy = stat_buffer.dist_buffer
    dist_buffer_yx = stat_buffer.dist_buffer_aux
    ratio_buffer = stat_buffer.ratio_buffer

    sort_max = maximum( neighbors ) + 1
    n_rows = size( dist_buffer_xy, 1 )

    data_X = @view R0[ :, x_inds ]
    data_Y = @view Rsim[ :, y_inds ]

    pairwise!( dist_buffer_xy, Euclidean(), data_X, data_Y )
    copyto!( dist_buffer_yx, dist_buffer_xy )

    # Sort each row from x->y and each column from y->x of the distance buffers and keep only specified neighbors
    for row in eachrow( dist_buffer_xy )
        partialsort!( row, 1:sort_max )
    end
    for col in eachcol( dist_buffer_yx )
        partialsort!( col, 1:sort_max )
    end

    # Compute the eCDF summary statistic
    for ii in eachindex(neighbors)
        dist_ind = neighbors[ii]

        # Apply transformation u = 1 - 1/ratio, mapping values to [0, 1] instead of [1, Inf)
        # X --> Y
        @views ratio_buffer[ 1:n_rows ] .= 1.0 .-  dist_buffer_xy[:, dist_ind] ./ dist_buffer_xy[:, dist_ind+1]
        # Y --> X
        @views ratio_buffer[ n_rows+1:end ] .= 1.0 .-  dist_buffer_yx[ dist_ind, : ] ./ dist_buffer_yx[ dist_ind+1, :]

        bins_ii = bins[ii]
        cdf_view = @view view_out[ (ii-1)*nbin+1 : ii*nbin ]
        empcdf!( cdf_view, ratio_buffer, nbin, bins_ii )
    end

    return nothing
end

function get_bin_quantity( summary_statistic::ID, data::DataContainer, inds_X, inds_Y )
    neighbors = summary_statistic.neighbors
    sort_max = maximum( neighbors ) + 1

    data_X = @view data.observations[ :, inds_X ]
    data_Y = @view data.observations[ :, inds_Y ]

    dists_xy = pairwise( Euclidean(), data_X, data_Y )
    dists_yx = copy( dists_xy )

    # Sort each row of the distance buffer and keep only specified neighbors
    for row in eachrow( dists_xy )
        partialsort!( row, 1:sort_max )
    end
    for col in eachcol( dists_yx )
        partialsort!( col, 1:sort_max )
    end

    n_rows, n_cols = size( dists_xy )
    id_ratios = Matrix{Float64}( undef, n_rows+n_cols, length(neighbors) )
    for ii in eachindex(neighbors)
        dist_ind = neighbors[ii]

        @views id_ratios[ 1:n_rows, ii ] .= 1.0 .- dists_xy[:, dist_ind] ./ dists_xy[:, dist_ind+1]
        @views id_ratios[ n_rows+1:end, ii ] .= 1.0 .- dists_yx[ dist_ind, : ] ./ dists_yx[ dist_ind+1, :]
    end

    return id_ratios
end

function allocate_buffer( statistic::ID, data::DataContainer )
    ndata = data.options.effective_N_obs
    rows, cols = resample_sizes( data.options.resampling_type, ndata )

    dist_buffer = Matrix{Float64}( undef, rows, cols )
    dist_buffer_aux = similar( dist_buffer )
    ratio_buffer = Vector{Float64}( undef, rows+cols )

    return (
        dist_buffer=dist_buffer,
        dist_buffer_aux=dist_buffer_aux,
        ratio_buffer=ratio_buffer
        )
end

required_diff_order(stat::ID) = 0

function generate_stat_name( stat::ID )
    return "ID_neighbors=$(stat.neighbors)"
end

get_summary_length(stat::ID, data::DataContainer) = stat.summary_length

function finalize_summary( stat::ID, data::DataContainer, buffers::BufferContainer )
    return @set stat.buffer = buffers.summary_buffers[ Symbol( generate_stat_name( stat ) ) ]
end
