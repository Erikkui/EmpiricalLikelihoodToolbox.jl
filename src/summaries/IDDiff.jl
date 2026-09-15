
"""
    IDDiff{B, T, BUF}

Intrinsic-dimension nearest-neighbour ratio ECDF computed on a numerical derivative of the data.

Identical to [`ID`](@ref), except the nearest-neighbour distances are taken between points of the
`diff_order`-th central difference of the series rather than of the raw series. `TargetData` trims
`diff_order` points from each end of the index cache, because central differences fabricate their
boundary columns by padding.

!!! note "Degenerate on constant derivatives"
    A linear ramp has a constant first derivative, which makes every pairwise distance zero and
    every ratio undefined. Use with data whose derivative actually varies.

# Fields
- `bins::B`: One vector of bin edges per requested neighbour. If `nothing`, calculated from the data.
- `nbin::Int`: The number of bins per neighbour.
- `dt_obs::Float64`: Time step between observations, the denominator of the difference.
- `diff_order::Int`: How many times to differentiate. `1` is the first derivative.
- `neighbors::T`: The neighbour indices to use, always stored as a vector.
- `summary_length::Int`: Equal to `length(neighbors) * nbin`.
- `buffer::BUF`: Scratch space, filled in by `TargetData`. `nothing` until then.

# Examples
```julia
stat = IDDiff(10, 1, 1, 1.0)        # 10 bins, 1st/2nd neighbour ratio, first derivative
stat = IDDiff(10, [1, 3], 1, 1.0)   # two ratios, summary_length == 20
```

See also [`ID`](@ref), [`CILDiff`](@ref).
"""
struct IDDiff{B, T, BUF} <: IDSummary
    bins::B
    nbin::Int
    dt_obs::Float64
    diff_order::Int
    neighbors::T
    summary_length::Int
    buffer::BUF
end

function IDDiff( nbin::Int, neighbors::Int, diff_order::Int, dt_obs::Float64 )
    summary_len = length( neighbors )*nbin
    return IDDiff( nothing, nbin, dt_obs, diff_order, [neighbors], summary_len, nothing )
end

function IDDiff( nbin::Int, neighbors::AbstractVector{<:Int}, diff_order::Int, dt_obs::Float64 )
    summary_len = length( neighbors )*nbin
    return IDDiff( nothing, nbin, dt_obs, diff_order, vec(neighbors), summary_len, nothing )
end

function IDDiff( bins::AbstractVector{<:Real}, neighbors::Int, diff_order::Int, dt_obs::Float64 )
    bins_vec = collect( vec(bins) )
    nbin = length( bins_vec )
    return IDDiff( [bins_vec], nbin, dt_obs, diff_order, [neighbors], nbin, nothing )
end

function IDDiff( bins::AbstractVector{<:AbstractVector{<:Real}}, neighbors::AbstractVector{<:Int}, diff_order::Int, dt_obs::Float64 )
    length(bins) == length(neighbors) || throw( ArgumentError(
        "bins and neighbors must have the same length, got $(length(bins)) and $(length(neighbors))" ) )

    bins_vecs = [ collect(vec(b)) for b in bins ]
    nbin = length( bins_vecs[1] )
    all( length(b) == nbin for b in bins_vecs ) || throw( ArgumentError(
        "all bins vectors must have the same length" ) )

    return IDDiff( bins_vecs, nbin, dt_obs, diff_order, vec(neighbors), length(neighbors)*nbin, nothing )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::IDDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    empcdf! = data.options.ecdf_function

    nbin = summary_statistic.nbin
    bins = summary_statistic.bins
    neighbors = summary_statistic.neighbors
    diff_order = summary_statistic.diff_order

    stat_buffer = summary_statistic.buffer
    dist_buffer_xy = stat_buffer.dist_buffer
    dist_buffer_yx = stat_buffer.dist_buffer_aux
    ratio_buffer = stat_buffer.ratio_buffer

    sort_max = maximum( neighbors ) + 1
    n_rows = size( dist_buffer_xy, 1 )

    data_X = @view data.differences[ diff_order ][ :, x_inds ]
    data_Y = @view data.differences[ diff_order ][ :, y_inds ]

    pairwise!( dist_buffer_xy, Euclidean(), data_X, data_Y )
    copyto!( dist_buffer_yx, dist_buffer_xy )

    # Sort each row of the distance buffer and keep only specified neighbors
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
    summary_statistic::IDDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    empcdf! = target.data.options.ecdf_function

    nbin = summary_statistic.nbin
    bins = summary_statistic.bins
    neighbors = summary_statistic.neighbors
    diff_order = summary_statistic.diff_order

    R0_diff = target.data.differences[ diff_order ]
    Rsim_diff = sim_data_all.differences[ diff_order ]

    stat_buffer = summary_statistic.buffer
    dist_buffer_xy = stat_buffer.dist_buffer
    dist_buffer_yx = stat_buffer.dist_buffer_aux
    ratio_buffer = stat_buffer.ratio_buffer

    sort_max = maximum( neighbors ) + 1
    n_rows = size( dist_buffer_xy, 1 )

    data_X = @view R0_diff[ :, x_inds ]
    data_Y = @view Rsim_diff[ :, y_inds ]

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

function get_bin_quantity( summary_statistic::IDDiff, data::DataContainer, inds_X, inds_Y )
    neighbors = summary_statistic.neighbors
    diff_ind = summary_statistic.diff_order
    sort_max = maximum( neighbors ) + 1

    data_X = @view data.differences[ diff_ind ][ :, inds_X ]
    data_Y = @view data.differences[ diff_ind ][ :, inds_Y ]

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

        # Apply transformation u = 1 - 1/ratio, mapping values to [0, 1] instead of [1, Inf)
        # X --> Y
        @views id_ratios[ 1:n_rows, ii ] .= 1.0 .- dists_xy[:, dist_ind] ./ dists_xy[:, dist_ind+1]
        # Y --> X
        @views id_ratios[ n_rows+1:end, ii ] .= 1.0 .- dists_yx[ dist_ind, : ] ./ dists_yx[ dist_ind+1, : ]
    end

    return id_ratios
end

function allocate_buffer( statistic::IDDiff, data::DataContainer )
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

required_diff_order(stat::IDDiff) = stat.diff_order

function generate_stat_name( stat::IDDiff )
    return "IDDiff_k=$(stat.neighbors)_diff_order=$(stat.diff_order)"
end

function finalize_summary( stat::IDDiff, data::DataContainer, buffers::BufferContainer )
    return @set stat.buffer = buffers.summary_buffers[ Symbol( generate_stat_name( stat ) ) ]
end

get_summary_length(stat::IDDiff, data::DataContainer) = stat.summary_length
