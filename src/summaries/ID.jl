struct ID{B, T} <: IDSummary
    bins::B
    nbin::Int
    neighbors::T
    summary_length::Int
end

function ID( nbin::Int, neighbors::Int )
    summary_len = length( neighbors )*nbin
    return ID( nothing, nbin, [neighbors], summary_len )
end

function ID( nbin::Int, neighbors::AbstractVector{<:Int} )
    return ID( nothing, nbin, vec(neighbors), length(neighbors)*nbin )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::ID,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    nbin = summary_statistic.nbin
    bins = summary_statistic.bins
    neighbors = summary_statistic.neighbors

    key = Symbol( generate_stat_name( summary_statistic ) )
    dist_buffer_xy = buffers.summary_buffers[ key ].dist_buffer
    dist_buffer_yx = buffers.summary_buffers[ key ].dist_buffer_aux
    ratio_buffer = buffers.summary_buffers[ key ].ratio_buffer

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

        # X --> Y
        @views ratio_buffer[ 1:n_rows ] .= dist_buffer_xy[:, dist_ind+1] ./ dist_buffer_xy[:, dist_ind]
        # Y --> X
        @views ratio_buffer[ n_rows+1:end ] .= dist_buffer_yx[ dist_ind+1, :] ./ dist_buffer_yx[ dist_ind, : ]

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
    obs_data_all::DataContainer,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    nbin = summary_statistic.nbin
    bins = summary_statistic.bins
    neighbors = summary_statistic.neighbors

    R0 = obs_data_all.observations
    Rsim = sim_data_all.observations

    key = Symbol( generate_stat_name( summary_statistic ) )
    dist_buffer_xy = buffers.summary_buffers[ key ].dist_buffer
    dist_buffer_yx = buffers.summary_buffers[ key ].dist_buffer_aux
    ratio_buffer = buffers.summary_buffers[ key ].ratio_buffer

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

        # X --> Y
        @views ratio_buffer[ 1:n_rows ] .= dist_buffer_xy[:, dist_ind+1] ./ dist_buffer_xy[:, dist_ind]
        # Y --> X
        @views ratio_buffer[ n_rows+1:end ] .= dist_buffer_yx[ dist_ind+1, :] ./ dist_buffer_yx[ dist_ind, : ]

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

        @views id_ratios[ 1:n_rows, ii ] .= dists_xy[:, dist_ind+1] ./ dists_xy[:, dist_ind]
        @views id_ratios[ n_rows+1:end, ii ] .= dists_yx[ dist_ind+1, :] ./ dists_yx[ dist_ind, : ]
    end

    return id_ratios
end

function allocate_buffer( statistic::ID, data::DataContainer )
    if data.options.resampling_type isa TimeseriesResampling
        rows = data.options.resampling_type.timeseries_block_size
        cols = size( data.observations, 2 ) - rows
    else
        rows = round( Int, size( data.observations, 2 ) / 2 )
        cols = rows
    end

    dist_buffer = Matrix{Float64}( undef, rows, cols )
    ratio_buffer = Vector{Float64}( undef, rows+cols )
    return (
        dist_buffer=dist_buffer,
        dist_buffer_aux=similar(dist_buffer),
        ratio_buffer=ratio_buffer
        )
end

required_diff_order(stat::ID) = 0

function generate_stat_name( stat::ID )
    return "ID_neighbors=$(stat.neighbors)"
end
