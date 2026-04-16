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
    sort_max = maximum( neighbors ) + 1

    key = nameof( typeof(summary_statistic) )
    dist_buffer = buffers.summary_buffers[ key ].dist_buffer
    ratio_buffer = buffers.summary_buffers[ key ].ratio_buffer

    data_X = @view data.observations[ :, x_inds ]
    data_Y = @view data.observations[ :, y_inds ]

    pairwise!( dist_buffer, Euclidean(), data_X, data_Y )

    # Sort each row of the distance buffer and keep only specified neighbors
    for row in eachrow(dist_buffer)
        partialsort!( row, 1:sort_max )
    end

    # Calculate the ratio of distances to specified neighbors
    for ii in eachindex(neighbors)
        col = neighbors[ii]
        bins_ii = bins[ii]
        @views ratio_buffer .= dist_buffer[:, col+1] ./ dist_buffer[:, col]
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
    sort_max = maximum( neighbors ) + 1

    R0 = obs_data_all.observations
    Rsim = sim_data_all.observations

    key = nameof( typeof(summary_statistic) )
    dist_buffer = buffers.summary_buffers[ key ].dist_buffer
    ratio_buffer = buffers.summary_buffers[ key ].ratio_buffer

    data_X = @view R0[ :, x_inds ]
    data_Y = @view Rsim[ :, y_inds ]

    pairwise!( dist_buffer, Euclidean(), data_X, data_Y )

    # Sort each row of the distance buffer and keep only specified neighbors
    for row in eachrow(dist_buffer)
        partialsort!( row, 1:sort_max )
    end

    # Calculate the ratio of distances to specified neighbors
    for ii in eachindex(neighbors)
        col = neighbors[ii]
        bins_ii = bins[ii]
        @views ratio_buffer .= dist_buffer[:, col+1] ./ dist_buffer[:, col]
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

    dists = pairwise( Euclidean(), data_X, data_Y )

    # Sort each row of the distance buffer and keep only specified neighbors
    for row in eachrow(dists)
        partialsort!( row, 1:sort_max )
    end
    data_len = length( inds_X )
    id_ratios = Matrix{Float64}( undef, data_len, length(neighbors) )
    for ii in eachindex(neighbors)
        col = neighbors[ii]
        @views id_ratios[:, ii] .= dists[:, col+1] ./ dists[:, col]
    end

    return id_ratios
end

function allocate_buffer( statistic::ID, data::DataContainer )
    if data.options.resampling_type isa TimeseriesResampling
        rows = data.options.timeseries_block_size
        cols = size( data.observations, 2 ) - rows
    else
        rows = round( Int, size( data.observations, 2 ) / 2 )
        cols = rows
    end

    dist_buffer = Matrix{Float64}( undef, rows, cols )
    ratio_buffer = Vector{Float64}( undef, rows )
    return ( dist_buffer=dist_buffer, ratio_buffer=ratio_buffer )
end

required_diff_order(stat::ID) = 0
