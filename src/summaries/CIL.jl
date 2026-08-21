#------------CIL
struct CIL{B} <: AbstractECDFSummary
    bins::B
    nbin::Int
    summary_length::Int
end

function CIL( nbin::Int)
    return CIL( nothing, nbin, nbin )
end

function CIL( bins::AbstractVector{<:Real} )
    return CIL( collect(vec(bins)), length(bins), length(bins) )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary::CIL,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    nbins = summary.nbin
    bins = summary.bins

    key = Symbol( generate_stat_name( summary ) )
    buffer = buffers.summary_buffers[ key ]

    data_X = @view data.observations[ :, x_inds ]
    data_Y = @view data.observations[ :, y_inds ]

    pairwise!( buffer, Euclidean(), data_X, data_Y )

    empcdf!( view_out, vec(buffer), nbins, bins )
    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary::CIL,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    obs_data_all::DataContainer,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    nbins = summary.nbin
    bins = summary.bins

    R0 = obs_data_all.observations
    Rsim = sim_data_all.observations

    key = Symbol( generate_stat_name( summary ) )
    buffer = buffers.summary_buffers[ key ]

    data_X = @view R0[ :, x_inds ]
    data_Y = @view Rsim[ :, y_inds ]

    pairwise!( buffer, Euclidean(), data_X, data_Y ) |> vec

    empcdf!(view_out, buffer, nbins, bins)
    return nothing
end

function get_bin_quantity( summary_statistic::CIL, data::DataContainer, inds_X, inds_Y )
    data_X = @view data.observations[ :, inds_X ]
    data_Y = @view data.observations[ :, inds_Y ]
    distances = pairwise( Euclidean(), data_X, data_Y ) |> vec
    return distances
end

function allocate_buffer( statistic::CIL, data::DataContainer )

    ndata = size( data.observations, 2 )
    rows, cols = resample_sizes( data.options.resampling_type, ndata )

    # if data.options.resampling_type isa TimeseriesResampling
    #     rows = data.options.resampling_type.timeseries_block_size
    #     println(data.options.resampling_type)
    #     cols = size( data.observations, 2 ) - rows
    # else
    #     rows = round( Int, size( data.observations, 2 ) / 2 )
    #     cols = rows
    # end

    buffer = Matrix{Float64}( undef, rows, cols )
    return buffer
end

required_diff_order(stat::CIL) = 0

function generate_stat_name( stat::CIL )
    return "CIL_nbin=$(stat.nbin)"
end
