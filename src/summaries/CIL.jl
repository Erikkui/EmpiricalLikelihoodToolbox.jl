
struct CIL{B, BUF} <: CILSummary
    bins::B
    nbin::Int
    summary_length::Int
    buffer::BUF
end

function CIL( nbin::Int)
    return CIL( nothing, nbin, nbin, nothing )
end

function CIL( bins::AbstractVector{<:Real} )
    bins_vec = collect( vec(bins) )
    return CIL( [bins_vec], length(bins_vec), length(bins_vec), nothing )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary::CIL,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    empcdf! = data.options.ecdf_function

    nbins = summary.nbin
    bins = summary.bins[1]

    buffer = summary.buffer

    data_X = @view data.observations[ :, x_inds ]
    data_Y = @view data.observations[ :, y_inds ]

    pairwise!( buffer, Euclidean(), data_X, data_Y )

    empcdf!( view_out, vec( buffer ), nbins, bins )
    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary::CIL,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    empcdf! = target.data.options.ecdf_function

    nbins = summary.nbin
    bins = summary.bins[1]

    R0 = target.data.observations
    Rsim = sim_data_all.observations

    buffer = summary.buffer

    data_X = @view R0[ :, x_inds ]
    data_Y = @view Rsim[ :, y_inds ]

    pairwise!( buffer, Euclidean(), data_X, data_Y )

    empcdf!( view_out, vec( buffer ), nbins, bins )
    return nothing
end

function get_bin_quantity( summary_statistic::CIL, data::DataContainer, inds_X, inds_Y )
    data_X = @view data.observations[ :, inds_X ]
    data_Y = @view data.observations[ :, inds_Y ]
    distances = pairwise( Euclidean(), data_X, data_Y )
    distances = reshape( distances, :, 1 )
    return distances
end

function allocate_buffer( statistic::CIL, data::DataContainer )

    ndata = data.options.effective_N_obs
    rows, cols = resample_sizes( data.options.resampling_type, ndata )

    buffer = Matrix{Float64}( undef, rows, cols )
    return buffer
end

required_diff_order(stat::CIL) = 0

function generate_stat_name( stat::CIL )
    return "CIL_nbin=$(stat.nbin)"
end

get_summary_length(stat::CIL, data::DataContainer) = stat.summary_length

function finalize_summary( stat::CIL, data::DataContainer, buffers::BufferContainer )
    return @set stat.buffer = buffers.summary_buffers[ Symbol( generate_stat_name( stat ) ) ]
end
