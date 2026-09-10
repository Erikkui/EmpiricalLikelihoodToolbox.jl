#------------CIL of differences
struct CILDiff{B} <: CILSummary
    bins::B
    nbin::Int
    dt_obs::Float64
    diff_order::Int
    summary_length::Int
end

function CILDiff( nbin::Int, diff_order::Int, dt_obs::Float64 )
    return CILDiff( nothing, nbin, dt_obs, diff_order, nbin )
end

function CILDiff( bins::AbstractVector{<:Real}, diff_order::Int, dt_obs::Float64 )
    bins_vec = collect( vec(bins) )
    return CILDiff( [bins_vec], length(bins_vec), dt_obs, diff_order, length(bins_vec) )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary::CILDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    empcdf! = data.options.ecdf_function

    nbins = summary.nbin
    bins = summary.bins[1]
    diff_order = summary.diff_order

    key = Symbol( generate_stat_name( summary ) )
    buffer = buffers.summary_buffers[ key ]

    data_X = @view data.differences[ diff_order ][ :, x_inds ]
    data_Y = @view data.differences[ diff_order ][ :, y_inds ]

    pairwise!( buffer, Euclidean(), data_X, data_Y )

    empcdf!(view_out, vec( buffer ), nbins, bins)
    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary::CILDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    empcdf! = target.data.options.ecdf_function

    nbins = summary.nbin
    bins = summary.bins[1]
    diff_order = summary.diff_order

    R0_diff = target.data.differences[ diff_order ]
    Rsim_diff = sim_data_all.differences[ diff_order ]

    key = Symbol( generate_stat_name( summary ) )
    buffer = buffers.summary_buffers[ key ]

    data_X = @view R0_diff[ :, x_inds ]
    data_Y = @view Rsim_diff[ :, y_inds ]

    pairwise!( buffer, Euclidean(), data_X, data_Y )

    empcdf!(view_out, vec( buffer ), nbins, bins)
    return nothing
end


function get_bin_quantity( summary::CILDiff, data::DataContainer, inds_X, inds_Y )
    diff_ind = summary.diff_order
    data_X = @view data.differences[ diff_ind ][ :, inds_X ]
    data_Y = @view data.differences[ diff_ind ][ :, inds_Y ]
    distances = pairwise( Euclidean(), data_X, data_Y )
    distances = reshape( distances, :, 1 )
    return distances
end

function allocate_buffer( statistic::CILDiff, data::DataContainer )
    ndata = size(data.observations, 2)
    rows, cols = resample_sizes( data.options.resampling_type, ndata )

    buffer = Matrix{Float64}( undef, rows, cols )
    return buffer
end

required_diff_order(stat::CILDiff) = stat.diff_order

function generate_stat_name( stat::CILDiff )
    return "CILDiff_diff_order=$(stat.diff_order)_nbin=$(stat.nbin)"
end

get_summary_length(stat::CILDiff, data::DataContainer) = stat.summary_length

function finalize_summary( stat::CILDiff, data::DataContainer, buffers::BufferContainer )
    return stat
end
