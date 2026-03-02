#------------Standard ECDF from differences
struct StandardECDFDiff{B} <: DifferenceECDFSummary
    bins::B
    nbin::Int
    dt_obs::Float64
    diff_order::Int
    summary_length::Int
end

function StandardECDFDiff( nbin::Int, diff_order::Int, dt_obs::Float64 )
    return StandardECDFDiff( nothing, nbin, dt_obs, diff_order, nbin )
end

function calculate_summary_statistic!(
    view_out::AbstractVector{Float64}, summary_statistic::StandardECDFDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins
    diff_order = summary_statistic.diff_order

    diff_data_X = @view data.differences[ diff_order ][ :, y_inds ]

    empcdf!( view_out, diff_data_X, nbins, bins )
    return nothing
end

function get_bin_quantity( summary_statistic::StandardECDFDiff, data::DataContainer, inds_X, inds_Y )
    diff_ind = summary_statistic.diff_order
    data_X = data.differences[ diff_ind ]
    # println( "binquant size: ", typeof(data.differences), "    ", diff_ind)
    data_X =  data_X[ :, inds_X ]
    return vec(data_X)
end

function allocate_buffer( statistic::StandardECDFDiff, len::Int )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end

required_diff_order(stat::StandardECDFDiff) = stat.diff_order
