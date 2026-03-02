#------------CIL of differences
struct CILDiff{B} <: AbstractECDFSummary
    bins::B
    nbin::Int
    dt_obs::Float64
    diff_order::Int
    summary_length::Int
end

function CILDiff( nbin::Int, diff_order::Int, dt_obs::Float64 )
    return CILDiff( nothing, nbin, dt_obs, diff_order, nbin )
end

function calculate_summary_statistic!(
    view_out::AbstractVector{Float64},
    summary::CILDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    nbins = summary.nbin
    bins = summary.bins
    diff_order = summary.diff_order

    key = nameof( typeof(summary) )
    buffer = buffers.summary_buffers[ key ]

    data_X = @view data.differences[ diff_order ][ :, x_inds ]
    data_Y = @view data.differences[ diff_order ][ :, y_inds ]

    pairwise!( buffer, Euclidean(), data_X, data_Y ) |> vec

    empcdf!(view_out, buffer, nbins, bins)
    return nothing
end

function get_bin_quantity( summary::CILDiff, data::DataContainer, inds_X, inds_Y )
    diff_ind = summary.diff_order
    data_X = @view data.differences[ diff_ind ][ :, inds_X ]
    data_Y = @view data.differences[ diff_ind ][ :, inds_Y ]
    distances = pairwise( Euclidean(), data_X, data_Y ) |> vec
    return distances
end

function allocate_buffer( statistic::CILDiff, len::Int )
    buffer = Matrix{Float64}( undef, len, len )
    return buffer
end

required_diff_order(stat::CILDiff) = stat.diff_order
