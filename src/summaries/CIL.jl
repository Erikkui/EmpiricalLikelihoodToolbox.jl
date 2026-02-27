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

function calculate_summary_statistic!(
    view_out::AbstractVector{Float64},
    summary::CIL,
    y_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    nbins = summary.nbin
    bins = summary.bins

    key = nameof( typeof(summary) )
    buffer = buffers.summary_buffers[ key ]

    data_X = @view data.observations[ :, x_inds ]
    data_Y = @view data.observations[ :, y_inds ]

    pairwise!( buffer, Euclidean(), data_X, data_Y )

    empcdf!( view_out, vec(buffer), nbins, bins )
    return nothing
end

function get_bin_quantity( summary_statistic::CIL, data::DataContainer, inds_X, inds_Y )
    data_X = @view data.observations[ :, inds_X ]
    data_Y = @view data.observations[ :, inds_Y ]
    distances = pairwise( Euclidean(), data_X, data_Y ) |> vec
    return distances
end

function allocate_buffer( statistic::CIL, len::Int )
    buffer = Matrix{Float64}( undef, len, len )
    return buffer
end
