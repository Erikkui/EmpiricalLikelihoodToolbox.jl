#------------Standard ECDF
struct StandardECDF{B} <: AbstractECDFSummary
    bins::B
    nbin::Int
    summary_length::Int
end

function StandardECDF( nbin::Int)
    return StandardECDF( nothing, nbin, nbin )
end

function StandardECDF( bins::AbstractVector{<:Real} )
    return StandardECDF( collect(vec(bins)), length(bins), length(bins) )
end

function calculate_summary_statistic!(
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDF,
    y_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins

    data_X = @view data.observations[ :, y_inds ]

    empcdf!( view_out, data_X, nbins, bins )
    return nothing
end

function get_bin_quantity( summary_statistic::StandardECDF, data::DataContainer, inds_X, inds_Y )
    data_X = @view data.observations[ :, inds_X ]
    return vec(data_X)
end

function allocate_buffer( statistic::StandardECDF, len::Int )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end
