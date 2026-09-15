
"""
    CILDiff{B, BUF}

Correlation Integral Likelihood computed on a numerical derivative of the data.

Identical to [`CIL`](@ref), except the pairwise distances are taken between points of the
`diff_order`-th central difference of the series rather than of the raw series. Pairing `CIL` with
`CILDiff` in a [`JointSummaryStatistics`](@ref) is a way to capture both the amplitude and
the local slope structure of a time series.

Because central differences fabricate their boundary columns by padding, `TargetData` trims
`diff_order` points from each end of the index cache.

# Fields
- `bins::B`: The bin edges (radii) for the ECDF. If `nothing`, calculated from the data.
- `nbin::Int`: The number of bins for the ECDF. Mandatory if `bins` is `nothing`.
- `dt_obs::Float64`: Time step between observations, the denominator of the difference.
- `diff_order::Int`: How many times to differentiate. `1` is the first derivative.
- `summary_length::Int`: The length of the summary statistic vector, equal to `nbin`.
- `buffer::BUF`: Scratch space, filled in by `TargetData`. `nothing` until then.

# Examples
```julia
stats = JointSummaryStatistics(CIL(10), CILDiff(10, 1, 1.0))
```

See also [`CIL`](@ref), [`StandardECDFDiff`](@ref).
"""
struct CILDiff{B, BUF} <: CILSummary
    bins::B
    nbin::Int
    dt_obs::Float64
    diff_order::Int
    summary_length::Int
    buffer::BUF
end

function CILDiff( nbin::Int, diff_order::Int, dt_obs::Float64 )
    return CILDiff( nothing, nbin, dt_obs, diff_order, nbin, nothing )
end

function CILDiff( bins::AbstractVector{<:Real}, diff_order::Int, dt_obs::Float64 )
    bins_vec = collect( vec(bins) )
    return CILDiff( [bins_vec], length(bins_vec), dt_obs, diff_order, length(bins_vec), nothing )
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

    buffer = summary.buffer

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

    buffer = summary.buffer

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
    ndata = data.options.effective_N_obs
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
    return @set stat.buffer = buffers.summary_buffers[ Symbol( generate_stat_name( stat ) ) ]
end
