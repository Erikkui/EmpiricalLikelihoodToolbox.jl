#------------Standard ECDF from differences
"""
    StandardECDFDiff{B}

Empirical CDF of a numerical derivative of the data, rather than of the raw series.

The derivatives are central differences, `(x[j+1] - x[j-1]) / (2*dt_obs)`, applied `diff_order`
times. `TargetData` computes them once and shares them with every summary that asks for the same
order. Because the boundary columns of a central difference are fabricated by padding, `TargetData`
also trims `diff_order` points from each end of the index cache.

# Fields
- `bins::B`: The bin edges for the ECDF. If `nothing`, the bins are calculated from the data.
- `nbin::Int`: The number of bins for the ECDF. Mandatory if `bins` is `nothing`.
- `dt_obs::Float64`: Time step between observations, the denominator of the difference.
- `diff_order::Int`: How many times to differentiate. `1` is the first derivative.
- `summary_length::Int`: The length of the summary statistic vector, equal to `nbin`.

# Examples
```julia
stat = StandardECDFDiff(10, 1, 1.0)   # 10 bins, first derivative, unit time step
stat = StandardECDFDiff(10, 2, 0.5)   # second derivative of data sampled every 0.5 time units
```

!!! warning "Supplied bin edges are currently discarded"
    `TargetData` recomputes bin edges from the data for every eCDF-based summary and overwrites
    whatever was passed to the constructor, so the `bins` constructor has no effect on a full
    pipeline run. It does take effect when the summary is evaluated directly.

See also [`StandardECDF`](@ref), [`CILDiff`](@ref).
"""
struct StandardECDFDiff{B} <: StandardECDFSummary
    bins::B
    nbin::Int
    dt_obs::Float64
    diff_order::Int
    summary_length::Int
end

function StandardECDFDiff( nbin::Int, diff_order::Int, dt_obs::Float64 )
    return StandardECDFDiff( nothing, nbin, dt_obs, diff_order, nbin )
end

function StandardECDFDiff( bins::AbstractVector{<:Real}, diff_order::Int, dt_obs::Float64 )
    bins_vec = collect( vec(bins) )
    return StandardECDFDiff( [bins_vec], length(bins_vec), dt_obs, diff_order, length(bins_vec) )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDFDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    empcdf! = data.options.ecdf_function

    diff_order = summary_statistic.diff_order

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins[1]

    data_X = @view data.differences[ diff_order ][ :, x_inds ]

    empcdf!( view_out, data_X, nbins, bins )
    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDFDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    empcdf! = target.data.options.ecdf_function

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins[1]
    diff_order = summary_statistic.diff_order

    Rsim_diff = sim_data_all.differences[ diff_order ]

    data_X = @view Rsim_diff[ :, x_inds ]
    empcdf!( view_out, data_X, nbins, bins )

    return nothing
end

function get_bin_quantity( summary_statistic::StandardECDFDiff, data::DataContainer, inds_X, inds_Y )
    diff_ind = summary_statistic.diff_order
    data_X = data.differences[ diff_ind ]
    data_X =  data_X[ :, inds_X ]
    return vec(data_X)
end

function allocate_buffer( statistic::StandardECDFDiff, data::DataContainer )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end

required_diff_order(stat::StandardECDFDiff) = stat.diff_order

function generate_stat_name( stat::StandardECDFDiff )
    return "StandardECDFDiff_k=$(stat.nbin)_diff_order=$(stat.diff_order)"
end

get_summary_length(stat::StandardECDFDiff, data::DataContainer) = stat.summary_length

function finalize_summary( stat::StandardECDFDiff, data::DataContainer, buffers::BufferContainer )
    return stat
end
