#------------Standard ECDF from differences
"""
    StandardECDFDiff{B}

Empirical CDF of a numerical derivative of the data, rather than of the raw series.

The derivatives are central differences, `(x[j+1] - x[j-1]) / (2*dt_obs)`, applied `diff_order`
times, where `dt_obs` is the observation time step of the model passed to `TargetData`. The
summary itself carries no time step. `TargetData` computes the derivatives once and shares them with
every summary that asks for the same order. Because the boundary columns of a central difference
are fabricated by padding, `TargetData` also trims `diff_order` points from each end of the index
cache.

# Fields
- `bins::B`: The bin edges for the ECDF. If `nothing`, the bins are calculated from the differences of the data
  (the same series the ECDF is evaluated on), not from the raw data.
- `nbin::Int`: The number of bins for the ECDF. Mandatory if `bins` is `nothing`.
- `diff_order::Int`: How many times to differentiate. `1` is the first derivative.
- `summary_length::Int`: The length of the summary statistic vector, equal to `nbin`.
- `name::String`: The statistic's name from `generate_stat_name`, set by the constructors.

# Examples
```julia
stat = StandardECDFDiff(10, 1)   # 10 bins, first derivative
stat = StandardECDFDiff(10, 2)   # 10 bins, second derivative
```

See also [`StandardECDF`](@ref), [`CILDiff`](@ref).
"""
struct StandardECDFDiff{B} <: StandardECDFSummary
    bins::B
    nbin::Int
    diff_order::Int
    summary_length::Int
    name::String
end

function StandardECDFDiff( nbin::Int, diff_order::Int )
    return with_name( StandardECDFDiff( nothing, nbin, diff_order, nbin, "" ) )
end

function StandardECDFDiff( bins::AbstractVector{<:Real}, diff_order::Int )
    bins_vec = collect( vec(bins) )
    return with_name( StandardECDFDiff( [bins_vec], length(bins_vec), diff_order, length(bins_vec), "" ) )
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
    return "ECDFDiff=$(stat.diff_order)_nbin=$(stat.nbin)"
end

get_summary_length(stat::StandardECDFDiff, data::DataContainer) = stat.summary_length

function finalize_summary( stat::StandardECDFDiff, data::DataContainer, buffers::BufferContainer )
    return stat
end
