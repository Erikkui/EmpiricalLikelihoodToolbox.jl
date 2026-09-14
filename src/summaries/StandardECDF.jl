"""
    StandardECDF{B}

Standard empirical cumulative distribution function summary statistic, with a fixed number of bins.

# Fields
- `bins::B`: The bin edges for the ECDF. If `nothing`, the bins will be automatically calculated from the data.
- `nbin::Int`: The number of bins for the ECDF. Mandatory if `bins` is `nothing`.
- `summary_length::Int`: The length of the summary statistic vector, equal to `nbin`.

# Examples
```julia
stat = StandardECDF(10)  # Create a StandardECDF with 10 bins
stat = StandardECDF([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])  # Create a StandardECDF with specified bin edges
```

!!! warning "Supplied bin edges are currently discarded"
    `TargetData` recomputes bin edges from the data for every eCDF-based summary and overwrites
    whatever was passed to the constructor, so the `bins` constructor has no effect on a full
    pipeline run. It does take effect when the summary is evaluated directly. Prefer the `nbin`
    constructor until this is fixed.

See also [`StandardECDFDiff`](@ref), [`StandardECDFMultiDimensional`](@ref), [`CIL`](@ref).
"""
struct StandardECDF{B} <: StandardECDFSummary
    bins::B
    nbin::Int
    summary_length::Int
end

function StandardECDF( nbin::Int)
    return StandardECDF( nothing, nbin, nbin )
end

function StandardECDF( bins::AbstractVector{<:Real} )
    bins_vec = collect( vec(bins) )
    return StandardECDF( [bins_vec], length(bins_vec), length(bins_vec) )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDF,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    empcdf! = data.options.ecdf_function

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins[1]

    data_X = @view data.observations[ :, x_inds ]

    empcdf!( view_out, data_X, nbins, bins )

    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDF,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    empcdf! = target.data.options.ecdf_function

    Rsim = sim_data_all.observations

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins[1]

    data_X = @view Rsim[ :, x_inds ]
    empcdf!( view_out, data_X, nbins, bins )

    return nothing
end

function get_bin_quantity( summary_statistic::StandardECDF, data::DataContainer, inds_X, inds_Y )
    data_X = @view data.observations[ :, inds_X ]
    return vec( data_X )
end

function allocate_buffer( statistic::StandardECDF, data::DataContainer )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end

required_diff_order(stat::StandardECDF) = 0

function generate_stat_name( stat::StandardECDF )
    return "StandardECDF_k=$(stat.nbin)"
end

function finalize_summary( stat::StandardECDF, data::DataContainer, buffers::BufferContainer )
    return stat
end

get_summary_length(stat::StandardECDF, data::DataContainer) = stat.summary_length
