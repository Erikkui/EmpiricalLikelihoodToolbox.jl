#------------Standard ECDF
"""
    StandardECDFMultiDimensional{B}

Per-dimension empirical CDF for data with more than one row (dimension).

One separate ECDF is computed for each of the `ndim` rows of the data, each with its own bin edges,
and the results are concatenated into a single vector of length `ndim * nbin`. This is a marginal
summary: it captures each coordinate separately and not their joint distribution.

Constructed through [`StandardECDF`](@ref) with two arguments. `StandardECDF(nbin, ndim)` returns a
plain `StandardECDF` when `ndim < 2`, so one-dimensional data does not pay for the multidimensional
code path.

# Fields
- `bins::B`: One vector of bin edges per dimension. If `nothing`, calculated from the data.
- `nbin::Int`: The number of bins per dimension.
- `ndim::Int`: The number of data dimensions (rows) summarized.
- `summary_length::Int`: The length of the summary statistic vector, equal to `ndim * nbin`.

# Examples
```julia
stat = StandardECDF(10, 3)   # 3-dimensional data, 10 bins each, summary_length == 30
stat = StandardECDF(10, 1)   # falls back to a plain StandardECDF
```

See also [`StandardECDF`](@ref), [`StandardECDFDiffMultiDimensional`](@ref).
"""
struct StandardECDFMultiDimensional{B} <: StandardECDFSummary
    bins::B
    nbin::Int
    ndim::Int
    summary_length::Int
end

function StandardECDF( nbin::Int, ndim::Int )
    if ndim < 2
        return StandardECDF( nothing, nbin, nbin )
    else
        return StandardECDFMultiDimensional( nothing, nbin, ndim, ndim*nbin )
    end
end



function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDFMultiDimensional,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    empcdf! = data.options.ecdf_function

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins

    data_X = @view data.observations[ :, x_inds ]

    start_ind = 1
    for (ii, row) in enumerate( eachrow(data_X) )
        end_ind = start_ind + summary_statistic.nbin - 1

        ecdf_view_out = @view view_out[ start_ind:end_ind ]
        empcdf!( ecdf_view_out, row, nbins, bins[ii] )

        start_ind += summary_statistic.nbin
    end

    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDFMultiDimensional,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    empcdf! = target.data.options.ecdf_function

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins

    ndata = size(sim_data_all.observations, 2)
    Rsim = sim_data_all.observations

    start_ind = 1

    data_X = @view Rsim[ :, x_inds ]
    for (ii, row) in enumerate( eachrow(data_X) )
        end_ind = start_ind + summary_statistic.nbin - 1

        ecdf_view_out = @view view_out[ start_ind:end_ind ]
        empcdf!( ecdf_view_out, row, nbins, bins[ii] )

        start_ind += summary_statistic.nbin
    end

    return nothing
end

function get_bin_quantity( summary_statistic::StandardECDFMultiDimensional, data::DataContainer, inds_X, inds_Y )
    data_X = @view data.observations[ :, inds_X ]
    return data_X
end

function allocate_buffer( statistic::StandardECDFMultiDimensional, data::DataContainer )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end

required_diff_order(stat::StandardECDFMultiDimensional) = 0

function generate_stat_name( stat::StandardECDFMultiDimensional )
    return "StandardECDF_k=$(stat.nbin)_ndim=$(stat.ndim)"
end

function finalize_summary( stat::StandardECDFMultiDimensional, data::DataContainer, buffers::BufferContainer )
    return stat
end

get_summary_length(stat::StandardECDFMultiDimensional, data::DataContainer) = stat.summary_length
