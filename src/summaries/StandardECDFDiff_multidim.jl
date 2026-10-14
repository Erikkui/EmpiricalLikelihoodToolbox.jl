#------------Standard ECDF
"""
    StandardECDFDiffMultiDimensional{B}

Per-dimension empirical CDF of a numerical derivative, for data with more than one row.

Combines [`StandardECDFMultiDimensional`](@ref) and [`StandardECDFDiff`](@ref): each of the `ndim`
rows is differentiated `diff_order` times and summarized by its own ECDF, and the results are
concatenated into a vector of length `ndim * nbin`.

Constructed through [`StandardECDFDiff`](@ref) with three arguments, which returns a plain
`StandardECDFDiff` when `ndim < 2`.

# Fields
- `bins::B`: One vector of bin edges per dimension. If `nothing`, calculated from each row of the differences.
- `nbin::Int`: The number of bins per dimension.
- `ndim::Int`: The number of data dimensions (rows) summarized.
- `diff_order::Int`: How many times to differentiate. `1` is the first derivative.
- `summary_length::Int`: The length of the summary statistic vector, equal to `ndim * nbin`.
- `name::String`: The statistic's name from `generate_stat_name`, set by the constructors.

# Examples
```julia
stat = StandardECDFDiff(10, 3, 1)   # 3 dimensions, first derivative, summary_length == 30
```

See also [`StandardECDFDiff`](@ref), [`StandardECDFMultiDimensional`](@ref).
"""
struct StandardECDFDiffMultiDimensional{B} <: StandardECDFSummary
    bins::B
    nbin::Int
    ndim::Int
    diff_order::Int
    summary_length::Int
    name::String
end

function StandardECDFDiff( nbin::Int, ndim::Int, diff_order::Int )
    if ndim < 2
        return with_name( StandardECDFDiff( nothing, nbin, diff_order, nbin, "" ) )
    else
        return with_name( StandardECDFDiffMultiDimensional( nothing, nbin, ndim, diff_order, ndim*nbin, "" ) )
    end
end



function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDFDiffMultiDimensional,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    empcdf! = data.options.ecdf_function

    diff_order = summary_statistic.diff_order

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins

    data_X = @view data.differences[ diff_order ][ :, x_inds ]

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
    summary_statistic::StandardECDFDiffMultiDimensional,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    empcdf! = target.data.options.ecdf_function

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins
    diff_order = summary_statistic.diff_order

    Rsim_diff = sim_data_all.differences[ diff_order ]

    start_ind = 1
    data_X = @view Rsim_diff[ :, x_inds ]
    for (ii, row) in enumerate( eachrow(data_X) )
        end_ind = start_ind + summary_statistic.nbin - 1

        ecdf_view_out = @view view_out[ start_ind:end_ind ]
        empcdf!( ecdf_view_out, row, nbins, bins[ii] )

        start_ind += summary_statistic.nbin
    end

    return nothing
end

function get_bin_quantity( summary_statistic::StandardECDFDiffMultiDimensional, data::DataContainer, inds_X, inds_Y )
    diff_ind = summary_statistic.diff_order
    data_X = data.differences[ diff_ind ]
    data_X =  data_X[ :, inds_X ]
    return data_X
end

function allocate_buffer( statistic::StandardECDFDiffMultiDimensional, data::DataContainer )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end

required_diff_order(stat::StandardECDFDiffMultiDimensional) = stat.diff_order

function generate_stat_name( stat::StandardECDFDiffMultiDimensional )
    return "ECDFDiff=$(stat.diff_order)_nbin=$(stat.nbin)_dim=$(stat.ndim)"
end

get_summary_length(stat::StandardECDFDiffMultiDimensional, data::DataContainer) = stat.summary_length

function finalize_summary( stat::StandardECDFDiffMultiDimensional, data::DataContainer, buffers::BufferContainer )
    return stat
end
