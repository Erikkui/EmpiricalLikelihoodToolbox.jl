
"""
    CIL{B, BUF}

Correlation Integral Likelihood summary statistic (Haario et al. 2015).

Computes every pairwise Euclidean distance between the sets "x" and "y" and returns the
empirical CDF of those distances, evaluated at `nbin` bin edges. The correlation integral is the
fraction of point pairs closer than a given radius, so this is that integral sampled at `nbin` radii.

This is a two-set summary: it consumes the "x" and "y" index sets produced by the configured
resampler (`MethodsOptions.resampling_type`), not the raw series. Which resampler you choose
therefore changes what this statistic measures.

Cost is O(N^2) in both time and memory, since the full pairwise distance matrix is materialized.

# Fields
- `bins::B`: The bin edges (radii) for the ECDF. If `nothing`, calculated from the data.
- `nbin::Int`: The number of bins for the ECDF. Mandatory if `bins` is `nothing`.
- `summary_length::Int`: The length of the summary statistic vector, equal to `nbin`.
- `buffer::BUF`: Scratch space, filled in by `TargetData`. `nothing` until then.

# Examples
```julia
stat = CIL(10)   # correlation integral evaluated at 10 radii (bins)
```

!!! warning "Supplied bin edges are currently discarded"
    `TargetData` recomputes bin edges from the data for every eCDF-based summary and overwrites
    whatever was passed to the constructor, so the `bins` constructor has no effect on a full
    pipeline run. It does take effect when the summary is evaluated directly.

See also [`CILDiff`](@ref), [`ID`](@ref).
"""
struct CIL{B, BUF} <: CILSummary
    bins::B
    nbin::Int
    summary_length::Int
    buffer::BUF
end

function CIL( nbin::Int)
    return CIL( nothing, nbin, nbin, nothing )
end

function CIL( bins::AbstractVector{<:Real} )
    bins_vec = collect( vec(bins) )
    return CIL( [bins_vec], length(bins_vec), length(bins_vec), nothing )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary::CIL,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    empcdf! = data.options.ecdf_function

    nbins = summary.nbin
    bins = summary.bins[1]

    buffer = summary.buffer

    data_X = @view data.observations[ :, x_inds ]
    data_Y = @view data.observations[ :, y_inds ]

    pairwise!( buffer, Euclidean(), data_X, data_Y )

    empcdf!( view_out, vec( buffer ), nbins, bins )
    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary::CIL,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    empcdf! = target.data.options.ecdf_function

    nbins = summary.nbin
    bins = summary.bins[1]

    R0 = target.data.observations
    Rsim = sim_data_all.observations

    buffer = summary.buffer

    data_X = @view R0[ :, x_inds ]
    data_Y = @view Rsim[ :, y_inds ]

    pairwise!( buffer, Euclidean(), data_X, data_Y )

    empcdf!( view_out, vec( buffer ), nbins, bins )
    return nothing
end

function get_bin_quantity( summary_statistic::CIL, data::DataContainer, inds_X, inds_Y )
    data_X = @view data.observations[ :, inds_X ]
    data_Y = @view data.observations[ :, inds_Y ]
    distances = pairwise( Euclidean(), data_X, data_Y )
    distances = reshape( distances, :, 1 )
    return distances
end

function allocate_buffer( statistic::CIL, data::DataContainer )

    ndata = data.options.effective_N_obs
    rows, cols = resample_sizes( data.options.resampling_type, ndata )

    buffer = Matrix{Float64}( undef, rows, cols )
    return buffer
end

required_diff_order(stat::CIL) = 0

function generate_stat_name( stat::CIL )
    return "CIL_nbin=$(stat.nbin)"
end

get_summary_length(stat::CIL, data::DataContainer) = stat.summary_length

function finalize_summary( stat::CIL, data::DataContainer, buffers::BufferContainer )
    return @set stat.buffer = buffers.summary_buffers[ Symbol( generate_stat_name( stat ) ) ]
end
