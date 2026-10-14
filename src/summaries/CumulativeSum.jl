"""
    CumulativeSum{F, S, SC, BUF}

Normalized cumulative sum of the data, coarse-grained into non-overlapping windows.

Takes the running cumulative sum of an "x" set, sums it within consecutive windows of
`contracting_window` observations, and divides through by a normalization factor. Being an
integral of the series, it responds to trends and drift that distribution-shaped summaries like
[`StandardECDF`](@ref) discard.

Works on multidimensional data. Each row (dimension) is accumulated, contracted and normalized
separately, and the results are concatenated row after row, so the output is
`[row 1 windows; row 2 windows; ...]`. All rows share the same resampled indices.

A ragged tail is dropped: each row contributes `div(n, contracting_window) - 1` entries, where `n`
is the length the resampler produces. The final window is always dropped, since the normalization
makes it carry no information for the observed data (it is `1.0` there).

The normalization factor of each row is, by default, that row's final contracted value of the
observed data, so the observed summary ends at `1.0`. Passing `scale` replaces it with fixed
constants, which keeps the summary on a fixed scale regardless of the observed data.

`transform` is applied elementwise to the data before it is accumulated, for observed and
simulated data alike. For example `transform = abs` accumulates absolute increments.

Unlike the other summaries, `summary_length` and `normalization_factor` are not known at
construction. Both are filled in by `TargetData`, which is also when `buffer` is bound. A
`CumulativeSum` that has not been through `TargetData` has `summary_length == 0` and would produce
a zero-width slice inside a [`JointSummaryStatistics`](@ref).

# Fields
- `contracting_window::Int`: Number of observations summed into each output entry.
- `summary_length::Int`: `ndim * (div(n, contracting_window) - 1)`. `0` until `TargetData` fills it in.
- `normalization_factor::S`: One divisor per data row. `[NaN]` until filled in by `TargetData`, unless
  `scale` was given.
- `transform::F`: Elementwise function applied to the data before accumulating. Default `identity`.
- `fixed_scale::SC`: The user-supplied `scale` vector, or `nothing` to normalize by the observed data.
- `buffer::BUF`: Scratch space, filled in by `TargetData`. `nothing` until then.
- `name::String`: The statistic's name from `generate_stat_name`, set by the constructors.

# Examples
```julia
stat = CumulativeSum(7)   # sum the running total in blocks of 7 observations

# On data [1 2 3 4] with window 2: cumsum [1 3 6 10] -> blocks [4, 16] -> normalized [0.25, 1.0],
# of which the final entry is dropped -> [0.25]

# Three series, absolute increments, fixed per-series scales
stat = CumulativeSum(20; transform = abs, scale = [1200, 1200, 4500])
```

See also [`StandardECDF`](@ref), [`JointSummaryStatistics`](@ref), [`MovingBlockBootstrap`](@ref).
"""
struct CumulativeSum{F, S<:AbstractVector{Float64}, SC, BUF} <: AbstractCumulativeSumSummary
    contracting_window::Int
    summary_length::Int
    normalization_factor::S
    transform::F
    fixed_scale::SC
    buffer::BUF
    name::String
end

function CumulativeSum( contracting_window::Int; transform = identity, scale = nothing )
    if isnothing( scale )
        return with_name( CumulativeSum( contracting_window, 0, [NaN], transform, nothing, nothing, "" ) )
    end

    if !all( s -> isfinite( s ) && s != 0, scale )
        throw( ArgumentError( "scale must contain finite, non-zero entries; got $scale" ) )
    end
    scale_vec = Float64.( collect( scale ) )
    return with_name( CumulativeSum( contracting_window, 0, copy( scale_vec ), transform, scale_vec, nothing, "" ) )
end

# The cumsum is taken over the resampled x set, so its length - and hence the contracted output
# length - follows the resampler, not the raw observation count.
function cumsum_data_length( stat::CumulativeSum, data::DataContainer )
    nx, _ = resample_sizes( data.options.resampling_type, data.options.effective_N_obs )
    return nx
end

# Running sum of `f` applied to `observations[:, inds]`, written row by row into `buffer`.
function transformed_cumsum!( buffer::AbstractMatrix{Float64}, f::F, observations::AbstractMatrix{Float64}, inds ) where {F}
    for rr in axes( buffer, 1 )
        acc = 0.0
        for (cc, ind) in enumerate( inds )
            acc += f( observations[ rr, ind ] )
            buffer[ rr, cc ] = acc
        end
    end
    return buffer
end

function cumulative_summary!( view_out::AbstractVector{Float64}, stat::CumulativeSum, observations, x_inds )
    buffer = stat.buffer
    nrows = size( buffer, 1 )
    factors = stat.normalization_factor

    transformed_cumsum!( buffer, stat.transform, observations, x_inds )

    # contract! lays the output out row after row (series-major), matching the summary layout.
    contract!( view_out, buffer, stat.contracting_window )

    n_windows = div( length( view_out ), nrows )
    for rr in 1:nrows
        factor = factors[ rr ]
        offset = (rr - 1) * n_windows
        for ww in 1:n_windows
            view_out[ offset + ww ] /= factor
        end
    end

    return nothing
end

function calculate_summary_statistic!(      # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::CumulativeSum,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    cumulative_summary!( view_out, summary_statistic, data.observations, x_inds )

    return nothing
end

function calculate_summary_statistic!(      # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::CumulativeSum,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    cumulative_summary!( view_out, summary_statistic, sim_data_all.observations, x_inds )

    return nothing
end


function allocate_buffer( statistic::CumulativeSum, data::DataContainer )
    ndim = size( data.observations, 1 )

    if !isnothing( statistic.fixed_scale ) && length( statistic.fixed_scale ) != ndim
        throw( ArgumentError(
            "CumulativeSum scale has $(length( statistic.fixed_scale )) entries but the data has $ndim rows" ) )
    end

    return zeros( ndim, cumsum_data_length( statistic, data ) )
end

get_summary_length(stat::CumulativeSum, data::DataContainer) =
    size( data.observations, 1 ) * ( div( cumsum_data_length( stat, data ), stat.contracting_window ) - 1 )


required_diff_order(stat::CumulativeSum) = 0

function generate_stat_name( stat::CumulativeSum )
    return "CumSum_len$(stat.contracting_window)"
end

function finalize_summary( stat::CumulativeSum, data::DataContainer, buffers::BufferContainer )
    nx = cumsum_data_length( stat, data )
    ndim = size( data.observations, 1 )
    n_windows = div( nx, stat.contracting_window )
    inds = @view buffers.index_cache[ 1:nx ]

    stat = @set stat.summary_length = ndim * ( n_windows - 1 )

    if isnothing( stat.fixed_scale )
        sum_cumul = transformed_cumsum!( zeros( ndim, nx ), stat.transform, data.observations, inds )
        sum_cumul_contracted = contract( sum_cumul, stat.contracting_window )
        stat = @set stat.normalization_factor = sum_cumul_contracted[ n_windows:n_windows:end ]
    end

    return @set stat.buffer = buffers.summary_buffers[ Symbol( stat.name ) ]
end
