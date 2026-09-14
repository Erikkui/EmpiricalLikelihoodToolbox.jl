#------------Chamfer Distance
"""
    ChamferDistance{T}

Chamfer distance between point sets "x" and "y".

For neighbour index `k`, the Chamfer distance is the mean `k`-th nearest-neighbour distance from
every x point to the y set, plus the mean `k`-th nearest-neighbour distance from every y point back
to the x set. It is symmetric in its two sets, non-decreasing in `k`, and zero at `k = 1` when the
two sets are identical.

Unlike the eCDF-based summaries this returns the distances themselves, not a distribution over them,
so it contributes only `length(neighbors)` numbers. Requesting several neighbours returns one value
per neighbour, in the order given.

This is a two-set summary: it consumes the "x" and "y" index sets produced by the configured
resampler (`MethodsOptions.resampling_type`), not the raw series.

# Fields
- `neighbors::T`: The neighbour index (or indices) to evaluate. An `Int` or a vector of `Int`.
- `highest_neighbor::Int`: `maximum(neighbors)`, the depth the k-d tree search must reach.
- `summary_length::Int`: `1` for a scalar `neighbors`, otherwise `length(neighbors)`.

# Examples
```julia
stat = ChamferDistance(1)        # nearest-neighbour Chamfer distance, summary_length == 1
stat = ChamferDistance([1, 3])   # 1st and 3rd neighbour, summary_length == 2
```

See also [`ChamferDistanceDiff`](@ref), [`ChamferECDF`](@ref).
"""
struct ChamferDistance{T} <: AbstractChamferSummary
    neighbors::T
    highest_neighbor::Int
    summary_length::Int
end

function ChamferDistance( neighbors::Int )
    return ChamferDistance( neighbors, maximum(neighbors), 1 )
end

function ChamferDistance( neighbors::AbstractVector{<:Int} )
    return ChamferDistance( collect(vec(neighbors)), maximum(neighbors), length(neighbors) )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferDistance,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    data_X = @view data.observations[ :, x_inds ]
    data_Y = @view data.observations[ :, y_inds ]

    kvals = summary_statistic.neighbors

    chamfer_distance!( view_out, data_X, data_Y, kvals )

    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferDistance,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    data_X = @view target.data.observations[ :, x_inds ]
    Rsim = @view sim_data_all.observations[ :, y_inds ]
    ytree = KDTree( Rsim )

    kvals = summary_statistic.neighbors

    # key = nameof( typeof(summary_statistic) )
    # buffer = buffers.summary_buffers[ key ]

    chamfer_distance!( view_out, data_X, Rsim, ytree, kvals )

    return nothing
end

function allocate_buffer( statistic::ChamferDistance, data::DataContainer )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end

required_diff_order(stat::ChamferDistance) = 0

function generate_stat_name( stat::ChamferDistance )
    return "ChamferDistance_k=$(stat.neighbors)"
end

get_summary_length(stat::ChamferDistance, data::DataContainer) = stat.summary_length

function finalize_summary( stat::ChamferDistance, data::DataContainer, buffers::BufferContainer )
    return stat
end
