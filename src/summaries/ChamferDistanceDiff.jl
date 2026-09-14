"""
    ChamferDistanceDiff{T}

Chamfer distance computed on a numerical derivative of the data.

Identical to [`ChamferDistance`](@ref), except the point sets compared are the `diff_order`-th
central difference of the series rather than the raw series. `TargetData` trims `diff_order` points
from each end of the index cache, because central differences fabricate their boundary columns by
padding.

# Fields
- `neighbors::T`: The neighbour index (or indices) to evaluate. An `Int` or a vector of `Int`.
- `highest_neighbor::Int`: `maximum(neighbors)`, the depth the k-d tree search must reach.
- `diff_order::Int`: How many times to differentiate. `1` is the first derivative.
- `dt_obs::Float64`: Time step between observations, the denominator of the difference.
- `summary_length::Int`: `1` for a scalar `neighbors`, otherwise `length(neighbors)`.

# Examples
```julia
stat = ChamferDistanceDiff(1, 1, 1.0)        # first derivative, nearest neighbour
stat = ChamferDistanceDiff([1, 3], 1, 1.0)   # summary_length == 2
```

See also [`ChamferDistance`](@ref).
"""
struct ChamferDistanceDiff{T} <: AbstractChamferSummary
    neighbors::T
    highest_neighbor::Int
    diff_order::Int
    dt_obs::Float64
    summary_length::Int
end

function ChamferDistanceDiff( neighbors::Int, diff_order::Int, dt_obs::Float64 )
    return ChamferDistanceDiff( neighbors, maximum(neighbors), diff_order, dt_obs, 1 )
end

function ChamferDistanceDiff( neighbors::AbstractVector{<:Int}, diff_order::Int, dt_obs::Float64 )
    return ChamferDistanceDiff( collect(vec(neighbors)), maximum(neighbors), diff_order, dt_obs, length(neighbors) )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferDistanceDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    diff_order = summary_statistic.diff_order

    data_X = @view data.differences[ diff_order ][ :, x_inds ]
    data_Y = @view data.differences[ diff_order ][ :, y_inds ]

    kvals = summary_statistic.neighbors

    chamfer_distance!( view_out, data_X, data_Y, kvals )

    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::ChamferDistanceDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    target::TargetData,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    diff_order = summary_statistic.diff_order

    data_X = @view target.data.differences[ diff_order ][ :, x_inds ]
    Rsim = @view sim_data_all.differences[ diff_order ][ :, y_inds ]
    ytree = KDTree( Rsim )

    kvals = summary_statistic.neighbors

    chamfer_distance!( view_out, data_X, Rsim, ytree, kvals )

    return nothing
end

function allocate_buffer( statistic::ChamferDistanceDiff, data::DataContainer )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end

required_diff_order(stat::ChamferDistanceDiff) = stat.diff_order

function generate_stat_name( stat::ChamferDistanceDiff )
    return "ChamferDistanceDiff_k=$(stat.neighbors)_diff=$(stat.diff_order)"
end

get_summary_length(stat::ChamferDistanceDiff, data::DataContainer) = stat.summary_length

function finalize_summary( stat::ChamferDistanceDiff, data::DataContainer, buffers::BufferContainer )
    return stat
end
