#------------SummaryStatistics: wrapping all summary statistics into one struct
"""
    JointSummaryStatistics{T<:Tuple}

Container that concatenates several summary statistics into one vector.

Every summary is evaluated in the order given and written into consecutive slices of the output,
each `summary.summary_length` long, so the combined summary has length
`sum(s.summary_length for s in statistics)`. `TargetData` requires its summaries wrapped in this
type even when there is only one.

Slicing uses the `summary_length` *field*, which for [`CumulativeSum`](@ref) is only correct after
`TargetData` has finalized it. Constructing a `JointSummaryStatistics` by hand and evaluating it
without going through `TargetData` can therefore produce zero-width slices.

!!! warning "Summaries must have distinct names"
    Buffers are keyed by `generate_stat_name`, so two summaries that generate the same name
    collide. `JointSummaryStatistics(StandardECDF(5), StandardECDF(5))` throws when passed to
    `TargetData`. Vary the parameters, or use a single summary.

# Fields
- `statistics::T`: A tuple of the wrapped summary statistics, in output order.

# Examples
```julia
stats = JointSummaryStatistics(CIL(10), CILDiff(10, 1, 1.0))
stats = JointSummaryStatistics(StandardECDF(10))   # required even for a single summary
```
"""
struct JointSummaryStatistics{T<:Tuple}
    statistics::T
end

# Slurps all summary statistics into a single tuple when calling SummaryStatistics
function JointSummaryStatistics( args::AbstractSummaryStatistic... )
    return JointSummaryStatistics( args )
end

# Called during training
function (SS::JointSummaryStatistics)(
    view_in,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    summaries = SS.statistics

    start_ind = 1
    foreach( summaries ) do summary
        summary_length = summary.summary_length
        end_ind = start_ind + summary_length - 1
        view_out = @view view_in[ start_ind:end_ind ]

        calculate_summary_statistic!( view_out, summary, x_inds, y_inds, data, buffers )

        start_ind += summary_length
    end

    return nothing
end

# Called during MCMC. obs_data is the full TargetData (not just its DataContainer), since summaries
# need access to the fixed observed data alongside options/buffers, eg. R0 = target.data.observations.
function (SS::JointSummaryStatistics)(
    view_in,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    obs_data::TargetData,
    sim_data::DataContainer,
    buffers::BufferContainer )

    summaries = SS.statistics

    start_ind = 1
    foreach(summaries) do summary
        summary_length = summary.summary_length
        end_ind = start_ind + summary_length - 1
        view_out = @view view_in[ start_ind:end_ind ]

        calculate_summary_statistic!( view_out, summary, x_inds, y_inds, obs_data, sim_data, buffers )

        start_ind += summary_length
    end

    return nothing
end
