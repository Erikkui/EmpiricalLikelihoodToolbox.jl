#------------SummaryStatistics: wrapping all summary statistics into one struct
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

# Called during MCMC
function (SS::JointSummaryStatistics)(
    view_in,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    obs_data::DataContainer,
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
