struct CumulativeSum{S, BUF} <: AbstractCumulativeSumSummary
    contracting_window::Int
    summary_length::Int
    normalization_factor::S
    buffer::BUF
end

function CumulativeSum( contracting_window::Int )
    return CumulativeSum( contracting_window, 0, NaN, nothing )
end

# The cumsum is taken over the resampled x set, so its length - and hence the contracted output
# length - follows the resampler, not the raw observation count.
function cumsum_data_length( stat::CumulativeSum, data::DataContainer )
    nx, _ = resample_sizes( data.options.resampling_type, data.options.effective_N_obs )
    return nx
end

function calculate_summary_statistic!(      # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::CumulativeSum,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    contracting_window = summary_statistic.contracting_window
    R0 = data.observations
    buffer = summary_statistic.buffer

    data_X = @view R0[ :, x_inds ]
    cumsum!( buffer, data_X, dims = 2 )
    contract!( view_out, buffer, contracting_window )

    view_out ./= summary_statistic.normalization_factor

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

    contracting_window = summary_statistic.contracting_window
    Rsim = sim_data_all.observations
    buffer = summary_statistic.buffer

    data_X = @view Rsim[ :, x_inds ]
    cumsum!( buffer, data_X, dims = 2 )
    contract!( view_out, buffer, contracting_window )

    view_out ./= summary_statistic.normalization_factor

    return nothing
end


function allocate_buffer( statistic::CumulativeSum, data::DataContainer )
    size( data.observations, 1 ) == 1 || throw( ArgumentError(
        "CumulativeSum supports one-dimensional data only, shaped (1, N); got $(size(data.observations))" ) )

    return zeros( 1, cumsum_data_length( statistic, data ) )
end

get_summary_length(stat::CumulativeSum, data::DataContainer) =
    div( cumsum_data_length( stat, data ), stat.contracting_window )


required_diff_order(stat::CumulativeSum) = 0

function generate_stat_name( stat::CumulativeSum )
    return "CumulativeSum_l=$(stat.contracting_window)"
end

function finalize_summary( stat::CumulativeSum, data::DataContainer, buffers::BufferContainer )
    nx = cumsum_data_length( stat, data )
    inds = @view buffers.index_cache[ 1:nx ]
    sum_cumul = cumsum( @view( data.observations[ :, inds ] ), dims = 2 )
    sum_cumul_contracted = contract( sum_cumul, stat.contracting_window )

    stat = @set stat.summary_length = div( nx, stat.contracting_window )
    stat = @set stat.normalization_factor = sum_cumul_contracted[end]
    return @set stat.buffer = buffers.summary_buffers[ Symbol( generate_stat_name( stat ) ) ]
end
