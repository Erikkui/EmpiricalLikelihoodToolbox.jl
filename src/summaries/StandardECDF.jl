#------------Standard ECDF
struct StandardECDF{B} <: AbstractECDFSummary
    bins::B
    nbin::Int
    summary_length::Int
end

function StandardECDF( nbin::Int)
    return StandardECDF( nothing, nbin, nbin )
end

function StandardECDF( bins::AbstractVector{<:Real} )
    return StandardECDF( collect(vec(bins)), length(bins), length(bins) )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDF,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins

    data_X = @view data.observations[ :, x_inds ]
    empcdf!( view_out, data_X, nbins, bins )

    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDF,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    obs_data_all::DataContainer,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    use_ecdf_sampling = obs_data_all.options.use_ecdf_sampling
    Rsim = sim_data_all.observations

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins

    if use_ecdf_sampling
        ndata = size( Rsim, 2)
        data_X = @view Rsim[ :, : ]
        yax_values = rand( ndata )  # uniform random values for y-axis

        xmin, xmax = minimum( data_X ), maximum( data_X )
        bins_dense_temp = range( xmin, xmax, length = ndata ) |> collect

        ecdf_sim = empcdf( data_X, nbins, bins_dense_temp )    # ecdf of simulation according to observed data bins
        data_X_new = invcdf( yax_values, ecdf_sim, nbins, 1 )  # inverse cdf to get simulated data values according to observed data bins
        empcdf!( view_out, data_X_new, nbins, bins )    # ecdf of simulation according to observed data bins
    else
        data_X = @view Rsim[ :, x_inds ]
        empcdf!( view_out, data_X, nbins, bins )
    end

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
