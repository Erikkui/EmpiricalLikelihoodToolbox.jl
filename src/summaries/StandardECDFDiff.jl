#------------Standard ECDF from differences
struct StandardECDFDiff{B} <: DifferenceECDFSummary
    bins::B
    nbin::Int
    dt_obs::Float64
    diff_order::Int
    summary_length::Int
end

function StandardECDFDiff( nbin::Int, diff_order::Int, dt_obs::Float64 )
    return StandardECDFDiff( nothing, nbin, dt_obs, diff_order, nbin )
end

function calculate_summary_statistic!(  # To be used in target and bin initialization
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDFDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    data::DataContainer,
    buffers::BufferContainer )

    diff_order = summary_statistic.diff_order

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins

    data_X = @view data.differences[ diff_order ][ :, x_inds ]

    empcdf!( view_out, data_X, nbins, bins )
    return nothing
end

function calculate_summary_statistic!(  # To be used in MCMC
    view_out::AbstractVector{Float64},
    summary_statistic::StandardECDFDiff,
    x_inds::AbstractVector{<:Integer},
    y_inds::AbstractVector{<:Integer},
    obs_data_all::DataContainer,
    sim_data_all::DataContainer,
    buffers::BufferContainer )

    use_ecdf_sampling = obs_data_all.options.use_ecdf_sampling

    nbins = summary_statistic.nbin
    bins = summary_statistic.bins
    diff_order = summary_statistic.diff_order

    Rsim_diff = sim_data_all.differences[ diff_order ]

    if use_ecdf_sampling
        ndata = size( Rsim_diff, 2 )
        data_X = @view Rsim_diff[ :, : ]
        yax_values = rand( ndata )  # uniform random values for y-axis

        xmin, xmax = minimum( data_X ), maximum( data_X )
        bins_dense_temp = range( 1.01*xmin, 0.99*xmax, length = ndata ) |> collect

        ecdf_sim = empcdf( data_X, nbins, bins_dense_temp )    # ecdf of simulation according to observed data bins
        data_X_new = invcdf( yax_values, ecdf_sim, nbins, 1 )  # inverse cdf to get simulated data values according to observed data bins
        empcdf!( view_out, data_X_new, nbins, bins )    # ecdf of simulation according to observed data bins
    else
        data_X = @view Rsim_diff[ :, x_inds ]
        empcdf!( view_out, data_X, nbins, bins )
    end

    return nothing
end

function get_bin_quantity( summary_statistic::StandardECDFDiff, data::DataContainer, inds_X, inds_Y )
    diff_ind = summary_statistic.diff_order
    data_X = data.differences[ diff_ind ]
    data_X =  data_X[ :, inds_X ]
    return vec(data_X)
end

function allocate_buffer( statistic::StandardECDFDiff, data::DataContainer )
    buffer = Vector{Float64}( undef, statistic.summary_length )
    return buffer
end

required_diff_order(stat::StandardECDFDiff) = stat.diff_order

function generate_stat_name( stat::StandardECDFDiff )
    return "StandardECDFDiff_k=$(stat.nbin)_diff_order=$(stat.diff_order)"
end
