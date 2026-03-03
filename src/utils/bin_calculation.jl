#----------Main bin calculation function
function bin_select( minmax, nbin, axis_uniform, data )
    # Generate bins for empirical cdf calculation

    if axis_uniform == :xax
        a = minmax[1]
        b = minmax[2]
        bins = collect( range(a, b, length=nbin) )

    elseif axis_uniform == :yax
        nbin_temp = 100
        a = minmax[1]
        b = minmax[2]
        bins_temp = collect( range(a, b, length=nbin_temp) )

        # Dense ecdf for inversion
        cdf = empcdf( data, nbin_temp, bins_temp )

        # Inverse CDF for final bins
        bins = invcdf( bins_temp, cdf, nbin, 1)

    elseif axis_uniform == :log
        R0 = b
        bb = (R0 / a / 1.01)^(1 / nbin)
        bins = R0 .* bb .^ (-nbin:-1)
    end

    return bins
end


# For basic cdf and cil summaries
function initialize_bins(
    data::DataContainer,
    statistic::AbstractECDFSummary,
    options::MethodsOptions )

    resampler = options.resampling_type
    bins_resamplings = options.bins_resamplings
    nbin = statistic.nbin

    minmax = zeros(2)
    mins = zeros( bins_resamplings )
    maxs = zeros( bins_resamplings )

    resampled_summaries_all = Vector{ AbstractVector{Float64} }( undef, bins_resamplings )
    ind_size = get_index_size( resampler, data.observations, options )
    index_cache = collect( 1:ind_size )
    for ii in 1:bins_resamplings
        # Half-half random split
        x_inds, y_inds = resampler( data, options, index_cache )
        summary = get_bin_quantity( statistic, data, x_inds, y_inds )

        resampled_summaries_all[ii] = summary
        mins[ii] = minimum( summary )
        maxs[ii] = maximum( summary )
    end
    minmax[1] = maximum( mins )
    minmax[2] = minimum( maxs )
    resampled_summaries_all = vcat( resampled_summaries_all... ) |> vec

    # Create bins
    bins = bin_select( minmax, nbin, options.axis_uniform, resampled_summaries_all )

    new_statistic = @set statistic.bins = bins
    return new_statistic
end


# Bin initialization for ChamferECDF
function initialize_bins(
    data::DataContainer,
    statistic::ChamferECDF,
    options::MethodsOptions )

    resampler = options.resampling_type
    bins_resamplings = options.bins_resamplings
    nbin = statistic.nbin

    minmax = zeros( 2, length(statistic.neighbors) )
    mins = zeros( bins_resamplings, length(statistic.neighbors) )
    maxs = zeros( bins_resamplings, length(statistic.neighbors) )

    resampled_summaries_all = Vector{ Matrix{Float64} }( undef, bins_resamplings )
    ind_size = get_index_size( resampler, data.observations, options )
    index_cache = collect( 1:ind_size )
    for ii in 1:bins_resamplings
        # Half-half random split
        x_inds, y_inds = resampler( data, options, index_cache )
        summary = get_bin_quantity( statistic, data, x_inds, y_inds )

        resampled_summaries_all[ii] = summary
        mins[ii, :] = minimum( summary, dims = 1 )
        maxs[ii, :] = maximum( summary, dims = 1 )
    end
    minmax[1, :] = maximum( mins, dims = 1 )
    minmax[2, :] = minimum( maxs, dims = 1 )
    resampled_summaries_all = vcat( resampled_summaries_all... )

    # Create bins
    bins = Vector{ Vector{Float64} }( undef, length(statistic.neighbors) )
    for ii in 1:length(statistic.neighbors)
        bins[ii] = bin_select( minmax[:, ii], nbin, options.axis_uniform, resampled_summaries_all[:, ii] )
    end
    # bins = bin_select( minmax, nbin, options.axis_uniform, resampled_summaries_all )

    new_statistic = @set statistic.bins = bins
    return new_statistic
end


# Bin initialization for ID summaries
function initialize_bins(
    data::DataContainer,
    statistic::IDSummary,
    options::MethodsOptions )

    resampler = options.resampling_type
    bins_resamplings = options.bins_resamplings
    nbin = statistic.nbin

    minmax = zeros( 2, length(statistic.neighbors) )
    mins = zeros( bins_resamplings, length(statistic.neighbors) )
    maxs = zeros( bins_resamplings, length(statistic.neighbors) )

    resampled_summaries_all = Vector{ Matrix{Float64} }( undef, bins_resamplings )
    ind_size = get_index_size( resampler, data.observations, options )
    index_cache = collect( 1:ind_size )
    for ii in 1:bins_resamplings
        # Half-half random split
        x_inds, y_inds = resampler( data, options, index_cache )
        summary = get_bin_quantity( statistic, data, x_inds, y_inds )

        resampled_summaries_all[ii] = summary
        mins[ii, :] = minimum( summary, dims = 1 )
        maxs[ii, :] = maximum( summary, dims = 1 )
    end
    minmax[1, :] = maximum( mins, dims = 1 )
    minmax[2, :] = minimum( maxs, dims = 1 )
    resampled_summaries_all = vcat( resampled_summaries_all... )

    # Create bins
    bins = Vector{ Vector{Float64} }( undef, length(statistic.neighbors) )
    for ii in 1:length(statistic.neighbors)
        bins[ii] = bin_select( minmax[:, ii], nbin, options.axis_uniform, resampled_summaries_all[:, ii] )
    end
    # bins = bin_select( minmax, nbin, options.axis_uniform, resampled_summaries_all )

    new_statistic = @set statistic.bins = bins
    return new_statistic
end


# For other summaries, we do not need to initialize bins
function initialize_bins( data::DataContainer, statistic::AbstractSummaryStatistic, options::MethodsOptions )
    return statistic
end
