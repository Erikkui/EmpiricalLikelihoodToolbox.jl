function calculate_bin_bounds( data::AbstractVector{<:Real} )

    # Calculate bin bounds for empirical cdf calculation
    quantiles = quantile( data, [0.005, 0.25, 0.75, 0.995])
    q_low, q1, q3, q_high = quantiles[1], quantiles[2], quantiles[3], quantiles[4]
    iqr = max( q3 - q1, 1e-9 )

    bin_min = q_low - 0.25 * iqr
    bin_max = q_high + 0.25 * iqr
    if !( isfinite(bin_max) ) || bin_max <= bin_min
        bin_max = bin_min + 1.0
    end

    return bin_min, bin_max
end

#----------Main bin calculation function
function bin_select( data, nbin, axis_uniform )
    # Generate bins for empirical cdf calculation
    a, b = calculate_bin_bounds( data )
    # println( "a = $a, b = $b" )
    if axis_uniform == :xax
        bins = collect( range(a, b, length=nbin) )

    elseif axis_uniform == :yax
        nbin_temp = 1000
        bins_temp = collect( range(a, b, length=nbin_temp) )

        # println( "Calculating ECDF for bin selection, nbin_temp = $bins_temp \n" )


        # Dense ecdf for inversion
        cdf = empcdf_raw( data, nbin_temp, bins_temp )

        # println( "Calculating ECDF for bin selection, nbin_temp = $cdf \n" )

        # Inverse CDF for final bins
        bins = invcdf( bins_temp, cdf, nbin, 1)

    elseif axis_uniform == :log
        R0 = b
        bb = (R0 / a / 1.01)^(1 / nbin)
        bins = R0 .* bb .^ (-nbin:-1)
    end

    return bins
end



# For basic and multidimensional cdf summaries
function initialize_bins(
    data::DataContainer,
    statistic::StandardECDFSummary,
    options::MethodsOptions,
    index_cache::Vector{Int} )

    nbin = statistic.nbin
    R0 = data.observations
    axis_uniform = options.axis_uniform

    ndim = size( R0, 1 )
    bins = Vector{ Vector{Float64} }( undef, ndim )

    # Create bins
    for ii in 1:ndim
        data_ii = @view R0[ii, :]
        bins[ii] = bin_select( data_ii, nbin, axis_uniform )
    end

    new_statistic = @set statistic.bins = bins
    return new_statistic
end



# Abstract ECDF summaries: when the ECDFs are calculated from other than raw data
function initialize_bins(
    data::DataContainer,
    statistic::AbstractECDFSummary,
    options::MethodsOptions,
    index_cache::Vector{Int} )

    resampler = options.resampling_type
    bins_resamplings = options.bins_resamplings
    nbin = statistic.nbin
    axis_uniform = options.axis_uniform

    resampled_summaries_all = Vector{ Matrix{Float64} }( undef, bins_resamplings )
    for ii in 1:bins_resamplings
        x_inds, y_inds = resampler( data, options, index_cache )
        summary = get_bin_quantity( statistic, data, x_inds, y_inds )
        resampled_summaries_all[ii] = summary
    end
    resampled_summaries_all = vcat( resampled_summaries_all... )

    # Create bins
    ndim = size( resampled_summaries_all, 2 )
    if ndim == 1
        resampled_summaries_all = vec( resampled_summaries_all )
        bins = bin_select( resampled_summaries_all, nbin, axis_uniform )
    else
        bins = Vector{ Vector{Float64} }( undef, ndim )
        for ii in 1:ndim
            data_ii = filter( isfinite, resampled_summaries_all[:, ii] )
            bins[ii] = bin_select( data_ii, nbin, axis_uniform )
        end
    end

    new_statistic = @set statistic.bins = bins
    return new_statistic
end



# For other summaries, we do not need to initialize bins
function initialize_bins(
    data::DataContainer,
    statistic::AbstractSummaryStatistic,
    options::MethodsOptions,
    index_cache::Vector{Int} )
    return statistic
end
