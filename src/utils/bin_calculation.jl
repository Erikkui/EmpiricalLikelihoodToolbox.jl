# Robust range of the data for the dense grid of the bin calculation. The 0.5 % / 99.5 % quantiles lie inside the
# data, so the grid never leaves the quantity's support (eg. negative distances or counts).
function _calculate_bin_bounds( data::AbstractVector{<:Real} )

    bin_min, bin_max = quantile( data, [0.005, 0.995] )
    if !( isfinite(bin_max) ) || bin_max <= bin_min
        bin_max = bin_min + 1.0
    end

    return bin_min, bin_max
end

# Discrete (tied) data: several invcdf levels can fall inside one jump of the ECDF, so their bins sit next to the same
# atom and give identical ECDF values, or a bin lands on the maximum, where the ECDF is identically 1. In that case every
# bin is snapped onto a distinct data value whose ECDF level P(X <= u) is nearest the bin's target level. The maximum is
# never used. Data whose bins already give distinct ECDF values strictly inside (0, 1) is returned unchanged.
function _snap_tied_bins( data::AbstractVector{<:Real}, bins::Vector{Float64}, levels )
    nbin = length( bins )
    F = empcdf_raw( data, nbin, bins )
    if allunique( F ) && all( f -> 0.0 < f < 1.0, F )
        return bins
    end

    # Distinct values u and their ECDF levels Fu
    s = sort( data )
    n = length( s )
    u = Float64[]
    Fu = Float64[]
    for jj in 1:n
        if jj == n || s[jj+1] != s[jj]
            push!( u, s[jj] )
            push!( Fu, jj / n )
        end
    end

    m = length( u ) - 1     # candidates exclude the maximum
    if m < 1
        @warn "Data has a single distinct value; ECDF bins cannot be separated." maxlog=1
        return bins
    end
    if m < nbin
        @warn "Data has only $m distinct values below its maximum for $nbin ECDF bins; some bins share an ECDF value." maxlog=1
    end

    snapped = similar( bins )
    prev = 0
    for ii in 1:nbin
        # Candidate with the nearest level
        k = searchsortedfirst( view( Fu, 1:m ), levels[ii] )
        if k > m || ( k > 1 && levels[ii] - Fu[k-1] <= Fu[k] - levels[ii] )
            k -= 1
        end

        if m >= nbin
            # Strictly increasing, leaving room for the remaining bins
            k = clamp( k, prev + 1, m - (nbin - ii) )
        else
            k = max( k, prev, 1 )
        end
        snapped[ii] = u[k]
        prev = k
    end

    return snapped
end

#----------Main bin calculation function
function _bin_select( data, nbin, axis_uniform )
    # Generate bins for empirical cdf calculation
    data = filter( isfinite, vec( data ) )
    isempty( data ) && throw( ArgumentError( "No finite values to calculate ECDF bins from." ) )

    # Extreme levels of the outermost bins, the same as the first and last invcdf_levels
    p = 0.5 / nbin

    if axis_uniform == :xax
        a, b = quantile( data, [p, 1 - p] )
        if !( b > a )
            a, b = _calculate_bin_bounds( data )
        end
        bins = collect( range(a, b, length=nbin) )

    elseif axis_uniform == :yax
        a, b = _calculate_bin_bounds( data )
        nbin_temp = 1000
        bins_temp = collect( range(a, b, length=nbin_temp) )

        # Dense ecdf for inversion
        cdf = empcdf_raw( data, nbin_temp, bins_temp )

        # Inverse CDF for final bins
        bins = invcdf( bins_temp, cdf, nbin, 1)
        bins = _snap_tied_bins( data, bins, invcdf_levels( cdf, nbin ) )

    elseif axis_uniform == :log
        # Log bins are only meaningful for a positive range
        positive = filter( >(0), data )
        isempty( positive ) && throw( ArgumentError( "axis_uniform = :log needs positive data, eg. not differences." ) )
        a, b = quantile( data, [p, 1 - p] )
        a = max( a, minimum( positive ) )
        if !( b > a )
            b = 2 * a
        end
        bins = exp.( range( log(a), log(b), length=nbin ) )
    else
        throw( ArgumentError( "Unknown axis_uniform $(axis_uniform). Must be :xax, :yax, or :log." ) )
    end

    return bins
end

function _has_existing_bins( statistic::AbstractSummaryStatistic )
    if isnothing( statistic.bins )
        return false
    else
        return true
    end
end

# For basic and multidimensional cdf summaries
function create_bins(
    data::DataContainer,
    statistic::StandardECDFSummary,
    options::MethodsOptions,
    index_cache::Vector{Int} )

    if _has_existing_bins( statistic )
        return statistic
    end

    nbin = statistic.nbin
    axis_uniform = options.axis_uniform

    # Bin the series the ECDF is evaluated on: the raw observations, or their differences for the Diff summaries
    diff_order = required_diff_order( statistic )
    R0 = diff_order == 0 ? data.observations : data.differences[ diff_order ]

    ndim = size( R0, 1 )

    # Create bins, from the trimmed columns in index_cache (the same range the ECDF is evaluated on)
    bins = Vector{ Vector{Float64} }( undef, ndim )
    for ii in 1:ndim
        data_ii = @view R0[ ii, index_cache ]
        bins[ii] = _bin_select( data_ii, nbin, axis_uniform )
    end

    new_statistic = @set statistic.bins = bins
    return new_statistic
end



# Abstract ECDF summaries: when the ECDFs are calculated from other than raw data
function create_bins(
    data::DataContainer,
    statistic::AbstractECDFSummary,
    options::MethodsOptions,
    index_cache::Vector{Int} )

    if _has_existing_bins( statistic )
        return statistic
    end

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
    bins = Vector{ Vector{Float64} }( undef, ndim )
    for ii in 1:ndim
        data_ii = filter( isfinite, resampled_summaries_all[:, ii] )
        bins[ii] = _bin_select( data_ii, nbin, axis_uniform )
    end

    new_statistic = @set statistic.bins = bins
    return new_statistic
end


# For other summaries, we do not need to initialize bins
function create_bins(
    data::DataContainer,
    statistic::AbstractSummaryStatistic,
    options::MethodsOptions,
    index_cache::Vector{Int} )
    return statistic
end
