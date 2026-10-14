# Chi-squared test: the squared Mahalanobis distances of the rows of Y are compared with Chisq(k).
# Tests Y against N(mu, C). mu is a vector or a 1 x k matrix. C is either a covariance matrix or the
# upper triangular Cholesky factor U of one, C = U'U (eg. `cholesky(C).U`). mu and C are given
# together; with neither given they are estimated from Y, so this is a normality test.
# The returned mu and stdev are the 1 x k mean and standard deviations of the tested distribution.
function chi2_test( Y::AbstractMatrix{<:Real}; mu = nothing, C = nothing )

    (mu === nothing) == (C === nothing) || throw(ArgumentError("give both mu and C, or neither"))

    n, k = size( Y )

    if mu === nothing
        Y_mean = mean( Y, dims=1 )
        Y_std = std( Y, dims=1)
        Y0 = Y .- Y_mean
        C = cov(Y)
        U = cholesky( Symmetric( C ) ).U
    else
        if C isa UpperTriangular
            U = C
        elseif C isa Matrix
            U = cholesky( Symmetric( C ) ).U
        else
            throw( ArgumentError( "C must be a covariance Matrix or an UpperTriangular Cholesky factor, got $(typeof(C))" ) )
        end
        Y_mean = reshape( mu, 1, : )
        Y_std = sqrt.( sum( abs2, U, dims=1 ) )  # sqrt(diag(C)), as C = U'U
        Y0 = Y .- Y_mean
    end

    # With Z = (Y - mu) / U the squared Mahalanobis distance of each row is just its squared norm.
    Z = Y0 / U
    D_sq = vec( sum( abs2, Z, dims=2 ) )

    # Theoretical Chi-square distribution
    chisq_dist = Chisq(k)

    # Emprical Histogram (pdf) calculation
    h = fit( Histogram, D_sq, nbins = 100 )
    edges = h.edges[1]
    x = (edges[1:end-1] .+ edges[2:end]) ./ 2  # Midpoints of histogram bins
    khi_n = h.weights ./ ( sum(h.weights) .* diff(edges) )

    # Chi-square theoretical pdf
    chi_pf = pdf.( chisq_dist, x )

    D_sorted = sort(D_sq)
    theo_q = quantile.( chisq_dist, (1:n) ./ (n + 1) )

    output = (
        bins = x,
        pdf_emp = khi_n,
        pdf_theor = chi_pf,
        dists_sorted = D_sorted,
        quantiles_theor = theo_q,
        mu = Y_mean,
        stdev = Y_std,
    )

    return output
end


function chi2_plot( chi2_results )
    fig = Figure( size = ( 1100, 450 ) )
    ax1 = Axis( fig[1, 1], title = "Empirical vs Theoretical PDF" )
    lines!(
        ax1,
        chi2_results.bins,
        chi2_results.pdf_theor;
        color = :blue,
        label = "Chisq($(length(chi2_results.mu))) pdf"
        )
    lines!(
        ax1,
        chi2_results.bins,
        chi2_results.pdf_emp;
        color = :red,
        label = "simulated summaries"
        )
    axislegend( ax1 )

    ax2 = Axis( fig[1, 2]; title = "QQ plot" )
    scatter!(
        ax2,
        chi2_results.quantiles_theor,
        chi2_results.dists_sorted;
        markersize = 4,
        color = :red
        )
    lines!(
        ax2,
        [0, maximum( chi2_results.quantiles_theor )],
        [0, maximum( chi2_results.quantiles_theor )];
        color = :blue
        )
    return fig
end
