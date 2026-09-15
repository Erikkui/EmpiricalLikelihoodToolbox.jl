# src/summaries/empirical_cdf.jl and src/utils/bin_calculation.jl

@testset "empirical cdf" begin

    @testset "empcdf_raw! counts data <= bin" begin
        data = collect(1.0:10.0)
        bins = [0.5, 1.0, 5.5, 10.0, 99.0]
        out = zeros(length(bins))
        empcdf_raw!(out, data, length(bins), bins)

        @test out == [0.0, 0.1, 0.5, 1.0, 1.0]
        @test all(v -> 0.0 <= v <= 1.0, out)
        @test issorted(out)                                  # non-decreasing in b
        @test empcdf_raw(data, 1, [maximum(data)])[1] == 1.0  # b >= max
        @test empcdf_raw(data, 1, [prevfloat(minimum(data))])[1] == 0.0
    end

    @testset "empcdf_raw! honours nbins over length(bins)" begin
        data = collect(1.0:4.0)
        bins = [1.0, 2.0, 3.0, 4.0]
        out = fill(-1.0, 4)
        empcdf_raw!(out, data, 2, bins)      # only the first two bins are written
        @test out[1:2] == [0.25, 0.5]
        @test out[3:4] == [-1.0, -1.0]
    end

    @testset "empcdf_kernelsmoothed!" begin
        rng = StableRNG(21)
        data = randn(rng, 400)
        bins = collect(-2.0:0.5:2.0)
        out = zeros(length(bins))
        empcdf_kernelsmoothed!(out, data, length(bins), bins)

        @test all(v -> 0.0 <= v <= 1.0, out)
        @test issorted(out)
        # Smoothing a large sample tracks the raw ECDF closely.
        @test out ≈ empcdf_raw(data, length(bins), bins) atol = 0.05

        @testset "empty data fills zeros" begin
            o = fill(9.0, 3)
            empcdf_kernelsmoothed!(o, Float64[], 3, [1.0, 2.0, 3.0])
            @test all(iszero, o)
        end

        @testset "iterates eachindex(bins), ignoring nbins" begin
            # Deliberately different from empcdf_raw!, which loops 1:nbins.
            o = fill(-1.0, 4)
            empcdf_kernelsmoothed!(o, collect(1.0:4.0), 2, [1.0, 2.0, 3.0, 4.0])
            @test all(>(-1.0), o)
        end
    end

    @testset "resolve_ecdf" begin
        @test resolve_ecdf(:default) === empcdf_raw!
        @test resolve_ecdf(:kernel_smoothed) === empcdf_kernelsmoothed!
        @test_throws ArgumentError resolve_ecdf(:bogus)
        # MethodsOptions resolves the function at construction, so a bad symbol fails early.
        @test_throws ArgumentError MethodsOptions(N_obs = 10, ecdf_calculation_type = :bogus)
        @test MethodsOptions(N_obs = 10).ecdf_function === empcdf_raw!
    end

    @testset "_calculate_bin_bounds" begin
        @testset "brackets the data" begin
            data = collect(1.0:100.0)
            lo, hi = _calculate_bin_bounds(data)
            @test lo < hi
            @test isfinite(lo) && isfinite(hi)
        end

        @testset "constant data collapses to a hairline range" begin
            # Every quantile coincides, so the width is driven entirely by the 1e-9 iqr floor:
            # hi - lo == 0.5 * 1e-9. Note this keeps hi > lo, so the `bin_max <= bin_min` guard
            # in _calculate_bin_bounds never fires on constant data.
            lo, hi = _calculate_bin_bounds(fill(5.0, 20))
            @test hi > lo
            @test hi - lo ≈ 0.5e-9 rtol = 1e-6
        end

        @testset "symmetric data gives symmetric bounds" begin
            lo, hi = _calculate_bin_bounds(collect(-10.0:10.0))
            @test lo ≈ -hi
        end

        @testset "shifting the data shifts the bounds" begin
            data = collect(1.0:50.0)
            lo, hi = _calculate_bin_bounds(data)
            lo2, hi2 = _calculate_bin_bounds(data .+ 100.0)
            @test lo2 ≈ lo + 100.0
            @test hi2 ≈ hi + 100.0
        end
    end

    @testset "_bin_select" begin
        data = collect(1.0:100.0)

        @testset ":xax is uniformly spaced" begin
            bins = _bin_select(data, 8, :xax)
            @test length(bins) == 8
            @test issorted(bins)
            @test all(≈(bins[2] - bins[1]), diff(bins))
        end

        @testset ":yax is roughly equal-probability" begin
            nbin = 8
            bins = _bin_select(data, nbin, :yax)
            @test length(bins) == nbin
            @test issorted(bins)
            # Equal-probability bins put roughly 1/nbin of the mass between consecutive edges.
            f = empcdf_raw(data, nbin, collect(Float64, bins))
            @test maximum(diff(f)) < 3 / nbin
        end

        @testset ":log is positive and geometrically spaced" begin
            # :log needs a strictly positive lower bin bound. `data` (1:100) has a robust lower
            # bound below zero and would throw -- see test_known_issues.jl -- so use data that
            # sits well away from the origin.
            bins = _bin_select(collect(1000.0:1100.0), 6, :log)
            @test length(bins) == 6
            @test all(>(0), bins)
            @test issorted(bins)
            @test all(≈(bins[2] / bins[1]), bins[2:end] ./ bins[1:end-1])
        end
    end

    @testset "create_bins dispatch" begin
        data = row(1.0:40.0)
        dc = make_container(data)
        opts = dc.options
        cache = collect(1:40)

        @testset "StandardECDFSummary gets one bin vector per data row" begin
            stat = create_bins(dc, StandardECDF(5), opts, cache)
            @test stat.bins isa Vector{Vector{Float64}}
            @test length(stat.bins) == 1
            @test length(stat.bins[1]) == 5
        end

        @testset "multidimensional data gets one bin vector per row" begin
            dc2 = make_container(vcat(row(1.0:40.0), row(41.0:80.0)))
            stat = create_bins(dc2, StandardECDF(5, 2), dc2.options, cache)
            @test length(stat.bins) == 2
            @test all(b -> length(b) == 5, stat.bins)
        end

        @testset "non-ECDF summaries pass through untouched" begin
            for stat in (ChamferDistance(1), ChamferDistanceDiff(1, 1, 1.0), CumulativeSum(4))
                @test create_bins(dc, stat, opts, cache) === stat
            end
        end
    end
end
