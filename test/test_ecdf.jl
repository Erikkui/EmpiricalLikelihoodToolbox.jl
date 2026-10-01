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
        @testset "is the robust quantile range, inside the data" begin
            data = collect(1.0:100.0)
            lo, hi = _calculate_bin_bounds(data)
            @test lo < hi
            @test isfinite(lo) && isfinite(hi)
            @test (lo, hi) == Tuple(quantile(data, [0.005, 0.995]))
            @test minimum(data) <= lo && hi <= maximum(data)
        end

        @testset "never leaves the support of non-negative data" begin
            # The old IQR padding pushed the lower bound below zero for counts and distances.
            lo, _ = _calculate_bin_bounds(vcat(zeros(40), collect(1.0:60.0)))
            @test lo >= 0.0
        end

        @testset "constant data falls back to a unit range" begin
            lo, hi = _calculate_bin_bounds(fill(5.0, 20))
            @test (lo, hi) == (5.0, 6.0)
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
            # `data` (1:100) has a robust lower bound below zero, which :log clamps to a small
            # positive value rather than rejecting (see the testset below) -- but the clamped
            # bins are numerically degenerate, so this uses data well away from the origin for a
            # clean example.
            bins = _bin_select(collect(1000.0:1100.0), 6, :log)
            @test length(bins) == 6
            @test all(>(0), bins)
            @test issorted(bins)
            @test all(≈(bins[2] / bins[1]), bins[2:end] ./ bins[1:end-1])
        end

        @testset ":log clamps the lower end to the smallest positive value" begin
            bins = _bin_select(vcat(zeros(30), collect(1.0:70.0)), 5, :log)
            @test length(bins) == 5
            @test all(>(0), bins)
            @test issorted(bins)
            @test bins[1] >= 1.0
        end

        @testset ":log rejects data without positive values" begin
            @test_throws ArgumentError _bin_select(collect(-10.0:-1.0), 5, :log)
        end

        @testset "an unknown axis_uniform raises ArgumentError" begin
            @test_throws ArgumentError _bin_select(data, 5, :bogus)
        end
    end

    @testset "_bin_select end bins and ties" begin
        # ECDF values at the bins, as the summaries evaluate them
        ecdf_at(data, bins) = empcdf_raw(data, length(bins), collect(Float64, bins))

        @testset ":yax levels are equal-mass midpoints on continuous data" begin
            data = randn(StableRNG(5), 4000)
            nbin = 10
            bins = _bin_select(data, nbin, :yax)
            F = ecdf_at(data, bins)
            @test F ≈ ((1:nbin) .- 0.5) ./ nbin atol = 0.02
            @test all(f -> 0.0 < f < 1.0, F)
            @test minimum(data) <= bins[1] && bins[end] <= maximum(data)
        end

        @testset ":yax leaves bins of data without ties untouched by the tie pass" begin
            data = randn(StableRNG(6), 500)
            a, b = _calculate_bin_bounds(data)
            grid = collect(range(a, b, length = 1000))
            cdf = empcdf_raw(data, 1000, grid)
            @test _bin_select(data, 8, :yax) == invcdf(grid, cdf, 8, 1)
        end

        @testset ":yax on tied data gives distinct ECDF values" begin
            # A staircase ECDF with one tall step: two invcdf levels fall inside the jump at 3
            data = vcat(fill(0.0, 5), fill(1.0, 10), fill(2.0, 10), fill(3.0, 40),
                        fill(4.0, 15), fill(5.0, 12), fill(6.0, 8))
            bins = _bin_select(data, 5, :yax)
            F = ecdf_at(data, bins)
            @test bins == [1.0, 2.0, 3.0, 4.0, 5.0]       # snapped onto data values, never the maximum
            @test allunique(F)
            @test all(f -> 0.0 < f < 1.0, F)

            # Counts with a large atom at zero
            counts = vcat(zeros(200), Float64.(rand(StableRNG(7), 1:60, 300)))
            F = ecdf_at(counts, _bin_select(counts, 10, :yax))
            @test allunique(F)
            @test all(f -> 0.0 < f < 1.0, F)
        end

        @testset ":yax with fewer distinct values than bins warns and keeps nbin bins" begin
            data = Float64.(repeat(0:3, 25))
            bins = @test_logs (:warn,) match_mode = :any _bin_select(data, 10, :yax)
            @test length(bins) == 10
            @test issorted(bins)
            @test all(<(3.0), bins)                       # the maximum is never a bin
        end

        @testset ":xax end bins are inside the data" begin
            data = randn(StableRNG(8), 1000)
            F = ecdf_at(data, _bin_select(data, 10, :xax))
            @test 0.0 < F[1] && F[end] < 1.0
        end

        @testset "non-finite values are dropped" begin
            data = vcat(collect(1.0:50.0), [NaN, Inf, -Inf])
            @test _bin_select(data, 5, :yax) == _bin_select(collect(1.0:50.0), 5, :yax)
            @test_throws ArgumentError _bin_select([NaN, Inf], 5, :yax)
        end
    end

    @testset "Diff summaries take bins from the differences" begin
        # Regression for 59b9a3e: StandardECDFDiff bins were taken from the raw observations.
        data = wiggly(200) .* 10 .+ 500                  # raw values far from the derivative scale
        for order in (1, 2)
            stat, dc, buffers, _ = prepare_summary(data, StandardECDFDiff(6, order, 1.0))
            d = vec(dc.differences[order][:, buffers.index_cache])
            bins = stat.bins[1]
            @test minimum(d) <= bins[1] && bins[end] <= maximum(d)
            F = empcdf_raw(d, 6, bins)
            @test all(f -> 0.0 < f < 1.0, F)
            @test bins != _bin_select(vec(data), 6, :xax)
        end

        @testset "multidimensional Diff bins each row of the differences" begin
            data2 = vcat(wiggly(200; seed = 3) .* 10 .+ 500, wiggly(200; seed = 4) .- 300)
            stat, dc, buffers, _ = prepare_summary(data2, StandardECDFDiff(6, 2, 1, 1.0))
            @test stat isa StandardECDFDiffMultiDimensional
            for ii in 1:2
                d = dc.differences[1][ii, buffers.index_cache]
                @test minimum(d) <= stat.bins[ii][1] && stat.bins[ii][end] <= maximum(d)
            end
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
