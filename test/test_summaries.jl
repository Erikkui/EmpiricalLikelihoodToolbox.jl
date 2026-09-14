# src/summaries/*.jl -- review section 22.1: every summary against a tiny hand-calculated
# dataset, covering neighbors=1, 3 and [1,3], odd N, multidimensional data and Diff variants.

@testset "summaries" begin

    NR = NoResampling()    # deterministic: x_inds == y_inds == the whole index cache

    @testset "hand-calculated values" begin

        @testset "StandardECDF" begin
            # data {1,2,3,4}, bins at 1.5/2.5/3.5 -> 1/4, 2/4, 3/4 of the mass below each.
            @test summarize(row(1:4), StandardECDF([1.5, 2.5, 3.5]); resampling_type = NR) ==
                  [0.25, 0.5, 0.75]
        end

        @testset "CumulativeSum" begin
            # cumsum([1 2 3 4]) = [1 3 6 10]; contracted in windows of 2 -> [1+3, 6+10] = [4, 16];
            # normalized by the final contracted value (16) -> [0.25, 1.0].
            @test summarize(row(1:4), CumulativeSum(2); resampling_type = NR) == [0.25, 1.0]
        end

        @testset "CIL" begin
            # Pairwise Euclidean distances of {1,2,3} against itself:
            #   [0 1 2; 1 0 1; 2 1 0] -> nine values {0,0,0,1,1,1,1,2,2}.
            # ECDF at 0, 1, 2 -> 3/9, 7/9, 9/9.
            @test summarize(row(1:3), CIL([0.0, 1.0, 2.0]); resampling_type = NR) ≈
                  [3 / 9, 7 / 9, 1.0]
        end

        @testset "ChamferDistance" begin
            # x_inds == y_inds under NoResampling, so each point's nearest neighbour is itself.
            @test summarize(row(1:6), ChamferDistance(1); resampling_type = NR) ≈ [0.0] atol = 1e-12

            # k=2 over the evenly spaced set {1..6}: the second-nearest neighbour is at distance 1
            # for the two endpoints and 1 for everyone else, so the mean is 1 in each direction.
            @test summarize(row(1:6), ChamferDistance([2]); resampling_type = NR) ≈ [2.0]
        end

        @testset "StandardECDFDiff reduces to a constant on a ramp" begin
            # A ramp of slope 2 differentiates to exactly 2 everywhere, so the ECDF is a step.
            ramp = row(2.0 .* (0:9))
            out = summarize(ramp, StandardECDFDiff([1.0, 3.0], 1, 1.0); resampling_type = NR)
            @test out == [0.0, 1.0]
        end
    end

    @testset "neighbour counts: 1, 3 and [1,3]" begin
        data = row(1.0:20.0)

        @testset "scalar neighbours give one output per summary" begin
            for k in (1, 3)
                @test length(summarize(data, ChamferDistance(k); resampling_type = NR)) == 1
            end
        end

        @testset "vector neighbours give one output per k" begin
            out = summarize(data, ChamferDistance([1, 3]); resampling_type = NR)
            @test length(out) == 2
            # Component k must equal the scalar summary for that same k.
            @test out[1] ≈ summarize(data, ChamferDistance(1); resampling_type = NR)[1]
            @test out[2] ≈ summarize(data, ChamferDistance(3); resampling_type = NR)[1]
            @test out[1] <= out[2]                      # non-decreasing in k
        end

        @testset "ECDF-based neighbour summaries scale their length by the k count" begin
            @test ChamferECDF(5, 1).summary_length == 5
            @test ChamferECDF(5, [1, 3]).summary_length == 10
            @test ID(5, 1).summary_length == 5
            @test ID(5, [1, 3]).summary_length == 10
        end
    end

    @testset "odd-length data" begin
        # Review section 13: odd N used to produce wrong buffer dimensions for CIL/ID.
        for n in (15, 21)
            for (stat, len) in ((CIL(4), 4), (ID(4, 1), 4), (ChamferDistance(1), 1))
                for sampler in (NR, RademacherSplit())
                    out = summarize(row(1.0:n), stat; resampling_type = sampler)
                    @test length(out) == len
                    @test all(isfinite, out)
                end
            end
        end
    end

    @testset "multidimensional data" begin
        data = vcat(row(1.0:20.0), row(21.0:40.0))

        @testset "StandardECDF(nbin, ndim) builds a multidimensional summary" begin
            stat = StandardECDF(5, 2)
            @test stat isa StandardECDFMultiDimensional
            @test stat.summary_length == 10                # ndim * nbin
        end

        @testset "ndim < 2 falls back to the one-dimensional summary" begin
            @test StandardECDF(5, 1) isa StandardECDF
            @test StandardECDFDiff(5, 1, 1, 1.0) isa StandardECDFDiff
        end

        @testset "two-set summaries handle multidimensional input" begin
            for stat in (CIL(4), ID(4, 1), ChamferDistance(1))
                out = summarize(data, stat; resampling_type = NR)
                @test all(isfinite, out)
            end
        end
    end

    @testset "Diff variants use derivatives, not raw data" begin
        ramp = row(3.0 .* (0:19))

        @testset "index cache is trimmed by the diff order" begin
            _, dc, buffers, _ = prepare_summary(ramp, StandardECDFDiff(4, 1, 1.0); resampling_type = NR)
            # calculate_diffs fabricates the boundary columns, so max_diff_order columns are
            # dropped from each end.
            @test buffers.index_cache == collect(2:19)
            @test dc.options.effective_N_obs == 18
        end

        @testset "derivative of a ramp is its slope" begin
            _, dc, _, _ = prepare_summary(ramp, StandardECDFDiff(4, 1, 1.0); resampling_type = NR)
            @test all(≈(3.0), dc.differences[1])
        end

        @testset "every Diff variant reports its diff order" begin
            @test required_diff_order(StandardECDFDiff(4, 2, 1.0)) == 2
            @test required_diff_order(CILDiff(4, 1, 1.0)) == 1
            @test required_diff_order(IDDiff(4, 1, 1, 1.0)) == 1
            @test required_diff_order(ChamferDistanceDiff(1, 1, 1.0)) == 1
            # Non-Diff variants require none.
            @test required_diff_order(StandardECDF(4)) == 0
            @test required_diff_order(CIL(4)) == 0
            @test required_diff_order(CumulativeSum(3)) == 0
        end
    end

    @testset "interface conformance" begin
        # The summary interface is informal -- nothing in the type system enforces it -- which is
        # exactly why it needs a test. Each entry is (constructor, number of data rows).
        cases = (
            (() -> StandardECDF(4), 1),
            (() -> StandardECDF([1.0, 2.0, 3.0]), 1),
            (() -> StandardECDFDiff(4, 1, 1.0), 1),
            (() -> StandardECDF(4, 2), 2),
            (() -> StandardECDFDiff(4, 2, 1, 1.0), 2),
            (() -> CIL(4), 1),
            (() -> CILDiff(4, 1, 1.0), 1),
            (() -> ID(4, 1), 1),
            (() -> ID(4, [1, 3]), 1),
            (() -> IDDiff(4, 1, 1, 1.0), 1),
            (() -> ChamferDistance(1), 1),
            (() -> ChamferDistance([1, 3]), 1),
            (() -> ChamferDistanceDiff(1, 1, 1.0), 1),
            (() -> ChamferECDF(4, 1), 1),
            (() -> CumulativeSum(3), 1),
        )

        for (make, nrows) in cases
            stat = make()
            data = nrows == 1 ? wiggly(24) : vcat(wiggly(24; seed = 1), wiggly(24; seed = 2))

            @testset "$(nameof(typeof(stat)))" begin
                @test required_diff_order(stat) isa Int
                @test generate_stat_name(stat) isa String
                @test stat isa AbstractSummaryStatistic

                target, _ = make_target(data, make(); resampling_type = RademacherSplit())
                finalized = target.summary_statistics.statistics[1]
                dc = target.data

                # The field is what JointSummaryStatistics slices with; the function is what
                # allocate_buffers sizes with. They must agree or the layout silently shifts.
                @test finalized.summary_length == get_summary_length(finalized, dc)
                @test get_summary_length(finalized, dc) isa Int
                @test get_summary_length(finalized, dc) > 0

                # A single evaluation must fill every slot it was given.
                out = fill(NaN, get_summary_length(finalized, dc))
                cache = target.buffers.index_cache
                x_inds, y_inds = dc.options.resampling_type(dc, dc.options, cache)
                calculate_summary_statistic!(out, finalized, x_inds, y_inds, dc, target.buffers)
                @test !any(isnan, out)
            end
        end
    end

    @testset "ID allocates two genuinely distinct distance buffers" begin
        # Regression for the review's critical aliasing bug: dist_buffer and dist_buffer_aux were
        # the same matrix, so sorting rows then columns destroyed the reverse-direction distances.
        for stat in (ID(4, 1), IDDiff(4, 1, 1, 1.0))
            _, _, buffers, _ = prepare_summary(wiggly(20), stat)
            buf = buffers.summary_buffers[Symbol(generate_stat_name(stat))]
            @test buf.dist_buffer !== buf.dist_buffer_aux
            @test size(buf.dist_buffer) == size(buf.dist_buffer_aux)
        end
    end

    @testset "JointSummaryStatistics" begin
        data = row(1.0:24.0)

        @testset "slurps varargs into a tuple" begin
            j = JointSummaryStatistics(StandardECDF(4), CumulativeSum(3))
            @test j.statistics isa Tuple
            @test length(j.statistics) == 2
        end

        @testset "concatenates component summaries in order" begin
            target, _ = make_target(data, StandardECDF(4), CumulativeSum(3);
                                    resampling_type = NR, inference_method = BSL(2))
            stats = target.summary_statistics.statistics
            cache = target.buffers.index_cache

            lengths = map(s -> get_summary_length(s, target.data), stats)
            @test target.summary_length == sum(lengths)

            joint = fill(NaN, target.summary_length)
            target.summary_statistics(joint, cache, cache, target.data, target.buffers)

            offset = 0
            for (s, len) in zip(stats, lengths)
                piece = fill(NaN, len)
                calculate_summary_statistic!(piece, s, cache, cache, target.data, target.buffers)
                @test joint[offset+1:offset+len] ≈ piece
                offset += len
            end
            @test offset == target.summary_length
        end
    end

    @testset "show prints a compact constructor-like form" begin
        @test string(StandardECDF(5)) == "StandardECDF(-, 5, 5)"
        @test occursin("CumulativeSum", string(CumulativeSum(3)))
    end
end
