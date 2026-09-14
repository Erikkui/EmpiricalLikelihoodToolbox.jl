# src/mcmc/target.jl -- review section 22.4: empirical mean/covariance against independently
# calculated reference results.

@testset "TargetData" begin

    data = wiggly(40)
    reg(C) = C + 1e-2 * Diagonal(diag(C)) + 1e-6 * I

    @testset "target mean and covariance match the training summaries" begin
        # TargetData hands back the training summaries it used, so the reference can be recomputed
        # exactly. No RNG control needed: whatever the resampler drew, these identities must hold.
        target, tsum = make_target(data, StandardECDF(5); training_resamplings = 200)

        @test size(tsum) == (target.summary_length, 200)
        @test target.obs_mean ≈ vec(mean(tsum, dims = 2))
        @test Matrix(target.cov_factorization) ≈ reg(cov(tsum'))
    end

    @testset "NoResampling + GSL gives an exactly degenerate covariance" begin
        # Every training draw uses the same fixed indices, so all columns are identical, the
        # empirical covariance is exactly zero, and only the ridge term survives.
        target, tsum = make_target(data, StandardECDF(5);
                                   resampling_type = NoResampling(), training_resamplings = 25)
        L = target.summary_length

        @test all(col -> col ≈ tsum[:, 1], eachcol(tsum))
        @test cov(tsum') ≈ zeros(L, L) atol = 1e-15
        @test Matrix(target.cov_factorization) ≈ 1e-6 * Matrix(I, L, L)
        @test target.obs_mean ≈ tsum[:, 1]
    end

    @testset "NoResampling + BSL takes a single deterministic pass" begin
        target, tsum = make_target(data, StandardECDF(5);
                                   resampling_type = NoResampling(),
                                   inference_method = BSL(4), training_resamplings = 25)

        @test size(tsum, 2) == 1                       # the training loop is skipped entirely
        @test target.obs_mean ≈ vec(tsum)
        # The reference summary is just the summary of all the data.
        @test target.obs_mean ≈ summarize(data, StandardECDF(5); resampling_type = NoResampling())
    end

    @testset "BSL with a resampler reuses GSL's training loop" begin
        target, tsum = make_target(data, StandardECDF(5);
                                   inference_method = BSL(4), training_resamplings = 60)
        @test size(tsum, 2) == 60
        @test target.obs_mean ≈ vec(mean(tsum, dims = 2))
        # BSL's covariance comes from simulations at MCMC time; here it is only a sized placeholder.
        @test Matrix(target.cov_factorization) ≈ Matrix(I, target.summary_length, target.summary_length)
    end

    @testset "covariance_type = :donsker" begin
        target, tsum = make_target(data, StandardECDF(5);
                                   covariance_type = :donsker, training_resamplings = 100)
        expected = donsker_covariance(vec(mean(tsum, dims = 2)), size(data, 2))
        @test Matrix(target.cov_factorization) ≈ reg(expected)
    end

    @testset "index cache trimming" begin
        # RademacherSplit shuffles the cache in place during training, so after construction only
        # the cache's *contents* are stable, not its order. NoResampling leaves the order intact.
        @testset "no derivatives keeps every index" begin
            target, _ = make_target(data, StandardECDF(5))
            @test sort(target.buffers.index_cache) == collect(1:40)
            @test target.data.options.effective_N_obs == 40

            target_nr, _ = make_target(data, StandardECDF(5); resampling_type = NoResampling())
            @test target_nr.buffers.index_cache == collect(1:40)
        end

        @testset "derivatives drop max_diff_order columns from each end" begin
            # calculate_diffs fabricates boundary columns; each order propagates that one column
            # further inward, so trimming d from each end excludes every fabricated point.
            for d in (1, 2)
                target, _ = make_target(wiggly(40), StandardECDFDiff(5, d, 1.0))
                @test sort(target.buffers.index_cache) == collect(d+1:40-d)
                @test target.data.options.effective_N_obs == 40 - 2d

                target_nr, _ = make_target(wiggly(40), StandardECDFDiff(5, d, 1.0);
                                           resampling_type = NoResampling())
                @test target_nr.buffers.index_cache == collect(d+1:40-d)
            end
        end
    end

    @testset "buffer shapes" begin
        n_train, n_summ = 80, 3

        @testset "GSL" begin
            target, _ = make_target(data, StandardECDF(5), CumulativeSum(4);
                                    training_resamplings = n_train, n_summaries = n_summ)
            b = target.buffers
            L = target.summary_length

            @test size(b.training_buffer) == (L, n_train)
            @test size(b.mcmc_buffer) == (L, n_summ)
            @test length(b.simulation_statistic) == L
            @test size(b.bsl_buffer) == (0, 0)             # unused under GSL
            @test L == sum(s -> get_summary_length(s, target.data),
                           target.summary_statistics.statistics)
        end

        @testset "BSL sizes bsl_buffer by n_sim" begin
            target, _ = make_target(data, StandardECDF(5);
                                    inference_method = BSL(7), training_resamplings = n_train)
            @test size(target.buffers.bsl_buffer) == (target.summary_length, 7)
        end

        @testset "simulation_diffs has max_diff_order + 2 entries" begin
            target, _ = make_target(wiggly(40), StandardECDFDiff(5, 2, 1.0))
            d = target.buffers.simulation_diffs
            @test length(d) == 4                            # 2 orders + 2 scratch
            @test size(d[1]) == (0, 0)                      # order 1 was not requested
            @test size(d[2]) == (1, 40)
            @test size(d[end-1]) == (1, 40) && size(d[end]) == (1, 40)
        end

        @testset "no derivatives means no difference buffers" begin
            target, _ = make_target(data, StandardECDF(5))
            @test isempty(target.buffers.simulation_diffs)
        end
    end

    @testset "summary buffers are keyed by generate_stat_name" begin
        target, _ = make_target(data, StandardECDF(5), CumulativeSum(4))
        keys_found = keys(target.buffers.summary_buffers)
        @test Symbol("StandardECDF_k=5") in keys_found
        @test Symbol("CumulativeSum_l=4") in keys_found
    end

    @testset "standardization" begin
        @testset "calibrates from the spread of training losses" begin
            loss = LogLikelihood()
            target, tsum = make_target(data, StandardECDF(5);
                                       standardize = true, loss = loss, training_resamplings = 100)
            losses = [loss(col, target.obs_mean, target.cov_factorization) for col in eachcol(tsum)]
            @test target.standardization_mean ≈ mean(losses)
            @test target.standardization_sd ≈ std(losses; corrected = false)
        end

        @testset "is rejected under NoResampling" begin
            # Every training draw is identical, so there is no variability to standardize against.
            @test_throws ArgumentError make_target(data, StandardECDF(5);
                                                   resampling_type = NoResampling(), standardize = true)
        end

        @testset "is nothing when not requested" begin
            target, _ = make_target(data, StandardECDF(5))
            @test isnothing(target.standardization_mean)
            @test isnothing(target.standardization_sd)
        end
    end

    @testset "priors default to a one-element nothing tuple" begin
        target, _ = make_target(data, StandardECDF(5))
        @test target.priors == (nothing,)

        priors = (a = Normal(0.0, 1.0),)
        target2, _ = make_target(data, StandardECDF(5); priors = priors)
        @test target2.priors === priors
    end

    @testset "summaries are finalized during construction" begin
        # CumulativeSum arrives with summary_length 0 and a NaN normalization factor; TargetData
        # must fill both in, or JointSummaryStatistics would slice a zero-width view.
        raw = CumulativeSum(4)
        @test raw.summary_length == 0 && isnan(raw.normalization_factor)

        target, _ = make_target(data, CumulativeSum(4); resampling_type = NoResampling())
        finalized = target.summary_statistics.statistics[1]
        @test finalized.summary_length == 10            # 40 observations / window 4
        @test isfinite(finalized.normalization_factor)
        @test !isnothing(finalized.buffer)
    end
end
