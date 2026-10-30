# src/inference/{GSL,BSL}.jl -- the per-method interface, the show strings and the option checks.

@testset "inference methods" begin

    data = wiggly(40)

    @testset "constructors" begin
        @test GSL().loss == LogLikelihood()
        @test GSL(loss = RobustChamfer(2.0)).loss == RobustChamfer(2.0)

        @test BSL(500).n_sim == 500
        @test BSL(n_sim = 500).n_sim == 500
        @test BSL().n_sim == 10
        @test BSL(500).loss == LogLikelihood()
        @test BSL(500; loss = LogLikelihood(2.0)).loss == LogLikelihood(2.0)
        @test BSL(500, LogLikelihood(2.0)) == BSL(500; loss = LogLikelihood(2.0))
    end

    @testset "show strings are stable (used in file names)" begin
        @test string(GSL()) == "GSL()"
        @test string(BSL(500)) == "BSL(500)"
        @test string(BSL(n_sim = 500)) == "BSL(500)"
        @test string(BSL()) == "BSL(10)"
        @test string(GSL(loss = LogLikelihood())) == "GSL()"
        @test string(GSL(loss = RobustChamfer())) == "GSL(RobustChamfer(1.0))"
        @test string(GSL(loss = LogLikelihood(2.0))) == "GSL(LogLikelihood(2.0))"
        @test string(BSL(500; loss = LogLikelihood(2.0))) == "BSL(500, LogLikelihood(2.0))"
    end

    @testset "BSL rejects RobustChamfer" begin
        @test_throws ArgumentError BSL(5; loss = RobustChamfer())
        @test_throws ArgumentError BSL(5, RobustChamfer())
        @test_throws ArgumentError BSL(; n_sim = 5, loss = RobustChamfer())
    end

    @testset "standardize" begin
        @testset "throws under BSL" begin
            @test_throws ArgumentError make_target(data, StandardECDF(5); standardize = true,
                                                   inference_method = BSL(5))
            @test_throws ArgumentError make_target(data, StandardECDF(5); standardize = true,
                                                   inference_method = BSL(5),
                                                   resampling_type = NoResampling())
        end

        @testset "throws under GSL with NoResampling, works with a resampler" begin
            @test_throws ArgumentError make_target(data, StandardECDF(5); standardize = true,
                                                   resampling_type = NoResampling())
            target, _ = make_target(data, StandardECDF(5); standardize = true,
                                    inference_method = GSL(loss = LogLikelihood(2.0)))
            @test !isnothing(target.standardization_sd)
        end
    end

    @testset "allocate_method_buffer" begin
        @test size(EmpiricalLikelihoodToolbox.allocate_method_buffer(GSL(), 6)) == (0, 0)
        @test size(EmpiricalLikelihoodToolbox.allocate_method_buffer(BSL(7), 6)) == (6, 7)
    end

    @testset "the method's loss is used by evaluate_log_target" begin
        model = NormalModel(mu = 0.0, sigma = 1.0)
        opts = MCMCOptions(mcmc_algorithm = AM(), nsteps = 5)
        seed = UInt64(31)
        values = map((1.0, 2.0)) do scaling
            target, _ = make_target(data, StandardECDF(5); resampling_type = NoResampling(),
                                    inference_method = GSL(loss = LogLikelihood(scaling)))
            evaluate_log_target([0.0, 1.0], target, model, opts; rng_seed = seed)
        end
        @test values[2] ≈ values[1] / 2
    end

    @testset "deprecated TargetData(data, stats, options, loss, model)" begin
        model = NormalModel()
        opts = make_options(; N_obs = 40, resampling_type = NoResampling())
        stats() = JointSummaryStatistics(StandardECDF(5))

        old_target, _ = @test_deprecated TargetData(data, stats(), opts, RobustChamfer(2.0), model)
        @test old_target.options.inference_method == GSL(loss = RobustChamfer(2.0))

        new_opts = make_options(; N_obs = 40, resampling_type = NoResampling(),
                                  inference_method = GSL(loss = RobustChamfer(2.0)))
        new_target, _ = TargetData(data, stats(), new_opts, model)
        @test old_target.obs_mean == new_target.obs_mean

        bsl_opts = make_options(; N_obs = 40, resampling_type = NoResampling(),
                                  inference_method = BSL(3))
        bsl_target, _ = @test_deprecated TargetData(data, stats(), bsl_opts, LogLikelihood(2.0), model)
        @test bsl_target.options.inference_method == BSL(3; loss = LogLikelihood(2.0))
        @test size(bsl_target.buffers.method_buffer, 2) == 3
    end
end
