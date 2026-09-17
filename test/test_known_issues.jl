# Review section 22.8. Each @test_broken states the DESIRED behaviour of an issue that is still
# open. They report as Broken today; the moment one is fixed CI turns red, which is the signal to
# promote it to a plain @test.

@testset "known issues" begin

    data = wiggly(40)

    @testset "standardize = true is a silent no-op" begin
        # standardize! is a pure function, but loss_functions.jl:24 and :59 discard its return
        # value onto an immutable Float64, so the standardization never reaches the loss.
        target, _ = make_target(data, StandardECDF(5);
                                standardize = true, loss = LogLikelihood(), training_resamplings = 100)
        sim = target.obs_mean .+ 0.05
        d = sim - target.obs_mean
        unstandardized = -0.5 * dot(d, target.cov_factorization \ d)

        @test isfinite(target.standardization_mean)     # calibration happens...
        @test target.standardization_sd > 0
        @test_broken LogLikelihood()(target, sim) != unstandardized   # ...but is never applied
    end

    @testset "BlowflyModel can be constructed and updated" begin
        # x0::E now ties BlowflyModel{E} to a field, so E is inferred from the default and
        # Accessors' positional rebuild works -- the model is usable from MCMC.
        @test (try; BlowflyModel(); true; catch; false; end)
        @test (try
            update_model_parameters(BlowflyModel{Int}(), [0.2, 6.0, 400.0, 0.1, 14.0, 0.1])
            true
        catch; false; end)
    end

    @testset "PredatorModel is exported but undefined" begin
        @test :PredatorModel in names(EmpiricalLikelihoodToolbox)
        @test_broken isdefined(EmpiricalLikelihoodToolbox, :PredatorModel)
        @test_broken all(n -> isdefined(EmpiricalLikelihoodToolbox, n),
                         names(EmpiricalLikelihoodToolbox))
    end

    @testset "mcmcrun silently replaces a wrong-length prior tuple" begin
        # Review section 19: a scientific package should fail loudly here, e.g.
        # ArgumentError("Expected 2 priors for (:mu, :sigma), received 1").
        target, _ = make_target(data, StandardECDF(5);
                                priors = (mu = Normal(0.0, 1.0),),   # 1 prior, 2 active params
                                resampling_type = NoResampling(), training_resamplings = 20)
        opts = MCMCOptions(nsteps = 20, mcmc_algorithm = AM(), initial_params = [0.0, 1.0],
                           likelihood_noise_scale = 0.0)
        @test_broken (try
            mcmcrun(target, NormalModel(), opts)
            false
        catch e; e isa ArgumentError; end)
    end

    @testset "priors are paired with parameters positionally, not by name" begin
        # A differently ordered NamedTuple silently associates the wrong prior with the wrong
        # parameter, because evaluate_log_prior zips keys(priors) against params[ii] by position.
        params = [0.5, 5.0]
        in_order = (a = Uniform(0.0, 1.0), b = Uniform(0.0, 10.0))
        swapped  = (b = Uniform(0.0, 10.0), a = Uniform(0.0, 1.0))
        @test_broken evaluate_log_prior(params, in_order) == evaluate_log_prior(params, swapped)
    end

    @testset "invalid option symbols fail late and opaquely" begin
        @testset "axis_uniform" begin
            # _bin_select has no else branch, so `bins` is never assigned.
            @test_broken (try
                _bin_select(collect(1.0:50.0), 5, :bogus)
                false
            catch e; e isa ArgumentError; end)
        end

        @testset "covariance_type" begin
            # train_target leaves C === nothing, which only fails later inside regularized_cholesky.
            @test_broken (try
                make_target(data, StandardECDF(5); covariance_type = :bogus)
                false
            catch e; e isa ArgumentError; end)
        end
    end

    @testset "_bin_select(:log) throws on data whose robust lower bound is not positive" begin
        # _calculate_bin_bounds pads by 0.25*iqr, which pushes the lower bound of ordinary data
        # below zero; (b/a/1.01)^(1/nbin) then raises a negative number to a fractional power.
        @test_broken (try; _bin_select(collect(1.0:100.0), 5, :log); true; catch; false; end)
    end

    @testset "embedding mishandles multi-row matrices" begin
        # It takes Nobs from `length(data_in)` and indexes linearly, so a (2, N) matrix is treated
        # as one flat 2N-element sequence.
        M = vcat(row(1.0:10.0), row(11.0:20.0))
        @test size(embedding(M, [1])) == (2, 19)                 # current behaviour
        @test_broken size(embedding(M, [1]), 2) == size(M, 2) - 1  # desired: 9 columns
    end

    @testset "two summaries with the same generated name collide" begin
        # allocate_buffers keys summary_buffers by generate_stat_name, so duplicates cannot
        # coexist in the NamedTuple.
        @test_broken (try
            make_target(data, StandardECDF(5), StandardECDF(5))
            true
        catch; false; end)
    end

    @testset "target.options is a stale copy of target.data.options" begin
        # allocate_buffers writes effective_N_obs into the DataContainer's options, but TargetData
        # also stores the pre-trim options separately. Code reading target.options sees the
        # untrimmed count.
        target, _ = make_target(wiggly(40), StandardECDFDiff(5, 1, 1.0))
        @test target.data.options.effective_N_obs == 38          # post-trim, correct
        @test target.options.effective_N_obs == 40               # stale
        @test_broken target.options.effective_N_obs == target.data.options.effective_N_obs
    end

    @testset "ChamferECDFSummary is declared but unused" begin
        @test ChamferECDF <: AbstractECDFSummary
        @test_broken ChamferECDF <: TwoSetSummary
    end

    @testset "RobustChamfer's three-argument form ignores x_star" begin
        C = regularized_cholesky(cov(randn(StableRNG(61), 3, 20)'))
        x = [1.0, 2.0, 3.0]
        @test_broken RobustChamfer()(x, zeros(3), C) != RobustChamfer()(x, ones(3), C)
    end

    @testset "likelihood_noise_scale defaults to a NaN sentinel" begin
        # MCMCOptions defaults it to NaN and only mcmcrun rewrites it to 0.0, so calling
        # calculate_loss directly with default options poisons every loss with NaN.
        target, _ = make_target(data, StandardECDF(5);
                                resampling_type = NoResampling(), training_resamplings = 20)
        defaults = MCMCOptions(mcmc_algorithm = AM(), nsteps = 10)
        @test isnan(defaults.likelihood_noise_scale)
        @test_broken isfinite(calculate_loss([0.0, 1.0], target, NormalModel(), defaults))
    end

    @testset "NormalModel's first observation is the initial state, not a draw" begin
        # The generic solver writes x0 into column 1, so the first "sample" is deterministically
        # x0 regardless of mu and sigma -- despite the field being documented as unused.
        firsts = [solve_model(NormalModel(mu = 5.0, sigma = 2.0), 10.0; rng = StableRNG(s))[1, 1]
                  for s in 1:5]
        @test all(==(0.0), firsts)                  # current behaviour
        @test_broken length(unique(firsts)) > 1     # desired: every observation is an iid draw
    end

    @testset "examples/main.jl does not run as committed" begin
        # It passes embedding_dim to RickerModel, which has no such field, and additionally
        # depends on CairoMakie, which is not in Project.toml.
        @test_broken (try; RickerModel(embedding_dim = 0); true; catch; false; end)
    end

    @testset "ResultsBuffer still cannot hold an Int alongside a Ref" begin
        # current_iter::I and stuck_kicks::I share one type parameter, so the struct cannot express
        # the recovery path directly. mcmcrun works around this by wrapping the count in a Ref at
        # the call site; giving the two fields separate parameters would remove the need.
        @test_broken (try
            EmpiricalLikelihoodToolbox.ResultsBuffer(zeros(2, 3), zeros(3), zeros(2, 2),
                                                     3, zeros(1), Ref(0))
            true
        catch; false; end)
    end

    @testset "supplied bin edges survive TargetData" begin
        # create_bins now skips recomputation when the summary already carries bins (see
        # _has_existing_bins), so a caller-supplied bins vector is honoured through the full
        # pipeline, not just on a direct summarize() call.
        custom = [-5.0, -2.0, 0.0, 2.0, 5.0]

        direct = summarize(data, StandardECDF(custom); resampling_type = NoResampling())
        @test direct == empcdf_raw(data, length(custom), custom)   # honoured on a direct call

        for stat in (StandardECDF(custom), CIL(custom), ID(custom, 1))
            target, _ = make_target(data, stat; training_resamplings = 20)
            kept = target.summary_statistics.statistics[1].bins[1]
            @test kept == custom
        end
    end
end
