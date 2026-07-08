# In VSCode terminal:
#   0. Start Julia REPL: ctrl+shift+p, "Julia: Start REPL"
#   1. ] activate .
#   2. Press backspace
#   3. using Revise
#   4. include("examples/main.jl")
#   5. run run_test_mcmc() in REPL: res = run_test_mcmc()

using EmpiricalLikelihoodToolbox
using Distributions
using CairoMakie

function run_test_mcmc()
    axis_unif = :yax
    nrep_training = 5000
    chain_length = 20000

    nbin = 10
    knn = 1

    n_loss_evals = 1
    n_summaries = 1

    # t_end = 60
    # dt_obs = 1.0
    # Ndata = t_end / dt_obs

    Ndata = 200
    dt_obs = 1.0
    active_parameters = ( :sigma, :rho, :beta )

    timeseries_block_size = 100
    standardize = true

    likelihood_noise_scale = 0.0

    # model = NegExpModel( dt_obs = dt_obs, theta1 = 1.0, theta2 = 0.1, noise_scale = 0.01 )
    model = Lorenz63Model( dt_obs = dt_obs, active_parameters = active_parameters )
    # model = BlowflyModel( dt_obs = dt_obs, active_parameters = active_parameters )
    # model = RickerModel( r = 3.7, K = 1000.0, x0 = 10.0, dt_obs = dt_obs )
    # model = PredatorModel( dt_obs=dt_obs)
    data = solve_model( model, Ndata*dt_obs )

    npar = length( get_active_model_params( model ) )
    default_params = collect( get_active_model_params( model ) )
    initial_params = default_params .+ (1 .+ 0.1 .* randn( npar ) )

    println( initial_params)
    sleep(3)


    ########
    # fig = Figure()
    # ax = Axis(fig[1, 1], xlabel="Time", ylabel="Observation")
    # xplot = collect( range( 0.0, stop = Ndata*dt_obs-dt_obs, length=Int(Ndata) ) )
    # lines!(ax, xplot, vec(data[1, :]))
    # lines!(ax, xplot, vec(data[2, :]), color=:red)
    # display(fig)
    # sleep(10)
    ########


    resampler = StandardResampling()
    # resampler = TimeseriesResampling()
    lossfun = LogLikelihood( scaling_parameter = 1.0)
    println(lossfun)
    sampler = AM( proposal_width = 0.01, adaptation_interval = 50 )
    # sampler = DRAM( proposal_width = 0.01, adaptation_interval = 50, n_stages = 2, proposal_scale = [1.0, 0.01] )
    priors = tuple( [Uniform(0.0, 10*default_params[i]) for i in 1:npar]... )


    summary_statistics = JointSummaryStatistics(
        # StandardECDF(10),
        # StandardECDFDiff(10, 1, dt_obs),
        # StandardECDFDiff(10, 2, dt_obs),
        ChamferDistance(1),
        # ChamferDistance(1:10)
        # ChamferECDF( 10, 1 )
        # CIL(10),
        # ID(10, 1:2)
        )

    methods_options = MethodsOptions(
        resampling_type=resampler,
        axis_uniform=axis_unif,
        training_resamplings=nrep_training,
        timeseries_block_size=timeseries_block_size,
        n_loss_evals = n_loss_evals,
        n_summaries = n_summaries,
        standardize = standardize,
        verbose = true
        )

    mcmc_options = MCMCOptions(
        initial_params = initial_params,
        nsteps = chain_length,
        mcmc_algorithm = sampler,
        update_interval = 30,
        loss_function = lossfun,
        discard_noisy_updates = false,
        likelihood_noise_scale = likelihood_noise_scale
        )

    target, training_summaries = TargetData( data, summary_statistics, methods_options; priors = priors, loss = lossfun )

    results, state = mcmcrun( target, model, mcmc_options )

    true_params = collect( get_active_model_params( model ) )
    npara = size(results.chain, 1)
    fig = Figure(size=(800, 200*npara))
    for i in 1:npara
        ax = Axis(fig[i, 1], xlabel="Iteration", ylabel="Parameter $i")
        lines!(ax, results.chain[i, :])
        hlines!(ax, [true_params[i]], color=:red, linestyle=:dash)
    end
    display(fig)

    # fig = Figure( size=(600, 600) )
    # ax = Axis(fig[1, 1], xlabel="Parameter 1", ylabel="Parameter 2")
    # scatter!( ax, results.chain[1, 5000:end], results.chain[2, 5000:end], markersize=6 )
    # scatter!( ax, [true_params[1]], [true_params[2]], markersize=10, color=:red )
    # display(fig)

    return results, state
end
