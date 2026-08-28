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
    covariance_type = :cov
    use_ecdf_sampling = false

    nrep_training = 5000
    chain_length = 30000

    nbin = 10
    knn = 1

    n_loss_evals = 10
    n_summaries = 1

    Ndata = 50
    dt_obs = 1.0
    t_end = Ndata * dt_obs

    timeseries_block_size = 100
    standardize = true
    reevaluate_current_loss = true

    likelihood_noise_scale = 0.0

    model = RickerModel( embedding_dim = 2 )
    data = solve_model( model, t_end )

    _, default_params = get_active_model_params( model )
    npar = length( default_params )
    initial_params = default_params .+ (1 .+ 0.1 .* randn( npar ) )
    initial_params = exp.( [2.8, -2.3, 1.79] )

    #######
    # fig = Figure()
    # ax = Axis(fig[1, 1], xlabel="Time", ylabel="Observation")
    # xplot = collect( range( 0.0, stop = Ndata*dt_obs-dt_obs, length=Int(Ndata) ) )
    # lines!(ax, xplot, vec(data[1, :]))
    # # lines!(ax, xplot, vec(data[2, :]), color=:red)
    # display(fig)
    # sleep(3)
    #######


    resampler = StandardResampling()
    # resampler = TimeseriesResampling()

    lossfun = LogLikelihood( scaling_parameter = 1.0 )

    sampler = AM( proposal_width = 0.01, adaptation_interval = 50 )
    # sampler = DRAM( proposal_width = 0.01, adaptation_interval = 50, n_stages = 2, proposal_scale = [1.0, 0.01] )

    param_names, _ = get_active_model_params( model )
    prior_distributions = tuple( [Uniform(0.0, 10e6) for i in 1:npar]... )
    prior_distributions = (
        Uniform( exp(2.0), exp(5.0) ),
        Uniform( exp(-3.0), exp(-0.22) ),
        Uniform( exp(1.61), exp(3.0) ),
    )
    priors = NamedTuple{ param_names }( prior_distributions)

    summary_statistics = JointSummaryStatistics(
        StandardECDF( 10, 2 ),
        )


    methods_options = MethodsOptions(
        N_obs = Ndata,
        resampling_type=resampler,
        covariance_type=covariance_type,
        axis_uniform=axis_unif,
        use_ecdf_sampling=use_ecdf_sampling,
        training_resamplings=nrep_training,
        n_loss_evals = n_loss_evals,
        n_summaries = n_summaries,
        standardize = standardize,
        verbose = true
        )

    mcmc_options = MCMCOptions(
        initial_params = initial_params,
        nsteps = chain_length,
        mcmc_algorithm = sampler,
        reevaluate_current_loss = reevaluate_current_loss,
        update_interval = 30,
        loss_function = lossfun,
        discard_noisy_updates = false,
        likelihood_noise_scale = likelihood_noise_scale
        )

    target, training_summaries = TargetData( data, summary_statistics, methods_options; priors = priors, loss = lossfun )

    results, state = mcmcrun( target, model, mcmc_options )

    fig = Figure(size=(800, 200*npar))
    for i in 1:npar
        ax = Axis(fig[i, 1], xlabel="Iteration", ylabel="Parameter $i")
        lines!(ax, results.chain[i, :])
        hlines!(ax, [default_params[i]], color=:red, linestyle=:dash)
    end
    display(fig)

    fig = Figure( size = ( 300*(npar-1), 300*(npar-1) ) )
    row = 1
    col = 1
    for ii in 1:npar
        for jj in col+1:npar
            ax = Axis(fig[row, col], xlabel="Parameter $ii", ylabel="Parameter $jj")
            scatter!(ax, results.chain[ii, :], results.chain[jj, :], markersize=2.0)
            row += 1
        end
        row = 1
        col += 1
    end
    display(fig)

    return results, state
end
