# In VSCode terminal:
#   0. Strt Julia REPL
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
    chain_length = 10000

    n_loss_evals = 1
    n_summaries = 1

    Ndata = 400
    dt_obs = 1.0
    timeseries_block_size = 100

    model = BlowflyModel( dt_obs = dt_obs, burn_in = 100 )
    data = solve_model( model, Ndata*dt_obs )

    npar = length( get_params( model ) )
    default_params = get_params( model )
    initial_params = default_params .+ 0.01 .* abs.(randn( npar ))

    # fig = Figure()
    # ax = Axis(fig[1, 1], xlabel="Time", ylabel="Observation")
    # lines!(ax, collect(1:Ndata), vec(data))
    # display(fig)
    # sleep(10)

    # resampler = StandardResampling()
    resampler = TimeseriesResampling()
    lossfun = LogLikelihood( scaling_parameter = 1.0 )
    sampler = AM( proposal_width = 0.01, adaptation_interval = 50 )
    # sampler = DRAM( proposal_width = 0.01, adaptation_interval = 50, n_stages = 2, proposal_scale = [1.0, 1/100] )
    priors = tuple( [Uniform(0.0, 10*default_params[i]) for i in 1:npar]...)


    summary_statistics = JointSummaryStatistics(
        StandardECDF(10),
        StandardECDFDiff(10, 1, dt_obs),
        ChamferDistance( 1:10 )
        )

    methods_options = MethodsOptions(
        resampling_type=resampler,
        axis_uniform=axis_unif,
        training_resamplings=nrep_training,
        timeseries_block_size=timeseries_block_size,
        n_loss_evals = n_loss_evals,
        n_summaries = n_summaries,
        verbose = true
        )

    mcmc_options = MCMCOptions(
        initial_params = initial_params,
        nsteps = chain_length,
        mcmc_algorithm = sampler,
        update_interval = 30,
        loss_function = lossfun,
        discard_noisy_updates = false,
        )

    target, training_summaries = TargetData( data, summary_statistics, methods_options; priors = priors )

    results, state = mcmcrun( target, model, mcmc_options )

    true_params = get_params( model )
    npara = size(results.chain, 1)
    fig = Figure(size=(1200, 200*npara))
    for i in 1:npara
        ax = Axis(fig[i, 1], xlabel="Iteration", ylabel="Parameter $i")
        lines!(ax, results.chain[i, :])
        hlines!(ax, [true_params[i]], color=:red, linestyle=:dash)
    end
    display(fig)

    return results, state
end
