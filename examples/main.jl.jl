# In VSCode terminal:
#   0. Strt Julia REPL
#   1. ] activate .
#   2. Press backspace
#   3. ] add EmpiricalLikelihoodToolbox
#   4. include("examples/main.jl")
#   5. run run_test_mcmc(): res = run_test_mcmc()
using CairoMakie


function run_test_mcmc()
    axis_unif = :yax
    nrep_training = 200
    nrep_sampling = nrep_training
    chain_length = 300

    model = Lorenz63Model( dt = dt, x0 = x0 )
    resampler = StandardResampling()

    summary_statistics = JointSummaryStatistics(
        ChamferECDF( 10, 1 ),
        )

    mcmc_options = MCMCOptions( nsteps = chain_length, mcmc_algorithm = AM( 0.01, 50 ) )

    options = MethodsOptions(
        resampling_type=resampler,
        axis_uniform=axis_unif,
        training_resamplings=nrep_training,
        mcmc_resamplings=nrep_sampling )

    data = solve_model( model, Ndata*dt )

    results, state = mcmcrun( data, summary_statistics, mcmc_options )

    npara = size(results[:chain], 1)
    fig = Figure(size=(1200, 300*npara))
    for i in 1:npara
        ax = Axis(fig[i, 1], xlabel="Iteration", ylabel="Parameter $i")
        lines!(ax, results[:chain][i, :])
    end
    save("mcmc_chains.png", fig)
    fig

    return results
end
