#------------MCMCState struct
struct MCMCState{T, M<:AbstractMatrix{<:Real}}
    current_params::T
    ss_current::Float64
    proposal_cov::M
end

struct ResultsBuffer{M<:AbstractMatrix{<:Real}, I, T}
    chain::M
    sschain::Vector{Float64}
    prop_cov::M
    current_iter::I
    acceptance::T
    stuck_kicks::I
end

function allocate_results_buffer( npar, chain_length, n_stages )
    chain_buffer = zeros( npar, chain_length )
    ss_buffer = zeros( chain_length )
    prop_cov = Matrix{Float64}( I, npar, npar )
    results_buffer = ResultsBuffer(
        chain_buffer,
        ss_buffer,
        prop_cov,
        Ref(0),
        zeros( Int, n_stages ),
        Ref(0),
    )
    return results_buffer
end

function mcmcrun( target::TargetData, model::AbstractSimulationModel, mcmc_options::MCMCOptions )

    # Setup
    MCMCRun = mcmc_options.mcmc_algorithm
    proposal_width = MCMCRun.proposal_width

    chain_length = mcmc_options.nsteps

    active_params_names = model.active_parameters
    active_param_values = [ getfield( model, param ) for param in active_params_names ]

    npar_active = length( active_param_values )

    # Set uninformative priors if not provided
    if length( target.priors ) != npar_active
        uninformative_priors = ntuple( _ -> nothing, Val(npar_active) )
        target = @set target.priors = uninformative_priors
    end

    if isnothing( mcmc_options.initial_params )
        current_params = active_param_values .+ 0.1 .* randn( npar_active )
    else
        current_params = mcmc_options.initial_params
    end

    proposal_cov = Matrix{Float64}( I, npar_active, npar_active )*proposal_width
    ss_current = calculate_loss( current_params, target, model, mcmc_options )
    ss_current += evaluate_log_prior( current_params, target.priors )

    n_stages = isa( MCMCRun, DRAM) ? MCMCRun.n_stages : 1
    results_buffers = allocate_results_buffer( npar_active, chain_length, n_stages )

    state = MCMCState( current_params, ss_current, proposal_cov )

    try
        results, state = MCMCRun( target, model, state, mcmc_options, results_buffers )
        results = @set results.acceptance = results.acceptance ./ chain_length
        results = @set results.prop_cov = state.proposal_cov

        acceptance_print = round.( results.acceptance.*100, digits=2 )
        stuck_print_percentage = round.( results.stuck_kicks.*100 ./ chain_length, digits=2 )
        println("MCMC completed")
        println("Acceptance rate: ", acceptance_print, "%")
        println( "Stuck kicks: ", stuck_print_percentage, "%\n")

        return results, state
    catch e
        last_successful_step = results_buffers.current_iter[]
        partial_chain = results_buffers.chain[:, 1:last_successful_step]
        partial_sschain = results_buffers.sschain[1:last_successful_step]
        acceptance = results_buffers.acceptance

        results = ResultsBuffer(
            partial_chain,
            partial_sschain,
            state.proposal_cov,
            last_successful_step,
            acceptance ./ last_successful_step,
            results_buffers.stuck_kicks,
            )

        acceptance_print = round.( results.acceptance.*100, digits=2 )
        stuck_print_percentage = round.( results.stuck_kicks.*100 ./ chain_length, digits=2 )

        println("\nMCMC crashed at iteration ", last_successful_step, " due to: \n", e)
        println("Returning partial chain.")
        println("Acceptance rate: ", acceptance_print, "%")
        println( "Stuck kicks: ", stuck_print_percentage, "%\n")

        Base.showerror(stdout, e, catch_backtrace())
        println() # Add a newline for readability

        return results, state
    end
end
