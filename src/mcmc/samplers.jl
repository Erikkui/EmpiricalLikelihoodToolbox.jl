#------------MCMCState struct
struct MCMCState{T, M<:AbstractMatrix{<:Real}}
    current_params::T
    ss_current::Float64
    proposal_cov::M
    acceptance_rate::Float64
    stuck_kicks::Int
end

Base.@kwdef struct AM
    proposal_width::Float64 = 0.01
    adaptation_interval::Int = 50
end

function (AM::AM)( target, model, options )
    # Initialization
    adaptation_interval = AM.adaptation_interval

    state = options.state
    chain_length = options.nsteps
    update_interval = options.update_interval
    loss = options.loss_function

    npar = get_params( model ) |> length
    chain = zeros( npar, chain_length )
    sschain = zeros( chain_length )

    am_scaling_parameter = 2.38^2 / npar
    epsilon = 1e-6

    noise_buffer = zeros( npar )
    params_proposal = zeros( npar )

    # Initial Cholesky
    proposal_cov = copy( state.proposal_cov )
    proposal_cov_L = cholesky( state.proposal_cov ).L

    # Bookkeeping
    n_accepted = 0
    n_stuck = 0

    println( "Starting MCMC with AM algorithm for ", chain_length, " iterations..." )
    iter = ProgressBar( 1:chain_length, printing_delay=0.01 )
    try
        for ii in iter

            # 1. Propose
            # params_proposal = state.current_params + proposal_cov_L * noise_buffer
            randn!( noise_buffer )
            mul!( params_proposal, proposal_cov_L, noise_buffer )
            params_proposal .+= state.current_params

            ss_proposal = calculate_loss( params_proposal, target, model, loss )

            # 2. Metropolis accept/reject
            log_ratio = ss_proposal - state.ss_current
            # println( "Old ss: ", state.ss_current, " New ss: ", ss_proposal, " Log Ratio: ", log_ratio )
            # println("\nState before update: ", state)

            rd = log( rand() )
            accepted = rd < log_ratio

            if accepted
                copyto!( state.current_params, params_proposal )
                state = @set state.ss_current = ss_proposal
                n_accepted += 1
                n_stuck = 0 # Reset stuck counter on move
            else
                # Rejected
                n_stuck += 1
                if n_stuck > update_interval
                    # If mcmc gets stuck in "too good" proposal
                    ss_recalc = calculate_loss( state.current_params, target, model, loss )
                    updated_fields = ( ss_current = ss_recalc, stuck_kicks = state.stuck_kicks + 1 )
                    state = setproperties( state, updated_fields )
                    n_stuck = 0 # Reset counter
                end
            end
            # println("State after update: ", state)
            # sleep(0.5)

            # Store
            chain[ :, ii ] = state.current_params
            sschain[ii] = state.ss_current

            # 3. Adaptive Update
            if ii > 100 && ii % adaptation_interval == 0
                proposal_cov = cov( chain[:, 1:ii]' ) .* am_scaling_parameter + epsilon * I
                proposal_cov_L = cholesky( Symmetric(proposal_cov) ).L
                state = @set state.proposal_cov = proposal_cov
            end

            description = "Acc: $(round(n_accepted/ii, digits=2)) SS: $(round(state.ss_current, digits=2))"
            set_description(iter, description)
        end
    catch e
        println( "Error occurred during MCMC:: ", e, "\nReturning results up to the last successful iteration.")

        # Remove cols from chain that are full zeros
        nonzero_inds = sum( abs.(chain), dims=1 ) .> 0
        chain = chain[ :, nonzero_inds[1, :] ]
        chain_length = size(chain, 2)
        sschain = sschain[ 1:chain_length ]

        # Save results
        results_new = Dict(
            :accept    => n_accepted / chain_length,
            :last      => chain[ :, end],
            :qcov      => proposal_cov,
            :chain     => chain,
            :sschain   => sschain
        )

        acceptance_rate = n_accepted / chain_length
        state = setproperties( state, ( acceptance_rate = acceptance_rate, stuck_kicks = n_stuck ) )

        # if !isnothing(results_prev)
        #     results_new[:chain] = vcat(results_prev[:chain], chain)
        # end
        println( "Acceptance rate: $(acceptance_rate).\nPercentage of stuck re-evaluations: ", n_stuck/chain_length )

        return results_new, state
    end

    # Save results
    results_new = Dict(
            :accept    => n_accepted / chain_length,
            :last      => chain[:, end],
            :qcov      => proposal_cov,
            :chain     => chain,
            :sschain   => sschain
        )

    acceptance_rate = n_accepted / chain_length
    state = setproperties( state, ( acceptance_rate = acceptance_rate, stuck_kicks = n_stuck ) )

    # if !isnothing(results_prev)
    #     results_new[:Chain] = vcat(results_prev[:Chain], chain)
    # end

    println( "MCMC finished with acceptance rate: $(acceptance_rate).\nPercentage of stuck re-evaluations: ", n_stuck/chain_length )
    return results_new, state
end
