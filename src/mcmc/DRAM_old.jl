function log_alpha_eval!( alphas, stage, step, log_post_values, log_proposaldist_values )

    log_alpha_eval = 0.0

    if stage == 2   # First delayed rejection stage: pi(y_1) / pi(y_2)
        alpha_step = min( 0.0, log_post_values[2] - log_post_values[3] )
        alphas[2] = alpha_step
        log_alpha_eval = log1mexp( alpha_step )  # log( 1 - exp( alpha_step ) )
        return log_alpha_eval
    end

    if step == 1
        alpha_step = min( 0.0, log_post_values[ stage ] - log_post_values[ stage + 1 ] )
        alphas[ sum( 1:(stage-1) ) + 1 ] = alpha_step
        log_alpha_eval!( alphas, stage, step+1, log_post_values, log_proposaldist_values )

        alpha_inds_start = sum( 1:(stage-1) ) + 1
        alpha_inds_end = sum( 1:stage ) - 1
        stage_alphas = @view alphas[ alpha_inds_start:alpha_inds_end ]
        return log_alpha_eval = sum( a -> log1mexp(a), stage_alphas )
    else
        if step < stage

            N_temp = log_post_values[ stage + 1 - step]
            D_temp = log_post_values[ stage + 1 ]
            for kk in 1:step-1
                # delta_N = log_proposaldist_values[ stage ] - log_proposaldist_values[ stage-kk ]    # q_(kk-1), nominator
                # delta_D = log_proposaldist_values[ stage ] - log_proposaldist_values[ stage+1 ]         # q_(kk-1), denominator
                N_ind = sum( 1:( stage-step-kk+1 ) ) + kk
                D_ind = sum( 1:( stage-1 ) ) + kk

                log_proposaldist_N = log_proposaldist_values[ N_ind ]
                alpha_N = -alphas[ N_ind ]
                N_temp += log_proposaldist_N + log1mexp( alpha_N )

                log_proposaldist_D = log_proposaldist_values[ D_ind ]
                alpha_D = -alphas[ D_ind ]
                D_temp += log_proposaldist_D + log1mexp( alpha_D )
            end
            alphas[ sum( 1:(stage-1) ) + step ] = N_temp - D_temp

            log_alpha_eval!( alphas, stage, step+1, log_post_values, log_proposaldist_values )
        end
    end


    return log_alpha_eval
end

function log_proposal_eval!( q_values, stage, proposals, mv_normals )
    ind = sum( 1:stage-1 ) + 1
    stage_proposal = proposals[ stage + 1 ]
    log_prob = 0.0
    for kk in 1:stage
        distribution = mv_normals[kk]
        y_kk = proposals[ end - kk ]
        prob_val_kk = logpdf( distribution, stage_proposal - y_kk )
        q_values[ind] = prob_val_kk
        log_prob += prob_val_kk
    end
    return log_prob
end

function delayed_rejection_stage(
    stage,
    state,
    target,
    model,
    loss,
    params_proposal,
    proposal_cov_L,
    dram_buffers
    )

    # Unpack buffers for readability
    N_values = dram_buffers.N_values
    D_values = dram_buffers.D_values
    alphas = dram_buffers.alphas
    log_proposaldist_values = dram_buffers.log_proposaldist_values
    log_post_values = dram_buffers.log_post_values
    proposals = dram_buffers.proposals
    mv_normals = dram_buffers.mv_normals
    gamma_values = dram_buffers.gamma_values
    noise_buffer = dram_buffers.noise_buffer

    randn!( noise_buffer )
    noise_buffer .*= sqrt( gamma_values[ stage ] )  # Scale the noise for the next proposal

    # Generate the next proposal. We do not condition on th previous rejected
    # proposal; however, we could consider a more complex proposal that does so in future iterations.
    mul!( params_proposal, proposal_cov_L, noise_buffer )
    params_proposal .+= state.current_params
    proposals[ stage + 1 ] .= params_proposal    # Store proposal

    ss_proposal = calculate_loss( params_proposal, target, model, loss )
    ss_proposal += evaluate_log_prior( params_proposal, target.priors )
    log_post_values[ stage + 1 ] = ss_proposal

    # Nominator
    log_proposal_path = log_proposal_eval!( log_proposaldist_values, stage, proposals, mv_normals )
    log_accept_path = log_alpha_eval!( alphas, stage, 1, log_post_values, log_proposaldist_values )
    log_N_kk = log_proposal_path + log_accept_path
    N_values[ stage ] = log_N_kk

    # Denominator
    a = D_values[ stage - 1 ]
    b = N_values[ stage - 1 ]
    pdf_val_kk = log_proposaldist_values[ sum( 1:stage ) ]
    log_D_kk = pdf_val_kk + ( a + log1mexp( b - a ) )
    D_values[ stage ] = log_D_kk

    # Final acceptance probability for this stage
    log_ratio = log_N_kk - log_D_kk
    alphas[ sum( 1:stage ) ] = log_ratio

    return log_ratio, ss_proposal
end

Base.@kwdef struct DRAM{P}
    proposal_width::Float64 = 0.01
    adaptation_interval::Int = 50
    n_stages::Int = 3
    proposal_scale::P = 0.5
end

struct DRAMBuffers{T}
    N_values::Vector{Float64}
    D_values::Vector{Float64}
    alphas::Vector{Float64}
    log_proposaldist_values::Vector{Float64}
    log_post_values::Vector{Float64}
    proposals::Vector{Vector{Float64}}
    mv_normals::Vector{T}
    gamma_values::Vector{Float64}
    noise_buffer::Vector{Float64}
end


function (DRAM::DRAM)( target, model, state, mcmc_options, results )
    # Initialization
    adaptation_interval = DRAM.adaptation_interval
    n_stages = DRAM.n_stages
    gamma = DRAM.proposal_scale

    if isa( gamma, Number )
        gamma_values = vcat( 1, map( k -> gamma^(2*k+1), 0:n_stages-2 ) )
    else
        gamma_values = gamma
    end

    chain_length = mcmc_options.nsteps
    loss = mcmc_options.loss_function

    npar = get_params( model ) |> length
    chain = results.chain
    sschain = results.sschain
    current_iter = results.current_iter

    am_scaling_parameter = 2.38^2 / npar
    epsilon = 1e-6

    noise_buffer = zeros( npar )
    params_proposal = zeros( npar )

    # Buffers for the Welford covariance update
    running_mean = copy(state.current_params)
    running_cov = copy(state.proposal_cov)
    diff1 = zeros(npar)
    diff2 = zeros(npar)

    # Buffers for the delayed rejection
    N_values = zeros(Float64, n_stages)
    D_values = similar( N_values )
    alphas = zeros( sum( 1:n_stages ) )
    log_proposaldist_values = similar( alphas )
    log_post_values = zeros( Float64, n_stages+1 )
    proposals = [ zeros(npar) for _ in 0:n_stages ]
    mv_normals = [ MvNormal( state.proposal_cov ) for _ in gamma_values ]
    dram_buffers = DRAMBuffers(
        N_values,
        D_values,
        alphas,
        log_proposaldist_values,
        log_post_values,
        proposals,
        mv_normals,
        gamma_values,
        noise_buffer
    )

    # Initial Cholesky
    proposal_cov = copy( state.proposal_cov )
    proposal_cov_L = cholesky( state.proposal_cov ).L

    # Bookkeeping
    n_accepted = 0
    n_stuck = 0

    is_master_thread = Threads.nthreads() == 1 || Threads.threadid() == 1

    if is_master_thread
        println( "Starting MCMC with AM algorithm for ", chain_length, " iterations..." )
        pbar = ProgressBar( 1:chain_length, printing_delay=0.1 )
    else
        pbar = nothing
    end

    ii = 1
    while ii <= chain_length

        # 1. Propose
        # params_proposal = state.current_params + proposal_cov_L * noise_buffer
        randn!( noise_buffer )
        mul!( params_proposal, proposal_cov_L, noise_buffer )
        params_proposal .+= state.current_params

        ss_proposal = calculate_loss( params_proposal, target, model, loss )
        ss_proposal += evaluate_log_prior( params_proposal, target.priors )

        # 2. Metropolis accept/reject
        log_ratio = ss_proposal - state.ss_current
        rd = log( rand() )
        accepted = rd < log_ratio

        if accepted
            copyto!( state.current_params, params_proposal )
            n_accepted += 1
            n_stuck = 0 # Reset stuck counter on move

            new_fields = ( ss_current = ss_proposal, accepted = n_accepted )
            state = setproperties( state, new_fields )
        else
            # Delayed rejection

            # Update buffers for the first stage
            proposals[1] .= state.current_params
            proposals[2] .= params_proposal
            alphas[1] = log_ratio
            log_post_values[1:2] .= [ state.ss_current, ss_proposal ]
            @inbounds log_proposaldist_values[1] = logpdf( mv_normals[1], state.current_params - params_proposal )
            N_values[1] = ss_proposal + log_proposaldist_values[1]
            D_values[1] = state.ss_current + log_proposaldist_values[1]

            for kk in 2:n_stages
                log_ratio, ss_proposal = delayed_rejection_stage(
                    kk,
                    state,
                    target,
                    model,
                    loss,
                    params_proposal,
                    proposal_cov_L,
                    dram_buffers
                    )

                rd = log( rand() )
                accepted = rd < log_ratio

                if accepted
                    copyto!( state.current_params, params_proposal )
                    n_accepted += 1
                    n_stuck = 0 # Reset stuck counter on move

                    new_fields = ( ss_current = ss_proposal, accepted = n_accepted )
                    state = setproperties( state, new_fields )
                    break # Exit the delayed rejection loop on acceptance
                end
            end
        end

        # Store
        chain[:, ii] .= state.current_params
        sschain[ii] = state.ss_current
        current_iter[] = ii

        # 3. Adaptive Update
        # Update running mean and covariance using Welford's algorithm
        recursive_welford!( running_mean, running_cov, state.current_params, diff1, diff2, npar, ii )
        if ii > 100 && ii % adaptation_interval == 0
            for col in 1:npar
                for row in 1:npar
                    proposal_cov[row, col] = running_cov[row, col] * am_scaling_parameter
                end
                # Add epsilon to diagonal without allocating a new Matrix(I)
                proposal_cov[col, col] += epsilon
            end
            proposal_cov_L = cholesky(Symmetric(proposal_cov)).L
            state.proposal_cov .= proposal_cov

            # Initialize proposal distributions for rejection stages
            C_chol = Cholesky( parent(proposal_cov_L), 'L', 0 )
            C_pd = PDMat( proposal_cov, C_chol )
            mv_normals .= (MvNormal(Symmetric(gamma * C_pd.mat)) for gamma in gamma_values)
        end

        # Progress display update
        if is_master_thread
            if !isnothing(pbar)
                set_description(pbar, "Acc: $(round(n_accepted/ii, digits=2)) SS: $(round(state.ss_current, digits=2))")
                update(pbar) # Requires manual update in a while loop
            end
        end

        # Advance the loop
        ii += 1
    end

    if is_master_thread
        println() # Clear the manual \r line
    end

    return results, state
end
