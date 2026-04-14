"""
    DRAM

A struct for the delayed rejection adaptive Metropolis MCMC algorithm.

# Fields
- `proposal_width::Float64`: The width of the proposal distribution (default: 0.01).
- `adaptation_interval::Int`: The interval at which to adapt the proposal distribution (default: 50).
- `n_stages::Int`: The number of proposal stages, INCLUDING the initial proposal) (default: 3).
- `proposal_scale`: The scaling factor for the proposal covariance at each stage. Can be a single number (applied as gamma^(2k+1), k = 0, ..., n_stages-2) or a vector of length n_stages-1 specifying the scale for each stage (default: 0.5).

"""
Base.@kwdef struct DRAM{P}
    proposal_width::Float64 = 0.01
    adaptation_interval::Int = 50
    n_stages::Int = 2
    proposal_scale::P = [1.0, 0.5]
end

function compute_log_path( states, log_pi, idx_range, zero_mean_dists )
    n = length( idx_range )
    idx_anchor = idx_range[1]

    # Base Case: Sequence of exactly 2 states
    if n == 2
        idx_dest = idx_range[2]
        diff = states[idx_dest] .- states[ idx_anchor ]
        log_q = logpdf( zero_mean_dists[1], diff )
        return log_pi[ idx_anchor ] + log_q
    end

    # Recursive Step
    # 1. Forward sub-path
    sub_fwd = idx_range[1:end-1]
    log_D_sub = compute_log_path( states, log_pi, sub_fwd, zero_mean_dists )

    # 2. Reverse sub-path
    sub_rev = reverse( sub_fwd )
    log_N_sub = compute_log_path( states, log_pi, sub_rev, zero_mean_dists )

    # 3. Calculate rejection probability: log(1 - alpha)
    log_alpha = min( 0.0, log_N_sub - log_D_sub )
    if log_alpha == 0.0
        return -Inf
    end
    log_rej = log1mexp( log_alpha )

    # 4. Calculate the final jump
    idx_final = idx_range[end]
    diff = states[ idx_final ] .- states[ idx_anchor ]

    log_q = logpdf( zero_mean_dists[n-1], diff )

    return log_D_sub + log_rej + log_q
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

    # Unpack buffers
    log_post_values = dram_buffers.log_post_values
    proposals = dram_buffers.proposals
    mv_normals = dram_buffers.mv_normals
    gamma_values = dram_buffers.gamma_values
    noise_buffer = dram_buffers.noise_buffer

    randn!( noise_buffer )
    noise_buffer .*= sqrt( gamma_values[ stage ] )  # Scale the noise for the next proposal

    # Generate the next proposal
    mul!( params_proposal, proposal_cov_L, noise_buffer )
    params_proposal .+= state.current_params
    proposals[ stage + 1 ] .= params_proposal

    ss_proposal = calculate_loss( params_proposal, target, model, loss )
    ss_proposal += evaluate_log_prior( params_proposal, target.priors )
    log_post_values[ stage + 1 ] = ss_proposal

    # Define exact index ranges
    forward_range = 1:(stage + 1)
    reverse_range = (stage + 1):-1:1

    # Calculate probabilities
    log_D_kk = compute_log_path( proposals, log_post_values, forward_range, mv_normals )
    log_N_kk = compute_log_path( proposals, log_post_values, reverse_range, mv_normals )

    # Final acceptance probability for this stage
    log_ratio = log_N_kk - log_D_kk

    # Return ss_proposal so the main loop captures the correct density if accepted
    return log_ratio, ss_proposal
end

# Removed unused flat buffers (alphas, D_values, N_values, etc.)
struct DRAMBuffers{T}
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
        gamma_values = vcat( 1.0, map( k -> Float64(gamma^(2*k+1)), 0:n_stages-2 ) )
    else
        gamma_values = Float64.(gamma)
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
    log_post_values = zeros( Float64, n_stages+1 )
    proposals = [ zeros(npar) for _ in 0:n_stages ]

    # Properly initialize the zero-mean distributions using Symmetric
    mv_normals = [ MvNormal(Symmetric(g * state.proposal_cov)) for g in gamma_values ]

    dram_buffers = DRAMBuffers(
        log_post_values,
        proposals,
        mv_normals,
        gamma_values,
        noise_buffer
    )

    # Initial Cholesky
    proposal_cov = copy( state.proposal_cov )
    proposal_cov_L = cholesky( Symmetric(state.proposal_cov) ).L

    is_master_thread = Threads.nthreads() == 1 || Threads.threadid() == 1

    if target.options.verbose
        pbar = ProgressBar( 1:chain_length, printing_delay=0.1 )
    else
        pbar = nothing
    end

    ii = 1
    while ii <= chain_length

        # 1. Propose
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
            # Update state with the proposal and its specific target density
            copyto!( state.current_params, params_proposal )
            @reset state.ss_current = ss_proposal

            # Update acceptance count
            results.acceptance[1] += 1
        else
            # Delayed rejection

            # Anchor the sequence buffers
            proposals[1] .= state.current_params
            proposals[2] .= params_proposal
            log_post_values[1] = state.ss_current
            log_post_values[2] = ss_proposal

            for kk in 2:n_stages
                log_ratio, ss_proposal_kk = delayed_rejection_stage(
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
                    @reset state.ss_current = ss_proposal_kk

                    results.acceptance[kk] += 1
                    break
                end
            end
        end

        # Store
        chain[:, ii] .= state.current_params
        sschain[ii] = state.ss_current
        current_iter[] = ii

        # 3. Adaptive Update
        recursive_welford!( running_mean, running_cov, state.current_params, diff1, diff2, npar, ii )
        if ii > 100 && ii % adaptation_interval == 0
            for col in 1:npar
                for row in 1:npar
                    proposal_cov[row, col] = running_cov[row, col] * am_scaling_parameter
                end
                proposal_cov[col, col] += epsilon
            end

            # Use Symmetric to ensure numeric stability during factorization
            proposal_cov_L = cholesky( Symmetric(proposal_cov) ).L
            state.proposal_cov .= proposal_cov

            # Rebuild scaled distributions
            for (idx, g) in enumerate(gamma_values)
                mv_normals[idx] = MvNormal( Symmetric(g * proposal_cov) )
            end
        end

        # Progress display update
        if !isnothing(pbar)
            accepted_text = round.( results.acceptance ./ ii, digits=2 )
            set_description(pbar, "Acc: $(accepted_text) SS: $(round(state.ss_current, digits=2))")
            update(pbar)
        end

        # Advance the loop
        ii += 1
    end

    if is_master_thread
        println()
    end

    return results, state
end
