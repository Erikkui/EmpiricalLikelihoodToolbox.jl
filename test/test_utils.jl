# Pure helpers in src/utils/utils.jl. Review section 22.3 asks specifically for numerical
# differentiation tested on known polynomials and sine waves.

@testset "utils" begin

    @testset "calculate_diffs" begin
        dt = 0.1
        t = collect(0.0:dt:2.0)

        @testset "linear ramp is exact" begin
            a = 3.5
            R = row(a .* t)
            d = calculate_diffs(R, (1,), dt)
            # Central differences are exact on a ramp, and the padded boundaries copy an
            # interior value that is also exactly `a`, so every entry equals the slope.
            @test all(≈(a), d[1])
        end

        @testset "quadratic: 1st derivative exact everywhere, 2nd exact in the interior" begin
            R = row(t .^ 2)
            d = calculate_diffs(R, (1, 2), dt)

            # (x[j+1]-x[j-1])/2dt = ((t+dt)^2-(t-dt)^2)/2dt = 2t, exactly.
            n = length(t)
            @test d[1][1, 2:n-1] ≈ 2 .* t[2:n-1]

            # The 2nd order pass consumes the padded 1st order array, so columns 2 and n-1 are
            # contaminated by the fabricated boundary. Everything further in is exactly 2.
            @test all(≈(2.0), d[2][1, 3:n-2])
        end

        @testset "sine: error is second order in dt" begin
            err = map((0.1, 0.05)) do h
                tt = collect(0.0:h:2pi)
                d = calculate_diffs(row(sin.(tt)), (1,), h)
                n = length(tt)
                maximum(abs.(d[1][1, 2:n-1] .- cos.(tt[2:n-1])))
            end
            # Halving dt must cut the error by ~4x. Far stronger than an absolute tolerance.
            @test err[1] / err[2] ≈ 4.0 rtol = 0.05
        end

        @testset "boundary padding copies the neighbouring column" begin
            R = row(t .^ 3)
            d = calculate_diffs(R, (1,), dt)
            @test d[1][:, 1] == d[1][:, 2]
            @test d[1][:, end] == d[1][:, end-1]
        end

        @testset "input is not mutated" begin
            R = row(t .^ 2)
            R_before = copy(R)
            calculate_diffs(R, (1, 2), dt)
            @test R == R_before
        end

        @testset "unrequested orders come back empty" begin
            d = calculate_diffs(row(t .^ 2), (2,), dt)
            @test size(d[1]) == (0, 0)
            @test size(d[2]) == (1, length(t))
        end

        @testset "multidimensional input differentiates each row" begin
            R = vcat(row(2.0 .* t), row(t .^ 2))
            d = calculate_diffs(R, (1,), dt)
            @test all(≈(2.0), d[1][1, :])
            @test d[1][2, 2:end-1] ≈ 2 .* t[2:end-1]
        end
    end

    @testset "calculate_diffs! matches calculate_diffs" begin
        # The MCMC hot loop relies on the in-place version being identical to the allocating one.
        dt = 0.1
        t = collect(0.0:dt:3.0)
        R = vcat(row(sin.(t)), row(t .^ 2))

        for orders in ((1,), (2,), (1, 2))
            expected = calculate_diffs(R, orders, dt)
            max_order = maximum(orders)

            # Buffer shaped exactly as allocate_buffers builds it: max_order + 2 entries,
            # the last two being scratch.
            buf = Vector{Matrix{Float64}}(undef, max_order + 2)
            for ii in 1:max_order
                buf[ii] = ii in orders ? zeros(size(R)) : Matrix{Float64}(undef, 0, 0)
            end
            buf[end-1] = zeros(size(R))
            buf[end] = zeros(size(R))

            calculate_diffs!(buf, R, orders, dt)
            for ii in orders
                @test buf[ii] ≈ expected[ii]
            end
        end
    end

    @testset "recursive_welford!" begin
        @testset "reproduces mean and unbiased cov" begin
            rng = StableRNG(11)
            for (npar, n) in ((3, 40), (1, 12), (2, 2))
                X = randn(rng, npar, n)

                running_mean = zeros(npar)
                running_cov = zeros(npar, npar)
                d1, d2 = zeros(npar), zeros(npar)
                for ii in 1:n
                    recursive_welford!(running_mean, running_cov, view(X, :, ii), d1, d2, npar, ii)
                end

                @test running_mean ≈ vec(mean(X, dims = 2))
                n > 1 && @test running_cov ≈ cov(X')
            end
        end

        @testset "first observation sets the mean and leaves the covariance alone" begin
            npar = 2
            m, C = zeros(npar), zeros(npar, npar)
            recursive_welford!(m, C, [4.0, -1.0], zeros(npar), zeros(npar), npar, 1)
            @test m == [4.0, -1.0]
            @test all(iszero, C)
        end
    end

    @testset "regularized_cholesky" begin
        reg(C) = C + 1e-2 * Diagonal(diag(C)) + 1e-6 * I

        @testset "factorizes the ridge-regularized matrix" begin
            rng = StableRNG(3)
            A = randn(rng, 5, 12)
            C = cov(A')
            F = regularized_cholesky(copy(C))
            @test Matrix(F) ≈ reg(C)
        end

        @testset "solving matches the explicit inverse" begin
            rng = StableRNG(4)
            C = cov(randn(rng, 4, 20)')
            b = randn(StableRNG(5), 4)
            @test regularized_cholesky(copy(C)) \ b ≈ inv(reg(C)) * b
        end

        @testset "succeeds on a rank-deficient matrix" begin
            # This is the whole point of the ridge term: a singular covariance must still factorize.
            v = [1.0, 2.0, 3.0]
            C = v * v'
            @test isposdef(regularized_cholesky(copy(C)))
        end

        @testset "zero covariance regularizes to 1e-6 I" begin
            # The degenerate case a NoResampling target produces.
            F = regularized_cholesky(zeros(3, 3))
            @test Matrix(F) ≈ 1e-6 * Matrix(I, 3, 3)
        end

        @testset "does not mutate its argument" begin
            # `cholesky!` mutates the freshly-built regularized copy, not the caller's matrix.
            C = cov(randn(StableRNG(6), 3, 10)')
            C_before = copy(C)
            regularized_cholesky(C)
            @test C == C_before
        end
    end

    @testset "donsker_covariance" begin
        F = [0.1, 0.4, 0.9]
        n = 25
        C = donsker_covariance(F, n)

        @test C ≈ [(min(F[i], F[j]) - F[i] * F[j]) / n for i in 1:3, j in 1:3]
        @test issymmetric(C)
        @test diag(C) ≈ F .* (1 .- F) ./ n
        @test all(≥(0), diag(C))
        @test minimum(eigvals(Symmetric(C))) > -1e-12          # positive semidefinite
        @test donsker_covariance(F, 2n) ≈ C ./ 2               # scales as 1/ndata
    end

    @testset "embedding" begin
        @testset "identity when no embedding is requested" begin
            x = collect(1.0:10.0)
            @test embedding(x, 0) === x
            @test embedding(x, Int[]) === x
        end

        @testset "lagged copies have the documented shape and content" begin
            x = collect(1.0:10.0)
            dims = [1, 2]
            E = embedding(x, dims)

            @test size(E) == (1 + length(dims), length(x) - maximum(dims))
            @test E[1, :] == x[3:10]
            @test E[2, :] == x[2:9]      # lag 1
            @test E[3, :] == x[1:8]      # lag 2
            # Each successive row is the previous one shifted by a single step.
            @test E[2, 2:end] == E[1, 1:end-1]
            @test E[3, 2:end] == E[2, 1:end-1]
        end

        @testset "diff embedding pairs the series with its lagged difference" begin
            x = collect(1.0:10.0)
            tau = 2
            E = embedding(x, tau, :diff)

            @test size(E) == (2, length(x) - tau)
            @test E[1, :] == x[tau+1:end]
            @test E[2, :] == x[tau+1:end] .- x[1:end-tau]
        end

        @testset "unknown embedding_type raises an explicit error" begin
            x = collect(1.0:10.0)
            @test_throws ArgumentError embedding(x, 2, :bogus)
        end
    end

    @testset "contract / contract!" begin
        M = row(1:10)

        @testset "sums non-overlapping windows" begin
            @test contract(M, 2) == [3.0, 7.0, 11.0, 15.0, 19.0]
            @test contract(M, 5) == [15.0, 40.0]
        end

        @testset "a ragged tail is dropped" begin
            c = contract(M, 3)                       # div(10,3) == 3 windows, last point unused
            @test length(c) == 3
            @test sum(c) == sum(1:9)
        end

        @testset "totals agree with the covered slice" begin
            for w in (2, 3, 4)
                nwin = div(size(M, 2), w)
                @test sum(contract(M, w)) ≈ sum(M[:, 1:nwin*w])
            end
        end

        @testset "sums across rows as well as columns" begin
            @test contract(vcat(row(1:4), row(10:10:40)), 2) == [3.0 + 30.0, 7.0 + 70.0]
        end

        @testset "in-place agrees with allocating" begin
            out = zeros(5)
            contract!(out, M, 2)
            @test out == contract(M, 2)
        end
    end

    @testset "invcdf" begin
        x = collect(0.0:0.1:1.0)
        cdf = x .^ 2                      # monotone
        r = invcdf(x, cdf, 7, 1)

        @test length(r) == 7
        @test issorted(r)                 # monotone in -> monotone out
        @test all(v -> minimum(x) <= v <= maximum(x), r)

        # cont=2 skips interpolation and returns grid values verbatim.
        @test all(in(x), invcdf(x, cdf, 7, 2))
    end

    @testset "standardize!" begin
        # Despite the bang, this is a pure function returning a value (see test_known_issues.jl).
        @test standardize!(10.0, 4.0, 2.0) == 3.0
        @test standardize!([3.0, 5.0], [1.0, 1.0], 2.0) == [1.0, 2.0]
    end

    @testset "model parameter accessors" begin
        @testset "round-trips through update_model_parameters" begin
            for m in updatable_models()
                names, values = get_active_model_params(m)
                @test length(names) == length(values)
                @test names == m.active_parameters

                new_values = collect(values) .+ 0.25
                m2 = update_model_parameters(m, new_values)

                @test get_active_model_params(m2)[2] == new_values
                @test get_active_model_params(m2)[1] == names
                # Accessors returns a new instance; the original must be untouched.
                @test get_active_model_params(m)[2] == values
            end
        end

        @testset "inactive parameters are preserved" begin
            m = NormalModel()
            m2 = update_model_parameters(m, [7.0, 3.0])
            @test m2.mu == 7.0 && m2.sigma == 3.0
            @test m2.dt_obs == m.dt_obs && m2.x0 == m.x0 && m2.dim == m.dim
        end

        @testset "get_all_model_params covers all_parameters" begin
            for m in all_models()
                names, values = get_all_model_params(m)
                @test names == m.all_parameters
                @test values == [getfield(m, n) for n in m.all_parameters]
            end
        end
    end
end
