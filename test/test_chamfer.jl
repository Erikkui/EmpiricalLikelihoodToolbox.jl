# src/summaries/chamfer_distance.jl -- checked against a brute-force reference that shares no code
# with the k-d tree implementation.

# All pairwise distances, sorted per point; the k-th one is the k-th nearest-neighbour distance.
function brute_force_chamfer(A, B, kvals)
    distances = [norm(a - b) for a in eachcol(A), b in eachcol(B)]   # n × m
    sorted_A_to_B = sort(distances; dims = 2)
    sorted_B_to_A = sort(distances; dims = 1)
    return [mean(sorted_A_to_B[:, k]) + mean(sorted_B_to_A[k, :]) for k in kvals]
end

@testset "chamfer_distance" begin
    rng = StableRNG(31)
    A = randn(rng, 2, 30)
    B = randn(rng, 2, 25)

    @testset "matches the brute-force reference" begin
        for kvals in (1, 3, [1, 3], [1, 2, 3])
            @test chamfer_distance(A, B, kvals) ≈ brute_force_chamfer(A, B, kvals)
        end
        A_1d = randn(rng, 1, 18)
        B_1d = randn(rng, 1, 22)
        @test chamfer_distance(A_1d, B_1d, [1, 4]) ≈ brute_force_chamfer(A_1d, B_1d, [1, 4])
    end

    @testset "a scalar k behaves like a one-element vector" begin
        @test chamfer_distance(A, B, 3) == chamfer_distance(A, B, [3])
        @test length(chamfer_distance(A, B, 2)) == 1
    end

    @testset "identical sets have zero 1-nearest-neighbour distance" begin
        # Each point's nearest neighbour in the other set is itself.
        @test chamfer_distance(A, A, 1) ≈ [0.0] atol = 1e-12
    end

    @testset "symmetric in its two sets" begin
        for kvals in (1, [1, 3])
            @test chamfer_distance(A, B, kvals) ≈ chamfer_distance(B, A, kvals)
        end
    end

    @testset "non-decreasing in k" begin
        @test issorted(chamfer_distance(A, B, [1, 2, 3, 4]))
    end

    @testset "hand-computed value on 1-D point sets" begin
        # A = {0, 10}, B = {1, 11}: every 1-NN distance is exactly 1 in both directions.
        Ah = reshape([0.0, 10.0], 1, 2)
        Bh = reshape([1.0, 11.0], 1, 2)
        # sum_AB/n + sum_BA/m = (1+1)/2 + (1+1)/2 = 2
        @test chamfer_distance(Ah, Bh, 1) ≈ [2.0]

        # k=2 reaches the far point: distances from 0 are {1, 11}, from 10 are {1, 9}.
        # sum_AB(k=2) = 11 + 9 = 20 over n=2 -> 10; by symmetry the reverse is also 10.
        @test chamfer_distance(Ah, Bh, [2]) ≈ [20.0 / 2 + 20.0 / 2]
    end

    @testset "scales with the data" begin
        @test chamfer_distance(2A, 2B, 1) ≈ 2 .* chamfer_distance(A, B, 1)
    end
end
