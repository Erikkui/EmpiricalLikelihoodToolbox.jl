# src/summaries/chamfer_distance.jl -- four methods that must all agree.

@testset "chamfer_distance" begin
    rng = StableRNG(31)
    A = randn(rng, 2, 30)
    B = randn(rng, 2, 25)

    @testset "all four methods agree" begin
        for kvals in (1, 3, [1, 3], [1, 2, 3])
            expected = chamfer_distance(A, B, kvals)

            out = similar(expected)
            chamfer_distance!(out, A, B, kvals)
            @test out == expected

            tree_B = EmpiricalLikelihoodToolbox.KDTree(B)
            @test chamfer_distance(A, B, tree_B, kvals) == expected

            out2 = similar(expected)
            chamfer_distance!(out2, A, B, tree_B, kvals)
            @test out2 == expected
        end
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
