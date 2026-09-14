# src/utils/resamplers.jl -- review section 22.2: index validity, disjointness where required,
# correct block lengths, all valid blocks reachable.

@testset "resamplers" begin

    samplers() = (
        RademacherSplit(),
        ContiguousBlockSplit(timeseries_block_size = 4),
        NoResampling(),
        StandardBootstrap(),
        MovingBlockBootstrap(block_size = 4),
    )

    @testset "resample_sizes matches the lengths actually returned" begin
        # Every buffer in allocate_buffers is sized from resample_sizes, so any disagreement
        # here shows up downstream as a silently wrong-shaped buffer. Odd N included on purpose.
        for ndata in (20, 21)
            dc = make_container(row(1.0:ndata))
            for s in samplers()
                cache = collect(1:ndata)
                x, y = s(dc, dc.options, cache)
                nx, ny = resample_sizes(s, ndata)

                @test length(x) == nx
                @test length(y) == ny
                @test all(∈(1:ndata), x)
                @test all(∈(1:ndata), y)
                @test get_index_size(s, dc.observations, dc.options) == ndata
            end
        end
    end

    @testset "RademacherSplit" begin
        ndata = 21
        dc = make_container(row(1.0:ndata))
        cache = collect(1:ndata)
        x, y = RademacherSplit()(dc, dc.options, cache)

        @test length(x) == div(ndata, 2)
        @test length(y) == ndata - div(ndata, 2)
        @test isempty(intersect(x, y))                  # a 50/50 split, no overlap
        @test sort(vcat(collect(x), collect(y))) == 1:ndata   # and it covers everything

        @testset "shuffles the cache in place and returns views into it" begin
            # Live footgun: a second call rewrites the arrays a previous call handed back.
            cache2 = collect(1:ndata)
            x2, _ = RademacherSplit()(dc, dc.options, cache2)
            @test parent(x2) === cache2
            snapshot = collect(x2)
            for _ in 1:20
                RademacherSplit()(dc, dc.options, cache2)
                collect(x2) == snapshot || break
            end
            @test collect(x2) != snapshot
        end
    end

    @testset "ContiguousBlockSplit" begin
        ndata, block = 20, 4
        dc = make_container(row(1.0:ndata))
        s = ContiguousBlockSplit(timeseries_block_size = block)

        @testset "x is one contiguous block, y is everything else" begin
            for _ in 1:50
                cache = collect(1:ndata)
                x, y = s(dc, dc.options, cache)
                @test length(x) == block
                @test collect(x) == x[1]:(x[1]+block-1)      # contiguous
                @test sort(vcat(collect(x), collect(y))) == 1:ndata
                @test isempty(intersect(x, y))
            end
        end

        @testset "every valid start index is reachable" begin
            Random.seed!(1234)
            starts = Set{Int}()
            for _ in 1:4000
                cache = collect(1:ndata)
                x, _ = s(dc, dc.options, cache)
                push!(starts, x[1])
            end
            @test starts == Set(1:(ndata-block+1))
        end

        @testset "resample_sizes validates the block size" begin
            @test resample_sizes(s, ndata) == (block, ndata - block)
            @test_throws ArgumentError resample_sizes(ContiguousBlockSplit(timeseries_block_size = ndata), ndata)
            @test_throws ArgumentError resample_sizes(ContiguousBlockSplit(timeseries_block_size = 0), ndata)
        end
    end

    @testset "NoResampling returns the cache itself, twice" begin
        ndata = 12
        dc = make_container(row(1.0:ndata))
        cache = collect(1:ndata)
        x, y = NoResampling()(dc, dc.options, cache)

        @test x === cache && y === cache     # aliasing: mutating one mutates "both" sets
        @test collect(x) == 1:ndata
        # Deterministic: repeated calls give identical results.
        @test NoResampling()(dc, dc.options, cache) == (x, y)
    end

    @testset "StandardBootstrap samples with replacement" begin
        ndata = 30
        dc = make_container(row(1.0:ndata))
        cache = collect(1:ndata)
        Random.seed!(7)
        x, y = StandardBootstrap()(dc, dc.options, cache)

        @test length(x) == ndata && length(y) == ndata
        @test all(∈(cache), x) && all(∈(cache), y)
        @test x != y                                  # two independent draws
        # With replacement, some index almost surely repeats across 30 draws from 30.
        @test length(unique(x)) < ndata
    end

    @testset "MovingBlockBootstrap" begin
        ndata, block = 20, 4
        dc = make_container(row(1.0:ndata))
        s = MovingBlockBootstrap(block_size = block)

        @testset "preserves length exactly" begin
            for _ in 1:50
                cache = collect(1:ndata)
                x, y = s(dc, dc.options, cache)
                @test length(x) == ndata && length(y) == ndata
                @test all(∈(cache), x) && all(∈(cache), y)
            end
        end

        @testset "is built from contiguous runs" begin
            Random.seed!(99)
            cache = collect(1:ndata)
            x, _ = s(dc, dc.options, cache)
            for start in 1:block:ndata
                stop = min(start + block - 1, ndata)
                seg = x[start:stop]
                @test seg == seg[1]:(seg[1]+length(seg)-1)
            end
        end

        @testset "a ragged tail is truncated, not dropped" begin
            # 20 is not a multiple of 6, so the final block is cut short to land exactly on ndata.
            cache = collect(1:ndata)
            x, _ = MovingBlockBootstrap(block_size = 6)(dc, dc.options, cache)
            @test length(x) == ndata
        end

        @testset "resample_sizes validates the block size" begin
            @test resample_sizes(s, ndata) == (ndata, ndata)
            @test resample_sizes(MovingBlockBootstrap(block_size = ndata), ndata) == (ndata, ndata)
            @test_throws ArgumentError resample_sizes(MovingBlockBootstrap(block_size = ndata + 1), ndata)
            @test_throws ArgumentError resample_sizes(MovingBlockBootstrap(block_size = 0), ndata)
        end
    end

    @testset "sampler type hierarchy" begin
        @test RademacherSplit() isa LengthChangingSampler
        @test ContiguousBlockSplit() isa LengthChangingSampler
        @test NoResampling() isa LengthPreservingSampler
        @test StandardBootstrap() isa LengthPreservingSampler
        @test MovingBlockBootstrap() isa LengthPreservingSampler
        @test all(s -> s isa AbstractResampler, samplers())
    end
end
