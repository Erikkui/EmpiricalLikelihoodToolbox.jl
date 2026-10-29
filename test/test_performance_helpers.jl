# Allocation-free helpers that replace allocating library calls in the hot loop. Each must give the
# same values as the call it replaces.

using EmpiricalLikelihoodToolbox: BlockComplement

@testset "performance helpers" begin

    @testset "BlockComplement equals setdiff" begin
        index_cache = collect( 3:22 )
        block_length = 5
        for block_start in 1:( length( index_cache ) - block_length + 1 )
            block = @view index_cache[ block_start:block_start+block_length-1 ]
            complement = BlockComplement( index_cache, block_start, block_length )
            @test complement == setdiff( index_cache, block )
            @test length( complement ) == length( index_cache ) - block_length
        end
        @test_throws BoundsError BlockComplement( index_cache, 1, block_length )[ 16 ]
    end

    @testset "ContiguousBlockSplit y set allocates nothing" begin
        data = wiggly( 40 )
        resampler = ContiguousBlockSplit( timeseries_block_size = 8 )
        dc = make_container( data; resampling_type = resampler )
        index_cache = collect( 1:40 )
        # Measured inside a function: at testset scope @allocated also counts untyped-variable boxing
        resample_allocations( resampler, dc, cache ) = @allocated resampler( dc, dc.options, cache )
        resample_allocations( resampler, dc, index_cache )
        @test resample_allocations( resampler, dc, index_cache ) == 0
    end
end
