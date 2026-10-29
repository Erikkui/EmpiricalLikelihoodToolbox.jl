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

    @testset "JointSummaryStatistics adds no allocations to its components" begin
        data = row( 1.0:24.0 )
        target, _ = make_target( data, StandardECDF( 4 ), CumulativeSum( 3 );
                                 resampling_type = NoResampling(), inference_method = BSL( 2 ) )
        cache = target.buffers.index_cache
        joint_out = fill( NaN, target.summary_length )
        joint_allocations( out, target, cache ) =
            @allocated target.summary_statistics( out, cache, cache, target.data, target.buffers )
        component_allocations( out, summary, target, cache ) =
            @allocated calculate_summary_statistic!( out, summary, cache, cache, target.data, target.buffers )

        joint_allocations( joint_out, target, cache )
        component_total = 0
        for summary in target.summary_statistics.statistics
            component_out = fill( NaN, summary.summary_length )
            component_allocations( component_out, summary, target, cache )
            component_total += component_allocations( component_out, summary, target, cache )
        end
        @test joint_allocations( joint_out, target, cache ) == component_total
    end
end
