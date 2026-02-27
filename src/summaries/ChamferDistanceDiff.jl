struct ChamferDistanceDiff{T} <: AbstractChamferSummary
    neighbors::T
    highest_neighbor::Int
    diff_order::Int
    summary_length::Int
end
