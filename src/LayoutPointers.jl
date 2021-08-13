module LayoutPointers

using ArrayInterface, Static, LinearAlgebra
using ArrayInterface: CPUPointer
using SIMDTypes: Bit, FloatingTypes
using Static: Zero, One
using ArrayInterface: contiguous_axis, contiguous_axis_indicator, contiguous_batch_size,
  stride_rank

"""
  abstract type AbstractStridedPointer{T,N,C,B,R,X,O} end

T: element type
N: dimensionality
C: contiguous dim
B: batch size
R: rank of strides
X: strides
O: offsets
"""
abstract type AbstractStridedPointer{T,N,C,B,R,X<:Tuple{Vararg{Any,N}},O<:Tuple{Vararg{Any,N}}} end

include("utils.jl")
include("cartesianvindex.jl")
include("stridedpointers.jl")
include("cartesian_indexing.jl")
include("cse_stridemultiples.jl")
include("grouped_strided_pointers.jl")

end
