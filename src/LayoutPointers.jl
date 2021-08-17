module LayoutPointers

using ArrayInterface, Static, LinearAlgebra
using ArrayInterface: CPUPointer, StrideIndex, offsets
using SIMDTypes: Bit, FloatingTypes
using Static: Zero, One
using ArrayInterface: contiguous_axis, contiguous_axis_indicator, contiguous_batch_size,
  stride_rank, offsets, offset1

export stridedpointer

"""
  abstract type AbstractStridedPointer{T,N,R,C,B,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}},O1} end


T: element type
N: dimensionality
C: contiguous dim
B: batch size
R: rank of strides
X: strides
O: offsets
"""
abstract type AbstractStridedPointer{T,N,R,C,B,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}},O1} end



include("utils.jl")
include("cartesianvindex.jl")
include("stridedpointers.jl")
include("cse_stridemultiples.jl")
include("grouped_strided_pointers.jl")

end
