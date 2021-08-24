module LayoutPointers

using ArrayInterface, Static, LinearAlgebra
using ArrayInterface: CPUPointer, StrideIndex, offsets
using SIMDTypes: Bit, FloatingTypes
using Static: Zero, One
using ArrayInterface: contiguous_axis, contiguous_axis_indicator, contiguous_batch_size,
  stride_rank, offsets, offset1, CPUTuple, static_first, static_step, strides
using ManualMemory: preserve_buffer, offsetsize

export stridedpointer

"""
  abstract type AbstractStridedPointer{T,N,C,B,R,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}}} end

T: element type
N: dimensionality
C: contiguous dim
B: batch size
R: rank of strides
X: strides
O: offsets
"""
abstract type AbstractStridedPointer{T,N,C,B,R,X<:Tuple{Vararg{Integer,N}},O<:Tuple{Vararg{Integer,N}}} end


include("stridedpointers.jl")
include("grouped_strided_pointers.jl")
include("precompile.jl")

end
