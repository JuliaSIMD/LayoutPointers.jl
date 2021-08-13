@inline asvalbool(r) = Val(map(Bool, r))
@inline asvalint(r) = Val(map(Int, r))
# @generated function asvalint(r::T) where {T<:Tuple{Vararg{StaticInt}}}
#     t = Expr(:tuple)
#     for s ∈ T.parameters
#         push!(t.args, s.parameters[1])
#     end
#     Expr(:call, Expr(:curly, :Val, t))
# end
# @generated function asvalbool(r::T) where {T<:Tuple{Vararg{StaticBool}}}
#     t = Expr(:tuple)
#     for b ∈ T.parameters
#         push!(t.args, b === True)
#     end
#     Expr(:call, Expr(:curly, :Val, t))
# end
@inline val_stride_rank(A) = asvalint(stride_rank(A))
@inline val_dense_dims(A) = asvalbool(ArrayInterface.dense_dims(A))

