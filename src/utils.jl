@inline val_stride_rank(A) = Val(known(stride_rank(A)))
@inline val_dense_dims(A) = Val(known(ArrayInterface.dense_dims(A)))

