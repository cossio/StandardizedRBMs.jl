CudaRBMs.gpu(rbm::StandardizedRBM) = StandardizedRBM(
    gpu(rbm.visible), gpu(rbm.hidden), gpu(rbm.w),
    gpu(rbm.offset_v), gpu(rbm.offset_h), gpu(rbm.scale_v), gpu(rbm.scale_h)
)

CudaRBMs.cpu(rbm::StandardizedRBM) = StandardizedRBM(
    cpu(rbm.visible), cpu(rbm.hidden), cpu(rbm.w),
    cpu(rbm.offset_v), cpu(rbm.offset_h), cpu(rbm.scale_v), cpu(rbm.scale_h)
)
