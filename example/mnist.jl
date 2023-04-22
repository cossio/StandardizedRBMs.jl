using StandardizedRBMs: standardize
using RestrictedBoltzmannMachines: RBM, Binary, xReLU, pcd!, initialize!,
    sample_from_inputs, sample_h_from_v, sample_v_from_v, mean_from_inputs
using CudaRBMs: gpu, cpu
using MLDatasets: MNIST
using Optimisers: Adam
using Statistics: mean, std, var, cov, cor
using Random: bitrand
import Makie
import CairoMakie
using Images: Gray, gray
using MosaicViews: mosaicview
using EllipsisNotation: (..)
import StatsBase

train_x = MNIST(split=:train).features .> 0.5
train_y = MNIST(split=:train).targets
tests_x = MNIST(split=:test).features .> 0.5
tests_y = MNIST(split=:test).targets

rbm = RBM(Binary((28,28)), xReLU((300,)), randn(28,28,300))
initialize!(rbm, train_x)
rbm = gpu(standardize(rbm))

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=200, height=200)
    Makie.scatter!(ax,
        cpu(vec(mean(train_x; dims=3))),
        vec(mean_from_inputs(cpu(rbm).visible, zeros(28, 28)))
    )
    Makie.resize_to_layout!(fig)
    fig
end

optim = Adam(5f-4, (0f0, 999f-3), 1f-6)
batchsize = 256
vm = sample_from_inputs(rbm.visible, gpu(zeros(28, 28, batchsize)))
vm = gpu(bitrand(28, 28, batchsize))
training_time = @elapsed begin
    state, ps = pcd!(
        rbm, gpu(train_x);
        optim, steps=20, batchsize, iters=20000, vm,
        ϵv=1f-1, ϵh=0f0, l2l1_weights=0.01, damping=1f-1
    )
end

data_h = sample_h_from_v(rbm, gpu(train_x))
std(data_h; dims=2)

sampled_v = cpu(sample_v_from_v(rbm, gpu(tests_x); steps=500));
mosaicview(Gray.(sampled_v[.., 1:36]), nrow=6)'

sample_v = sample_from_inputs(rbm.visible, gpu(zeros(28, 28, 5000)));
sample_v = gpu(bitrand(28, 28, 5000));
sampling_time = @elapsed begin
    sampled_v = cpu(sample_v_from_v(rbm, gpu(sampled_v); steps=10000));
end
mosaicview(Gray.(sampled_v[.., rand(1:size(sampled_v,3), 36)]), nrow=6)'

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=200, height=200, yscale=log10)
    Makie.hist!(ax, vec(rbm.w), bins=-2:0.01:2, fillto=1e-4)
    Makie.ylims!(ax, 1e-3, 1e5)
    Makie.resize_to_layout!(fig)
    fig
end

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=200, height=200)
    Makie.scatter!(ax,
        vec(mean(train_x; dims=3)),
        vec(mean(sampled_v; dims=3)),
    )
    Makie.resize_to_layout!(fig)
    fig
end

let fig = Makie.Figure()
    ax = Makie.Axis(fig[1,1], width=200, height=200)
    Makie.scatter!(ax,
        vec(cov(reshape(tests_x, 28*28, :); dims=2)),
        vec(cov(reshape(sampled_v, 28*28, :); dims=2)),
    )
    Makie.resize_to_layout!(fig)
    fig
end
