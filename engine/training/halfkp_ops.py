"""Custom CUDA kernel for HalfKP sparse L1 layer.

One thread block per position, 256 threads (=HL1), cooperative accumulation.
L1 weights in fp16 — halves memory bandwidth.
feat_offset enables own vs opponent perspective sharing a single weight matrix.
"""

import torch
from torch.utils.cpp_extension import load_inline

FEAT_TOTAL = 22680
HL1 = 256

# --- CUDA source (compiled by nvcc — can use <<<>>> syntax) ---
_cuda_source = r"""
#include <cstdint>
#include <cuda_fp16.h>

const int HL1 = 256;

extern "C" __global__ void halfkp_l1_forward_kernel(
    const int64_t* __restrict__ feat_flat,
    const int64_t* __restrict__ feat_starts,
    const int*    __restrict__ feat_lens,
    const half*   __restrict__ l1_weight,
    const float*  __restrict__ l1_bias,
    float*        __restrict__ output,
    int batch_size,
    int64_t feat_offset)
{
    int pos_idx = blockIdx.x;
    if (pos_idx >= batch_size) return;
    int tid = threadIdx.x;

    int64_t start = feat_starts[pos_idx];
    int nf = feat_lens[pos_idx];
    float acc = l1_bias[tid];

    for (int f = 0; f < nf; f++) {
        int64_t feat_idx = feat_offset + feat_flat[start + f];
        acc += __half2float(l1_weight[feat_idx * HL1 + tid]);
    }
    output[pos_idx * HL1 + tid] = acc;
}

extern "C" __global__ void halfkp_l1_backward_kernel(
    const int64_t* __restrict__ feat_flat,
    const int64_t* __restrict__ feat_starts,
    const int*    __restrict__ feat_lens,
    const float*  __restrict__ grad_output,
    float*        __restrict__ grad_weight,
    int batch_size,
    int64_t feat_offset)
{
    int pos_idx = blockIdx.x;
    if (pos_idx >= batch_size) return;
    int tid = threadIdx.x;

    int64_t start = feat_starts[pos_idx];
    int nf = feat_lens[pos_idx];
    float grad = grad_output[pos_idx * HL1 + tid];

    for (int f = 0; f < nf; f++) {
        int64_t feat_idx = feat_offset + feat_flat[start + f];
        atomicAdd(grad_weight + feat_idx * HL1 + tid, grad);
    }
}

// Launch wrappers (called from C++ code, compiled by nvcc)

void halfkp_l1_forward_launch(
    const int64_t* feat_flat, const int64_t* feat_starts, const int* feat_lens,
    const void* l1_weight, const float* l1_bias, float* output,
    int batch_size, int64_t feat_offset)
{
    halfkp_l1_forward_kernel<<<batch_size, 256>>>(
        feat_flat, feat_starts, feat_lens,
        (const half*)l1_weight, l1_bias, output, batch_size, feat_offset);
}

void halfkp_l1_backward_launch(
    const int64_t* feat_flat, const int64_t* feat_starts, const int* feat_lens,
    const float* grad_output, float* grad_weight,
    int batch_size, int64_t feat_offset)
{
    halfkp_l1_backward_kernel<<<batch_size, 256>>>(
        feat_flat, feat_starts, feat_lens, grad_output, grad_weight,
        batch_size, feat_offset);
}
"""

# --- C++ source (compiled by g++ — no CUDA syntax, pure PyTorch tensor API) ---
_cpp_source = r"""
#include <torch/extension.h>

// Forward declarations for CUDA launch wrappers (compiled separately by nvcc)
void halfkp_l1_forward_launch(
    const int64_t* feat_flat, const int64_t* feat_starts, const int* feat_lens,
    const void* l1_weight, const float* l1_bias, float* output,
    int batch_size, int64_t feat_offset);

void halfkp_l1_backward_launch(
    const int64_t* feat_flat, const int64_t* feat_starts, const int* feat_lens,
    const float* grad_output, float* grad_weight,
    int batch_size, int64_t feat_offset);

// PyTorch-callable wrappers

void launch_halfkp_l1_forward(
    at::Tensor feat_flat,
    at::Tensor feat_starts,
    at::Tensor feat_lens,
    at::Tensor l1_weight,
    at::Tensor l1_bias,
    at::Tensor output,
    int64_t batch_size,
    int64_t feat_offset)
{
    halfkp_l1_forward_launch(
        feat_flat.data_ptr<int64_t>(),
        feat_starts.data_ptr<int64_t>(),
        feat_lens.data_ptr<int>(),
        l1_weight.data_ptr(),           // void* — fp16 half, cast in .cu
        l1_bias.data_ptr<float>(),
        output.data_ptr<float>(),
        (int)batch_size,
        feat_offset);
}

void launch_halfkp_l1_backward(
    at::Tensor feat_flat,
    at::Tensor feat_starts,
    at::Tensor feat_lens,
    at::Tensor grad_output,
    at::Tensor grad_weight,
    int64_t batch_size,
    int64_t feat_offset)
{
    halfkp_l1_backward_launch(
        feat_flat.data_ptr<int64_t>(),
        feat_starts.data_ptr<int64_t>(),
        feat_lens.data_ptr<int>(),
        grad_output.data_ptr<float>(),
        grad_weight.data_ptr<float>(),
        (int)batch_size,
        feat_offset);
}
"""

_ops = None


def _get_ops():
    global _ops
    if _ops is None:
        _ops = load_inline(
            name="halfkp_ops",
            cpp_sources=_cpp_source,
            cuda_sources=_cuda_source,
            functions=["launch_halfkp_l1_forward", "launch_halfkp_l1_backward"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=True,
        )
    return _ops


class HalfKpL1Function(torch.autograd.Function):
    """Sparse HalfKP L1: output[b,i] = bias[i] + sum_{f in features[b]} weight[f+offset,i]

    l1_weight is fp32 parameter → converted to fp16 for compute, saving BW.
    Gradients flow in fp32, compatible with AMP GradScaler.
    """

    @staticmethod
    def forward(ctx, feat_flat, feat_starts, feat_lens, l1_weight, l1_bias, offset):
        ops = _get_ops()
        batch_size = feat_starts.numel()
        output = torch.empty(batch_size, HL1, device=feat_flat.device, dtype=torch.float32)
        # Convert to fp16 for the CUDA kernel (saves memory bandwidth)
        l1_w_fp16 = l1_weight.half().contiguous()
        ops.launch_halfkp_l1_forward(
            feat_flat.contiguous(),
            feat_starts.contiguous(),
            feat_lens.to(torch.int32).contiguous(),
            l1_w_fp16,
            l1_bias.contiguous(),
            output,
            batch_size,
            int(offset),
        )
        ctx.save_for_backward(feat_flat, feat_starts, feat_lens)
        ctx.offset = int(offset)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ops = _get_ops()
        feat_flat, feat_starts, feat_lens = ctx.saved_tensors
        batch_size = feat_starts.numel()
        # fp32 gradient for fp32 parameter (GradScaler-compatible)
        grad_weight = torch.zeros(FEAT_TOTAL * HL1, device=feat_flat.device, dtype=torch.float32)
        ops.launch_halfkp_l1_backward(
            feat_flat.contiguous(),
            feat_starts.contiguous(),
            feat_lens.to(torch.int32).contiguous(),
            grad_output.contiguous(),
            grad_weight,
            batch_size,
            ctx.offset,
        )
        return None, None, None, grad_weight, grad_output.sum(dim=0), None


def halfkp_l1(feat_flat, feat_starts, feat_lens, l1_weight, l1_bias, offset=0):
    return HalfKpL1Function.apply(feat_flat, feat_starts, feat_lens, l1_weight, l1_bias, offset)
