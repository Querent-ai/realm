// RMS Normalization compute shader
// Normalizes input by RMS (Root Mean Square) and scales by weights

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

struct RMSNormParams {
    batch_size: u32,
    hidden_dim: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(3) var<uniform> params: RMSNormParams;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.x;
    let tid = local_id.x;
    let hidden_dim = params.hidden_dim;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * hidden_dim;

    // Phase 1: Compute sum of squares
    var local_sum_sq = 0.0;
    for (var i = tid; i < hidden_dim; i += 256u) {
        let val = input[base_idx + i];
        local_sum_sq += val * val;
    }
    shared_sum[tid] = local_sum_sq;
    workgroupBarrier();

    // Reduce to get total sum
    var stride = 128u;
    while (stride >= 1u) {
        if (tid < stride && tid + stride < 256u) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let sum_sq = shared_sum[0];
    workgroupBarrier();

    // Compute RMS
    let rms = sqrt(sum_sq / f32(hidden_dim) + params.eps);

    // Phase 2: Normalize and scale
    for (var i = tid; i < hidden_dim; i += 256u) {
        let idx = base_idx + i;
        output[idx] = (input[idx] / rms) * weight[i];
    }
}
