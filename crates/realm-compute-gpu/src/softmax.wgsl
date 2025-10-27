// Softmax compute shader for attention scores
// Computes softmax along the last dimension with numerical stability

@group(0) @binding(0) var<storage, read_write> logits: array<f32>;

struct SoftmaxParams {
    batch_size: u32,    // Number of independent softmax operations
    dim_size: u32,      // Size of dimension to normalize
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(1) var<uniform> params: SoftmaxParams;

// Shared memory for reduction
var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let batch_idx = workgroup_id.x;
    let tid = local_id.x;
    let dim_size = params.dim_size;

    if (batch_idx >= params.batch_size) {
        return;
    }

    let base_idx = batch_idx * dim_size;

    // Phase 1: Find maximum value (for numerical stability)
    var local_max = -1e10;
    for (var i = tid; i < dim_size; i += 256u) {
        local_max = max(local_max, logits[base_idx + i]);
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    // Reduce to find global max
    var stride = 128u;
    while (stride >= 1u) {
        if (tid < stride && tid + stride < 256u) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let max_val = shared_max[0];
    workgroupBarrier();

    // Phase 2: Compute exp(x - max) and sum
    var local_sum = 0.0;
    for (var i = tid; i < dim_size; i += 256u) {
        let exp_val = exp(logits[base_idx + i] - max_val);
        logits[base_idx + i] = exp_val;
        local_sum += exp_val;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    // Reduce to find global sum
    stride = 128u;
    while (stride >= 1u) {
        if (tid < stride && tid + stride < 256u) {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let sum_val = shared_sum[0];
    workgroupBarrier();

    // Phase 3: Normalize by sum
    for (var i = tid; i < dim_size; i += 256u) {
        logits[base_idx + i] /= sum_val;
    }
}
