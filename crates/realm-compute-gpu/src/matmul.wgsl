// WebGPU compute shader for matrix multiplication
// C = A @ B where A is [M, K] and B is [K, N]

@group(0) @binding(0) var<storage, read> a: array<f32>;  // Input matrix A [M, K]
@group(0) @binding(1) var<storage, read> b: array<f32>;  // Input matrix B [K, N]
@group(0) @binding(2) var<storage, read_write> c: array<f32>;  // Output matrix C [M, N]

struct Dimensions {
    M: u32,  // Rows in A and C
    K: u32,  // Cols in A, rows in B
    N: u32,  // Cols in B and C
    _pad: u32,
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

// Workgroup size: 16x16 threads
@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    // Bounds check
    if (row >= dims.M || col >= dims.N) {
        return;
    }

    // Compute dot product for C[row, col]
    var sum = 0.0;
    for (var k = 0u; k < dims.K; k++) {
        let a_idx = row * dims.K + k;
        let b_idx = k * dims.N + col;
        sum += a[a_idx] * b[b_idx];
    }

    let c_idx = row * dims.N + col;
    c[c_idx] = sum;
}
