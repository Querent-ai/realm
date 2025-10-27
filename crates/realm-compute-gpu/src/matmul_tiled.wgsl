// Optimized tiled matrix multiplication with shared memory
// C = A @ B where A is [M, K] and B is [K, N]
//
// Uses 16x16 workgroup with shared memory tiles for better cache locality
// Expected speedup: 3-5x over naive implementation

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Dimensions {
    M: u32,
    K: u32,
    N: u32,
    _pad: u32,
}

@group(0) @binding(3) var<uniform> dims: Dimensions;

// Shared memory tiles (16x16)
var<workgroup> tile_a: array<f32, 256>;  // 16x16
var<workgroup> tile_b: array<f32, 256>;  // 16x16

const TILE_SIZE: u32 = 16u;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let row = global_id.x;
    let col = global_id.y;
    let local_row = local_id.x;
    let local_col = local_id.y;

    var sum = 0.0;

    // Number of tiles needed
    let num_tiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    // Process each tile
    for (var t = 0u; t < num_tiles; t++) {
        // Load tile from A into shared memory
        let a_row = workgroup_id.x * TILE_SIZE + local_row;
        let a_col = t * TILE_SIZE + local_col;

        if (a_row < dims.M && a_col < dims.K) {
            let a_idx = a_row * dims.K + a_col;
            tile_a[local_row * TILE_SIZE + local_col] = a[a_idx];
        } else {
            tile_a[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Load tile from B into shared memory
        let b_row = t * TILE_SIZE + local_row;
        let b_col = workgroup_id.y * TILE_SIZE + local_col;

        if (b_row < dims.K && b_col < dims.N) {
            let b_idx = b_row * dims.N + b_col;
            tile_b[local_row * TILE_SIZE + local_col] = b[b_idx];
        } else {
            tile_b[local_row * TILE_SIZE + local_col] = 0.0;
        }

        // Synchronize workgroup
        workgroupBarrier();

        // Compute partial dot product using shared memory
        for (var k = 0u; k < TILE_SIZE; k++) {
            sum += tile_a[local_row * TILE_SIZE + k] * tile_b[k * TILE_SIZE + local_col];
        }

        // Synchronize before loading next tile
        workgroupBarrier();
    }

    // Write result
    if (row < dims.M && col < dims.N) {
        let c_idx = row * dims.N + col;
        c[c_idx] = sum;
    }
}
