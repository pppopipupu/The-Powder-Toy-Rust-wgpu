// Shared 3D renderer code: voxel word decoding, palette lookup, hierarchical
// world traversal and ray/AABB helpers.
//
// `obb.wgsl` and `lighting.wgsl` are prefixed with this file when the pipelines
// are built, so every function and binding below is visible to both stages.
//
// The world is 256^3 voxels with `y` up and `x` as the fastest axis. Voxel
// words use the packed layout: type in bits 0..8, temperature in 8..24, life
// in 24..32. Occupancy masks are the brick-aligned leaf masks (one bit per 2^3
// block, two words per 8^3 brick) and the six-level coarse pyramid.

struct SceneUniforms {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // xyz = eye position, w = tan(fov_y * 0.5).
    eye: vec4<f32>,
    // xyz = normalized view direction, w = aspect ratio.
    forward: vec4<f32>,
    // xyz = camera right, w = near plane.
    right: vec4<f32>,
    // xyz = camera up, w = far plane.
    up: vec4<f32>,
    // xyz = direction towards the sun, w = exposure.
    sun_dir: vec4<f32>,
    // rgb = ambient sky colour, w = ambient intensity.
    ambient: vec4<f32>,
    // rgb = sun colour, w = sun intensity.
    sun_color: vec4<f32>,
    // xy = render target size, zw = reciprocal size.
    target_size: vec4<f32>,
    // x = rigid body count, y = feature flags (bit 0 shadows, bit 1 AO).
    body: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> voxel_a: array<u32>;
@group(0) @binding(1) var<storage, read> leaf_masks: array<u32>;
@group(0) @binding(2) var<storage, read> coarse_masks: array<u32>;
@group(0) @binding(3) var<storage, read> palette: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> body_desc: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> body_voxels: array<u32>;
@group(0) @binding(6) var<storage, read> body_masks: array<u32>;
@group(0) @binding(7) var<uniform> scene: SceneUniforms;

const VOXEL_DIM: u32 = 256u;
const VOXEL_LAST: f32 = 255.0;
const BRICK_SHIFT: u32 = 3u;
const LEAF_BLOCK: u32 = 2u;
const LEAF_WORD_SHIFT: u32 = 5u;
const PALETTE_STRIDE: u32 = 4u;

const BODY_GRID: f32 = 32.0;
const BODY_DESC_VEC4: u32 = 10u;
const BODY_GRID_VOXELS: u32 = 32768u;
const BODY_GRID_DIM: u32 = 32u;
const BODY_MASK_WORDS: u32 = 128u;
const MAX_BODIES: u32 = 64u;

const MAX_WORLD_STEPS: u32 = 512u;
const MAX_BODY_STEPS: u32 = 256u;
const TRANSMITTANCE_CUTOFF: f32 = 0.02;
const AO_DISTANCE: f32 = 16.0;
const SHADOW_DISTANCE: f32 = 512.0;

/// Palette flag: the element is a participating medium (volume accumulation).
const PALETTE_FLAG_VOLUME: u32 = 1u;
/// Palette flag: the emission term scales with the voxel life counter.
const PALETTE_FLAG_LIFE_EMISSION: u32 = 2u;

const WORLD_MIN: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const WORLD_MAX: vec3<f32> = vec3<f32>(256.0, 256.0, 256.0);

// The gamma-space sky. The host clears the presentation target to the same
// value, so a pixel without a hit is bit-identical to the clear colour.
const SKY_COLOR: vec3<f32> = vec3<f32>(0.53, 0.72, 0.92);

/// Element type of a packed voxel word.
fn voxel_type(word: u32) -> u32 {
    return word & 0xffu;
}

/// Remaining life of a packed voxel word (0..255).
fn voxel_life(word: u32) -> u32 {
    return (word >> 24u) & 0xffu;
}

/// Albedo of `ty` from the render palette.
fn palette_albedo(ty: u32) -> vec3<f32> {
    return palette[ty * PALETTE_STRIDE].rgb;
}

/// Packed palette flags of `ty`.
fn palette_flags(ty: u32) -> u32 {
    return u32(palette[ty * PALETTE_STRIDE].a + 0.5);
}

/// Volume absorption coefficient of `ty`; zero for opaque elements.
fn palette_sigma(ty: u32) -> f32 {
    return palette[ty * PALETTE_STRIDE + 1u].x;
}

/// Emission radiance of `ty`, optionally modulated by the voxel life.
fn palette_emission(ty: u32, life: u32) -> vec3<f32> {
    var emission = palette[ty * PALETTE_STRIDE + 1u].yzw;
    if ((palette_flags(ty) & PALETTE_FLAG_LIFE_EMISSION) != 0u) {
        emission *= 0.5 + 0.5 * (f32(life) / 255.0);
    }
    return emission;
}

/// Whether `ty` participates in volume accumulation instead of stopping the ray.
fn palette_is_volume(ty: u32) -> bool {
    return (palette_flags(ty) & PALETTE_FLAG_VOLUME) != 0u;
}

/// Leaf-mask word of the 8^3 brick containing `(x, y, z)`.
fn leaf_word(x: u32, y: u32, z: u32, high: bool) -> u32 {
    let brick = ((y >> BRICK_SHIFT) << 10u) | ((z >> BRICK_SHIFT) << 5u) | (x >> BRICK_SHIFT);
    let block = (((y & 7u) >> 1u) << 4u) | (((z & 7u) >> 1u) << 2u) | ((x & 7u) >> 1u);
    var word = brick * 2u;
    if (high) {
        word = word + 1u;
    }
    return word;
}

/// Bit of the 2^3 leaf block containing `(x, y, z)`.
fn leaf_bit(x: u32, y: u32, z: u32) -> u32 {
    let block = (((y & 7u) >> 1u) << 4u) | (((z & 7u) >> 1u) << 2u) | ((x & 7u) >> 1u);
    return 1u << (block & 31u);
}

/// Whether the 2^3 leaf block containing `(x, y, z)` has any non-empty voxel.
fn leaf_occupied(x: u32, y: u32, z: u32) -> bool {
    let block = (((y & 7u) >> 1u) << 4u) | (((z & 7u) >> 1u) << 2u) | ((x & 7u) >> 1u);
    let word = leaf_word(x, y, z, block >= 32u);
    return (leaf_masks[word] & (1u << (block & 31u))) != 0u;
}

/// Word offset of coarse pyramid level `level` (0 = 4^3 cells, 5 = 128^3 cells).
///
/// These are `tpt_core::voxel3d::COARSE_LEVEL_BASES`: the levels are stored
/// back-to-back and level 5 owns the single word at 9362.
fn coarse_base(level: u32) -> u32 {
    if (level == 0u) {
        return 0u;
    }
    if (level == 1u) {
        return 8192u;
    }
    if (level == 2u) {
        return 9216u;
    }
    if (level == 3u) {
        return 9344u;
    }
    if (level == 4u) {
        return 9360u;
    }
    return 9362u;
}

/// Whether the coarse cell of `level` containing `(x, y, z)` is occupied.
fn coarse_occupied(level: u32, x: u32, y: u32, z: u32) -> bool {
    let shift = 2u + level;
    let dim_log2 = 6u - level;
    let index = ((y >> shift) << (2u * dim_log2)) | ((z >> shift) << dim_log2) | (x >> shift);
    let word = coarse_base(level) + (index >> 5u);
    return (coarse_masks[word] & (1u << (index & 31u))) != 0u;
}

/// Largest empty block size containing the voxel, if the coarse pyramid can
/// skip it; returns 1 when the voxel must be inspected.
fn coarse_skip_size(x: u32, y: u32, z: u32) -> u32 {
    var level = 6u;
    loop {
        if (level == 0u) {
            break;
        }
        level = level - 1u;
        if (!coarse_occupied(level, x, y, z)) {
            return 1u << (2u + level);
        }
    }
    return 1u;
}

/// Replaces near-zero ray components so slab tests never divide by zero.
fn safe_dir(rd: vec3<f32>) -> vec3<f32> {
    let eps = 1.0e-20;
    return vec3<f32>(
        select(rd.x, select(eps, -eps, rd.x < 0.0), abs(rd.x) < eps),
        select(rd.y, select(eps, -eps, rd.y < 0.0), abs(rd.y) < eps),
        select(rd.z, select(eps, -eps, rd.z < 0.0), abs(rd.z) < eps),
    );
}

struct SlabHit {
    hit: bool,
    t_enter: f32,
    t_exit: f32,
};

/// Ray/AABB intersection clipped to `[t_min, t_max]`.
fn ray_aabb(
    ro: vec3<f32>,
    rd: vec3<f32>,
    box_min: vec3<f32>,
    box_max: vec3<f32>,
    t_min: f32,
    t_max: f32,
) -> SlabHit {
    let inv = vec3<f32>(1.0, 1.0, 1.0) / safe_dir(rd);
    let t0 = (box_min - ro) * inv;
    let t1 = (box_max - ro) * inv;
    let t_small = min(t0, t1);
    let t_big = max(t0, t1);
    let enter = max(max(max(t_small.x, t_small.y), t_small.z), t_min);
    let exit = min(min(min(t_big.x, t_big.y), t_big.z), t_max);

    var result: SlabHit;
    result.hit = exit >= enter;
    result.t_enter = enter;
    result.t_exit = exit;
    return result;
}

/// Parameter at which the ray leaves the axis-aligned block of `size` voxels
/// containing `cell`, starting from parameter `t`.
fn block_exit_t(
    ro: vec3<f32>,
    rd: vec3<f32>,
    cell: vec3<u32>,
    size: u32,
    t: f32,
) -> f32 {
    let mask = ~(size - 1u);
    var t_out = 1.0e30;
    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        let dir = rd[axis];
        if (abs(dir) > 1.0e-8) {
            let block_min = f32(cell[axis] & mask);
            let boundary = select(block_min, block_min + f32(size), dir > 0.0);
            let axis_t = (boundary - ro[axis]) / dir;
            t_out = min(t_out, axis_t);
        }
    }
    return max(t_out, t + 1.0e-4);
}

/// Unit normal of the voxel face the ray enters through.
fn voxel_normal(p: vec3<f32>, cell: vec3<u32>, rd: vec3<f32>) -> vec3<f32> {
    let local = p - vec3<f32>(cell);
    let d = min(local, vec3<f32>(1.0, 1.0, 1.0) - local);
    if (d.x <= d.y && d.x <= d.z) {
        return vec3<f32>(select(1.0, -1.0, rd.x > 0.0), 0.0, 0.0);
    }
    if (d.y <= d.z) {
        return vec3<f32>(0.0, select(1.0, -1.0, rd.y > 0.0), 0.0);
    }
    return vec3<f32>(0.0, 0.0, select(1.0, -1.0, rd.z > 0.0));
}

/// One opaque surface hit.
///
/// `life` is the packed 8-bit life of the hit voxel; the lighting pass needs it
/// so life-modulated emission (FIRE, PLSM) matches the volume path.
struct SurfaceHit {
    hit: bool,
    t: f32,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    emission: vec3<f32>,
    ty: u32,
    life: u32,
};

fn no_surface() -> SurfaceHit {
    var surface: SurfaceHit;
    surface.hit = false;
    surface.t = 0.0;
    surface.normal = vec3<f32>(0.0, 1.0, 0.0);
    surface.albedo = vec3<f32>(0.0);
    surface.emission = vec3<f32>(0.0);
    surface.ty = 0u;
    surface.life = 0u;
    return surface;
}

/// World traversal result: the first opaque surface plus the volume
/// inscatter/transmittance accumulated in front of it.
struct WorldTrace {
    surface: SurfaceHit,
    inscatter: vec3<f32>,
    transmittance: f32,
};

/// Hierarchical world DDA (plan sections 8.2 and 8.3).
///
/// The ray is only traversed between `t_enter`/`t_exit` (the world box) and
/// stops at `t_limit` (a rigid body hit or the box exit). Empty 2^3 leaf blocks
/// and empty coarse cells are skipped without touching the voxels; volume
/// elements accumulate radiance in `inscatter`, everything else is opaque.
fn trace_world(
    ro: vec3<f32>,
    rd: vec3<f32>,
    t_enter: f32,
    t_exit: f32,
    t_limit: f32,
) -> WorldTrace {
    var result: WorldTrace;
    result.surface = no_surface();
    result.inscatter = vec3<f32>(0.0);
    result.transmittance = 1.0;

    var t = max(t_enter, 0.0);
    let limit = min(t_exit, t_limit);
    var steps = 0u;

    loop {
        if (t >= limit || steps >= MAX_WORLD_STEPS) {
            break;
        }
        if (result.transmittance < TRANSMITTANCE_CUTOFF) {
            break;
        }
        steps = steps + 1u;

        let p = ro + rd * t;
        let cell = vec3<u32>(clamp(floor(p), vec3<f32>(0.0), vec3<f32>(VOXEL_LAST)));

        var size = coarse_skip_size(cell.x, cell.y, cell.z);
        if (size == 1u && !leaf_occupied(cell.x, cell.y, cell.z)) {
            size = 2u;
        }
        let t_next = block_exit_t(ro, rd, cell, size, t);

        if (size == 1u) {
            let word = voxel_a[(cell.y << 16u) | (cell.z << 8u) | cell.x];
            let ty = voxel_type(word);
            if (ty != 0u) {
                if (palette_is_volume(ty)) {
                    let sigma = palette_sigma(ty);
                    let ds = max(t_next - t, 0.0);
                    let step_t = exp(-sigma * ds);
                    let ambient_light = scene.ambient.rgb * scene.ambient.a;
                    let radiance = palette_albedo(ty) * ambient_light
                        + palette_emission(ty, voxel_life(word));
                    result.inscatter += result.transmittance * (1.0 - step_t) * radiance;
                    result.transmittance *= step_t;
                } else {
                    var surface: SurfaceHit;
                    surface.hit = true;
                    surface.t = t;
                    surface.normal = voxel_normal(p, cell, rd);
                    surface.albedo = palette_albedo(ty);
                    surface.emission = palette_emission(ty, voxel_life(word));
                    surface.ty = ty;
                    surface.life = voxel_life(word);
                    result.surface = surface;
                    break;
                }
            }
        }

        if (t_next <= t) {
            break;
        }
        t = t_next;
    }

    return result;
}

/// Traverses one rigid body's local 32^3 grid.
///
/// `ro`/`rd` are in the body's local voxel space; the returned `t` is the same
/// ray parameter as the world-space ray because the model transform is affine.
fn trace_body_local(
    ro: vec3<f32>,
    rd: vec3<f32>,
    t_min: f32,
    t_max: f32,
    grid_offset: u32,
    dims: vec3<u32>,
    mask_base: u32,
) -> SurfaceHit {
    var surface = no_surface();
    let dir = safe_dir(rd);
    let grid_max = vec3<f32>(dims);
    let slab = ray_aabb(ro, dir, vec3<f32>(0.0), grid_max, t_min, t_max);
    if (!slab.hit) {
        return surface;
    }

    var t = max(slab.t_enter, t_min);
    var steps = 0u;

    loop {
        if (t >= slab.t_exit || steps >= MAX_BODY_STEPS) {
            break;
        }
        steps = steps + 1u;

        let p = ro + dir * t;
        let cell = vec3<u32>(clamp(floor(p), vec3<f32>(0.0), grid_max - vec3<f32>(1.0)));
        let block = (((cell.y >> 1u) << 8u) | ((cell.z >> 1u) << 4u) | (cell.x >> 1u));
        let mask_word = body_masks[mask_base + (block >> 5u)];
        let occupied = (mask_word & (1u << (block & 31u))) != 0u;

        var size = 1u;
        if (!occupied) {
            size = 2u;
        }
        let t_next = block_exit_t(ro, dir, cell, size, t);

        if (size == 1u) {
            let local_index = (cell.y << 10u) | (cell.z << 5u) | cell.x;
            let word = body_voxels[grid_offset + local_index];
            let ty = voxel_type(word);
            if (ty != 0u) {
                surface.hit = true;
                surface.t = t;
                surface.normal = voxel_normal(p, cell, dir);
                surface.albedo = palette_albedo(ty);
                surface.emission = palette_emission(ty, voxel_life(word));
                surface.ty = ty;
                surface.life = voxel_life(word);
                return surface;
            }
        }

        if (t_next <= t) {
            break;
        }
        t = t_next;
    }

    return surface;
}

/// Nearest rigid body surface along `[t_min, t_max]`, if any.
///
/// Bodies are rejected with a cheap world-space AABB slab test before their
/// inverse model is read (plan section 8.2).
fn trace_bodies(ro: vec3<f32>, rd: vec3<f32>, t_min: f32, t_max: f32) -> SurfaceHit {
    var best = no_surface();
    var best_t = t_max;
    let count = min(u32(scene.body.x + 0.5), MAX_BODIES);

    for (var i = 0u; i < count; i = i + 1u) {
        let base = i * BODY_DESC_VEC4;
        let aabb_min = body_desc[base + 32u / 4u].xyz;
        let aabb_max = body_desc[base + 36u / 4u].xyz;
        let slab = ray_aabb(ro, rd, aabb_min, aabb_max, t_min, best_t);
        if (!slab.hit) {
            continue;
        }

        let inv_model = mat4x4<f32>(
            body_desc[base + 16u / 4u],
            body_desc[base + 20u / 4u],
            body_desc[base + 24u / 4u],
            body_desc[base + 28u / 4u],
        );
        let grid_offset = i * BODY_GRID_VOXELS;
        let dims = vec3<u32>(BODY_GRID_DIM, BODY_GRID_DIM, BODY_GRID_DIM);

        let local_ro = (inv_model * vec4<f32>(ro, 1.0)).xyz;
        let local_rd = (inv_model * vec4<f32>(rd, 0.0)).xyz;
        let hit = trace_body_local(
            local_ro,
            local_rd,
            slab.t_enter,
            min(slab.t_exit, best_t),
            grid_offset,
            dims,
            i * BODY_MASK_WORDS,
        );
        if (hit.hit && hit.t < best_t) {
            best_t = hit.t;
            best = hit;
            // Local normal -> world: inverse-model transposed (no scale in v1,
            // but the transpose keeps the transform correct for uniform scale).
            let model3 = mat3x3<f32>(
                inv_model[0].xyz,
                inv_model[1].xyz,
                inv_model[2].xyz,
            );
            best.normal = normalize(transpose(model3) * hit.normal);
        }
    }

    return best;
}

/// Whether an opaque voxel or rigid body blocks the ray within `max_t`.
fn trace_occluded(ro: vec3<f32>, rd: vec3<f32>, max_t: f32) -> bool {
    let world = trace_world(ro, rd, 0.0, max_t, max_t);
    if (world.surface.hit) {
        return true;
    }
    let body = trace_bodies(ro, rd, 0.0, max_t);
    return body.hit;
}

/// Two-ray cosine-weighted ambient occlusion.
fn ambient_occlusion(pos: vec3<f32>, normal: vec3<f32>, seed: f32) -> f32 {
    let up = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0),
        abs(normal.y) > 0.9,
    );
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    var occluded = 0.0;
    for (var i = 0u; i < 2u; i = i + 1u) {
        let angle = seed + f32(i) * 2.3999632;
        let dir = normalize(
            normal + 0.65 * (tangent * cos(angle) + bitangent * sin(angle)),
        );
        if (trace_occluded(pos + normal * 0.05, dir, AO_DISTANCE)) {
            occluded += 1.0;
        }
    }
    return 1.0 - 0.5 * (occluded * 0.5);
}

/// Cheap per-pixel hash used to decorrelate the AO rays.
fn hash12(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

/// Converts a ray parameter to the DirectX-style depth the projection expects.
fn ray_depth(t: f32, rd: vec3<f32>) -> f32 {
    let near = scene.right.w;
    let far = scene.up.w;
    let view_depth = max(t * dot(rd, scene.forward.xyz), near);
    return clamp((far / (far - near)) * (1.0 - near / view_depth), 0.0, 1.0);
}
