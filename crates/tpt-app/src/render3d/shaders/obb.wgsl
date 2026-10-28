// World OBB pass.
//
// Exactly one draw of the 36-vertex world cube with back-face culling: the
// camera is either outside the box (back faces cover the world projection) or
// inside it (back faces cover the screen). The fragment stage traces the world
// and the rigid bodies itself and writes the G-buffer plus `frag_depth`; the
// lighting pass reads those targets back (plan sections 8.1 and 8.2).

struct WorldObbVertex {
    @builtin(position) clip_position: vec4<f32>,
};

struct WorldObbFragment {
    @location(0) rt0: vec4<f32>,
    @location(1) rt1: vec4<f32>,
    @location(2) rt2: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

const CUBE_CORNERS: array<vec3<f32>, 8> = array<vec3<f32>, 8>(
    vec3<f32>(0.0, 0.0, 0.0),
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(1.0, 1.0, 1.0),
    vec3<f32>(0.0, 1.0, 1.0),
);

// Outward-facing counter-clockwise triangles: two per cube face.
const CUBE_INDICES: array<u32, 36> = array<u32, 36>(
    5u, 1u, 2u, 5u, 2u, 6u,
    0u, 4u, 7u, 0u, 7u, 3u,
    3u, 7u, 6u, 3u, 6u, 2u,
    0u, 1u, 5u, 0u, 5u, 4u,
    4u, 5u, 6u, 4u, 6u, 7u,
    0u, 3u, 2u, 0u, 2u, 1u,
);

@vertex
fn vs_world_obb(@builtin(vertex_index) vertex_index: u32) -> WorldObbVertex {
    let corner = CUBE_CORNERS[CUBE_INDICES[vertex_index]];
    var out: WorldObbVertex;
    out.clip_position = scene.view_proj * vec4<f32>(corner * WORLD_MAX, 1.0);
    return out;
}

@fragment
fn fs_world_obb(@builtin(position) frag_coord: vec4<f32>) -> WorldObbFragment {
    let uv = frag_coord.xy / scene.target_size.xy;
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let ray_dir = normalize(
        scene.forward.xyz
            + scene.right.xyz * (ndc.x * scene.eye.w * scene.forward.w)
            + scene.up.xyz * (ndc.y * scene.eye.w),
    );
    let ray_origin = scene.eye.xyz;

    var out: WorldObbFragment;
    out.rt0 = vec4<f32>(0.0);
    out.rt1 = vec4<f32>(0.0);
    out.rt2 = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.depth = 1.0;

    let box_hit = ray_aabb(ray_origin, ray_dir, WORLD_MIN, WORLD_MAX, 0.0, 1.0e30);
    if (!box_hit.hit) {
        return out;
    }

    let t_enter = max(box_hit.t_enter, 0.0);
    let body = trace_bodies(ray_origin, ray_dir, t_enter, box_hit.t_exit);
    let limit = select(box_hit.t_exit, body.t, body.hit);
    let world = trace_world(ray_origin, ray_dir, t_enter, box_hit.t_exit, limit);

    var surface = no_surface();
    if (world.surface.hit && (!body.hit || world.surface.t <= body.t)) {
        surface = world.surface;
    } else if (body.hit) {
        surface = body;
    }

    if (surface.hit) {
        // G-buffer layout: rt0.a carries type / 255 (Rgba8Unorm reproduces an
        // integer id exactly) and rt1.w carries the voxel life (Rgba16Float
        // represents every integer in 0..255 exactly). Moving `ty` out of
        // rt1.w frees that channel for `life`, which the lighting pass needs
        // for life-modulated emission.
        out.rt0 = vec4<f32>(surface.albedo, f32(surface.ty) / 255.0);
        out.rt1 = vec4<f32>(surface.normal, f32(surface.life));
        out.rt2 = vec4<f32>(world.inscatter, world.transmittance);
        out.depth = ray_depth(surface.t, ray_dir);
    } else {
        // Nothing opaque: keep the volume inscatter for the lighting pass and
        // leave depth at the clear value so the sky term is used.
        out.rt2 = vec4<f32>(world.inscatter, world.transmittance);
    }

    return out;
}
