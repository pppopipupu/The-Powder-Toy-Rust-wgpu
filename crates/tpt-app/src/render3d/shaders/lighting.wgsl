// Fullscreen lighting pass.
//
// Reads the G-buffer written by the world OBB pass, reconstructs the world
// position from depth, shoots one sun shadow ray and two ambient-occlusion
// rays through the same hierarchy, composites the volume inscatter and writes
// the tone-mapped, gamma-encoded frame into the presentation target (plan
// section 8.4).

@group(1) @binding(0) var depth_target: texture_depth_2d;
@group(1) @binding(1) var rt0_target: texture_2d<f32>;
@group(1) @binding(2) var rt1_target: texture_2d<f32>;
@group(1) @binding(3) var rt2_target: texture_2d<f32>;

struct LightingVertex {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_lighting(@builtin(vertex_index) vertex_index: u32) -> LightingVertex {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: LightingVertex;
    out.clip_position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    return out;
}

/// Tone maps and gamma-encodes a linear radiance value.
fn present(linear: vec3<f32>) -> vec3<f32> {
    let exposed = vec3<f32>(1.0, 1.0, 1.0) - exp(-max(linear, vec3<f32>(0.0)));
    return pow(exposed, vec3<f32>(1.0 / 2.2));
}

@fragment
fn fs_lighting(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let coord = vec2<i32>(frag_coord.xy);
    let rt2 = textureLoad(rt2_target, coord, 0);
    let depth = textureLoad(depth_target, coord, 0);

    if (depth >= 1.0) {
        // Sky plus whatever volume inscatter was accumulated along the ray.
        // `SKY_COLOR` is already gamma encoded and matches the host clear
        // colour, so an empty ray is bit-identical to the cleared target.
        let haze = present(rt2.rgb * scene.sun_dir.w);
        return vec4<f32>(SKY_COLOR * rt2.a + haze, 1.0);
    }

    let rt0 = textureLoad(rt0_target, coord, 0);
    let rt1 = textureLoad(rt1_target, coord, 0);

    let uv = frag_coord.xy / scene.target_size.xy;
    let ndc = vec3<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth);
    let clip = scene.inv_view_proj * vec4<f32>(ndc, 1.0);
    let pos = clip.xyz / clip.w;

    let normal = normalize(rt1.xyz);
    // `obb.wgsl` writes `ty / 255` into rt0.a and the voxel life into rt1.w.
    let ty = u32(rt0.a * 255.0 + 0.5);
    let life = u32(rt1.w + 0.5);
    let albedo = rt0.rgb;
    let emissive = palette_emission(ty, life);

    let features = u32(scene.body.y + 0.5);
    let sun = normalize(scene.sun_dir.xyz);

    var shadow = 1.0;
    if ((features & 1u) != 0u) {
        if (trace_occluded(pos + normal * 0.05, sun, SHADOW_DISTANCE)) {
            shadow = 0.0;
        }
    }

    var ao = 1.0;
    if ((features & 2u) != 0u) {
        ao = ambient_occlusion(pos, normal, hash12(frag_coord.xy));
    }

    let sun_term = max(dot(normal, sun), 0.0) * shadow;
    let ambient = scene.ambient.rgb * (scene.ambient.a * ao);
    let direct = scene.sun_color.rgb * (scene.sun_color.a * sun_term);
    let lit = albedo * (ambient + direct) + emissive;

    let linear = lit * rt2.a + rt2.rgb;
    let exposed = vec3<f32>(1.0, 1.0, 1.0) - exp(-max(linear * scene.sun_dir.w, vec3<f32>(0.0)));
    let encoded = pow(exposed, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(encoded, 1.0);
}
