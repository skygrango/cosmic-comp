#version 450
#extension GL_EXT_shader_image_load_formatted : enable

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) uniform image2D dst;
layout(binding = 1) uniform sampler2D tex;

layout(push_constant, std140) uniform PushConstants {
    vec4 dst_rect;               // x, y, width, height in target framebuffer
    vec2 geo_size;               // physical/logical size for corner rounding
    vec2 _pad0;
    vec4 corner_radius;          // [top-left, top-right, bottom-right, bottom-left]
    vec4 input_to_geo_col0;      // Column 0 of 3x3 matrix to map UV to geometry space [0..1]
    vec4 input_to_geo_col1;      // Column 1
    vec4 input_to_geo_col2;      // Column 2
    float noise;
    float alpha;
    uint is_hdr;
    uint is_bgr;
} params;

float rounding_alpha(vec2 coords, vec2 size, vec4 radii) {
    vec2 center;
    float radius;

    if (coords.x < radii.x && coords.y < radii.x) {
        radius = radii.x;
        center = vec2(radius, radius);
    } else if (size.x - radii.y < coords.x && coords.y < radii.y) {
        radius = radii.y;
        center = vec2(size.x - radius, radius);
    } else if (size.x - radii.z < coords.x && size.y - radii.z < coords.y) {
        radius = radii.z;
        center = vec2(size.x - radius, size.y - radius);
    } else if (coords.x < radii.w && size.y - radii.w < coords.y) {
        radius = radii.w;
        center = vec2(radius, size.y - radius);
    } else {
        return 1.0;
    }

    float dist = distance(coords, center);
    float half_px = 0.5;
    return 1.0 - smoothstep(radius - half_px, radius + half_px, dist);
}

float hash(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 727.727);
    p3 += dot(p3, p3.xyz + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    uvec2 local_coord = gl_GlobalInvocationID.xy;
    if (local_coord.x >= uint(params.dst_rect.z) || local_coord.y >= uint(params.dst_rect.w)) {
        return;
    }

    uvec2 fb_coord = local_coord + uvec2(params.dst_rect.xy);
    vec2 uv = (vec2(local_coord) + vec2(0.5)) / params.dst_rect.zw;

    mat3 input_to_geo = mat3(
        params.input_to_geo_col0.xyz,
        params.input_to_geo_col1.xyz,
        params.input_to_geo_col2.xyz
    );

    vec3 coords_geo = input_to_geo * vec3(uv, 1.0);
    if (coords_geo.x < 0.0 || coords_geo.x > 1.0 || coords_geo.y < 0.0 || coords_geo.y > 1.0) {
        return;
    }

    float round_a = rounding_alpha(coords_geo.xy * params.geo_size, params.geo_size, params.corner_radius);
    if (round_a <= 0.0) {
        return;
    }

    vec4 color = texture(tex, uv);
    float total_alpha = color.a * round_a * params.alpha;
    if (total_alpha <= 0.0001) {
        return;
    }

    if (params.noise > 0.0) {
        float noiseHash = hash(uv);
        float noiseAmount = (mod(noiseHash, 1.0) - 0.5) * params.noise;
        color.rgb = clamp(color.rgb + vec3(noiseAmount), 0.0, 1.0);
    }

    vec4 dst_color = imageLoad(dst, ivec2(fb_coord));
    if (params.is_bgr != 0) {
        dst_color = dst_color.bgra;
    }

    vec4 out_color;
    if (total_alpha >= 0.999) {
        out_color = vec4(color.rgb, 1.0);
    } else {
        out_color = vec4(color.rgb * total_alpha + dst_color.rgb * (1.0 - total_alpha), 1.0);
    }

    if (params.is_bgr != 0) {
        out_color = out_color.bgra;
    }
    imageStore(dst, ivec2(fb_coord), out_color);
}
