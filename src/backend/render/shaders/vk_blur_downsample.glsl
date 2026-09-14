#version 450
#extension GL_EXT_shader_image_load_formatted : enable

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0) uniform image2D dst;
layout(binding = 1) uniform sampler2D tex;

layout(push_constant, std140) uniform PushConstants {
    vec2 half_pixel;
    float offset;
    uint is_hdr;
    uvec2 dst_size;
    vec2 src_uv_scale;
} params;

// ST 2084 EOTF: PQ code value to linear luminance (normalized 0..1 = 10000 nits)
const float pq_m1 = 0.1593017578125;
const float pq_m2 = 78.84375;
const float pq_c1 = 0.8359375;
const float pq_c2 = 18.8515625;
const float pq_c3 = 18.6875;

vec3 pq_to_linear(vec3 code) {
    vec3 p = pow(max(code, vec3(0.0)), vec3(1.0 / pq_m2));
    return pow(max(p - pq_c1, vec3(0.0)) / (pq_c2 - pq_c3 * p), vec3(1.0 / pq_m1));
}

vec3 linear_to_pq(vec3 value) {
    vec3 p = pow(max(value, vec3(0.0)), vec3(pq_m1));
    return pow((pq_c1 + pq_c2 * p) / (1.0 + pq_c3 * p), vec3(pq_m2));
}

vec4 sample_color(vec2 uv) {
    vec4 c = texture(tex, uv);
    if (params.is_hdr != 0) {
        c.rgb = pq_to_linear(c.rgb);
    }
    return c;
}

void main() {
    uvec2 coord = gl_GlobalInvocationID.xy;
    if (coord.x >= params.dst_size.x || coord.y >= params.dst_size.y) {
        return;
    }

    vec2 uv = ((vec2(coord) + vec2(0.5)) / vec2(params.dst_size)) * params.src_uv_scale;
    vec2 half_pixel = params.half_pixel;
    float offset = params.offset;

    vec4 sum = sample_color(uv) * 4.0;
    sum += sample_color(uv - half_pixel * offset);
    sum += sample_color(uv + half_pixel * offset);
    sum += sample_color(uv + vec2(half_pixel.x, -half_pixel.y) * offset);
    sum += sample_color(uv - vec2(half_pixel.x, -half_pixel.y) * offset);

    vec4 color;
    if (sum.a <= 0.0001) {
        color = vec4(0.0);
    } else {
        if (params.is_hdr != 0) {
            vec3 lin = sum.rgb / 8.0;
            color = vec4(linear_to_pq(lin), sum.a / 8.0);
        } else {
            color = sum / sum.a;
        }
    }
    imageStore(dst, ivec2(coord), color);
}
