// Taken from niri and modified, licensed GPL-3.0

#version 100

//_DEFINES_

#if defined(EXTERNAL)
#extension GL_OES_EGL_image_external : require
#endif

precision highp float;
#if defined(EXTERNAL)
uniform samplerExternalOES tex;
#else
uniform sampler2D tex;
#endif

uniform float alpha;
varying vec2 v_coords;

#if defined(DEBUG_FLAGS)
uniform float tint;
#endif

uniform vec2 geo_size;
uniform vec4 corner_radius;
uniform mat3 input_to_geo;
uniform float noise;

uniform float hdr_enabled;
uniform float hdr_reference_white;
uniform float hdr_sdr_gamma;
uniform float hdr_gamut_stretch;

// Linear Rec.709 to linear BT.2020. GLSL constructors are column-major.
const mat3 rec709_to_bt2020 = mat3(
    0.6274040, 0.0690970, 0.0163916,
    0.3292820, 0.9195400, 0.0880132,
    0.0433136, 0.0113612, 0.8955950
);

// Linear Display P3 to linear BT.2020. GLSL constructors are column-major.
const mat3 p3_to_bt2020 = mat3(
    0.7538330, 0.0457438, -0.0012103,
    0.1985974, 0.9417772, 0.0176017,
    0.0475696, 0.0124789, 0.9836086
);

// Linear BT.2020 to linear Rec.709. GLSL constructors are column-major.
const mat3 bt2020_to_rec709 = mat3(
     1.6604903, -0.1245500, -0.0181511,
    -0.5876391,  1.1328999, -0.1005787,
    -0.0728516, -0.0083480,  1.1187299
);

// Primaries selector: 0.0 = Rec.709 / sRGB, 1.0 = Display P3, 2.0 = BT.2020 (identity)
uniform float hdr_input_primaries;

vec3 convert_primaries(vec3 linear_rgb) {
    if (hdr_input_primaries > 1.5) {
        return linear_rgb;
    } else if (hdr_input_primaries > 0.5) {
        return p3_to_bt2020 * linear_rgb;
    } else {
        return mix(
            rec709_to_bt2020 * linear_rgb,
            linear_rgb,
            clamp(hdr_gamut_stretch, 0.0, 1.0)
        );
    }
}

float decode_sdr(float value) {
    if (hdr_sdr_gamma == 1.0) {
        return value;
    }
    if (hdr_sdr_gamma > 0.0) {
        return pow(max(value, 0.0), hdr_sdr_gamma);
    }
    return value <= 0.04045
        ? value / 12.92
        : pow((value + 0.055) / 1.055, 2.4);
}

vec3 decode_sdr_v(vec3 value) {
    if (hdr_sdr_gamma == 1.0) {
        return value;
    }
    if (hdr_sdr_gamma > 0.0) {
        return pow(max(value, vec3(0.0)), vec3(hdr_sdr_gamma));
    }
    vec3 low = value / 12.92;
    vec3 high = pow((value + 0.055) / 1.055, vec3(2.4));
    vec3 cutoff = step(vec3(0.04045), value);
    return mix(low, high, cutoff);
}

float encode_sdr(float value) {
    if (hdr_sdr_gamma == 1.0) {
        return value;
    }
    if (hdr_sdr_gamma > 0.0) {
        return pow(max(value, 0.0), 1.0 / hdr_sdr_gamma);
    }
    return value <= 0.0031308
        ? value * 12.92
        : 1.055 * pow(max(value, 0.0), 1.0 / 2.4) - 0.055;
}

// ST 2084 inverse EOTF: absolute luminance in the normalized 10000 cd/m^2 domain to a PQ code value.
float encode_pq(float value) {
    const float m1 = 0.1593017578125; // 2610 / 16384
    const float m2 = 78.84375;        // 2523 / 4096 * 128
    const float c1 = 0.8359375;       // 3424 / 4096
    const float c2 = 18.8515625;      // 2413 / 4096 * 32
    const float c3 = 18.6875;         // 2392 / 4096 * 32
    float p = pow(max(value, 0.0), m1);
    return pow((c1 + c2 * p) / (1.0 + c3 * p), m2);
}

vec3 encode_pq_v(vec3 value) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 p = pow(max(value, vec3(0.0)), vec3(m1));
    return pow((c1 + c2 * p) / (1.0 + c3 * p), vec3(m2));
}

// ST 2084 EOTF: PQ code value to luminance in the normalized 10000 cd/m^2 domain.
float pq_to_linear(float code) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    float p = pow(max(code, 0.0), 1.0 / m2);
    return pow(max(p - c1, 0.0) / (c2 - c3 * p), 1.0 / m1);
}

vec3 pq_to_linear_v(vec3 code) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 p = pow(max(code, vec3(0.0)), vec3(1.0 / m2));
    return pow(max(p - c1, vec3(0.0)) / (c2 - c3 * p), vec3(1.0 / m1));
}

uniform float hdr_input_pq;          // 1.0 = buffer holds PQ code values
uniform float hdr_input_hlg;         // 1.0 = buffer holds HLG code values (BT.2100)
uniform float hdr_content_reference; // content reference white in cd/m²
uniform float hdr_max_content_luminance;
uniform float hdr_max_destination_luminance;
uniform float hdr_hardware_offload;  // 1.0 = CRTC hardware offloads PQ encoding via GAMMA_LUT
uniform float hdr_target_is_sdr;     // 1.0 = Target output is SDR display (sRGB)

// SDR to PQ/BT.2020, with the reference white defining where linear SDR 1.0 lands in absolute luminance.
vec3 sdr_to_pq(vec3 rgb) {
    if (hdr_target_is_sdr > 0.5) {
        if (hdr_input_primaries < 0.5 && hdr_sdr_gamma == 0.0) {
            return rgb;
        }
        vec3 linear_rgb = decode_sdr_v(rgb);
        vec3 bt2020_rgb = convert_primaries(linear_rgb);
        vec3 rec709_rgb = clamp(bt2020_to_rec709 * bt2020_rgb, 0.0, 1.0);
        return vec3(
            encode_sdr(rec709_rgb.r),
            encode_sdr(rec709_rgb.g),
            encode_sdr(rec709_rgb.b)
        );
    }
    vec3 linear_rgb = decode_sdr_v(rgb);
    if (hdr_hardware_offload > 0.5) {
        // Hardware offload path: CRTC GAMMA_LUT handles PQ encoding on scanout.
        // We output normalized linear BT.2020 in [0.0, 1.0] where 1.0 = reference_white.
        return max(convert_primaries(linear_rgb), 0.0);
    }
    vec3 absolute = max(convert_primaries(linear_rgb), 0.0)
        * (clamp(hdr_reference_white, 80.0, 10000.0) / 10000.0);
    return encode_pq_v(absolute);
}

// ITU-R BT.2100 / SMPTE RP 2092 ICtCp color space matrices.
const mat3 to_ictcp = mat3(
    0.5,  1.613769531250,   4.378173828125,
    0.5, -3.323486328125, -4.245605468750,
    0.0,  1.709716796875, -0.132568359375
);

const mat3 from_ictcp = mat3(
    1.0,               1.0,               1.0,
    0.00860903703793, -0.00860903703793,  0.56003133571068,
    0.11102962500303, -0.11102962500303, -0.32062717498732
);

// BT.2020 linear RGB to Dolby LMS (for ICtCp).
const mat3 bt2020_to_lms = mat3(
    0.412109375, 0.166748046875, 0.024169921875,
    0.52392578125, 0.720458984375, 0.075439453125,
    0.06396484375, 0.11279296875, 0.900390625
);

const mat3 lms_to_bt2020 = mat3(
    3.43660669, -0.79132956, -0.02594990,
    -2.50645212, 1.98360045, -0.09891371,
    0.06984542, -0.19227090, 1.12486361
);

// Performs Reinhard tonemapping on the Intensity channel in ICtCp space
vec3 tonemap_ictcp(vec3 linear_rgb) {
    if (hdr_max_content_luminance <= hdr_max_destination_luminance * 1.01) {
        return clamp(linear_rgb, vec3(0.0), vec3(hdr_max_destination_luminance / 10000.0));
    }

    vec3 lms = bt2020_to_lms * linear_rgb;
    vec3 lms_pq = encode_pq_v(lms);
    vec3 ictcp = to_ictcp * lms_pq;

    // Luminance in nits
    float lum = pq_to_linear(ictcp.r) * 10000.0;

    // Modified Reinhard roll-off
    float rel_lum = max(lum / hdr_reference_white, 0.0);
    float in_range = hdr_max_content_luminance / hdr_reference_white;
    float out_range = hdr_max_destination_luminance / hdr_reference_white;
    float v = (out_range * (1.0 + in_range) - in_range) / (in_range * in_range);
    rel_lum = rel_lum * (1.0 + rel_lum * v) / (1.0 + rel_lum);
    lum = rel_lum * hdr_reference_white;

    ictcp.r = encode_pq(lum / 10000.0);
    vec3 mapped_lms_pq = from_ictcp * ictcp;
    vec3 mapped_lms = pq_to_linear_v(mapped_lms_pq);
    vec3 mapped_rgb = lms_to_bt2020 * mapped_lms;
    return clamp(mapped_rgb, vec3(0.0), vec3(hdr_max_destination_luminance / 10000.0));
}

vec3 pq_rescale(vec3 code) {
    float ref_scale = clamp(hdr_reference_white, 80.0, 10000.0)
        / max(hdr_content_reference, 80.0);
    vec3 linear_rgb = pq_to_linear_v(code) * ref_scale;
    linear_rgb = tonemap_ictcp(linear_rgb);
    if (hdr_target_is_sdr > 0.5) {
        float inv_white = 10000.0 / clamp(hdr_reference_white, 80.0, 10000.0);
        vec3 linear_sdr_bt2020 = linear_rgb * inv_white;
        vec3 linear_sdr_rec709 = clamp(bt2020_to_rec709 * linear_sdr_bt2020, 0.0, 1.0);
        return vec3(
            encode_sdr(linear_sdr_rec709.r),
            encode_sdr(linear_sdr_rec709.g),
            encode_sdr(linear_sdr_rec709.b)
        );
    }
    if (hdr_hardware_offload > 0.5) {
        float inv_norm = 10000.0 / clamp(hdr_reference_white, 80.0, 10000.0);
        return linear_rgb * inv_norm;
    }
    return encode_pq_v(linear_rgb);
}

// HLG inverse OETF
float hlg_to_scene(float e) {
    const float a = 0.17883277;
    const float b = 0.28466892;
    const float c = 0.55991073;
    e = clamp(e, 0.0, 1.0);
    if (e <= 0.5) {
        return (e * e) / 3.0;
    } else {
        return (exp((e - c) / a) + b) / 12.0;
    }
}

// HLG (BT.2020 primaries) to PQ/BT.2020
vec3 hlg_to_pq(vec3 hlg) {
    vec3 scene = vec3(
        hlg_to_scene(hlg.r),
        hlg_to_scene(hlg.g),
        hlg_to_scene(hlg.b)
    );
    float ys = dot(scene, vec3(0.2627, 0.6780, 0.0593));
    float gain = pow(max(ys, 1e-6), 0.2) * 0.10;
    float ref_scale = clamp(hdr_reference_white, 80.0, 10000.0)
        / max(hdr_content_reference, 80.0);
    vec3 display = scene * (gain * ref_scale);
    display = tonemap_ictcp(display);
    if (hdr_target_is_sdr > 0.5) {
        float inv_white = 10000.0 / clamp(hdr_reference_white, 80.0, 10000.0);
        vec3 linear_sdr_bt2020 = display * inv_white;
        vec3 linear_sdr_rec709 = clamp(bt2020_to_rec709 * linear_sdr_bt2020, 0.0, 1.0);
        return vec3(
            encode_sdr(linear_sdr_rec709.r),
            encode_sdr(linear_sdr_rec709.g),
            encode_sdr(linear_sdr_rec709.b)
        );
    }
    if (hdr_hardware_offload > 0.5) {
        float inv_norm = 10000.0 / clamp(hdr_reference_white, 80.0, 10000.0);
        return display * inv_norm;
    }
    return encode_pq_v(display);
}

float optical_alpha_pq(float a) {
    if (hdr_target_is_sdr > 0.5 || hdr_hardware_offload > 0.5) {
        return a;
    }
    if (a <= 0.0001) return 0.0;
    if (a >= 0.9999) return 1.0;
    float white_norm = clamp(hdr_reference_white, 80.0, 10000.0) / 10000.0;
    float pq_white = encode_pq(white_norm);
    float pq_lum = encode_pq(white_norm * a);
    return clamp(pq_lum / max(pq_white, 0.001), 0.0, 1.0);
}

float rounding_alpha(vec2 coords, vec2 size) {
    vec2 center;
    float radius;

    if (coords.x < corner_radius.x && coords.y < corner_radius.x) {
        radius = corner_radius.x;
        center = vec2(radius, radius);
    } else if (size.x - corner_radius.y < coords.x && coords.y < corner_radius.y) {
        radius = corner_radius.y;
        center = vec2(size.x - radius, radius);
    } else if (size.x - corner_radius.z < coords.x && size.y - corner_radius.z < coords.y) {
        radius = corner_radius.z;
        center = vec2(size.x - radius, size.y - radius);
    } else if (coords.x < corner_radius.w && size.y - corner_radius.w < coords.y) {
        radius = corner_radius.w;
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
    if (alpha <= 0.0) {
        discard;
    }

    vec4 color = texture2D(tex, v_coords);
    #if defined(NO_ALPHA)
    color = vec4(color.rgb, 1.0);
    #endif
    if (color.a <= 0.0) {
        discard;
    }

    vec3 coords_geo = input_to_geo * vec3(v_coords, 1.0);
    if (coords_geo.x < 0.0 || 1.0 < coords_geo.x || coords_geo.y < 0.0 || 1.0 < coords_geo.y) {
        // Clip outside geometry.
        color = vec4(0.0);
    } else {
        // Apply corner rounding inside geometry.
        color = color * rounding_alpha(coords_geo.xy * geo_size, geo_size);
    }

    if (color.a <= 0.0) {
        discard;
    }

    if (noise > 0.0) {
        // Add noise fx
        // This can be used to achieve a glass look
        float noiseHash = hash(v_coords);
        float noiseAmount = (mod(noiseHash, 1.0) - 0.5);
        color.rgb += noiseAmount * noise;
    }

    if (hdr_enabled > 0.5) {
        vec3 rgb = color.a > 0.00001 ? color.rgb / color.a : vec3(0.0);
        if (hdr_input_pq > 0.5) {
            rgb = pq_rescale(rgb);
        } else if (hdr_input_hlg > 0.5) {
            rgb = hlg_to_pq(rgb);
        } else {
            rgb = sdr_to_pq(rgb);
        }
        float total_alpha = clamp(color.a * alpha, 0.0, 1.0);
        color.rgb = rgb * total_alpha;
        color.a = total_alpha;
    } else {
        color *= alpha;
    }

    #if defined(DEBUG_FLAGS)
    if (tint == 1.0)
        color = vec4(0.0, 0.2, 0.0, 0.2) + color * 0.8;
    #endif

    gl_FragColor = color;
}
