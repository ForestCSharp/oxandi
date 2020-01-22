#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(early_fragment_tests) in;

struct Light {
    vec3 color;
    float strength;
    float radius;
};

#define MAX_LIGHTS 10

layout(binding = 2) uniform LightUniform {
    Light point_lights[MAX_LIGHTS];
} lights;

layout(location = 0) in vec4 frag_color;
layout(location = 1) in vec3 frag_pos;
layout(location = 0) out vec4 color;

void main() {

    vec3 point_light_location = vec3(2, 2, 2);
    float point_light_radius = 6.0;

    float distance_to_light = length(point_light_location - frag_pos);
    float att = clamp(1.0 - distance_to_light/point_light_radius, 0.0, 1.0);
    att *= att;

    color = vec4(frag_color.xyz * att, frag_color.w);
}