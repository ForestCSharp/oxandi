#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(early_fragment_tests) in;

struct Light {
    vec3 position;
    float strength;
    vec3 color;
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

    //FIXME: iterate over all point lights, use their color to affect mesh's color
    vec4 accumulated_color = vec4(0.0);
    
    float distance_to_light = length(lights.point_lights[0].position - frag_pos);
    float att = clamp(1.0 - distance_to_light/lights.point_lights[0].radius, 0.0, 1.0);
    att *= att;

    color = vec4(vec3(0.0, 1.0, 0.0) * att * lights.point_lights[0].strength, 1.0);
}