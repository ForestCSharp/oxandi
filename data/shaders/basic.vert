#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformStruct {
    mat4 model_matrix;
    mat4 view_matrix;
    mat4 proj_matrix;
} ubo;

layout(location = 0) in vec3 pos;
layout(location = 1) in vec4 color;
layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec3 frag_pos;

void main() {
    frag_color = color;
    gl_Position  = ubo.proj_matrix * ubo.view_matrix * ubo.model_matrix * vec4(pos, 1.0);
    frag_pos = (ubo.model_matrix * vec4(pos, 1.0)).xyz;
}