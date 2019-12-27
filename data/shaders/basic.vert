#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformStruct {
    mat4 model_matrix;
    mat4 view_matrix;
    mat4 proj_matrix;
} ubo;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec4 in_col;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_norm;
//Adding Skinned inputs so layout matches gltf model vertices
layout(location = 4) in vec4 in_joint_indices;
layout(location = 5) in vec4 in_joint_weights;

layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec3 frag_pos;

void main() {
    frag_color = in_col;
    gl_Position  = ubo.proj_matrix * ubo.view_matrix * ubo.model_matrix * vec4(in_pos, 1.0);
    frag_pos = (ubo.model_matrix * vec4(in_pos, 1.0)).xyz;
}