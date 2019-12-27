#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformStruct {
    mat4 model_matrix;
    mat4 view_matrix;
    mat4 proj_matrix;
} ubo;

#define MAX_BONE_COUNT 500

layout(binding = 1) uniform SkeletonUniform {
    mat4 bones[MAX_BONE_COUNT];
} skeleton;

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec4 in_col;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec3 in_norm;
layout(location = 4) in vec4 in_joint_indices;
layout(location = 5) in vec4 in_joint_weights;

layout(location = 0) out vec4 frag_color;
layout(location = 1) out vec3 frag_pos;

void main() {

    mat4 skinMatrix = mat4(0.0);

    for (int i=0; i<4; ++i)
    {
        skinMatrix += (in_joint_weights[i] * skeleton.bones[int(in_joint_indices[i])]);
    }

    //Skin matrix is identity if joint weights are all zero
    if ((abs(in_joint_weights[0] - 0.0)) < 0.000001)
    {
        skinMatrix = mat4(1.0);
    }

    frag_color = in_col;
    gl_Position  = ubo.proj_matrix * ubo.view_matrix * ubo.model_matrix * skinMatrix * vec4(in_pos, 1.0);
    frag_pos = (ubo.model_matrix * skinMatrix * vec4(in_pos, 1.0)).xyz;
}