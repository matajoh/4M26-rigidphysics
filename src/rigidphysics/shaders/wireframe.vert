#version 330 core

uniform mat4 u_mv_matrix;
uniform mat4 u_proj_matrix;
uniform vec4 u_color;
uniform vec3 u_centroid;

layout(location=0) in vec3 v_position;

out vec4 v_color;

void main()
{
    v_color = u_color;
    gl_Position = u_proj_matrix * u_mv_matrix * vec4(v_position + u_centroid, 1.0);
}