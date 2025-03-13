#version 330 core

uniform mat4 u_mv_matrix;
uniform mat4 u_proj_matrix;
uniform mat4 u_norm_matrix;
uniform int u_transparent;
uniform vec2 u_checkerboard;

layout(location=0) in vec3 v_position;
layout(location=1) in vec3 v_normal;
layout(location=2) in vec2 v_uv;
layout(location=3) in vec3 v_instance_position;
layout(location=4) in vec4 v_instance_rotation;
layout(location=5) in vec3 v_instance_scale;
layout(location=6) in vec3 v_instance_color;
layout(location=7) in float v_instance_alpha;

out vec4 v_color;
out vec2 v_texture;
out vec3 v_position_view;
out vec3 v_normal_view;
out vec2 v_checkerboard;

void main()
{
    if(u_transparent == 1)
    {
        v_color = vec4(v_instance_color, v_instance_alpha);
    }
    else
    {
        v_color = vec4(v_instance_color, 1.0);
    }

    if(v_normal.x == 1.0 || v_normal.x == -1.0)
    {
        v_checkerboard = v_instance_scale.yz * u_checkerboard;
    }
    else if(v_normal.y == 1.0 || v_normal.y == -1.0)
    {
        v_checkerboard = v_instance_scale.zx * u_checkerboard;
    }
    else
    {
        v_checkerboard = v_instance_scale.yx * u_checkerboard;
    }

    v_texture = v_uv;    
    vec3 position = v_instance_scale * v_position;
    vec3 normal = v_normal;
    vec4 rotation = v_instance_rotation;
    position += 2.0 * cross(rotation.xyz, cross(rotation.xyz, position) + rotation.w * position);
    normal += 2.0 * cross(rotation.xyz, cross(rotation.xyz, normal) + rotation.w * normal);
    position += v_instance_position;
    gl_Position = u_proj_matrix * u_mv_matrix * vec4(position, 1.0);
    v_position_view = (u_mv_matrix * vec4(position, 1.0)).xyz;
    v_normal_view = normalize(u_norm_matrix * vec4(normal, 1.0)).xyz;
}