#version 330 core

uniform vec3 u_ambient_light_color;
uniform vec3 u_directional_light_dir;
uniform vec3 u_directional_light_color;
uniform int u_shading_type;

in vec4 v_color;
in vec2 v_texture;
in vec3 v_position_view;
in vec3 v_normal_view;
in vec2 v_checkerboard;

out vec4 f_color;

float FresBias = 0.0;
float FresStren = 1.0;
float FresPow = 2.0;
float GaussConstant = 100.0;
float PI = 3.1415926;

float saturate(float f)
{
  return clamp(f, 0.0, 1.0);  
}

float f_lambert(vec3 normal, vec3 light)
{
  return max(0.0, dot(normal, light));
}

float f_cook_torrance(vec3 normal, vec3 viewer, vec3 light, float roughness)
{  
    // Compute intermediary values
    vec3 half_vector = normalize(light + viewer);
    float NdotL = saturate(dot(normal, light));
    float NdotH = saturate(dot(normal, half_vector));
    float NdotV = saturate(dot(normal, viewer));
    float VdotH = saturate(dot(viewer, half_vector));

    // Approximate the Fresnel value
    float F = FresBias + FresStren * pow((1.0 - NdotV), FresPow);
    F = saturate(F);

    // Microfacet distribution
    float alpha = acos(NdotH);
    float D = GaussConstant * exp(-(alpha * alpha) / (roughness * roughness));

    // Geometric attenuation factor
    float G = min(1.0, min((2.0 * NdotH * NdotV / VdotH), (2.0 * NdotH * NdotL / VdotH)));

    return saturate(max(0.001, ((F * D * G) / (PI * NdotV * NdotL))));
}

void main()
{
  vec3 color = v_color.rgb;
  ivec2 cell = ivec2(v_checkerboard * v_texture);
  if(bool((cell.x + cell.y) & 1))
  {
    color = clamp(color * 1.1, 0.0, 1.0);
  }
  else
  {
    color = clamp(color * 0.9, 0.0, 1.0);
  }

  if(u_shading_type == 0)
  {
      vec3 diffuse_color = color;
      vec3 ambient_color = u_ambient_light_color;
      if(v_color.a < 0.75){
        ambient_color = clamp(ambient_color * 1.5, 0.0, 1.0);
      }
      vec3 L = normalize(u_directional_light_dir);
      vec3 V = normalize(-v_position_view);
      vec3 N = normalize(v_normal_view);
      vec3 specular_color = vec3(0.75, 0.75, 0.75);
      f_color = vec4(ambient_color * diffuse_color, v_color.a);
      f_color.rgb += u_directional_light_color * diffuse_color * f_lambert(N, L);
      f_color.rgb += u_directional_light_color * specular_color * f_cook_torrance(N, V, L, 0.2);
  }
  else
  {
      f_color = vec4(color, v_color.a);
  }
}