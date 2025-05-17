#version 460

layout(triangles) in;
layout(triangle_strip, max_vertices = 30) out;

#extension GL_ARB_shader_draw_parameters : enable

layout( location = 0 ) in vec3 g_position[];

//globals
struct LightData
{
    vec4 m_light_pos;
    vec4 m_radiance;
    vec4 m_attenuattion;
    mat4 m_view_projection;
};

layout( std140, set = 0, binding = 0 ) uniform PerFrameData
{
    vec4      m_camera_pos;
    mat4      m_view;
    mat4      m_projection;
    mat4      m_view_projection;
    mat4      m_inv_view;
    mat4      m_inv_projection;
    mat4      m_inv_view_projection;
    vec4      m_clipping_planes;
    LightData m_lights[ 10 ];
    uint      m_number_of_lights;
} per_frame_data;



void main() {
    
    for (int i = 0; i < per_frame_data.m_number_of_lights; ++i) {

        mat4 lightVP = per_frame_data.m_lights[i].m_view_projection;

        gl_Layer = i; // Renderizamos a la capa correspondiente al índice de la luz

        for (int j = 0; j < 3; ++j) {
            vec4 worldPos = vec4(g_position[j], 1.0);
            vec4 lightSpacePos = lightVP * worldPos;
            
            // Perspective divide and invert Z
            //lightSpacePos.w = 1 - lightSpacePos.w;  // Invert before perspective divide
            gl_Position = lightSpacePos;
            EmitVertex();
        }

        EndPrimitive();
    }
	/*
	gl_Layer = 0; // Only render to the first layer for testing

    // Vertex 1 (bottom-left)
    gl_Position = vec4(1.0, 1.0, 0.5, 1.0);
    EmitVertex();

    // Vertex 2 (top-middle)
    gl_Position = vec4(1.0, 0.9, 0.5, 1.0);
    EmitVertex();

    // Vertex 3 (bottom-right)
    gl_Position = vec4(0.9, 1.0, 0.5, 1.0);
    EmitVertex();

    EndPrimitive();
	
	gl_Layer = 0; // Only render to the first layer for testing

    // Vertex 1 (bottom-left)
    gl_Position = vec4(-1.0, -1.0, 0.5, 1.0);
    EmitVertex();

    // Vertex 2 (top-middle)
    gl_Position = vec4(0.0, 1.0, 0.5, 1.0);
    EmitVertex();

    // Vertex 3 (bottom-right)
    gl_Position = vec4(1.0, -1.0, 0.5, 1.0);
    EmitVertex();

    EndPrimitive();*/
}