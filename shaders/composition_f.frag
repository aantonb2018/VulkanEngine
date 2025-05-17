#version 460

#extension GL_ARB_shader_draw_parameters : enable
#define INV_PI 0.31830988618
#define PI   3.14159265358979323846264338327950288

layout( location = 0 ) in vec2 f_uvs;

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

layout ( set = 0, binding = 1 ) uniform sampler2D i_albedo;
layout ( set = 0, binding = 2 ) uniform sampler2D i_position_and_depth;
layout ( set = 0, binding = 3 ) uniform sampler2D i_normal;
layout ( set = 0, binding = 4 ) uniform sampler2D i_material;
//layout ( set = 0, binding = 5 ) uniform sampler2D i_blur;
layout ( set = 0, binding = 5 ) uniform sampler2DArray i_shadow_maps;


layout(location = 0) out vec4 out_color;

float ShadowCalculation(vec4 fragPosLightSpace, uint light)
{
    // perform perspective divide
    vec3 projCoords = fragPosLightSpace.xyz / fragPosLightSpace.w;
    // transform to [0,1] range
    projCoords = projCoords * 0.5 + 0.5;
    // get closest depth value from light's perspective (using [0,1] range fragPosLight as coords)
    float closestDepth = texture(i_shadow_maps, vec3(projCoords.xy, light)).r; 
    // get depth of current fragment from light's perspective
    float currentDepth = projCoords.z;
    // check whether current frag pos is in shadow
    float shadow = currentDepth > closestDepth  ? 1.0 : 0.0;

    return shadow;
}  

float evalVisibility(vec3 frag_pos, vec3 normal, uint light_id) {
	
    LightData light = per_frame_data.m_lights[light_id];
    uint light_type = uint(floor(light.m_light_pos.a));
    
    // Ambient lights are always visible
    if (light_type == 2) return 1.0;
    /*
    // For point lights, check if fragment is behind the light
    if (light_type == 1) {
        vec3 to_light = light.m_light_pos.xyz - frag_pos;
        if (dot(to_light, normal) < 0.0) return 0.0;
    }*/
    
    // Transform fragment position to light space
    vec4 frag_pos_light_space = light.m_view_projection * (vec4(frag_pos, 1.0));
	//vec4 frag_pos_light_space = light.m_view_projection * (vec4(frag_pos, 1.0) * per_frame_data.m_inv_projection * per_frame_data.m_inv_view);
	//return ShadowCalculation(frag_pos_light_space, light_id);
    // Perspective divide
    vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;
    
    // Transform to [0,1] range for texture sampling
    proj_coords = proj_coords * 0.5 + 0.5;
    
    // Check if fragment is outside light frustum
    if (proj_coords.z > 1.0 || 
        proj_coords.x < 0.0 || proj_coords.x > 1.0 || 
        proj_coords.y < 0.0 || proj_coords.y > 1.0) {
        return 0.0;
    }
    
    // Get depth from shadow map
    float closest_depth = texture(i_shadow_maps, vec3(proj_coords.xy, light_id)).r;
    float current_depth = proj_coords.z;
    
    // Add bias to prevent shadow acne
    float bias = max(0.005 * (1.0 - dot(normal, normalize(light.m_light_pos.xyz - frag_pos.xyz))), 0.005);
    
    // PCF for softer shadows
    float shadow = 0.0;
	
    vec2 texel_size = 1.0 / textureSize(i_shadow_maps, 0).xy;
    for(int x = -1; x <= 1; ++x) {
        for(int y = -1; y <= 1; ++y) {
            float pcf_depth = texture(i_shadow_maps, vec3(proj_coords.xy + vec2(x, y) * texel_size, light_id)).r;   
			shadow += current_depth - bias > pcf_depth ? 1.0 : 0.0;        
        }    
    }
	
    shadow /= 9.0;
    
    return 1.0 - shadow;
}

vec3 evalDiffuse()
{
    vec4  albedo       = texture( i_albedo  , f_uvs );
    vec3  n            = normalize( texture( i_normal, f_uvs ).rgb * 2.0 - 1.0 );    
    vec3  frag_pos     = texture( i_position_and_depth, f_uvs ).xyz;
    vec3  shading = vec3( 0.0 );


    for( uint id_light = 0; id_light < per_frame_data.m_number_of_lights; id_light++ )
    {
        LightData light = per_frame_data.m_lights[ id_light ];
        uint light_type = uint( floor( light.m_light_pos.a ) );
		
		float visibility = evalVisibility(frag_pos, n, id_light);

        switch( light_type )
        {
            case 0: //directional
            {
                vec3 l = normalize( -light.m_light_pos.xyz );
                shading += max( dot( n, l ), 0.0 ) * light.m_radiance.rgb * albedo.rgb * visibility;
                break;
            }
            case 1: //point
            {
                vec3 l = light.m_light_pos.xyz - frag_pos;
                float dist = length( l );
                float att = 1.0 / (light.m_attenuattion.x + light.m_attenuattion.y * dist + light.m_attenuattion.z * dist * dist );
                vec3 radiance = light.m_radiance.rgb * att;
				l = normalize(l);

                shading += max( dot( n, l ), 0.0 ) * albedo.rgb * radiance * visibility;
                break;
            }
            case 2: //ambient
            {
                shading += light.m_radiance.rgb * albedo.rgb;
                break;
            }
        }
    }

    return shading;
}

float normalDistribution(vec3 h, vec3 n, float roughness){
	float alpha = pow(roughness, 2.f);
    float dotNH = max(dot(n, h), 0.0);
	float pow_alpha = pow(alpha, 2.f);
	float ndf = pow_alpha / (PI * pow(pow(dotNH, 2.f) * (pow_alpha - 1) + 1, 2.f));
    return ndf;
}

float geometricTerm(vec3 v, vec3 n, float roughness){
	float k = pow(roughness + 1, 2) / 8;
	float dotNV = max(dot(n, v), 0.0);
	
	float g = dotNV/(dotNV * (1 - k) + k);
	
	return g;
}

float fresnelScalar(vec3 v, vec3 h, float F0){
	float dotVH = max(dot(v, h), 0.0);
	
	float fresnel = F0 + (1 - F0) * pow(2, -5.55573f * dotVH - 6.98316f * dotVH);
	//float fresnel = F0 + (1 - F0) * pow(1 - dotVH, 5);
	return fresnel;
}

vec3 fresnelVec(vec3 v, vec3 h, vec3 F0){
	float dotVH = max(dot(v, h), 0.0);
	
	vec3 fresnel = F0 + (1 - F0) * pow(2, -5.55573f * dotVH - 6.98316f * dotVH);
	//float fresnel = F0 + (1 - F0) * pow(1 - dotVH, 5);
	return fresnel;
}

vec3 evalMicrofacets(){
	vec4  albedo       = texture( i_albedo  , f_uvs );
    vec3  n            = normalize( texture( i_normal, f_uvs ).rgb * 2.0 - 1.0 ); 
	vec3  frag_pos     = texture( i_position_and_depth, f_uvs ).xyz;
	
	//vec3  l 		   = per_frame_data.m_lights[0].m_light_pos.xyz - frag_pos;
	
	//vec3  h 		   = normalize(l + v);	
    
	
    vec3  shading = vec3( 0.0 );
	vec3  ambient = vec3( 0.0 );
	
	vec3  v = normalize(per_frame_data.m_camera_pos.xyz - frag_pos);
	float roughness = texture(i_material, f_uvs).y;
	float metallic = texture(i_material, f_uvs).z;
	
	float F0 = mix(0.04, max(albedo.r, max(albedo.g, albedo.b)), metallic);
	vec3 F0_vec = mix(vec3(0.04), albedo.rgb, metallic);
	
	vec3 l = vec3(0.0f);
	
	for( uint id_light = 0; id_light < per_frame_data.m_number_of_lights; id_light++ )
    {
        LightData light = per_frame_data.m_lights[ id_light ];
        uint light_type = uint( floor( light.m_light_pos.a ) );
		
		float visibility = evalVisibility(frag_pos, n, id_light);
		
		vec3 diffuse = vec3(0.0f);
		vec3 specular = vec3(0.0f);

        switch( light_type )
        {
            case 0: //directional
            {
                l = normalize( -light.m_light_pos.xyz );
                diffuse = max( dot( n, l ), 0.0 ) * light.m_radiance.rgb * albedo.rgb;
				//diffuse = albedo.rgb * INV_PI;
				
                break;
            }
            case 1: //point
            {
                l = (per_frame_data.m_inv_view * light.m_light_pos).xyz - frag_pos;
                float dist = length( l );
                float att = 1.0 / (light.m_attenuattion.x + light.m_attenuattion.y * dist + light.m_attenuattion.z * dist * dist );
                vec3 radiance = light.m_radiance.rgb * att;
				l = normalize(l);

                diffuse = max( dot( n, l ), 0.0 ) * albedo.rgb * radiance;
		
                break;
            }
            case 2: //ambient
            {
                ambient += light.m_radiance.rgb * albedo.rgb;
                continue;
            }
        }
		
		vec3  h = normalize(l + v);
				
		float ndf = normalDistribution(h, n, roughness);
	
		float g = geometricTerm(v, n, roughness) 
			* geometricTerm(l, n, roughness);
					
		vec3 fresnel = fresnelVec(v, h, F0_vec);//fresnelScalar(v,h,F0) * vec3(1.0f);
		
		vec3 kD                 = vec3(1.0) - fresnel;
		kD                      *= 1.0 - metallic;
		
		diffuse *= kD;
	
		specular = (ndf * fresnel * g) / (4.0 * max(dot(n, l), 0.0) * max(dot(n, v), 0.0));
				
		shading += (diffuse + specular) * light.m_radiance.rgb * max(dot(n, l), 0.0) * visibility;
    }

	shading += ambient;
	
	return shading;
}

void main() 
{
	//vec3 n = normalize(f_normal);
	//vec3 h = normalize(m_lights[0].m_light_pos.xyz - m_camera_pos.xyz);
	
	//float ndf = normalDistribution(h, n, per_object_data.objects[ f_instance ].m_metallic_roughness);
	
    float gamma = 2.2f;
    float exposure = 1.0f;
    vec3 mapped;
	
	//float blur_value = texture(i_blur, f_uvs).r;

	int id_material = int(texture(i_material, f_uvs).x);
	
	switch(id_material){
		case 0:
			mapped = vec3( 1.0f ) - exp(-evalDiffuse() * exposure);
			//mapped *= (1.0 - blur_value * 0.5);
			break;
		case 1:
			mapped = vec3( 1.0f ) - exp(-evalMicrofacets() * exposure);
			//mapped = mix(mapped, mapped * 0.7, blur_value);
			break;
		default:
			mapped = vec3(1.0f);
			break;
	}	

    out_color = vec4( pow( mapped, vec3( 1.0f / gamma ) ), 1.0 );/*
	
	vec3  frag_pos     = texture( i_position_and_depth, f_uvs ).xyz;
	LightData light = per_frame_data.m_lights[ 1 ];
	
	vec4 frag_pos_light_space = light.m_view_projection * (vec4(frag_pos, 1.0));
	//return ShadowCalculation(frag_pos_light_space, light_id);
    // Perspective divide
    vec3 proj_coords = frag_pos_light_space.xyz / frag_pos_light_space.w;
    
    // Transform to [0,1] range for texture sampling
    proj_coords = proj_coords * 0.5 + 0.5;
    
    
    // Get depth from shadow map
    float closest_depth = texture(i_shadow_maps, vec3(proj_coords.xy, 1)).r;
	float current_depth = proj_coords.z;

	float depth = texture( i_position_and_depth, f_uvs ).w;
	depth = (depth - 0.1) / (100.0 - 0.1);
	
	float shadow = current_depth > closest_depth  ? 1.0 : 0.0;
    out_color = vec4(vec3(shadow), 1.0);*/
}