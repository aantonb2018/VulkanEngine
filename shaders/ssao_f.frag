#version 450

layout(location = 0) in vec2 uv;

layout(input_attachment_index = 0, binding = 0) uniform subpassInput inPosition;
layout(input_attachment_index = 1, binding = 1) uniform subpassInput inNormals;
layout(input_attachment_index = 2, binding = 2) uniform subpassInput inDepth;

layout(binding = 3) uniform SSAOParams {
    mat4 projection;
    mat4 invProjection;
    vec2 noiseScale;
    vec2 screenSize;  // Add this to replace textureSize()
    float radius;
    float bias;
    float power;
} params;


layout(location = 0) out float outSSAO;

const int kernelSize = 64;
layout(binding = 4) uniform sampler2D noiseTex;
layout(binding = 5) uniform SamplesBuffer {
    vec4 samples[kernelSize];
};

void main() {
    vec3 fragPos = subpassLoad(inPosition).xyz;
    vec3 normal = normalize(subpassLoad(inNormals).xyz);
    vec3 randomVec = texture(noiseTex, uv * params.noiseScale).xyz;
    
    vec3 tangent = normalize(randomVec - normal * dot(randomVec, normal));
    vec3 bitangent = cross(normal, tangent);
    mat3 TBN = mat3(tangent, bitangent, normal);

    float occlusion = 0.0;
    for (int i = 0; i < kernelSize; i++) {
        vec3 sampledPos = TBN * samples[i].xyz;
        sampledPos = fragPos + sampledPos * params.radius;
        
        vec4 offset = params.projection * vec4(sampledPos, 1.0);
        offset.xyz /= offset.w;
        offset.xy = offset.xy * 0.5 + 0.5;
        
        ivec2 texCoord = ivec2(offset.xy * params.screenSize);
        float sampleDepth = subpassLoad(inDepth).x;


        float rangeCheck = smoothstep(0.0, 1.0, params.radius / abs(fragPos.z - sampleDepth));
        occlusion += (sampleDepth <= sampledPos.z + params.bias ? 1.0 : 0.0) * rangeCheck;
    }
    
    outSSAO = 1.0 - (occlusion / kernelSize);
    outSSAO = pow(outSSAO, params.power);
}