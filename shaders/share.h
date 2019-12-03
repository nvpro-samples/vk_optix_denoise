struct PerRayData_pathtrace
{
  vec3 result;
  vec3 radiance;
  vec3 attenuation;
  vec3 origin;
  vec3 direction;
  uint seed;
  int  depth;
  int  countEmitted;
  int  done;
};

struct PerRayData_pick
{
  vec4 worldPos;
  vec4 barycentrics;
  uint instanceID;
  uint primitiveID;
};


struct Light
{
  vec4 position;
  vec4 color;
  //  float intensity;
  //  float _pad;
};

// Per Instance information
struct primInfo
{
  uint indexOffset;
  uint vertexOffset;
  uint materialIndex;
};

// Matrices buffer for all instances
struct InstancesMatrices
{
  mat4 world;
  mat4 worldIT;
};

struct Scene
{
  mat4  projection;
  mat4  model;
  vec4  camPos;
  int   nbLights;  // w = lightRadiance
  int   _pad1;
  int   _pad2;
  int   _pad3;
  Light lights[10];
};

struct Material
{
  vec4  baseColorFactor;
  vec3  emissiveFactor;
  float metallicFactor;  // 8
  vec3  specularFactor;
  float roughnessFactor;  // 12 -
  int   alphaMode;        // 0: opaque, 1: mask, 2: blend
  float alphaCutoff;
  float glossinessFactor;
  int   shadingModel;  // 16 - 0: metallic-roughness, 1: specular-glossiness
  int   doubleSided;
  int   pad0;
  int   pad1;
  int   pad2;
};
