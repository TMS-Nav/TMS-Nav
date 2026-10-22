// the mri slice of the aim 2 tab. one plane through the head that looks up the t1
// and |E| volumes at every fragment, so the picture follows the plane wherever it is
// moved. both volumes are uint8 3d textures on the same grid (efield/export_web.py),
// the field already divided by the colour scale's top, so 1.0 is the top colour.
import * as THREE from "three";

import { turboTable } from "./colormap.mjs";

const vertexShader = /* glsl */ `
  varying vec3 vWorld;
  void main() {
    vec4 world = modelMatrix * vec4(position, 1.0);
    vWorld = world.xyz;
    gl_Position = projectionMatrix * viewMatrix * world;
  }
`;

const fragmentShader = /* glsl */ `
  precision highp sampler3D;
  uniform sampler3D t1;
  uniform sampler3D field;
  uniform sampler2D lut;
  uniform mat4 rasToTex;
  uniform float cut;
  uniform float showField;
  varying vec3 vWorld;

  void main() {
    vec3 tc = (rasToTex * vec4(vWorld, 1.0)).xyz;
    if (any(lessThan(tc, vec3(0.0))) || any(greaterThan(tc, vec3(1.0)))) discard;
    float g = texture(t1, tc).r;
    // outside the head is see through, so the slice has the head's own outline
    if (g < 0.015) discard;
    float e = texture(field, tc).r;
    // fade the colour in just above the cut rather than switching it on, so the
    // edge of the hotspot reads as a gradient, not a stencil
    float a = smoothstep(cut, cut + 0.12, e) * showField;
    vec3 col = mix(vec3(g * 0.92), texture(lut, vec2(e, 0.5)).rgb, a * 0.88);
    gl_FragColor = vec4(col, 1.0);
  }
`;

/**
 * a 256 x 1 turbo lookup texture, shared by the slice shader. left without a colour
 * space on purpose: the shader writes these bytes straight out, as is the t1 grey,
 * so the slice shows the colormap's own sRGB values with no conversion either way
 */
export function makeLutTexture() {
  const tex = new THREE.DataTexture(turboTable(256), 256, 1, THREE.RGBAFormat);
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.needsUpdate = true;
  return tex;
}

/** a uint8 volume, x fastest, as a linearly filtered 3d texture. */
export function makeVolumeTexture(bytes, dims) {
  const tex = new THREE.Data3DTexture(bytes, dims[0], dims[1], dims[2]);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

/**
 * RAS mm to texture coordinates. the affine maps voxel indices to RAS, and a
 * texture puts voxel i's centre at (i + 0.5) / n, so invert, shift half a voxel
 * and divide by the size.
 */
export function rasToTexMatrix(affineRows, dims) {
  const affine = new THREE.Matrix4().set(...affineRows.flat());
  const toTex = new THREE.Matrix4()
    .makeScale(1 / dims[0], 1 / dims[1], 1 / dims[2])
    .multiply(new THREE.Matrix4().makeTranslation(0.5, 0.5, 0.5));
  return toTex.multiply(affine.invert());
}

export function createSliceMaterial({ t1, field, lut, rasToTex }) {
  return new THREE.ShaderMaterial({
    uniforms: {
      t1: { value: t1 },
      field: { value: field },
      lut: { value: lut },
      rasToTex: { value: rasToTex },
      cut: { value: 0.2 },
      showField: { value: 1.0 },
    },
    vertexShader,
    fragmentShader,
    side: THREE.DoubleSide,
  });
}
