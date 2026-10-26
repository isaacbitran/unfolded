// WebGPU matmul backend. Weights live on the GPU; each call uploads the (small) activation
// matrix, runs one kernel, and reads the result back so the trace can keep it.
//
// Both kernels are memory-bound GEMV-like products (T is small), so each thread owns one output
// column and keeps a tile of up to 8 tokens in registers: one pass over the weights serves 8 tokens.

/// <reference types="@webgpu/types" />
import type { Backend, Weights } from './gpt2';
import { CFG } from './gpt2';

const TT = 8; // tokens per thread

const LINEAR = /* wgsl */ `
struct Dims { T: u32, inDim: u32, outDim: u32, pad: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let o = gid.x;
  if (o >= dims.outDim) { return; }
  let t0 = gid.y * ${TT}u;
  var acc: array<f32, ${TT}>;
  for (var r = 0u; r < ${TT}u; r++) { acc[r] = 0.0; }
  for (var i = 0u; i < dims.inDim; i++) {
    let w = W[i * dims.outDim + o];
    for (var r = 0u; r < ${TT}u; r++) {
      let t = t0 + r;
      if (t < dims.T) { acc[r] += x[t * dims.inDim + i] * w; }
    }
  }
  for (var r = 0u; r < ${TT}u; r++) {
    let t = t0 + r;
    if (t < dims.T) { y[t * dims.outDim + o] = acc[r] + b[o]; }
  }
}`;

const UNEMBED = /* wgsl */ `
struct Dims { T: u32, vocab: u32, d4: u32, pad: u32 };
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> E: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= dims.vocab) { return; }
  let t0 = gid.y * ${TT}u;
  var acc: array<f32, ${TT}>;
  for (var r = 0u; r < ${TT}u; r++) { acc[r] = 0.0; }
  for (var i = 0u; i < dims.d4; i++) {
    let e = E[v * dims.d4 + i];
    for (var r = 0u; r < ${TT}u; r++) {
      let t = t0 + r;
      if (t < dims.T) { acc[r] += dot(x[t * dims.d4 + i], e); }
    }
  }
  for (var r = 0u; r < ${TT}u; r++) {
    let t = t0 + r;
    if (t < dims.T) { y[t * dims.vocab + v] = acc[r]; }
  }
}`;

export async function webgpuBackend(w: Weights): Promise<Backend | null> {
  const gpu = (globalThis.navigator as Navigator | undefined)?.gpu;
  if (!gpu) return null;
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return null;
  const need = w['wte.weight'].byteLength;
  if (adapter.limits.maxStorageBufferBindingSize < need || adapter.limits.maxBufferSize < need) return null;
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });

  const upload = (data: Float32Array, usage = GPUBufferUsage.STORAGE) => {
    const buf = device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    new Float32Array(buf.getMappedRange()).set(data);
    buf.unmap();
    return buf;
  };
  const gpuWeights = new Map<string, GPUBuffer>();
  for (const [name, data] of Object.entries(w)) {
    if (name.includes('.weight') && (name.includes('attn.c_') || name.includes('mlp.c_') || name === 'wte.weight')) gpuWeights.set(name, upload(data));
    else if (name.endsWith('.bias') && (name.includes('attn.c_') || name.includes('mlp.c_'))) gpuWeights.set(name, upload(data));
  }

  const linearPipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: LINEAR }), entryPoint: 'main' } });
  const unembedPipe = device.createComputePipeline({ layout: 'auto', compute: { module: device.createShaderModule({ code: UNEMBED }), entryPoint: 'main' } });

  async function run(pipe: GPUComputePipeline, uniforms: Uint32Array, inputs: GPUBuffer[], outLen: number, groups: [number, number]) {
    const ubuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(ubuf, 0, uniforms);
    const out = device.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size: outLen * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const entries = [ubuf, ...inputs, out].map((buffer, binding) => ({ binding, resource: { buffer } }));
    const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(groups[0], groups[1]);
    pass.end();
    enc.copyBufferToBuffer(out, 0, read, 0, outLen * 4);
    device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    for (const b of [ubuf, out, read]) b.destroy();
    return result;
  }

  return {
    name: 'webgpu',
    async linear(x, T, weight, bias, inDim, outDim) {
      const xb = upload(x);
      const y = await run(linearPipe, new Uint32Array([T, inDim, outDim, 0]), [xb, gpuWeights.get(weight)!, gpuWeights.get(bias)!], T * outDim, [Math.ceil(outDim / 64), Math.ceil(T / TT)]);
      xb.destroy();
      return y;
    },
    async unembed(x, T) {
      const xb = upload(x);
      const y = await run(unembedPipe, new Uint32Array([T, CFG.vocab, CFG.dModel / 4, 0]), [xb, gpuWeights.get('wte.weight')!], T * CFG.vocab, [Math.ceil(CFG.vocab / 64), Math.ceil(T / TT)]);
      xb.destroy();
      return y;
    },
  };
}
