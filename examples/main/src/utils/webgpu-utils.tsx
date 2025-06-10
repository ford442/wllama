let gpuDevice: GPUDevice | null = null;
let gpuAdapter: GPUAdapter | null = null;

export async function initializeWebGPU(): Promise<boolean> {
    if (!navigator.gpu) {
        console.warn("WebGPU not supported on this browser.");
        return false;
    }

    try {
        gpuAdapter = await navigator.gpu.requestAdapter();
        if (!gpuAdapter) {
            console.warn("No WebGPU adapter found.");
            return false;
        }

        gpuDevice = await gpuAdapter.requestDevice();
        if (!gpuDevice) {
            console.warn("No WebGPU device found.");
            return false;
        }

        gpuDevice.lost.then(() => {
            console.error("WebGPU device lost!");
            gpuDevice = null;
            gpuAdapter = null;
        });

        console.log("WebGPU initialized successfully!");
        return true;
    } catch (error) {
        console.error("Failed to initialize WebGPU:", error);
        gpuDevice = null;
        gpuAdapter = null;
        return false;
    }
}

export function getGPUDevice(): GPUDevice | null {
    return gpuDevice;
}

export function createGPUBuffer(data: Float32Array, usage: GPUBufferUsageFlags): GPUBuffer {
    if (!gpuDevice) {
        throw new Error("WebGPU device not available.");
    }
    const buffer = gpuDevice.createBuffer({
        size: data.byteLength,
        usage: usage | GPUBufferUsage.COPY_DST,
        mappedAtCreation: false,
    });
    gpuDevice.queue.writeBuffer(buffer, 0, data);
    return buffer;
}

export async function readGPUBuffer(buffer: GPUBuffer): Promise<Float32Array> {
    if (!gpuDevice) {
        throw new Error("WebGPU device not available.");
    }
    const size = buffer.size;
    const stagingBuffer = gpuDevice.createBuffer({
        size: size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const commandEncoder = gpuDevice.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
    gpuDevice.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange());
    const dataCopy = new Float32Array(result);
    stagingBuffer.unmap();
    stagingBuffer.destroy();
    return dataCopy;
}

// GELU Shader and Pipeline
const geluShaderWGSL = `
    @group(0) @binding(0) var<storage, read> input: array<f32>;
    @group(0) @binding(1) var<storage, write> output: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let i = global_id.x;
        if (i >= arrayLength(&input)) { return; }

        let x = input[i];
        // GELU approximation (check wllama's exact formula if it differs)
        let K = 0.044715;
        let M_SQRT2_OVER_PI = 0.7978845608028654; // sqrt(2 / PI)
        let cdf = 0.5 * (1.0 + tanh(M_SQRT2_OVER_PI * (x + K * x * x * x)));
        output[i] = x * cdf;
    }
`;

let geluComputePipeline: GPUComputePipeline | null = null;

export async function runGELUKernel(inputBuffer: GPUBuffer, outputBuffer: GPUBuffer, numElements: number): Promise<void> {
    const device = getGPUDevice();
    if (!device) {
        console.error("WebGPU device not initialized for GELU.");
        return;
    }

    if (!geluComputePipeline) {
        geluComputePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({ code: geluShaderWGSL }),
                entryPoint: 'main',
            },
        });
    }

    const bindGroup = device.createBindGroup({
        layout: geluComputePipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inputBuffer } },
            { binding: 1, resource: { buffer: outputBuffer } },
        ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(geluComputePipeline);
    passEncoder.setBindGroup(0, bindGroup);

    const workgroupSize = 64; // Matches @workgroup_size in shader
    const numWorkgroups = Math.ceil(numElements / workgroupSize);
    passEncoder.dispatchWorkgroups(numWorkgroups);
    passEncoder.end();

    device.queue.submit([commandEncoder.finish()]);
}

// Simple CPU GELU for comparison (for testing purposes only)
export function geluCPU(x: number): number {
    const K = 0.044715;
    const M_SQRT2_OVER_PI = 0.7978845608028654;
    const cdf = 0.5 * (1.0 + Math.tanh(M_SQRT2_OVER_PI * (x + K * x * x * x)));
    return x * cdf;
}
