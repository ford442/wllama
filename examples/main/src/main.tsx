// import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.tsx';
import './index.css';


// Import WebGPU utilities
import {
  initializeWebGPU,
  getGPUDevice, // You might need this later if you want to pass the device around
  createGPUBuffer,
  readGPUBuffer,
  runGELUKernel,
  geluCPU, // For comparison
} from './utils/webgpu-utils'; // <--- NEW IMPORT

// --- Test function for GELU GPU vs CPU (keep in main.tsx for testing app startup) ---
async function testGELU(): Promise<void> {
    const gpuDevice = getGPUDevice(); // Get the device from the utility
    if (!gpuDevice) {
        console.warn("WebGPU not initialized, skipping GELU test.");
        return;
    }

    const testSize = 1024;
    const inputData = new Float32Array(testSize);
    for (let i = 0; i < testSize; i++) {
        inputData[i] = Math.random() * 10 - 5; // Random values between -5 and 5
    }

    console.log("Input Data (first 5):", inputData.slice(0, 5));

    // --- CPU GELU ---
    const cpuOutput = new Float32Array(testSize);
    for (let i = 0; i < testSize; i++) {
        cpuOutput[i] = geluCPU(inputData[i]);
    }
    console.log("CPU GELU Output (first 5):", cpuOutput.slice(0, 5));

    // --- GPU GELU ---
    const inputBuffer = createGPUBuffer(inputData, GPUBufferUsage.STORAGE);
    const outputBuffer = gpuDevice.createBuffer({
        size: inputData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    console.log("Running GPU GELU kernel...");
    await runGELUKernel(inputBuffer, outputBuffer, testSize);
    console.log("GPU GELU kernel finished.");

    const gpuOutput = await readGPUBuffer(outputBuffer);
    console.log("GPU GELU Output (first 5):", gpuOutput.slice(0, 5));

    // --- Compare Results ---
    let differences = 0;
    let maxDiff = 0;
    for (let i = 0; i < testSize; i++) {
        const diff = Math.abs(cpuOutput[i] - gpuOutput[i]);
        if (diff > 1e-5) { // Check for a small tolerance
            differences++;
            if (diff > maxDiff) maxDiff = diff;
        }
    }

    if (differences === 0) {
        console.log(`GELU Test SUCCESS: CPU and GPU outputs match within tolerance.`);
    } else {
        console.warn(`GELU Test FAILED: ${differences} differences found. Max difference: ${maxDiff}`);
    }

    // Clean up GPU buffers
    inputBuffer.destroy();
    outputBuffer.destroy();
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  // TODO: we disable strict mode because some dispatchers are fired twice
  //<React.StrictMode>
  <App />
  //</React.StrictMode>,
);
