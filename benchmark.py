# %%
import torch
import time
import numpy as np

def benchmark_matrix_mul(size, dtype, device, num_iterations=10):
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)
    
    # Warm-up run
    torch.matmul(a, b)
    torch.cuda.synchronize()
    
    times = []
    for _ in range(num_iterations):
        start_time = time.perf_counter()
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        times.append(end_time - start_time)
    
    times = np.array(times)
    avg_time = np.mean(times)
    time_std_error = np.std(times, ddof=1) / np.sqrt(num_iterations)
    
    # Calculate TFLOPs
    ops = 2 * size * size * size
    tflops = (ops / times) / 1e12
    avg_tflops = np.mean(tflops)
    tflops_std_error = np.std(tflops, ddof=1) / np.sqrt(num_iterations)
    
    return avg_time, time_std_error, avg_tflops, tflops_std_error

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA not available, running on CPU")

    sizes = [
        512, 768, 1024,
        1536, 2048,
        3072, 4096,
        6144, 8192,
        12288, 16384
    ]
    dtypes = [torch.float16, torch.bfloat16]  # Changed to include bfloat16 instead of float32

    print(f"{'Size':>10} {'Dtype':>10} {'Time (s)':>20} {'TFLOPs':>20}")
    print("-" * 65)

    for dtype in dtypes:
        for size in sizes:
            avg_time, time_std_error, avg_tflops, tflops_std_error = benchmark_matrix_mul(size, dtype, device)
            dtype_str = 'FP16' if dtype == torch.float16 else 'BF16'  # Changed to reflect bfloat16
            time_result = f"{avg_time:.4f} ± {time_std_error:.4f}"
            tflops_result = f"{avg_tflops:.2f} ± {tflops_std_error:.2f}"
            print(f"{size:10d} {dtype_str:>10} {time_result:>20} {tflops_result:>20}")

if __name__ == "__main__":
    main()

# %%
