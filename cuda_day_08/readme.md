🔹 Brent-Kung Parallel Prefix Sum Algorithm

	•	Prefix Sum (Scan): Computes cumulative sums efficiently.
	•	Brent-Kung Algorithm: Uses a tree-based approach to perform the scan in O(log N) time.
	•	Two Main Phases:
	•	✅ Upsweep (Reduction Phase) → Builds a sum tree by summing adjacent values.
	•	✅ Downsweep (Distribution Phase) → Uses the sum tree to compute prefix sums.

🔹 Used cudaGetDeviceProperties() to analyze GPU capabilities. Important GPU features:
 
	•	Max Threads per Block (1024 typical)
	•	Warp Size (32)
	•	Shared Memory per Block (48 KB)
	•	Number of Multiprocessors (SMs)
 	•	Understanding GPU hardware lets you design CUDA kernels that match the architecture.
	•	More SMs = more parallelism.
	•	Optimizing memory access patterns can maximize throughput.
