Today I gained a deep understanding of how CUDA works, including thread execution, shared memory, kernel launches, and debugging common mistakes in CUDA programming. 
🔹 How CUDA Threads Work
	•	Threads execute in parallel but may reach different parts of the code at different times.
	•	Each thread has a fixed threadIdx.x, threadIdx.y, and threadIdx.z that does not change during execution.
	•	Multiple threads cooperate to load data, especially when using shared memory
 Threads must cooperate to load shared memory before computations
