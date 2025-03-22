## Kogge Stone algorithm 
The naive implementation where I do global memory access has runtime:

# Partial sums allocation time: 0.1755 ms
Kernel time: 0.3264 ms
Partial sums allocation time: 0.0118 ms
Kernel time: 0.0204 ms
Scan verified successfully!

# The shared memory access has access time: 
Partial sums allocation time: 0.0988 ms
Kernel time: 0.0000 ms

Instant Improvement!

# I used double buffer here, it shows clearly here i gave up space for time. Thats amazing! 
Partial sums allocation time: 0.1765 ms
Kernel time: 0.0000 ms
