### Normal Execution 
We load value of image pixel, identify its bin and update the bin by 1, but the accesses are global and main issue here is, race conditions caused by threads
because of this. 
Race conditions meaning a thread may read the old value before the previous thread gets to update and thus causing wrong values in the system. 
```GPU kernel execution time: 0.123264 ms```

### Atomic Add
To solve this we introduce, atomic add where read-modify-write happens in one single operation (ISA), this reduces the time by significant amount but not for CPU,
we just do, 
```atomicAdd(&bins[b], 1)```
```GPU kernel execution time: 0.141344 ms```

Next up, 
### Privatisation of Histogram
1. Where each block updates a private copy (doesn't necessarily mean in shared memory) of the histogram, and after that atomically it updates the global copy when done that is privatisation. If the num_bins fits its copy on shared memory, it reduces further computation time.
2. Even though we dont have the leverage of using shared memory, we still have benefit of reducing global memory latency.
   ```GPU kernel execution time: 0.749600 ms```

### Another optimisation is Thread Coarsening
1. If I have more thread blocks, the parallel operations are worth it, but if I have less blocks, the hardware is going to end up parallelising these operations and that's not worth it, so we are going to take matter in our hands and parallel the operation ourselves.
2. For each thread, we are going to assign multiple inputs, and that thread is responsible for the output of each input. So there are more than one input assigned to each thread to calculate output.
