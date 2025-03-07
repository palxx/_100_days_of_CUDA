Today I studied NVIDIA profiling, there are various aspects that are included in profiling: 

With command nvprof ./vecadd, it showed me information in two sections 
1. GPU activities included timing of CUDA Memcpy from Host to Device and from Device to Host and the call to the Kernel itself. 
2. API calls - other api calls such as cudaMemcpy, cudaMalloc, cudaGetAttribute, etc were provided by this API.

With the command: nvprof -m all -o metrics.prof ./vecadd
1. The above command will get all the attributes like how the kernel is executing, how much time it has taken, open both the files in NVIDIA Profiler. 
2. In which we can CUDA Application Analysis which tells us Duration, excution time of execution od CUDA. 
3. We have check overall GPU Usage, which tells us about the which factors are making my performance lag. 
4. We have whether our application compute, bandwidht or latency bound, along with the infornation like gridsize, blocksize, threads, registers per thread,
shared memory, etc.
5. More fine grained control over the application by knowing memory loads, memory stores, L1 and L2 cache accesses, local read, local write, etc.

It also concluded that having block dim as 32 and max block per SM as 32 can reduced our performance by 2x if our SM as max thread size 2048 where as the 
current configuration gives 1024 threads per SM. This can be a limiting factor and we should be careful about our hardware specifications and 
should provide the parameters according to the best fit for our architechture.

Happy CUDA Coding!!
