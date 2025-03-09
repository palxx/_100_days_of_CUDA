Today I coded, convolution for 2D matrix, by looping over constant memory conv mask, and implemented the basic version of convolution in 2D. So, the compute to memory ratio in this case M^2 is 0.5 OP/B, every thread loads M^2 components but with tiling we load the mask conv into shared memory and use the threads. 
I am going to optimize and  implement with tiling tommorow. 
