DAY 04: Layer Normalization using CUDA
Today I performed layer normalization in CUDA kernel, first we add rows and colums then copy our data on GPU we copy our data from hdata to tensor to GPU, perform layer normalization i.e. mean, variance, normalization on each row which is allocated to each thread in each block, copy the data back to host and free the memory on GPU.
