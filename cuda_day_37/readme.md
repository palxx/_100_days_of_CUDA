### Normal Execution 
We load value of image pixel, identify its bin and update the bin by 1, but the accesses are global and main issue here is, race conditions caused by threads
because of this. 
Race conditions meaning a thread may read the old value before the previous thread gets to update and thus causing wrong values in the system. 

### Atomic Add
To solve this we introduce, atomic add where read-modify-write happens in one single operation (ISA), this reduces the time by significant amount but not for CPU,
we just do, 
```atomicAdd(&bins[b], 1)```

Next up, 
### Privatisation of Histogram
where each block updates a private copy of the histogram, and after that atomically it updates the global copy when done. 

![alt text]([![image](https://github.com/user-attachments/assets/c9a08698-ffe4-47f3-80a6-4edd9c18671b)](https://plus.unsplash.com/premium_photo-1664474619075-644dd191935f?fm=jpg&q=60&w=3000&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8aW1hZ2V8ZW58MHx8MHx8fDA%3D))
