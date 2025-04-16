#include <iostream>
#include <vector>

extern "C" unsigned int enqueue_gpu(unsigned int* input, unsigned int* queue, unsigned int N);


int main(){
  std::vector<unsigned int>input = {3,2,4,6,7};
  unsigned int N = input.size();
  std::vector<unsigned int>queue(input.size());
  unsigned int queueSize = enqueue_gpu(input.data(), queue.data(), N);
  std::cout << "Enqueued " << queueSize << " elements:" << std::endl;
    for(unsigned int i = 0; i < queueSize; i++){
        std::cout << queue[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
