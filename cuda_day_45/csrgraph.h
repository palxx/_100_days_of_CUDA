#pragma once
#include <limits.h>

typedef struct{
unsigned int numVertices;
unsigned int numEdges;
unsigned int* scrPointers;
unsigned int* dst;
}CSRgraph;
