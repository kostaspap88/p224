#include "cuda_runtime.h"
#include <stdint.h>
__device__ void add256(uint32_t *a, uint32_t *b);
__device__ void add224(uint32_t *a, uint32_t *b);
