/* authors: Kostas Papagiannopoulos, RU
			Arturo Cedillo, TU/e
			Mathias Morbitzer, RU

			Project Lola Scam Van [Low-latency scalar multiplication for VANETs]

			This file is part of Project Lola Scam Van. The DIES research group, University of Twente
			is free to use and modify the project for research and educational purposes. 
*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "finite_field_arithmetic.cuh"
#include "EC_double_jacobian.cuh"
//#include "EC_double_projective_serial.cuh"
//#include "EC_add_projective_serial.cuh"
//#include "EC_mixed_add_jacobian.cuh"

#include "show_and_move.cuh"

cudaError_t cuda_kernel( uint32_t *r,  uint32_t *x1, uint32_t *y1, uint32_t *z1, uint32_t *x2, uint32_t *y2, size_t size);


__global__ void kernel(uint32_t* x1,uint32_t* y1,uint32_t* z1,uint32_t* x2,uint32_t* y2)
{
	 
	for (int j=0;j<224;j++){
	
	EC_double_jacobian_parallel(x1,y1,z1);
	 
	}
  //Here we can use EC_double, EC_add to do the scalar multiplication.
  //We should consider sliding window and mixed addition
}

int main()
{
    const uint32_t arraySize = 8;  
    uint32_t x1[arraySize] = { 2, 0, 0, 0, 0, 0, 0, 0  }; 
	uint32_t y1[arraySize] = { 3, 0, 0, 0, 0, 0, 0, 0  };  
	uint32_t z1[arraySize] = { 4, 0, 0, 0, 0, 0, 0, 0  }; 
	uint32_t x2[arraySize] = { 7, 0, 0, 0, 0, 0, 0, 0  };
	uint32_t y2[arraySize] = { 9 , 0, 0, 0, 0, 0, 0, 0  };
	uint32_t r[arraySize] = { 0 }; 
	
   
    cudaError_t cudaStatus = cuda_kernel(r,x1,y1,z1,x2,y2,arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cuda_kernel failed!");
        return 1;
    } 

    printf("End of Kernel\n");
	
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();  
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1; 
    }
	system("PAUSE");
    return 0;
} 


cudaError_t cuda_kernel( uint32_t *r,  uint32_t *x1 , uint32_t *y1, uint32_t *z1 , uint32_t *x2 , uint32_t *y2 , size_t size)
{
	
    uint32_t *dev_x1 = 0;
    uint32_t *dev_y1 = 0;
    uint32_t *dev_z1 = 0;
	uint32_t *dev_x2 = 0;
	uint32_t *dev_y2 = 0;
	uint32_t *dev_r = 0;
	
 

    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_r, size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_x1, size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_y1, size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_z1, size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_x2, size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	cudaStatus = cudaMalloc((void**)&dev_y2, size * sizeof(uint32_t));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


   

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_x1, x1, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_y1, y1, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_z1, z1, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

	cudaStatus = cudaMemcpy(dev_x2, x2, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
		
	cudaStatus = cudaMemcpy(dev_y2, y2, size * sizeof(uint32_t), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
		
		

	
	clock_t  start,end;
	start=clock();

	//CUrrent kernel config: 1block, 1thread per block. Decent parallel results with <<<16,64>>> on 525m GT GPU
    kernel<<<1, 2>>>(dev_x1,dev_y1,dev_z1,dev_x2,dev_y2);
	
    
    cudaStatus = cudaDeviceSynchronize();
	end=clock();
	float t=(float)(end-start)/CLOCKS_PER_SEC;
	printf("time ellapsed: %f\n",t);
    
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(r, dev_r, size * sizeof(uint32_t), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
	
    cudaFree(dev_r);
    cudaFree(dev_x1);
    cudaFree(dev_y1);
	cudaFree(dev_z1);
	cudaFree(dev_x2);
	cudaFree(dev_y2);

    return cudaStatus;
}
 