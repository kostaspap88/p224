#ifndef SHOW_MOVE
#define SHOW_MOVE


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//debugging function (not tested)
__device__ void show(uint32_t *x, char *name){
	printf("%c%c%c:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",name[0],name[1],name[2],
				x[7],x[6],x[5],x[4],x[3],x[2],x[1],x[0]); 
}

//tranfer x to y (not tested)
__device__ void transfer(uint32_t *x,uint32_t *y){
	for (int i=0;i<=7;i++){
		y[i]=x[i];
	}
}

#endif