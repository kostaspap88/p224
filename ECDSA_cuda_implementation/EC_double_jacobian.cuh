#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "show_and_move.cuh"
#include "finite_field_arithmetic.cuh"
__device__ void EC_double_jacobian_parallel(uint32_t* x, uint32_t* y, uint32_t* z){
   int i = threadIdx.x;
	
	uint32_t delta_gamma[8];
	
	if (i%2==0){
		transfer(z,delta_gamma);
	}
	else if (i%2==1){
		transfer(y,delta_gamma);
	}
	  
	square256(delta_gamma);
	//show(delta_gamma,"d_g");
	
	uint32_t t0_t5[8];
	uint32_t fop[8];
	uint32_t sop[8];

	
	if (i%2==0){
		transfer(x,t0_t5);
		transfer(delta_gamma,sop);
		modred_quick(sop,1);
	}

	if (i%2==1){
		transfer(y,t0_t5);
		transfer(z,sop);
	}
	

	positive_add224(t0_t5,sop);
	//show(t0_t5,"t05");

	uint32_t t1[8];
	
	
	if (i%2==0){
		transfer(x,t1);
		transfer(delta_gamma,sop);
		positive_add224(t1,sop);
		//show(t1,"t1");
	}

	uint32_t t2_beta[8];
	if (i%2==0){
		transfer(t0_t5,fop);
		transfer(t1,t2_beta);
	}
	if (i%2==1){
		transfer(x,fop);
		transfer(delta_gamma,t2_beta);
	}
	positive_mult256(fop,t2_beta);
	//show(t2_beta,"t2b");
	
	uint32_t scale[8]={0,0,0,0,0,0,0,0};
	uint32_t alpha_t4[8];
	
	if (i%2==0){
		scale[0]=3;
	}
	if (i%2==1){
		scale[0]=8;
	}

	transfer(scale,fop);
	transfer(t2_beta,alpha_t4);

	positive_mult256(fop,alpha_t4); //changed this will multPower
	//show(alpha_t4,"at4");

	uint32_t t3_t6[8];
	if (i%2==0){
		
		transfer(alpha_t4,t3_t6);
		
	}
	if (i%2==1){
		
		transfer(t0_t5,t3_t6);
	
	}
	square256(t3_t6);
	//show(t3_t6,"t36");

	uint32_t x3_t7[8];
	__shared__ uint32_t x3_shared[8];
	__shared__ uint32_t t4_shared[8];

	if(i%2==1){
		transfer(alpha_t4,t4_shared);
	}

	if (i%2==0){
		
		transfer(t4_shared,sop);
		
	}
	if (i%2==1){
		
		transfer(delta_gamma,sop);
	}
	transfer(t3_t6,x3_t7);
	signed_add224(x3_t7,sop,0,1);
	//show(x3_t7,"xt7");

	uint32_t t8[8];

	if (i%2==1){
		scale[0]=4;
		transfer(scale,fop);
		transfer(t2_beta,t8);
		positive_mult256(fop,t8);
		//show(t8,"t8");
	}
	
	

	uint32_t t9_z3[8];
	__shared__ uint32_t t8_shared[8];
	__shared__ uint32_t delta_shared[8];
	if (i%2==0){
		transfer(delta_gamma,delta_shared);
	}
	if (i%2==1){
		transfer(t8,t8_shared);
	}
	if (i%2==0){
		transfer(t8_shared, t9_z3);
		transfer(x3_t7,sop);
	}
	if (i%2==1){
		transfer(x3_t7,t9_z3);
		transfer(delta_shared,sop);
	}
	signed_add224(t9_z3,sop,0,1);
	//show(t9_z3,"t9z");

	uint32_t t12_t10[8];
	if (i%2==0){
		transfer(alpha_t4,fop);
		transfer(t9_z3,t12_t10);
	}
	if (i%2==1){
		transfer(delta_gamma,fop);
		transfer(delta_gamma,t12_t10);
	}
	
	positive_mult256(fop,t12_t10);
	//show(t12_t10,"tla");


	uint32_t  t11[8];
	
	__shared__ uint32_t y3_shared[8];
	
	if (i%2==1){
		scale[8]=8;
		transfer(scale,fop);
		transfer(t12_t10,t11);
		positive_mult256(fop,t11); //replace with powermult
		//show(t11,"t11");
	}

	__shared__ uint32_t t12_shared[8];
	if (i%2==0){
		transfer(t12_t10,t12_shared);
	}

	uint32_t y3[8];
	if (i%2==1){
		transfer(t12_shared,y3);
		transfer(t11,sop);
		signed_add224(y3,sop,0,1);
		transfer(y3,y3_shared);
		//show(y3,"y3");
	}

	
}