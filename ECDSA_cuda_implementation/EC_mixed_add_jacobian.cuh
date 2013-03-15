#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "show_and_move.cuh"
#include "finite_field_arithmetic.cuh"

__device__ void EC_mixed_add_jacobian_parallel(uint32_t* x1, uint32_t* y1, uint32_t* z1,uint32_t* x2,uint32_t* y2){

	

	 int i = threadIdx.x;

	 uint32_t z1z1_z1z1[8];
	 transfer(z1,z1z1_z1z1);

	 square256(z1z1_z1z1);
	 
	 uint32_t fop[8];
	 uint32_t sop[8];

	if (i%2==0){
		transfer(x2,fop);
	}
	else if (i%2==1){
		transfer(z1,fop);
	}
	uint32_t u2_t0[8];
	transfer(z1z1_z1z1,u2_t0);	  

	positive_mult256(fop,u2_t0);
    uint32_t h[8];
	
	if (i%2==0){
		transfer(u2_t0,h);
		transfer(x1,sop);
		signed_add224(h,sop,0,1);
	}

	uint32_t hh_s2[8];
	if (i%2==0){
		transfer(h,fop);
		transfer(h,hh_s2);
	}
	else if (i%2==1){
		transfer(y2,fop);
		transfer(u2_t0,hh_s2);
	}
	positive_mult256(fop,hh_s2);

	uint32_t t1[8];
	if (i%2==1){
		transfer(hh_s2,t1);
		transfer(y1,sop);
		signed_add224(t1,sop,0,1);
	}

	uint32_t i_r[8];
	uint32_t scale[8]={0,0,0,0,0,0,0,0};
	if (i%2==0){
		scale[0]=4;
		transfer(hh_s2,i_r);
	}
	else if (i%2==1){
		scale[0]=2;
		transfer(t1,i_r);
	}
	transfer(scale,fop);
	positive_mult256(fop,i_r); 
	//PERFORMANCE: change this with power2_mult once implemented
	__shared__ uint32_t i_shared[8];
	if  (i%2==0){
	transfer(i_r,i_shared);
	}

	uint32_t j_v[8];
	if (i%2==0){
		transfer(h,fop);
		transfer(i_r,j_v);
	}
	else if (i%2==1){
		transfer(x1,fop);
		transfer(i_shared,j_v);
	}
	positive_mult256(fop,j_v);
	uint32_t t6_t2[8];

	if (i%2==0){
		transfer(y1,fop);
		transfer(j_v,t6_t2);
	}
	else if (i%2==1){
		transfer(i_r,fop);
		transfer(i_r,t6_t2);
	}
	positive_mult256(fop,t6_t2);

	

	uint32_t t7_t3[8];
	if (i%2==0){
		transfer(t6_t2,t7_t3);
	}
	else if (i%2==1){
		transfer(j_v,t7_t3);
	}
	scale[0]=2;
	transfer(scale,fop);
	positive_mult256(fop,t7_t3);//change this with power2_mult

	__shared__ uint32_t j_shared[8];
	if(i%2==0){
		transfer(j_v,j_shared);
	}

	uint32_t t9_t4[8];
	if (i%2==0){
		transfer(z1,t9_t4);
		transfer(h,sop);
	}
	else if (i%2==1){
		transfer(t6_t2,t9_t4);
		transfer(j_shared,sop);
		modred_quick(sop,1);
	}

	positive_add224(t9_t4,sop);
	uint32_t x3[8];
	uint32_t t5[8];
	__shared__ uint32_t x3_shared[8];
	if (i%2==1){
		transfer(t9_t4,x3);
		transfer(t7_t3,sop);
		signed_add224(x3,sop,0,1);
		transfer(x3,x3_shared);
		transfer(j_v,t5);
		transfer(x3,sop);
		signed_add224(t5,sop,0,1);
	}

	uint32_t t10_t8[8];
	if (i%2==0){
		transfer(t9_t4,fop);
		transfer(t9_t4,t10_t8);
	}
	else if (i%2==1){
		transfer(i_r,fop);
		transfer(t5,t10_t8);
	}
	positive_mult256(fop,t10_t8);
	
	uint32_t t11_y3[8];
	__shared__ uint32_t t7_shared[8];
	if (i%2==0){
		transfer(t7_t3,t7_shared);
		transfer(z1z1_z1z1,sop);

	}
	else if (i%2==1){
		
		transfer(t7_shared,sop);
	}
	
	transfer(t10_t8,t11_y3);
	signed_add224(t11_y3,sop,0,1);
	__shared__ uint32_t y3_shared[8];
	if (i%2==1){
		transfer(t11_y3,y3_shared);
	}
	uint32_t z3[8];
	__shared__ uint32_t z3_shared[8];
	if (i%2==0){
		transfer(t11_y3,z3);
		transfer(hh_s2,sop);
		signed_add224(z3,sop,0,1);
		transfer(z3,z3_shared);
	}

	//x3,y3,z3 in shared memory x3/y3/z3_shared
	
}