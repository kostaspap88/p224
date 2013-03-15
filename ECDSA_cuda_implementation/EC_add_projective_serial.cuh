#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "finite_field_arithmetic.cuh"

/*Elliptic curve addition
Projective-3 coords for short Weirstrass Form
Cohen–Miyaji–Ono "Efficient elliptic curve exponentiation using mixed coordinates"
*/

__device__ void EC_add_serial(uint32_t* x1, uint32_t* y1, uint32_t* z1,uint32_t* x2, uint32_t* y2, uint32_t* z2){


uint32_t Z2_2[8],Z2_3[8],Z1_1[8],Z1_2[8];
for(int i=0;i<=7;i++){
	Z2_2[i]=z2[i];
	Z2_3[i]=z2[i];
	Z1_1[i]=z1[i];
	Z1_2[i]=z1[i];
}

//Y1Z2 = Y1*Z2
uint32_t Y1Z2_1[8];

positive_mult256(y1,z2);//y1 reserved for yfinal, y1z2 in z2 
for(int i=0;i<=7;i++){
	Y1Z2_1[i]=z2[i];
}

// printf("Y1Z2:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	z2[7],z2[6],z2[5],z2[4],z2[3],z2[2],z2[1],z2[0]); 


uint32_t X1Z2_1[8];
//X1Z2 = X1*Z2
positive_mult256(x1,Z2_2);//x1 reserved for xfinal, x1z2 in z2_2
for(int i=0;i<=7;i++){
	X1Z2_1[i]=Z2_2[i];
}

// printf("X1Z2:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	Z2_2[7],Z2_2[6],Z2_2[5],Z2_2[4],Z2_2[3],Z2_2[2],Z2_2[1],Z2_2[0]); 

uint32_t Z1Z2_1[8];
//Z1Z2 = Z1*Z2 
positive_mult256(z1,Z2_3);//z1 reserved for zfinal, z1z2 in z2_3
for(int i=0;i<=7;i++){
	Z1Z2_1[i]=Z2_3[i];
}

 //printf("Z1Z2:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	Z2_3[7],Z2_3[6],Z2_3[5],Z2_3[4],Z2_3[3],Z2_3[2],Z2_3[1],Z2_3[0]); 

uint32_t t0[8]; 
//t0 = Y2*Z1
positive_mult256(Z1_1,y2); //t0 in y2, Z1_1=free

//printf("t0:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	y2[7],y2[6],y2[5],y2[4],y2[3],y2[2],y2[1],y2[0]); 

uint32_t u[8];
//u = t0-Y1Z2
signed_add224(y2,z2,0,1); // u in y2, z2=free
for(int i=0;i<=7;i++){
	u[i]=y2[i];//u in u, u not free
}
//printf("u:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	y2[7],y2[6],y2[5],y2[4],y2[3],y2[2],y2[1],y2[0]); 

//uu = u^2
square256(y2); //uu in y2
//printf("uu:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	y2[7],y2[6],y2[5],y2[4],y2[3],y2[2],y2[1],y2[0]); 

//t1 = X2*Z1
positive_mult256(x2,Z1_2);// t1 in Z1_2, x2=free
	//printf("t1:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	Z1_2[7],Z1_2[6],Z1_2[5],Z1_2[4],Z1_2[3],Z1_2[2],Z1_2[1],Z1_2[0]); 


//v = t1-X1Z2
uint32_t v[8];
signed_add224(Z1_2,Z2_2,0,1);//v in Z1_2, Z2_2=free
for(int i=0;i<=7;i++){
	Z2_2[i]=Z1_2[i];// v in Z2_2, Z2_2 not free
	v[i]=Z1_2[i];//v in v, v not free
}
//printf("v:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		Z1_2[7],Z1_2[6],Z1_2[5],Z1_2[4],Z1_2[3],Z1_2[2],Z1_2[1],Z1_2[0]); 

//vv = v^2
square256(Z1_2); //vv in Z1_2
for(int i=0;i<=7;i++){
	Z1_1[i]=Z1_2[i];// vv in Z1_1, Z1_1  not free
}
//printf("vv:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		Z1_2[7],Z1_2[6],Z1_2[5],Z1_2[4],Z1_2[3],Z1_2[2],Z1_2[1],Z1_2[0]); 

//vvv = v*vv
positive_mult256(Z2_2,Z1_2);// vvv in Z1_2, Z2_2 = free
for(int i=0;i<=7;i++){
	Z2_2[i]=Z1_2[i];// vvv in Z2_2, Z2_2 not free
	x2[i]=Z1_2[i];// vvv in x2, x2 not free
	z2[i]=Z1_2[i];// vvv in z2, z2 not free REMOVE!
}
//printf("vvv:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	Z1_2[7],Z1_2[6],Z1_2[5],Z1_2[4],Z1_2[3],Z1_2[2],Z1_2[1],Z1_2[0]); 

//R = vv*X1Z2
positive_mult256(X1Z2_1,Z1_1); //x1z2_1=free, R in Z1_1
for(int i=0;i<=7;i++){
	X1Z2_1[i]=Z1_1[i];//R in X1Z2_1, X1Z2_1 not free
}
//printf("R:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		Z1_1[7],Z1_1[6],Z1_1[5],Z1_1[4],Z1_1[3],Z1_1[2],Z1_1[1],Z1_1[0]); 

uint32_t scale[8]={2,0,0,0,0,0,0,0};
//t2 = 2*R
positive_mult256(scale,Z1_1); //t2 in Z1_1, scale empty
//printf("t2:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		Z1_1[7],Z1_1[6],Z1_1[5],Z1_1[4],Z1_1[3],Z1_1[2],Z1_1[1],Z1_1[0]); 

//t3 = uu*Z1Z2
positive_mult256(Z1Z2_1,y2);// t3 in y2, Z1Z2_1 = free
//printf("t3:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		y2[7],y2[6],y2[5],y2[4],y2[3],y2[2],y2[1],y2[0]); 

//t4 = t3-vvv
signed_add224(y2,Z1_2,0,1);//t4 in y2, Z1_2 = free
//printf("t4:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		y2[7],y2[6],y2[5],y2[4],y2[3],y2[2],y2[1],y2[0]); 

//A = t4-t2
signed_add224(y2,Z1_1,0,1);//A in y2, Z1_1 = free
for(int i=0;i<=7;i++){
	Z1_1[i]=y2[i];// A in Z1_1, Z1_1 not free
}
//printf("A:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		y2[7],y2[6],y2[5],y2[4],y2[3],y2[2],y2[1],y2[0]); 

//X3 = v*A
positive_mult256(v,y2);
//printf("X3:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		y2[7],y2[6],y2[5],y2[4],y2[3],y2[2],y2[1],y2[0]); 
//t5 = R-A
signed_add224(Z1_1,X1Z2_1,1,0); //t5 in Z1_1, X1Z2_1 = free
//printf("t5:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	Z1_1[7],Z1_1[6],Z1_1[5],Z1_1[4],Z1_1[3],Z1_1[2],Z1_1[1],Z1_1[0]); 
//t6 = vvv*Y1Z2
positive_mult256(Y1Z2_1,Z2_2); //t6 in Z2_2, Y1Z2_1 = free
//printf("t6:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	Z2_2[7],Z2_2[6],Z2_2[5],Z2_2[4],Z2_2[3],Z2_2[2],Z2_2[1],Z2_2[0]); 
//t7 = u*t5
positive_mult256(u,Z1_1);
//printf("t7:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		Z1_1[7],Z1_1[6],Z1_1[5],Z1_1[4],Z1_1[3],Z1_1[2],Z1_1[1],Z1_1[0]);

//Y3 = t7-t6
signed_add224(Z1_1,Z2_2,0,1); //Y3 in Z1_1
//printf("Y3:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		Z1_1[7],Z1_1[6],Z1_1[5],Z1_1[4],Z1_1[3],Z1_1[2],Z1_1[1],Z1_1[0]);

//Z3 = vvv*Z1Z2
positive_mult256(x2,Z2_3); //Z3 in Z2_3
//printf("Z3:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	Z2_3[7],Z2_3[6],Z2_3[5],Z2_3[4],Z2_3[3],Z2_3[2],Z2_3[1],Z2_3[0]);

for(int i=0;i<=7;i++){
	x1[i]=y2[i];
	y1[i]=Z1_1[i];
	z1[i]=Z2_3[i];
}


}