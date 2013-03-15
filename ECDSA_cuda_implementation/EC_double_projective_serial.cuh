#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "finite_field_arithmetic.cuh"


/*Elliptic Curve doubling
Projective-3 coords for short Weirstrass Form
 "dbl-2007-bl-2" doubling formulas - 7M+3S
X1,Y1,Z1 < 2^224, but they are stored in 256 bit arrays */

__device__ void EC_double_serial(uint32_t* x, uint32_t* y, uint32_t* z){

	
	
	uint32_t t0[8];
	uint32_t t1[8];
	uint32_t z1[8];

	for(int i=0;i<=7;i++){
		t0[i]=x[i];
	}
	
	for(int i=0;i<=7;i++){
		z1[i]=z[i];
	}
	
	//t0 = X1-Z1
	signed_add224(t0,z1,0,1);
	


	for(int i=0;i<=7;i++){
		t1[i]=x[i];
	}
	
	for(int i=0;i<=7;i++){
		z1[i]=z[i];
	}
	

	//t0 = X1+Z1
	
	signed_add224(t1,z1,0,0);
	

	

	//t2 = t0*t1 
	
	positive_mult256(t0,t1);
	

	
	
	
	

	


	//perhaps I can do this slightly better - but only slightly
	uint32_t scale[8]={3,0,0,0,0,0,0,0};
 
	positive_mult256(scale,t1);

	//w = 3*t2
	
	
	
	uint32_t wt[8];
	for(int i=0;i<=7;i++){
		wt[i]=t1[i];
	}
	
	 //use t0 which is available
	 for(int i=0;i<=7;i++){
		 t0[i]=y[i];
	 }
	 
	
	 //t3 = Y1*Z1
	 positive_mult256(z1,t0);
	
	 

	 //printf("T3:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	t0[7],t0[6],t0[5],t0[4],t0[3],t0[2],t0[1],t0[0]); 

	 //now Z1 doesnt contain z

	 scale[0]=2;
	
	 
	 //s=2*t3
	positive_mult256(scale,t0);
	

	
	//printf("S:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
				//t0[7],t0[6],t0[5],t0[4],t0[3],t0[2],t0[1],t0[0]); 
	
	uint32_t st[8],stt[8],sttt[8];
	for(int i=0;i<=7;i++){
		st[i]=t0[i];
		stt[i]=t0[i];
		sttt[i]=t0[i];
	}
	
	
	//ss=s^2
	square256(t0);
	//printf("SS:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	t0[7],t0[6],t0[5],t0[4],t0[3],t0[2],t0[1],t0[0]); 
	
	//sss=ss*s
	positive_mult256(st,t0);
	
	
	//printf("SSS:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		t0[7],t0[6],t0[5],t0[4],t0[3],t0[2],t0[1],t0[0]); 

	//We have Zfinal
	for(int i=0;i<=7;i++){
		z[i]=t0[i];
	}
	

	//R=Y1*s
	uint32_t tr[8];
	
	for(int i=0;i<=7;i++){
		st[i]=y[i];
		
	}
	
	//printf("stt:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		stt[7],stt[6],stt[5],stt[4],stt[3],stt[2],stt[1],stt[0]); 
	//printf("st:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		st[7],st[6],st[5],st[4],st[3],st[2],st[1],st[0]); 
	
	positive_mult256(stt,st);
	//	printf("MULRES:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		STT.num[7],STT.num[6],STT.num[5],STT.num[4],STT.num[3],STT.num[2],STT.num[1],STT.num[0],
			//	ST.num[7],ST.num[6],ST.num[5],ST.num[4],ST.num[3],ST.num[2],ST.num[1],ST.num[0]); 
	


	
	
	for(int i=0;i<=7;i++){
		tr[i]=st[i];
	}
	
	
	//printf("R:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//		R.num[7],R.num[6],R.num[5],R.num[4],R.num[3],R.num[2],R.num[1],R.num[0]);

	//RR=R^2
	square256(st);
	//printf("RR:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	st[7],st[6],st[5],st[4],st[3],st[2],st[1],st[0]);
	
	//modred_long(tt,R.num);

	
	

	
	//t4=X1*R
	for(int i=0;i<=7;i++){
		stt[i]=x[i];
	}
	positive_mult256(stt,tr);
	//modred_long(stt,TR.num);
	

	//printf("T4:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	tr[7],tr[6],tr[5],tr[4],tr[3],tr[2],tr[1],tr[0]);

	
	//B=2*t4
	scale[0]=2;
	positive_mult256(scale,tr);
	//modred_long(scale,T4.num);
	
	
	uint32_t bt[8];
	for(int i=0;i<=7;i++){
		bt[i]=tr[i];
	}

	//printf("B:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	tr[7],tr[6],tr[5],tr[4],tr[3],tr[2],tr[1],tr[0]); 
	//
	//t5=w^2
	
	square256(t1);
	
	
	
	

	//printf("T5:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	t1[7],t1[6],t1[5],t1[4],t1[3],t1[2],t1[1],t1[0]); 
	
	//t6=2*B
	scale[0]=2;
	positive_mult256(scale,bt);

	//printf("T6:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
				//bt[7],bt[6],bt[5],bt[4],bt[3],bt[2],bt[1],bt[0]); 
	
	//h=t5-t6
	
	signed_add224(t1,bt,0,1);
	
	//printf("H:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	t1[7],t1[6],t1[5],t1[4],t1[3],t1[2],t1[1],t1[0]); 
	
	uint32_t ht[8];
	for(int i=0;i<=7;i++){
		ht[i]=t1[i];
	}
	//X3=h*s
	
	positive_mult256(sttt,t1);
	//modred_long(sttt,H.num);
	//printf("X1:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	t1[7],t1[6],t1[5],t1[4],t1[3],t1[2],t1[1],t1[0]); 
	
	

	
	//t7=B-h
	
	
	signed_add224(ht,tr,1,0);

	
	
	//printf("T8:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	st[7],st[6],st[5],st[4],st[3],st[2],st[1],st[0]);

	
	//t9=w*t7
	positive_mult256(wt,ht);
	
	//printf("T9:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
			//	ht[7],ht[6],ht[5],ht[4],ht[3],ht[2],ht[1],ht[0]);

	
	//Y1=t9-t8
	signed_add224(ht,st,0,1);
	
	
	
	//printf("Y1:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//	ht[7],ht[6],ht[5],ht[4],ht[3],ht[2],ht[1],ht[0]);
	for (int i=0;i<=7;i++){
		x[i]=t1[i];
		y[i]=ht[i];
	}

	//printf("X:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//	x[7],x[6],x[5],x[4],x[3],x[2],x[1],x[0]);
	//printf("Y:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//	y[7],y[6],y[5],y[4],y[3],y[2],y[1],y[0]);
		
	//printf("Z:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//	z[7],z[6],z[5],z[4],z[3],z[2],z[1],z[0]);

	



}
