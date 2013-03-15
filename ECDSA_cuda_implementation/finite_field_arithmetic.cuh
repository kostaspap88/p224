#ifndef FF_AR
#define FF_AR

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>



//constant group order
__constant__ uint32_t gorder[8]={1,0,0,4294967295,4294967295,4294967295,4294967295,0};
__constant__ uint32_t gorder224[7]={1,0,0,4294967295,4294967295,4294967295,4294967295};
__constant__ uint32_t gorder512[16]={1,0,0,4294967295,4294967295,4294967295,4294967295,0,0,0,0,0,0,0,0,0};
//constant large zero
__constant__ uint32_t zero[8]={0,0,0,0,0,0,0,0};
__constant__ uint32_t zero512[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

//curve parameters can be stored in constant memory for faster access
__constant__ uint32_t b256[8]={0x2355ffb4,  0x270b3943,  0xd7bfd8ba,  0x5044b0b7,  0xf5413256,  0x0c04b3ab,  0xb4050a85, 0x0};

__constant__ uint32_t xG256[8]={0x115c1d21, 0x343280d6, 0x56c21122, 0x4a03c1d3, 0x321390b9, 0x6bb4bf7f, 0xb70e0cbd, 0x0};

__constant__ uint32_t yG256[8]={0x85007e34, 0x44d58199, 0x5a074764, 0xcd4375a0, 0x4c22dfe6, 0xb5f723fb, 0xbd376388, 0x0};

__constant__ uint32_t zG256[8]={1,0,0,0,0,0,0,0};

__constant__ uint32_t a256[8]={3,0,0,0,0,0,0,0};



/* 32 bit multiplication, result in a and b (a is high part, b is low part) */
__device__ void mult32(uint32_t *b, uint32_t *a)
{
	
	uint32_t temp;
	asm("mul.lo.u32	 %0, %2, %1;\n\t"
		"mul.hi.u32	 %1, %2, %1;\n\t"
		: "=r"(temp), "+r"(*b)
		: "r"(*a));
	*a=temp;
	
	
}

/* 64 bit mult, result in a and b (a is high part, b is low part) */
__device__ void mult64(uint32_t *a, uint32_t *b)
{
	//size of a,b is 64 bits
	uint32_t al, ah, bl,bh,ali,ahi,bli,bhi;
	al=a[0];
	ah=a[1];
	bl=b[0];
	bh=b[1];
	ali=a[0];
	ahi=a[1];
	bli=b[0];
	bhi=b[1];
	
	//printf("al=%08x, bl=%08x, ah=%08x, bh=%08x\n",ali,bli,ahi,bhi);
	//printf("BEGIN: ah=%08x, bh=%08x, al=%08x, bl=%08x\n",ah,bh,al,bl);
	
	mult32(&al,&bl);
	uint32_t mult_al,mult_bl;
	mult_al=al;
	mult_bl=bl;
	//printf("MULLOW: al,bl= %08x %08x\n",al,bl);
	
	
	mult32(&ah,&bh);
	uint32_t mult_ah,mult_bh;
	mult_ah=ah;
	mult_bh=bh;
	//printf("MULHIGH: ah,bh= %08x %08x\n",ah,bh);


	 
	uint32_t tempah=0;
	uint32_t tempal=0;
	uint32_t tempbh=0;
	uint32_t tempbl=0;
	asm("add.cc.u32 %0, %4, %5;\n\t"
		"addc.u32 %1, %1, 0;\n\t"
		"add.cc.u32 %2, %6, %7;\n\t"
		"addc.u32 %3, %3, 0;\n\t"
		: "=r"(tempal), "+r"(tempah), "=r"(tempbl), "+r"(tempbh) //0,1,2,3
		: "r"(ahi), "r"(ali), "r"(bhi), "r"(bli)); //4,5,6,7
	//printf("TEMP: tempah,tempal= %08x %08x\n",tempah,tempal);printf("TEMP: tempbh,tempbl= %08x %08x\n",tempbh,tempbl);

	uint32_t tempcarry=0;
	uint32_t tempali=tempal;
	uint32_t tempbli=tempbl;

	mult32(&tempal,&tempbl);
	tempcarry=tempah&tempbh;//replace it with asm()
	//I hate using 'if' here, perhaps I will figure out another way
	if(tempah==1){
		asm("add.cc.u32 %0,%0,%2;\n\t"
			"addc.u32 %1,%1,0;\n\t"
			: "+r"(tempal), "+r"(tempcarry)
			: "r"(tempbli));
		
	}
	if(tempbh==1){
		asm("add.cc.u32 %0,%0,%2;\n\t"
			"addc.u32 %1,%1,0;\n\t"
			: "+r"(tempal), "+r"(tempcarry)
			: "r"(tempali));
	
	}

	
	//printf("TCARRY: tcarry= %d\n",tempcarry);

//	asm("mul.lo.u32 tempah, tempah, tempbh"
	


	//printf("AFTER MUL: tcarry,tempal,tempbl= %08x %08x %08x\n",tempcarry,tempal,tempbl);
	//printf("BEFORE (term1)*(term2) ADD: ah=%08x, bh=%08x, al=%08x, bl=%08x\n",ah,bh,al,bl);
	//printf("AFTER MUL: tempah,tempbh= %08x %08x\n",tempah,tempbh);
	asm("add.cc.u32 %0,%0,%3;\n\t"
		"addc.cc.u32 %1,%1,%4;\n\t"
		"addc.u32 %2,%2,%5;\n\t"
		: "+r"(al), "+r"(bh), "+r"(ah)
		: "r"(tempbl), "r"(tempal), "r"(tempcarry));
	
//	printf("AFTER (term1)*(term2) ADD:: ah=%08x, bh=%08x, al=%08x, bl=%08x\n",ah,bh,al,bl);

	asm("sub.cc.u32 %0, %0, %3;\n\t"
		"subc.cc.u32 %1, %1, %4;\n\t"
		"subc.u32 %2, %2, 0;\n\t"
		"sub.cc.u32 %0, %0, %6;\n\t"
		"subc.cc.u32 %1, %1, %5;\n\t"
		"subc.u32 %2, %2, 0;\n\t"
		: "+r"(al), "+r"(bh), "+r"(ah)
		: "r"(mult_bh), "r"(mult_ah), "r"(mult_al), "r"(mult_bl));
	/* //SCHOOLBOOK VERSION ALTERNATIVE
	mult32(&ahi,&bli);
	//printf("ahi,bli= %08x %08x\n", ahi,bli);
	
	
	mult32(&ali,&bhi);
	printf("ali,bhi= %08x %08x\n", ali,bhi);
	printf("BEFORE ADDITIONS: ah=%08x, bh=%08x, al=%08x, bl=%08x\n",ah,bh,al,bl);
	asm("add.cc.u32 %0, %0, %3;\n\t"
		"addc.cc.u32 %1, %1, %4;\n\t"
		"addc.u32 %2, %2, 0;\n\t"
		: "+r"(al), "+r"(bh), "+r"(ah)
		: "r"(bli), "r"(ahi));

	//printf("1st ADD: ah=%08x, bh=%08x, al=%08x, bl=%08x\n",ah,bh,al,bl);

	asm("add.cc.u32 %0, %0, %3;\n\t"
		"addc.cc.u32 %1, %1, %4;\n\t"
		"addc.u32 %2, %2, 0;\n\t"
		: "+r"(al), "+r"(bh), "+r"(ah)
		: "r"(bhi), "r"(ali));
*/
	//printf("END: ah=%08x, bh=%08x, al=%08x, bl=%08x\n",ah,bh,al,bl);


	
	 a[0]=bh;
	 a[1]=ah;
	 b[0]=bl;
	 b[1]=al;
}


/* 128bit mult, result in a and b (a is high part, b is low part) */
__device__ void mult128(uint32_t *a, uint32_t *b){
	uint32_t alow[2],blow[2],ahigh[2],bhigh[2];


	//printf("a128: %08x  %08x  %08x  %08x\n",a[3],a[2],a[1],a[0]);
	//printf("b128: %08x  %08x  %08x  %08x\n",b[3],b[2],b[1],b[0]);

	alow[0]=a[0];
	alow[1]=a[1];
	ahigh[0]=a[2];
	ahigh[1]=a[3];
	blow[0]=b[0];
	blow[1]=b[1];
	bhigh[0]=b[2];
	bhigh[1]=b[3];
	mult64(alow,blow);
	mult64(ahigh,bhigh);
//	printf("LOW MULT 128: %08x  %08x  %08x  %08x\n",alow[1],alow[0],blow[1],blow[0]);
//	printf("HIGH MUL 128: %08x  %08x  %08x  %08x\n",ahigh[1],ahigh[0],bhigh[1],bhigh[0]);

	uint32_t suma[3]={0,0,0};
	asm("add.cc.u32 %0, %3, %4;\n\t"
		"addc.cc.u32 %1, %5, %6;\n\t"
		"addc.u32 %2, %2, 0;\n\t"
		:"=r"(suma[0]), "=r"(suma[1]), "+r"(suma[2]) //0,1,2
		:"r"(a[2]), "r"(a[0]), "r"(a[3]), "r"(a[1]) ); //3,4,5,6

//	printf("A SUM 128: %08x  %08x  %08x\n",suma[2],suma[1],suma[0]);

	uint32_t sumb[3]={0,0,0};
	asm("add.cc.u32 %0, %3, %4;\n\t"
		"addc.cc.u32 %1, %5, %6;\n\t"
		"addc.u32 %2, %2, 0;\n\t"
		:"=r"(sumb[0]), "=r"(sumb[1]), "+r"(sumb[2]) //0,1,2
		:"r"(b[2]), "r"(b[0]), "r"(b[3]), "r"(b[1]) ); //3,4,5,6

//	printf("B SUM128: %08x  %08x  %08x\n",sumb[2],sumb[1],sumb[0]);

	uint32_t kar_high[2],kar_low[2];
	kar_high[0]=suma[0];
	kar_high[1]=suma[1];
	kar_low[0]=sumb[0];
	kar_low[1]=sumb[1];

	mult64(kar_high,kar_low);
//   printf("KAR MUL: %08x  %08x  %08x  %08x\n",kar_high[1],kar_high[0],kar_low[1],kar_low[0]);

   //as usual take care of the spilover carry multiplication ( I make it sound easy but it was not, I was confused at first)
   //again I dont like the 'if' will check again later
   uint32_t kar_top=0;
   

   if(sumb[2]==1){
   asm("add.cc.u32 %0,%0,%3;\n\t"
	   "addc.cc.u32 %1,%1,%4;\n\t"
	   "addc.u32 %2,%2,0;\n\t"
		:"+r"(kar_high[0]), "+r"(kar_high[1]), "+r"(kar_top)
		:"r"(suma[0]), "r"(suma[1]));
   }

   if(suma[2]==1){
   asm("add.cc.u32 %0,%0,%3;\n\t"
	   "addc.cc.u32 %1,%1,%4;\n\t"
	   "addc.u32 %2,%2,0;\n\t"
		:"+r"(kar_high[0]), "+r"(kar_high[1]), "+r"(kar_top)
		:"r"(sumb[0]), "r"(sumb[1]));
   }

   
 //   printf("KAR MUL and CARRY128: %08x  %08x  %08x  %08x  %08x\n",kar_top,kar_high[1],kar_high[0],kar_low[1],kar_low[0]);

	//restructuring results
	
	a[0]=bhigh[0];
	a[1]=bhigh[1];
	a[2]=ahigh[0];
	a[3]=ahigh[1];
	b[0]=blow[0];
	b[1]=blow[1];
	b[2]=alow[0];
	b[3]=alow[1];

	
	//double carry spilover
	if ((suma[2]==1)&&(sumb[2]==1)){
	//	printf("DOUBLE CARRY in KAR128\n");
	   asm("add.cc.u32 %0,%0,1;\n\t"
		   "addc.u32 %1,%1,0;\n\t"
		   :"+r"(a[2]), "+r"(a[3]));
   }

	//adding kar_mul to results
	asm("add.cc.u32 %0,%0,%6;\n\t"
		"addc.cc.u32 %1,%1,%7;\n\t"
		"addc.cc.u32 %2,%2,%8;\n\t"
		"addc.cc.u32 %3,%3,%9;\n\t"
		"addc.cc.u32 %4,%4,%10;\n\t"
		"addc.u32 %5,%5,0;\n\t"
		:"+r"(b[2]), "+r"(b[3]), "+r"(a[0]), "+r"(a[1]), "+r"(a[2]),"+r"(a[3]) //0,1,2,3,4
		:"r"(kar_low[0]), "r"(kar_low[1]), "r"(kar_high[0]), "r"(kar_high[1]), "r"(kar_top)); //5,6,7,8,9

//	printf("AFTER ADD 128: %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",a[3],a[2],a[1],a[0],b[3],b[2],b[1],b[0]);
	

	//subtractions

	

	
	asm("sub.cc.u32 %0,%0,%6;\n\t"
		"subc.cc.u32 %1,%1,%7;\n\t"
		"subc.cc.u32 %2,%2,%8;\n\t"
		"subc.cc.u32 %3,%3,%9;\n\t"
		"subc.cc.u32 %4,%4,0;\n\t"
		"subc.u32 %5,%5,0;\n\t"
		:"+r"(b[2]), "+r"(b[3]), "+r"(a[0]), "+r"(a[1]), "+r"(a[2]),"+r"(a[3]) //0,1,2,3,4
		:"r"(blow[0]), "r"(blow[1]), "r"(alow[0]), "r"(alow[1]));

	asm("sub.cc.u32 %0,%0,%6;\n\t"
		"subc.cc.u32 %1,%1,%7;\n\t"
		"subc.cc.u32 %2,%2,%8;\n\t"
		"subc.cc.u32 %3,%3,%9;\n\t"
		"subc.cc.u32 %4,%4,0;\n\t"
		"subc.u32 %5,%5,0;\n\t"
		:"+r"(b[2]), "+r"(b[3]), "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3]) //0,1,2,3,4
		:"r"(bhigh[0]), "r"(bhigh[1]), "r"(ahigh[0]), "r"(ahigh[1]));
	
	
//	 printf("FINAL128 %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",a[3],a[2],a[1],a[0],b[3],b[2],b[1],b[0]);

}

/* 256 bit mult, result in a and b (a is high part, b is low part) */
__device__ void mult256(uint32_t *a, uint32_t *b){
	uint32_t alow[4],blow[4],ahigh[4],bhigh[4];

	alow[0]=a[0];
	alow[1]=a[1];
	alow[2]=a[2];
	alow[3]=a[3];

	ahigh[0]=a[4];
	ahigh[1]=a[5];
	ahigh[2]=a[6];
	ahigh[3]=a[7];

	blow[0]=b[0];
	blow[1]=b[1];
	blow[2]=b[2];
	blow[3]=b[3];

	bhigh[0]=b[4];
	bhigh[1]=b[5];
	bhigh[2]=b[6];
	bhigh[3]=b[7];

	//printf("a256 %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
	//printf("b256 %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",b[7],b[6],b[5],b[4],b[3],b[2],b[1],b[0]);
	mult128(alow,blow);
	mult128(ahigh,bhigh);

	//printf("MUL LOW 256 %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",alow[3],alow[2],alow[1],alow[0],
	//	blow[3],blow[2],blow[1],blow[0]);

	//printf("MUL HIGH %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",ahigh[3],ahigh[2],ahigh[1],ahigh[0],
	//	bhigh[3],bhigh[2],bhigh[1],bhigh[0]);


	uint32_t suma[5]={0,0,0,0,0};
	asm("add.cc.u32 %0, %5, %6;\n\t"
		"addc.cc.u32 %1, %7, %8;\n\t"
		"addc.cc.u32 %2, %9, %10;\n\t"
		"addc.cc.u32 %3, %11, %12;\n\t"
		"addc.u32 %4, %4, 0;\n\t"
		:"=r"(suma[0]), "=r"(suma[1]), "=r"(suma[2]), "=r"(suma[3]), "+r"(suma[4]) 
		:"r"(a[4]), "r"(a[0]), "r"(a[5]), "r"(a[1]) , "r"(a[6]), "r"(a[2]), "r"(a[7]), "r"(a[3]));

	//printf("A SUM 256: %08x  %08x  %08x  %08x  %08x\n",suma[4],suma[3],suma[2],suma[1],suma[0]);

	uint32_t sumb[5]={0,0,0,0,0};
	asm("add.cc.u32 %0, %5, %6;\n\t"
		"addc.cc.u32 %1, %7, %8;\n\t"
		"addc.cc.u32 %2, %9, %10;\n\t"
		"addc.cc.u32 %3, %11, %12;\n\t"
		"addc.u32 %4, %4, 0;\n\t"
		:"=r"(sumb[0]), "=r"(sumb[1]), "=r"(sumb[2]), "=r"(sumb[3]), "+r"(sumb[4]) 
		:"r"(b[4]), "r"(b[0]), "r"(b[5]), "r"(b[1]) , "r"(b[6]), "r"(b[2]), "r"(b[7]), "r"(b[3]));

//	printf("B SUM 256: %08x  %08x  %08x  %08x  %08x\n",sumb[4],sumb[3],sumb[2],sumb[1],sumb[0]);

	
	uint32_t kar_high[4],kar_low[4];
	kar_high[0]=suma[0];
	kar_high[1]=suma[1];
	kar_high[2]=suma[2];
	kar_high[3]=suma[3];
	kar_low[0]=sumb[0];
	kar_low[1]=sumb[1];
	kar_low[2]=sumb[2];
	kar_low[3]=sumb[3];

	mult128(kar_high,kar_low);

  // printf("KAR MUL 256: %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",kar_high[3],kar_high[2],kar_high[1],kar_high[0],kar_low[3],kar_low[2],kar_low[1],kar_low[0]);

   //as usual take care of the spilover carry multiplication ( I make it sound easy but it was not, I was confused at first)
   //again I dont like the 'if' will check again later
   uint32_t kar_top=0;
   

   if(sumb[4]==1){
	  // printf("INSIDE: add suma\n");
   asm("add.cc.u32 %0,%0,%5;\n\t"
	   "addc.cc.u32 %1,%1,%6;\n\t"
	   "addc.cc.u32 %2,%2,%7;\n\t"
	   "addc.cc.u32 %3,%3,%8;\n\t"
	   "addc.u32 %4,%4,0;\n\t"
		:"+r"(kar_high[0]), "+r"(kar_high[1]), "+r"(kar_high[2]), "+r"(kar_high[3]), "+r"(kar_top)
		:"r"(suma[0]), "r"(suma[1]), "r"(suma[2]), "r"(suma[3]));
   }

   if(suma[4]==1){
	  // printf("INSIDE: add sumb\n");
   asm("add.cc.u32 %0,%0,%5;\n\t"
	   "addc.cc.u32 %1,%1,%6;\n\t"
	   "addc.cc.u32 %2,%2,%7;\n\t"
	   "addc.cc.u32 %3,%3,%8;\n\t"
	   "addc.u32 %4,%4,0;\n\t"
		:"+r"(kar_high[0]), "+r"(kar_high[1]), "+r"(kar_high[2]), "+r"(kar_high[3]), "+r"(kar_top)
		:"r"(sumb[0]), "r"(sumb[1]), "r"(sumb[2]), "r"(sumb[3]));
   }

    //double carry spilover
	
	if ((suma[4]==1)&&(sumb[4]==1)){
		//printf("INSIDE: DOUBLE CARRY\n");
	   asm("add.cc.u32 %0,%0,1;\n\t"  
		   :"+r"(kar_top));
   }

 // printf("KAR MUL and CARRY 256: %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",kar_top,kar_high[3],kar_high[2],kar_high[1],kar_high[0],kar_low[3],kar_low[2],kar_low[1],kar_low[0]);

	//restructuring results
	
	a[0]=bhigh[0];
	a[1]=bhigh[1];
	a[2]=bhigh[2];
	a[3]=bhigh[3];
	a[4]=ahigh[0];
	a[5]=ahigh[1];
	a[6]=ahigh[2];
	a[7]=ahigh[3];
	
	b[0]=blow[0];
	b[1]=blow[1];
	b[2]=blow[2];
	b[3]=blow[3];
	b[4]=alow[0];
	b[5]=alow[1];
	b[6]=alow[2];
	b[7]=alow[3];
	
  

	//printf("BEFORE BIG ADD 256: %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
//		a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0],b[7],b[6],b[5],b[4],b[3],b[2],b[1],b[0]);
	//adding kar_mul to results
	uint32_t temp_carry=0;
	asm("add.cc.u32 %0,%0,%5;\n\t"
		"addc.cc.u32 %1,%1,%6;\n\t"
		"addc.cc.u32 %2,%2,%7;\n\t"
		"addc.cc.u32 %3,%3,%8;\n\t"
		"addc.u32 %4,%4,0;\n\t"
		:"+r"(b[4]), "+r"(b[5]), "+r"(b[6]), "+r"(b[7]), "+r"(temp_carry) //0,1,2,3,4
		:"r"(kar_low[0]), "r"(kar_low[1]), "r"(kar_low[2]), "r"(kar_low[3])); //5,6,7,8,9
//	printf("ADD Carry is: %d\n",temp_carry);
//	printf("CarTOP is: %d\n",kar_top);
//	printf("LOW BIG ADD: %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x \n",
//			b[7],b[6],b[5],b[4],b[3],b[2],b[1],b[0]);
	uint32_t top_temp=0; //we need this only to do the big addittion. After subs it will be gone :X
	//BE CAREFUL with Carry
	
	asm("add.cc.u32 %0,%0,%13;\n\t"
		"addc.cc.u32 %0,%0,%9;\n\t"
		"addc.cc.u32 %1,%1,%10;\n\t"
		"addc.cc.u32 %2,%2,%11;\n\t"
		"addc.cc.u32 %3,%3,%12;\n\t"
		"addc.cc.u32 %4,%4,%14;\n\t"
		"addc.cc.u32 %5,%5,0;\n\t"
		"addc.cc.u32 %6,%6,0;\n\t"
		"addc.cc.u32 %7,%7,0;\n\t"
		"addc.u32 %8,%8,0;\n\t"
		:"+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3]), "+r"(a[4]), "+r"(a[5]),"+r"(a[6]),"+r"(a[7]),"+r"(top_temp) //0,1,2,3,4,5
		:"r"(kar_high[0]), "r"(kar_high[1]), "r"(kar_high[2]), "r"(kar_high[3]),"r"(temp_carry),"r"(kar_top)); //5,6,7,8,9


	


	//printf("AFTER BIG ADD 256: %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//	top_temp,a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0],b[7],b[6],b[5],b[4],b[3],b[2],b[1],b[0]);

	//subtractions

	
	temp_carry=0;
		asm("sub.cc.u32 %0,%0,%5;\n\t"
		"subc.cc.u32 %1,%1,%6;\n\t"
		"subc.cc.u32 %2,%2,%7;\n\t"
		"subc.cc.u32 %3,%3,%8;\n\t"
		//"addc.u32 %4,%4,0;\n\t"
		:"+r"(b[4]), "+r"(b[5]), "+r"(b[6]), "+r"(b[7]), "+r"(temp_carry) //0,1,2,3,4
		:"r"(blow[0]), "r"(blow[1]), "r"(blow[2]), "r"(blow[3]));
		//printf("SUB1 Carry is: %d\n",temp_carry);
		//BE SUPER-CAREFUl with carry here
		asm(
		"subc.cc.u32 %0,%0,%9;\n\t"
		"subc.cc.u32 %1,%1,%10;\n\t"
		"subc.cc.u32 %2,%2,%11;\n\t"
		"subc.cc.u32 %3,%3,%12;\n\t"
		"subc.cc.u32 %4,%4,0;\n\t"
		"subc.cc.u32 %5,%5,0;\n\t"
		"subc.cc.u32 %6,%6,0;\n\t"
		"subc.cc.u32 %7,%7,0;\n\t"
		"subc.u32 %8,%8,0;\n\t"
		:"+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3]), "+r"(a[4]), "+r"(a[5]), "+r"(a[6]), "+r"(a[7]), "+r"(top_temp) //0,1,2,3,4
		:"r"(alow[0]), "r"(alow[1]), "r"(alow[2]), "r"(alow[3]), "r"(temp_carry));


	//	printf("AFTER 1st SUB:%08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
	//		top_temp,a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0],b[7],b[6],b[5],b[4],b[3],b[2],b[1],b[0]);

		//second substraction

		//temp_carry =0;
		asm("sub.cc.u32 %0,%0,%5;\n\t"
		"subc.cc.u32 %1,%1,%6;\n\t"
		"subc.cc.u32 %2,%2,%7;\n\t"
		"subc.cc.u32 %3,%3,%8;\n\t"
		//"addc.u32 %4,%4,0;\n\t"
		:"+r"(b[4]), "+r"(b[5]), "+r"(b[6]), "+r"(b[7]), "+r"(temp_carry) //0,1,2,3,4
		:"r"(bhigh[0]), "r"(bhigh[1]), "r"(bhigh[2]), "r"(bhigh[3]));
	//	printf("SUB2 Carry is: %d\n",temp_carry);
	//	printf("LOW SUB2: %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x \n",
	//		b[7],b[6],b[5],b[4],b[3],b[2],b[1],b[0]);
		asm(
		
		"subc.cc.u32 %0,%0,%9;\n\t"
		"subc.cc.u32 %1,%1,%10;\n\t"
		"subc.cc.u32 %2,%2,%11;\n\t"
		"subc.cc.u32 %3,%3,%12;\n\t"
		"subc.cc.u32 %4,%4,0;\n\t"
		"subc.cc.u32 %5,%5,0;\n\t"
		"subc.cc.u32 %6,%6,0;\n\t"
		"subc.cc.u32 %7,%7,0;\n\t"
		"subc.u32 %8,%8,0;\n\t"
		:"+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3]), "+r"(a[4]), "+r"(a[5]), "+r"(a[6]), "+r"(a[7]), "+r"(top_temp) //0,1,2,3,4
		:"r"(ahigh[0]), "r"(ahigh[1]), "r"(ahigh[2]), "r"(ahigh[3]), "r"(temp_carry));


		//printf("FINAL256:%08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x\n",
		//	top_temp,a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0],b[7],b[6],b[5],b[4],b[3],b[2],b[1],b[0]);
		

}

__device__ void add224(uint32_t *a, uint32_t *b){

	
	asm("add.cc.u32		%0, %0, %8;\n\t"
		"addc.cc.u32	%1, %1, %9;\n\t"
		"addc.cc.u32	%2, %2, %10;\n\t"
		"addc.cc.u32	%3, %3, %11;\n\t"
		"addc.cc.u32	%4, %4, %12;\n\t"
		"addc.cc.u32	%5, %5, %13;\n\t"
		"addc.cc.u32	%6, %6, %14;\n\t"
		"addc.u32		%7,%7,0;\n\t"
		: "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3]), "+r"(a[4]), "+r"(a[5]), "+r"(a[6]), "+r"(a[7])
		: "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]));

	
	
}
//add 256 bit numbers, without overflow
__device__ void add256(uint32_t *a, uint32_t *b){

	asm("add.cc.u32		%0, %0, %4;\n\t"
		"addc.cc.u32	%1, %1, %5;\n\t"
		"addc.cc.u32	%2, %2, %6;\n\t"
		"addc.cc.u32	%3, %3, %7;\n\t"
		: "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
		: "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));


	asm("addc.cc.u32	%0, %0, %4;\n\t"
		"addc.cc.u32	%1, %1, %5;\n\t"
		"addc.cc.u32	%2, %2, %6;\n\t"
		"addc.cc.u32	%3, %3, %7;\n\t"
		: "+r"(a[4]), "+r"(a[5]), "+r"(a[6]), "+r"(a[7])
		: "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));

	asm("addc.cc.u32	%0, %0, %4;\n\t"
		"addc.cc.u32	%1, %1, %5;\n\t"
		"addc.cc.u32	%2, %2, %6;\n\t"
		"addc.cc.u32	%3, %3, %7;\n\t"
		: "+r"(a[8]), "+r"(a[9]), "+r"(a[10]), "+r"(a[11])
		: "r"(b[8]), "r"(b[9]), "r"(b[10]), "r"(b[11]));

	asm("addc.cc.u32	%0, %0, %4;\n\t"
		"addc.cc.u32	%1, %1, %5;\n\t"
		"addc.cc.u32	%2, %2, %6;\n\t"
		"addc.cc.u32	%3, %3, %7;\n\t"
		: "+r"(a[12]), "+r"(a[13]), "+r"(a[14]), "+r"(a[15])
		: "r"(b[12]), "r"(b[13]), "r"(b[14]), "r"(b[15]));
}


//return 2 if equal, 1 if a>b, 0 if a<b
__device__ uint8_t cmp224(uint32_t *a, uint32_t *b){

	if(a[7]>b[7])
		return 1;
	if(a[7]<b[7])
		return 0;
	//it would be nice if I replaced it with PTX
	if (a[6]>b[6])
		return 1;
	if (a[6]<b[6])
		return 0;
	if (a[5]>b[5])
		return 1;
	if (a[5]<b[5])
		return 0;
	if (a[4]>b[4])
		return 1;
	if (a[4]<b[4])
		return 0;
	if (a[3]>b[3])
		return 1;
	if (a[3]<b[3])
		return 0;
	if (a[2]>b[2])
		return 1;
	if (a[2]<b[2])
		return 0;
	if (a[1]>b[1])
		return 1;
	if (a[1]<b[1])
		return 0;
	if (a[0]>b[0])
		return 1;
	if (a[0]<b[0])
		return 0;

	return 2;
}

//return 2 if equal, 1 if a>b, 0 if a<b
__device__ uint8_t cmp512(uint32_t *a, uint32_t *b){
	
	if(a[15]>b[75])
		return 1;
	if(a[15]<b[15])
		return 0;
	//it would be nice if I replaced it with PTX
	if (a[14]>b[14])
		return 1;
	if (a[14]<b[14])
		return 0;
	if (a[13]>b[13])
		return 1;
	if (a[13]<b[13])
		return 0;
	if (a[12]>b[12])
		return 1;
	if (a[12]<b[12])
		return 0;
	if (a[11]>b[11])
		return 1;
	if (a[11]<b[11])
		return 0;
	if (a[10]>b[10])
		return 1;
	if (a[10]<b[10])
		return 0;
	if (a[9]>b[9])
		return 1;
	if (a[9]<b[9])
		return 0;
	if (a[8]>b[8])
		return 1;
	if (a[8]<b[8])
		return 0;
	
	if(a[7]>b[7])
		return 1;
	if(a[7]<b[7])
		return 0;
	//it would be nice if I replaced it with PTX
	if (a[6]>b[6])
		return 1;
	if (a[6]<b[6])
		return 0;
	if (a[5]>b[5])
		return 1;
	if (a[5]<b[5])
		return 0;
	if (a[4]>b[4])
		return 1;
	if (a[4]<b[4])
		return 0;
	if (a[3]>b[3])
		return 1;
	if (a[3]<b[3])
		return 0;
	if (a[2]>b[2])
		return 1;
	if (a[2]<b[2])
		return 0;
	if (a[1]>b[1])
		return 1;
	if (a[1]<b[1])
		return 0;
	if (a[0]>b[0])
		return 1;
	if (a[0]<b[0])
		return 0;

	return 2;
}


//assume that a>=b
__device__ void sub224(uint32_t *a, uint32_t *b){

	
	asm("sub.cc.u32		%0, %0, %8;\n\t"
		"subc.cc.u32	%1, %1, %9;\n\t"
		"subc.cc.u32	%2, %2, %10;\n\t"
		"subc.cc.u32	%3, %3, %11;\n\t"
		"subc.cc.u32	%4, %4, %12;\n\t"
		"subc.cc.u32	%5, %5, %13;\n\t"
		"subc.cc.u32	%6, %6, %14;\n\t"
		"subc.u32		%7,%7,0;\n\t"
		: "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3]), "+r"(a[4]), "+r"(a[5]), "+r"(a[6]), "+r"(a[7])
		: "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]), "r"(b[4]), "r"(b[5]), "r"(b[6]));

	
	
}

//subtraction b(512) from a(512), a>b so no need for extra overflow byte
__device__ void sub512(uint32_t *a, uint32_t *b){
	
	asm("sub.cc.u32		%0, %0, %4;\n\t"
		"subc.cc.u32	%1, %1, %5;\n\t"
		"subc.cc.u32	%2, %2, %6;\n\t"
		"subc.cc.u32	%3, %3, %7;\n\t"
		: "+r"(a[0]), "+r"(a[1]), "+r"(a[2]), "+r"(a[3])
		: "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]));


	asm("subc.cc.u32	%0, %0, %4;\n\t"
		"subc.cc.u32	%1, %1, %5;\n\t"
		"subc.cc.u32	%2, %2, %6;\n\t"
		"subc.cc.u32	%3, %3, %7;\n\t"
		: "+r"(a[4]), "+r"(a[5]), "+r"(a[6]), "+r"(a[7])
		: "r"(b[4]), "r"(b[5]), "r"(b[6]), "r"(b[7]));

	asm("subc.cc.u32	%0, %0, %4;\n\t"
		"subc.cc.u32	%1, %1, %5;\n\t"
		"subc.cc.u32	%2, %2, %6;\n\t"
		"subc.cc.u32	%3, %3, %7;\n\t"
		: "+r"(a[8]), "+r"(a[9]), "+r"(a[10]), "+r"(a[11])
		: "r"(b[8]), "r"(b[9]), "r"(b[10]), "r"(b[11]));

	asm("subc.cc.u32	%0, %0, %4;\n\t"
		"subc.cc.u32	%1, %1, %5;\n\t"
		"subc.cc.u32	%2, %2, %6;\n\t"
		"subc.cc.u32	%3, %3, %7;\n\t"
		: "+r"(a[12]), "+r"(a[13]), "+r"(a[14]), "+r"(a[15])
		: "r"(b[12]), "r"(b[13]), "r"(b[14]), "r"(b[15]));

	
}




//modulo reduction after multiplication of 256 bit numbers
//Special moduli reduction, see HaC chapter 14
//RESULT stored in b!
__device__ void modred_long(uint32_t *a,uint32_t *b){
	//a and b have 256 bits each so 512 in total

//	printf("MODREDLONG a:  %08x %08x %08x %08x %08x %08x %08x %08x\n",
	//	a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
	//printf("MODREDLONG b:  %08x %08x %08x %08x %08x %08x %08x %08x\n",
	//	b[7],b[6],b[5],b[4],b[3],b[2],b[1],b[0]);
	//q0=rounddown[x/b^t]
	//b^t=2^224, shift right ab for 224 bits = 7*32 , q0*b^t
	uint32_t x[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	
	uint32_t q[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	uint32_t tq[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	uint32_t tr[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	uint32_t r[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

	//unroll them ? I think the compiler should do it
	for(int i=0;i<=7;i++){
	
		x[i]=b[i];
	}
	for(int i=8;i<=15;i++){
	
		x[i]=a[i-8];
	}
//	printf("MODREDLONG x:  %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x\n",
	//	x[15],x[14],x[13],x[12],x[11],x[10],x[9],x[8],x[7],x[6],x[5],x[4],x[3],x[2],x[1],x[0]);

	if (cmp512(x,gorder512)==0){
		for(int i=0;i<=7;i++){
	
		a[i]=0;
		}
		//printf("no need to modreduce\n");
		return;
	}

	//create q0
	q[8]=a[7];
	q[7]=a[6];
	q[6]=a[5];
	q[5]=a[4];
	q[4]=a[3];
	q[3]=a[2];
	q[2]=a[1];
	q[1]=a[0];
	q[0]=b[7];

	

	//create r0
	r[6]=b[6];
	r[5]=b[5];
	r[4]=b[4];
	r[3]=b[3];
	r[2]=b[2];
	r[1]=b[1];
	r[0]=b[0];

	//printf("q[12] %08x \n",q[12]);

 
	int i=0;
	/*
	while(cmp512(q,zero512)==1){
		
	}*/
	
	
//	printf("q0:  %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x %08x \n",
	//	q[15],q[14],q[13],q[12],q[11],q[10],q[9],q[8],q[7],q[6],q[5],q[4],q[3],q[2],q[1],q[0]);

	//LOOP start
	while(cmp512(q,zero512)==1){

	//qnew=qold * c / b^t
	//qnew = qold *(2^96-1) / b^224
	//q*c = q<<96 - q , 96/32=3

	//tq= q<<96
	tq[0]=0;
	tq[1]=0;
	tq[2]=0;
	tq[3]=q[0];
	tq[4]=q[1];
	tq[5]=q[2];
	tq[6]=q[3];
	tq[7]=q[4];
	tq[8]=q[5];
	tq[9]=q[6];
	tq[10]=q[7];
	tq[11]=q[8];
	tq[12]=q[9];
	tq[13]=q[10];
	tq[14]=q[11];
	tq[15]=q[12];

	//q*c = tq - q , stored in tq
	sub512(tq,q);

	//I have tq=q*c, I need to to do >>224
	q[15]=0;
	q[14]=0;
	q[13]=0;
	q[12]=0;
	q[11]=0;
	q[10]=0;
	q[9]=0;
	q[8]=tq[15];
	q[7]=tq[14];
	q[6]=tq[13];
	q[5]=tq[12];
	q[4]=tq[11];
	q[3]=tq[10];
	q[2]=tq[9];
	q[1]=tq[8];
	q[0]=tq[7];

	//create rnew = q*c (already calculated and stored in tq) - qnew (just calculated, stored in q) <<224

	//tr=qnew<<224
	tr[0]=0;
	tr[1]=0;
	tr[2]=0;
	tr[3]=0;
	tr[4]=0;
	tr[5]=0;
	tr[6]=0;
	tr[7]=q[0];
	tr[8]=q[1];
	tr[9]=q[2];
	tr[10]=q[3];
	tr[11]=q[4];
	tr[12]=q[5];
	tr[13]=q[6];
	tr[14]=q[7];
	tr[15]=q[8];

	sub512(tq,tr);
	//now q*c - qnew<<224  is stored in tq

	//r=r+r1
	i++;
	add256(r,tq);

	}

	while(cmp512(r,gorder512)>=1){
		sub512(r,gorder512);
	}

	//result is in r[]
	//copyback to a,b
	for(int i=0;i<=7;i++){
		b[i]=r[i];
	}

	for(int i=0;i<=7;i++){
		a[i]=0;
	}


}


//Fast modulo reduction after addition
//overflow is small, so no need for modred_long
__device__ void modred_quick(uint32_t *a, uint32_t sign){
	uint32_t temp[8]={0,0,0,0,0,0,0,0};
	
	if(sign==1){
		while((!(cmp224(a,zero)==2))&&(cmp224(a,gorder)>=1)){
			sub224(a,gorder);
		}
		if(cmp224(a,zero)!=2){
	
			for(int i=0;i<=7;i++){
				temp[i]=gorder[i];
			}
			sub224(temp,a);
			for(int i=0;i<=7;i++){
				a[i]=temp[i];
			}
		}
		//sign=0;
		//not really needed
	}
}

//add two positive numbers, make sure result is modulo reduced. RESULT STORED in a!
__device__ void positive_add224(uint32_t* a,uint32_t* b){

			add224(a,b);
			//printf("a:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  \n",a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
			//printf("a:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  \n",a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
			//printf("GOrd:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  \n",gorder[7],gorder[6],gorder[5],gorder[4],gorder[3],gorder[2],gorder[1],gorder[0]);
			//if I exceed group order
			if(cmp224(a,gorder)>=1){
				sub224(a,gorder);
				//printf("MODRED1:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  \n",a.num[7],a.num[6],a.num[5],a.num[4],a.num[3],a.num[2],a.num[1],a.num[0]);
				//printf("Gord:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  \n",gorder[7],gorder[6],gorder[5],gorder[4],gorder[3],gorder[2],gorder[1],gorder[0]);
			}
			if(cmp224(a,gorder)>=1){
				sub224(a,gorder);
				//printf("MODRED2:  %08x  %08x  %08x  %08x  %08x  %08x  %08x  %08x  \n",a.num[7],a.num[6],a.num[5],a.num[4],a.num[3],a.num[2],a.num[1],a.num[0]);
			}
			return;

}

//multiply and then do modulo reduction. RESULT in b!
__device__ void positive_mult256(uint32_t* a,uint32_t* b){
	
	mult256(a,b);

	modred_long(a,b);
	
}

//add two numbers (+/-) - reduce first, add the positives after. RESULT in a!
__device__ void signed_add224(uint32_t* a,uint32_t* b, uint32_t asign, uint32_t bsign){
	
		if((asign==0)&&(bsign==0)){
			positive_add224(a,b);
			return;
		}
		if((asign==1)&&(bsign==1)){
			modred_quick(a,1);
			modred_quick(b,1);
			positive_add224(a,b);
			return;
		}
		if((asign==0)&&(bsign==1)){
			modred_quick(b,1);
			positive_add224(a,b);
			return;
		}
		if((asign==1)&&(bsign==0)){
			modred_quick(a,1);
			positive_add224(a,b);
			return;
		}
	
}


//REMENMBER TO update DAT!
__device__ void square256(uint32_t* a){
	
	
	uint32_t temp[8]={0,0,0,0,0,0,0,0};
	for(int i=0;i<=7;i++){
		temp[i]=a[i];
	}
	
	
	
	mult256(temp,a);

	modred_long(temp,a);

	
}


//Multiply by 2,4,8 ... No big need to implement it, since we hide it in parallelism :D */
__device__ void power2_mult(uint32_t* even, uint32_t* a){
}


#endif
