//============================================================================
// Name        : XXL-NS.cu
// Author      : BBR
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in CUDA
//============================================================================

/*
 * Copyright 2008, Karen Hains, UWA . All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws. Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * WE MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.
 */

/* HelloWorld Project
 * This project demonstrates the basics on how to setup
 * an example GPU Computing application.*
 * This file contains the CPU (host) code.
 */

// Host defines
#define NUM_THREADS 8
#define STR_SIZE 50

// Includes
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include "structures.h"
#include "/usr/local/cuda/include/curand.h"
#endif /* STRUCTURES_H_ */
#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "interfaces.h"
#endif /* INTERFACES_H_ */
#include "vars.h"
// GPU Kernels declarations

//#define _control_
//#define _printgraph_
#define __timing__
//#define __timing2__
#define __mobonly__

//////////////////////
// Program main
//////////////////////

/*************************/
/*** Random model ***/
/*************************/


extern Simulation_Parameters simulation_parameters;

int main(int argc, char** argv) {

	

	
	cudaEvent_t start, stop;
	curandGenerator_t gen;
	float elapsed, total_elapsed = 0.0, total_dest = 0.0, total_forwarded = 0.0;
	double elapsed_min = 10000000.0, elapsed_max = 0.0;
	float drop_probability;
	enum Initial_Distribution initial_distribution;
	struct Final_Data final;

	int N, nb_tours;
	struct Performances performance;
	int dwrite, nwrite;
	
	/* 
	 * This parameter is used to send different types of output (only numbers or with labels) according
	 * to whether we are going to use it to draw a graph (only numbers) or to show it in a gui (with labels)
	 * 0 = only numbers, 1 = with lebels 
	 */
	
	char output_format = 0;
	if (argc >= 2){
	   if (atoi(argv[1]) == 1)
		output_format = 1;
	}

	if (argc >= 3){
		dwrite = atoi(argv[2]);
	}

	/* Initialization of the random number generator */
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	checkCUDAError("cuda create random");

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*////////////////
	 float drop_probability = 0.7;
	 N = 16;
	 cudaEventRecord(start, 0);

	 performance = static_grid(N, 1, drop_probability, (void *) &gen);

	 cudaEventRecord(stop, 0);
	 cudaEventSynchronize(stop);
	 cudaEventElapsedTime(&elapsed, start, stop);


	 printf("* Messages received by %d : %d \n", N - 1, performance.total_Dest);


	 printf("* Number of packets generated : %d \n",performance.total_forwarded);

	 printf("* Time elapsed for N = %d is : %f\n", N, elapsed);
	 /////////////////
	 *
	 */

	//for (int p = 0; p < 11; p++) {
	

		/* Setting default values for the simulation parameters */
		Init_Simulation_Parameters();

		/* Getting back values that are necessary to launch the simulation */

		drop_probability = simulation_parameters.simulation_config.drop_probability;
		N = simulation_parameters.simulation_config.node_number;
		nb_tours = simulation_parameters.simulation_config.simulation_time;
		initial_distribution = simulation_parameters.topology_config.distribution.initial_distribution;
		
		performance.total_Dest = 0;

		//while (performance.total_Dest <= 600)

		//for (int H = 2; H < 280; (H < 65 ? (H < 20 ? H++ : H+= 2) : H += 10)) 
		{			
			//printf("\n\n* Probability threshold is : %f\n", drop_probability);
			//printf("%f ", drop_probability);


		    	if (output_format)			
			    //printf("* Node number: %d\n", N);
			    final.node_number = N;
		    	else
			    printf("%d ", N);

			total_elapsed = 0.0;
			total_dest = 0.0;
			elapsed_max = 0.0;
			elapsed_min = 10000000.0;
			total_forwarded = 0.0;

			for (int exec_nb = 0; exec_nb < 1; exec_nb++) {
				cudaEventRecord(start, 0);

				switch (initial_distribution) {
				
					case RANDOM_DISTRIBUTION:
				
					performance = random(N, 1, drop_probability,
							(curandGenerator_t *) &gen, (argc >= 3 ? dwrite : -3));

					break;

					case GRID:
				
					performance = static_grid(N, 1, drop_probability,
							(curandGenerator_t *) &gen, (argc >= 3 ? dwrite : -3));

					break;
				}

				cudaEventRecord(stop, 0);
				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&elapsed, start, stop);

				total_elapsed += elapsed;

				if (elapsed > elapsed_max) {
					elapsed_max = elapsed;
				}

				if (elapsed < elapsed_min) {
					elapsed_min = elapsed;
				}

				total_dest += (float) performance.total_Dest;

				total_forwarded += (float) performance.total_forwarded;
				

			}

			//total_elapsed = total_elapsed / 5.0;
			

			//total_dest = total_dest / 5.0;
		

			//total_forwarded = total_forwarded / 5.0;
			
			//memory = performance.used_memory;

			//printf("%d ", (int) total_dest);

		    	if (output_format)
			    //printf("* Loss rate: %f\n", 1.0 - (total_dest / (float) nb_tours));
			    final.loss = 1.0 - (total_dest / (float) nb_tours);
		    	else
			    printf("%f ", 1.0 - (total_dest / (float) nb_tours));
			

			//printf("* Number of packets generated : %d \n",(int) total_forwarded);
			
			if (output_format)			
			    //printf("* All forwarded packets: %d\n", (int)total_forwarded);
			    final.forwarded = (int)total_forwarded;
		    	else
			    printf("%d ", (int)total_forwarded);

			//printf("* Amount of memory we used : GPU = CPU = %d \n", memory);
			//But we don't need it now			
			
			if (output_format){			
			    /*printf("* Used memory [device, host]: [%d, %d]\n", performance.device_used_memory, 
				performance.host_used_memory);*/
			    final.device_memory = performance.device_used_memory;
			    final.host_memory = performance.host_used_memory;
		    	}else
	   		    printf("%d %d ", performance.device_used_memory, performance.host_used_memory);
	
			//printf("* Time elapsed for N = %d is : %f\n", N, total_elapsed);
			if (output_format)			
			    //printf("* Elapsed time (average): %f\n", total_elapsed);
			    final.average_time = total_elapsed;
		    	else
			    printf("%f ", total_elapsed);
			
			if (output_format)			
			    //printf("* Elapsed time (min): %f\n", elapsed_min);
			    final.min_time = elapsed_min;
		    	else
			    printf("%f ", elapsed_min);

			if (output_format)			
			    //printf("* Elapsed time (max): %f\n", elapsed_max);
			    final.max_time = elapsed_max;
		    	else
			    printf("%f\n", elapsed_max);
			
			if (argc >= 3){
				if( (nwrite = write( dwrite, &final, sizeof(struct Final_Data) )) == -1 )
            			    perror( "write main" );

				if (::close (dwrite) == -1 ) /* we close the read desc. */
	        			perror( "close on write" );
			}
		}
	    
	

	//Static_Network(argc, argv);
	//main2(argc, argv);
	//main3(argc, argv);
	//main4(argc, argv);
}

int main4(int argc, char** argv);
int main2(int argc, char** argv) {

 Simulation_Parameters *Device_simulation_parameters;
 Init_Simulation_Parameters();

 cudaMalloc((void**) &Device_simulation_parameters, sizeof(Simulation_Parameters));
	checkCUDAError("cudaDeviceSimParamMalloc");

 cudaMemcpy(Device_simulation_parameters, &simulation_parameters, sizeof(Simulation_Parameters),
	cudaMemcpyHostToDevice);

 // Host variables
 int i, j, k, l, /* nBytes, */ N, turn, tpb;
 //unsigned int num_threads;
 //char *cpu_odata;
 //char *string;
 //dim3 grid(1024,1,1);
 //dim3 threads(N/256,1,1);//  N/256 must be an int
 curandGenerator_t gen;
 cudaEvent_t start, stop, start2, stop2;
 //float time1 = 0.0;
 //float time2 = 0.0;
 float T[100][5][7];
 //float maxx[3], minn[0], avg[3];
 /*------------------------------------*/
 //
 struct Geo *Host_Geo, *Device_Geo;
 struct Geo2 *Host_Geo2, *Device_Geo2;
 //struct Phy *Host_Phy, *Device_Phy;
 struct Cell *Host_Cell, *Device_Cell;
 struct Node *Host_Node, *Device_Node;
 struct Buffer *Host_InPhy, *Host_OutPhy, *Device_InPhy, *Device_OutPhy;
 struct RouterBuffer *Host_RouterBuffer, *Device_RouterBuffer;
 float *Device_Randx, *Device_Randy, *Device_Randz, *Device_Randv,
 *Device_RouterProb, *Host_PosRandx, *Host_PosRandy, *Host_PosRandz,
 *Host_VRandx, *Host_VRandy, *Host_VRandz;
 /*------------------------------------*/

 curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
 checkCUDAError("cuda creator4");
 curandSetPseudoRandomGeneratorSeed( gen , time(NULL)) ;
 curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

 for (N = 1024 * 27; N <= (1024 * 32); N += 1024)
 for(turn=0;turn<8;turn++)
 {
 {
 printf("**************%d**************\n", N);
 for(tpb=32;tpb<=1024;tpb*=2)
 {
 tpb = 32;
 dim3 grid(tpb, 1, 1);

 //printf("**************%f**************\n",100*pow(2,turn-9));
 printf("<<<%d>>>\n", tpb);
 dim3 threads(N / tpb, 1, 1);//  N/256 must be an int


/*
 * Host memory allocation
 */


 //Host_Memory_Allocation(Host_Cell, Host_Geo, Host_Geo2, Host_PosRandx, Host_PosRandy, Host_PosRandz, Host_VRandx, Host_VRandy, Host_VRandz, Host_Node);

 ///Cell host memory allocation///
 cudaMallocHost((void**) &Host_Cell, B * sizeof(struct Cell));
 checkCUDAError("cudaCellMalloc");

 ///Geo host memory allocation///
 cudaMallocHost((void**) &Host_Geo, N * sizeof(struct Geo));
 checkCUDAError("cudaGeoMalloc");

 ///Geo2 host memory allocation///
 cudaMallocHost((void**) &Host_Geo2, N * sizeof(struct Geo2));
 checkCUDAError("cudaGeo2Malloc");

 /// Host Buffer allocation
 cudaMallocHost((void**) &Host_InPhy, N * sizeof(struct Buffer));
 checkCUDAError("cudaBufferInMalloc");
 cudaMallocHost((void**) &Host_OutPhy, N * sizeof(struct Buffer));
 checkCUDAError("cudaBufferOut2Malloc");

 ///Random position and speed parameters (needed for Geo) host memory allocation///
 cudaMallocHost((void **) &Host_PosRandx, N * sizeof(float));
 checkCUDAError("cudaPosRandxMalloc");
 cudaMallocHost((void **) &Host_PosRandy, N * sizeof(float));
 checkCUDAError("cudaPosRandyMalloc");
 cudaMallocHost((void **) &Host_PosRandz, N * sizeof(float));
 checkCUDAError("cudaPosRandzMalloc");
 cudaMallocHost((void **) &Host_VRandx, N * sizeof(float));
 checkCUDAError("cudaVRandxMalloc");
 cudaMallocHost((void **) &Host_VRandy, N * sizeof(float));
 checkCUDAError("cudaVRandyMalloc");
 cudaMallocHost((void **) &Host_VRandz, N * sizeof(float));
 checkCUDAError("cudaVRandzMalloc");

 ///Node host memory allocation///
 cudaMallocHost((void**) &Host_Node, N * sizeof(struct Node));
 checkCUDAError("cudaNodeMalloc");

 ///Host Router Buffer Allocation///
 cudaMallocHost((void **) &Host_RouterBuffer,
 N * sizeof(struct RouterBuffer));
 checkCUDAError("cudaRouterBufferMalloc");

/*
 * Host initialization and control
 */

 Init_Host(Host_Cell, Host_Geo, Host_Geo2, Host_PosRandx, Host_PosRandy,
 Host_PosRandz, Host_VRandx, Host_VRandy, Host_VRandz,
 Host_Node, Host_RouterBuffer, N);
 //Init_Control(Host_Cell, Host_Geo, Host_Geo2, Host_Node);
 /*
  * Device memory allocation
  */

 ///Cell Device memory allocation///
 cudaMalloc((void**) &Device_Cell, B * sizeof(struct Cell));
 checkCUDAError("cudaDeviceCellMalloc");

 ///Geo Device memory allocation///
 cudaMalloc((void**) &Device_Geo, N * sizeof(struct Geo));
 checkCUDAError("cudaDeviceGeoMalloc");

 ///Geo2 Device memory allocation///
 cudaMalloc((void**) &Device_Geo2, N * sizeof(struct Geo2));
 checkCUDAError("cudaDeviceGeo2Malloc");

 ///Buffer Device memory allocation///
 cudaMalloc((void**) &Device_InPhy, N * sizeof(struct Buffer));
 checkCUDAError("cudaDeviceBufferINMalloc");
 printf("adress of device_inphy :%p size of buffer %d\n", &Device_InPhy,
 sizeof(struct Buffer));
 cudaMalloc((void**) &Device_OutPhy, N * sizeof(struct Buffer));
 checkCUDAError("cudaDeviceBufferOutMalloc");

 ///Random speed parameters (needed for Mobility) Device memory allocation///
 cudaMalloc((void **) &Device_Randx, N * sizeof(float));
 checkCUDAError("cudaDeviceRandxMalloc");
 cudaMalloc((void **) &Device_Randy, N * sizeof(float));
 checkCUDAError("cudaDeviceRandyMalloc");
 cudaMalloc((void **) &Device_Randz, N * sizeof(float));
 checkCUDAError("cudaDeviceRandzMalloc");
 cudaMalloc((void **) &Device_Randv, N * sizeof(float));
 checkCUDAError("cudaDeviceRandvMalloc");

 ///Node Device memory allocation///
 cudaMalloc((void**) &Device_Node, N * sizeof(struct Node));
 checkCUDAError("cudaMalloc");

 ///Device Router Buffer Allocation///
 cudaMalloc((void **) &Device_RouterBuffer,
 N * sizeof(struct RouterBuffer));
 checkCUDAError("cudaDeviceRouterBufferMalloc");

 ///Device Router Transmission Probability Allocation///
 cudaMalloc((void **) &Device_RouterProb, N * sizeof(float));
 checkCUDAError("cudaDeviceBufferProbMalloc");

 /*
  * Device initialization
  */
 Init_Device(Host_Cell, Host_Geo, Host_Geo2, Host_Node,
 Host_RouterBuffer, Device_Cell, Device_Geo, Device_Geo2,
 Device_Node, Device_RouterBuffer, N);

 Init_buffer<<<threads,grid>>>(Device_OutPhy, N);
 Init_buffer<<<threads,grid>>>(Device_InPhy, N);

 #ifdef _control_
 printf("first controle\n");
 cudaMemcpy(Host_OutPhy,Device_OutPhy, N* sizeof(struct Buffer) , cudaMemcpyDeviceToHost);
 cudaMemcpy(Host_InPhy,Device_InPhy, N* sizeof(struct Buffer) , cudaMemcpyDeviceToHost);
 #endif
 //Initialisation du generateur de nombres aleatoires
 checkCUDAError("cuda creator5");
 checkCUDAError("cuda creator3");
 //Printposition(Host_Geo);
 cudaEventCreate(&start);
 cudaEventCreate(&stop);
 cudaEventCreate(&start2);
 cudaEventCreate(&stop2);
 cudaEventRecord(start2, 0);
 checkCUDAError("cuda creator2");
 creator<<<threads,grid>>>(Device_OutPhy,1,1);
 intirouter<<<threads,grid>>>(Device_RouterBuffer,N);
 //creator<<<1,1>>>(Device_InPhy);
 checkCUDAError("cuda creator");
 #ifdef _control_
 printf("second controle\n");
 cudaMemcpy(Host_OutPhy,Device_OutPhy, N* sizeof(struct Buffer) , cudaMemcpyDeviceToHost);
 #endif

 for(i=0;i<10;i++)
 {
 j=0;
 creator<<<threads,grid>>>(Device_OutPhy,1,3);
 #ifdef __timing__
 cudaEventRecord(start2, 0);
 #endif
 curandGenerateUniform ( gen , Device_Randx , N );
 curandGenerateUniform ( gen , Device_Randy , N );
 curandGenerateUniform ( gen , Device_Randz , N );
 curandGenerateUniform ( gen , Device_Randv , N );
 #ifdef __timing__
 cudaEventRecord(stop2, 0);
 cudaEventSynchronize(stop2);
 cudaEventElapsedTime(&T[i][j][0], start2, stop2);
 cudaEventRecord(start2, 0);
 #endif

 Mobility<<<threads,grid>>>(Device_Cell,Device_Geo,Device_Randx,Device_Randy,Device_Randz,Device_Randv,N,&simulation_parameters);
 #ifdef __timing__
 cudaEventRecord(stop2, 0);
 cudaEventSynchronize(stop2);
 cudaEventElapsedTime(&T[i][j][1], start2, stop2);
 cudaEventRecord(start2, 0);
 #endif
 checkCUDAError("cuda kernel error mob");
 Mobility_Control<<<threads,grid>>>(Device_Cell,Device_Geo,N,Device_simulation_parameters);
 checkCUDAError("cuda kernel error controle2");
 updateCell<<<32,(B/32)>>>(Visibility,Device_Cell, N);
 checkCUDAError("cuda kernel error controle3");
 #ifdef __timing__
 cudaEventRecord(stop2, 0);
 cudaEventSynchronize(stop2);
 cudaEventElapsedTime(&T[i][j][2], start2, stop2);
 cudaEventRecord(start2, 0);
 #endif
 Visible<<<threads,grid>>>(Device_Geo, Device_Cell,Device_Geo2);
 checkCUDAError("cuda kernel error controle4");
 #ifdef __timing__
 cudaEventRecord(stop2, 0);
 cudaEventSynchronize(stop2);
 cudaEventElapsedTime(&T[i][j][3], start2, stop2);
 cudaEventRecord(start2, 0);
 #endif
 #ifdef _control_
 printf("3td controle\n");
 cudaMemcpy(Host_Geo,Device_Geo, N * sizeof(struct Geo) , cudaMemcpyDeviceToHost);
 checkCUDAError("cuda kernel1234");
 Connectivity_Control(Host_Geo,N);
 checkCUDAError("cuda kernel123");
 #endif
 cudaThreadSynchronize();
 checkCUDAError("cuda kernel12");
 creator<<<threads,grid>>>(Device_OutPhy,i,i);
 checkCUDAError("cuda creator2");
 for(j=0;j<5;j++) //slot
 {
 cudaMemcpy(Host_Geo,Device_Geo, N* sizeof(struct Geo) , cudaMemcpyDeviceToHost);
 cudaMemcpy(Host_OutPhy,Device_OutPhy, N* sizeof(struct Buffer) , cudaMemcpyDeviceToHost);
 printpropagation(Host_Geo,Host_OutPhy,i,j,N);
 #ifdef _printgraph_
 printf("4td controle\n");
 printpropagation(Host_Geo,Host_OutPhy,i,j,N);
 checkCUDAError("cuda kernel5");
 #endif

/* Don't want to compile !!!! 
 Sender<<<threads,grid>>>(Device_Geo,Device_InPhy,Device_OutPhy);
*/

 #ifdef __timing__
 cudaEventRecord(stop2, 0);
 cudaEventSynchronize(stop2);
 cudaEventElapsedTime(&T[i][j][4], start2, stop2);
 cudaEventRecord(start2, 0);
 #endif
 checkCUDAError("cuda kernel error Sender");
 Receiver<<<threads,grid>>>(Device_InPhy, N);
 #ifdef __timing__
 cudaEventRecord(stop2, 0);
 cudaEventSynchronize(stop2);
 cudaEventElapsedTime(&T[i][j][5], start2, stop2);
 cudaEventRecord(start2, 0);
 #endif
 checkCUDAError("cuda kernel error Receiver");
 //simple_router<<<threads,grid>>>(Device_InPhy,Device_OutPhy);
 curandGenerateUniform ( gen , Device_RouterProb , N );
 router<<<threads,grid>>>(Device_InPhy,Device_OutPhy,Device_RouterBuffer,Device_RouterProb);
 checkCUDAError("cuda kernel error Router");
 #ifdef __timing__
 cudaEventRecord(stop2, 0);
 cudaEventSynchronize(stop2);
 cudaEventElapsedTime(&T[i][j][6], start2, stop2);
 cudaEventRecord(start2, 0);
 #endif
 //cleaner<<<threads,grid>>>(Device_OutPhy,Device_Randx);
 //cleaner<<<threads,grid>>>(Device_InPhy,Device_Randx);
 //ms
 //for K;10 //subslot
 #ifdef _printgraph_
 printf("5td controle\n");
 cudaMemcpy(Host_OutPhy,Device_OutPhy, N* sizeof(struct Buffer) , cudaMemcpyDeviceToHost);
 cudaMemcpy(Host_InPhy,Device_InPhy, N* sizeof(struct Buffer) , cudaMemcpyDeviceToHost);
 cudaMemcpy(Host_RouterBuffer,Device_RouterBuffer, N* sizeof(struct RouterBuffer) , cudaMemcpyDeviceToHost);

 //Buffer_Control(Host_InPhy, Host_OutPhy, N);
 //output
 printf("5.5td controle\n");
 for(k=0;k<N;k++)
 {
 if(Host_RouterBuffer[k].HeaderArray[0][3]!=0)
 {
 printf("Node %d Message seen %d\n\r",k,Host_RouterBuffer[k].HeaderArray[0][3]);
 printf(" [OUT %d %d  IN %d %d ]\n",Host_OutPhy[k].readIndex,Host_OutPhy[k].writeIndex,Host_InPhy[k].readIndex,Host_InPhy[k].writeIndex);

 }
 }

 //Printconnectivity(Host_Geo);
 //printpropagation(Host_Geo,Host_OutPhy,i,j);
 printf("round *********%d %d*****\n\r",i,j);
 for(k=0;k<N;k++)
 {
 if(Host_InPhy[k].writeIndex>0)
 {
 printf(" [OUT %d %d  IN %d %d ]",Host_OutPhy[k].readIndex,Host_OutPhy[k].writeIndex,Host_InPhy[k].readIndex,Host_InPhy[k].writeIndex);
 printf("node %d\n",k);
 for(int m=0;m<Host_InPhy[k].writeIndex;m++)
 {
 for(l=0;l<4;l++)
 printf(" [%d]",Host_InPhy[k].Element[m].Header[l]);
 printf("\n");
 }

 for(l=0;l<4;l++)
 printf(" [%d-%d]",Host_OutPhy[k].Element[0].Header[l],Host_InPhy[k].Element[0].Header[l]);
 printf("\n\r");
 }
 }
 #endif
 }

 cudaMemcpy(Host_RouterBuffer,Device_RouterBuffer, N * sizeof(struct RouterBuffer) , cudaMemcpyDeviceToHost);
 int m0=0,m1=0,m2=0,m3=0,m4=0;
 for(k=0;k<N;k++)
 {
 switch(Host_RouterBuffer[k].writeIndex)
 {
 case 0:
 m0++;
 break;
 case 1:
 m1++;
 break;
 case 2:
 m2++;
 break;
 case 3:
 m3++;
 break;
 case 4:
 m4++;
 break;
 }
 }
 printf("Messages seen: 0-%d 1-%d 2-%d 3-%d 4-%d\n",m0, m1, m2, m3, m4);
 for(k=0;k<N;k++)
 {
 printf("%d [",k);
 for(l=0;l<Host_Geo[k].Neighbors;l++)
 printf("  %d ",Host_Geo[k].Neighbor[l]);
 printf(" ]\n");
 }

 #ifdef _printgraph_
 printf("6td controle\n");
 cudaMemcpy(Host_Geo,Device_Geo, N * sizeof(struct Geo) , cudaMemcpyDeviceToHost);
 cudaMemcpy(Host_Geo2,Device_Geo2, N * sizeof(struct Geo2) , cudaMemcpyDeviceToHost);
 cudaMemcpy(Host_Cell,Device_Cell, B * sizeof(struct Cell) , cudaMemcpyDeviceToHost);
 cudaMemcpy(Host_InPhy,Device_InPhy, N* sizeof(struct Buffer) , cudaMemcpyDeviceToHost);
 checkCUDAError("cuda kernel7");
 #endif
// Momentqnement!!!! Connectivity_Control(Host_Geo,N); 

 for(int k=0;k<B;k++)
 printf("Cell:%d  size=%d, member 0= %d member 1 =%d \n\r",k,Host_Cell[k].size,Host_Cell[k].member[0],Host_Cell[k].member[1]);
 checkCUDAError("cudaMemcpy4");

 //Printposition(Host_Geo);
 //printf("%d\n",i);
 //Printmobility(Host_Geo,100, i,1000);
 //Printtrajictoire(Host_Geo,i, 1000);
 }
/*
 for(i=0;i<N;i++)
 {
 if(Host_InPhy[i].BufferIndex>(-1))
 printf("In index of %d =%d \r\n",i,Host_InPhy[i].BufferIndex);
 }
*/
 printf("process\n");

	k=0;
	 for(i=0;i<B;i++)
	 {
	 if(Host_Cell[i].size==0)
	 {
	 printf("Cell %d %d is empty\n",i,Host_Cell[i].size);
	 k+=Host_Cell[i].size;
	 }
	 }
	 printf("Total Nodes %d\n",k);
		for(i=0;i<N;i++)
	 {
	 //printf("Node %d %d\n",i,Host_Geo[i].MobMode);
	 for(j=0;j<Host_Geo[i].MobMode;j++)
	 if(Host_Geo2[i].Distance[j]<=100)printf(" node %d  --node%d = %f\n\r",i,Host_Geo2[i].NodeId[j],Host_Geo2[i].Distance[j]);
	 }
	for(i=0;i<N;i++)
	 {
	 //printf("Node %d %d\n",i,Host_Geo[i].MobMode);
	 printf("%d [",i);
	 for(j=0;j<Host_Geo[i].Neighbors;j++)
	 printf("  %d ",Host_Geo[i].Neighbor[j]);
	 printf(" ]\n\r");
	 }
	 printf("Done!\n");
	 i=0;
	 k=0;
	 for(j=0;j<N;j++)
	 {
	 if(Host_Geo[j].Neighbors==0)
	 {
	 printf("No Neighbors:%d\n",j);
	 i++;
	 }
	 k+=Host_Geo[j].Neighbors;
	 }
	 printf("Done! Total: %d\n",i);
	 printf("Average: %f\n",k/float(N));

	 //Free_Device(Device_Geo, Device_Geo2, Device_Cell, Device_Node, Device_InPhy, Device_OutPhy, Device_RouterBuffer, Device_Randx, Device_Randy, Device_Randz);
	 cudaFree(Device_Geo);
	 cudaFree(Device_Geo2);
	 cudaFree(Device_Cell);
	 cudaFree(Device_Node);
	 cudaFree(Device_InPhy);
	 cudaFree(Device_OutPhy);
	 cudaFree(Device_RouterBuffer);
	 cudaFree(Device_Randx);
	 cudaFree(Device_Randy);
	 cudaFree(Device_Randz);
	 cudaFree(Device_simulation_parameters);
	 checkCUDAError("cudaFreeDevice");
	 cudaFreeHost(Host_Geo);
	 cudaFreeHost(Host_Geo2);
	 cudaFreeHost(Host_Cell);
	 cudaFreeHost(Host_Node);
	 cudaFreeHost(Host_InPhy);
	 cudaFreeHost(Host_OutPhy);
	 cudaFreeHost(Host_RouterBuffer);
	 cudaFreeHost(Host_PosRandx);
	 cudaFreeHost(Host_PosRandy);
	 cudaFreeHost(Host_PosRandz);
	 cudaFreeHost(Host_VRandx);
	 cudaFreeHost(Host_VRandy);
	 cudaFreeHost(Host_VRandz);
	 //Free_Host(Host_Geo,Host_Geo2,Host_Cell,Host_Node,Host_InPhy,Host_OutPhy,Host_RouterBuffer,Host_PosRandx,Host_PosRandy,Host_PosRandz,Host_VRandx,Host_VRandy,Host_VRandz);
	 checkCUDAError("cudaFreeHost");
	 #ifdef __timing2__
	 cudaEventRecord(stop2, 0);
	 cudaEventSynchronize(stop2);
	 cudaEventElapsedTime(&time2, start2, stop2);
	 printf("total time for %d Nodes during %d mobility and %d slots is %f \n\r",N,i,j,time2/10);
	 //printf("asdfasdf\n");
	 //Printconnectivity(Host_Geo);
	 //	Printposition(Host_Geo);
	 #endif

	 checkCUDAError("cudaMemcpy4");
	 #ifdef __timing__
	 for(i=0;i<10;i++)
	 {
	 //printf("%d =>",i);
	 for(j=0;j<5;j++)
	 {
	 //	printf("=>%d",j);
	 for(k=0;k<7;k++)
	 {
	 // if((i==0)||((i>0)&&(j>3)))printf("[%f]",T[i][j][k]);
	 if((j==0)||((k>3)))printf("%f|",T[i][j][k]); else {printf("%f|",T[i][0][k]);}
	 }
	 printf("\n");
	 }
	 //printf("\n\r");
	 }
	 #endif
	 }
	 }
	 }
return 0;
}
void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}

/**
 * \dot
 * digraph mainarch  {
 * 		rankdir=LR;
 * 		main [shape=rect,style=filled,fillcolor=lightgrey];
 *		node [shape=rect];
 *		main->Init_Host;
 *		Init_Host->Init_Cell;
 *		Init_Host->Init_Geo;
 *		Init_Host->Init_Node;
 *
 * }
 * \enddot
 **/

int main3(int argc, char** argv) {

	Simulation_Parameters *Device_simulation_parameters;
 	Init_Simulation_Parameters();

 	cudaMalloc((void**) &Device_simulation_parameters, sizeof(Simulation_Parameters));
		checkCUDAError("cudaDeviceSimParamMalloc");

 	cudaMemcpy(Device_simulation_parameters, &simulation_parameters, sizeof(Simulation_Parameters),
		cudaMemcpyHostToDevice);

	// Host variables
	int i, j, /* k, l,  nBytes, turn,*/ N, tpb;
	//unsigned int num_threads;
	//char *cpu_odata;
	//char *string;
	//dim3 grid(1024,1,1);
	//dim3 threads(N/256,1,1);//  N/256 must be an int
	curandGenerator_t gen;
	cudaEvent_t start, stop, start2, stop2;
	//float time1 = 0.0;
	//float time2 = 0.0;
	float T[1000][5][7];
	float /*maxx, minn,*/ avg[7], avrage[500][7];
	/*------------------------------------*/
	//
	struct Geo *Host_Geo, *Device_Geo;
	struct Geo2 *Host_Geo2, *Device_Geo2;
	//struct Phy *Host_Phy, *Device_Phy;
	struct Cell *Host_Cell, *Device_Cell;
	struct Node *Host_Node, *Device_Node;
	//struct Buffer *Host_InPhy, *Host_OutPhy, *Device_InPhy, *Device_OutPhy;
	struct RouterBuffer *Host_RouterBuffer, *Device_RouterBuffer;
	float *Device_Randx, *Device_Randy, *Device_Randz, *Device_Randv,
			/* *Device_RouterProb,*/ *Host_PosRandx, *Host_PosRandy, *Host_PosRandz,
			*Host_VRandx, *Host_VRandy, *Host_VRandz;
	/*------------------------------------*/

	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	checkCUDAError("cuda creator4");
	//curandSetPseudoRandomGeneratorSeed( gen , time(NULL)) ;
	curandSetPseudoRandomGeneratorSeed(gen, time(NULL));

	for (N = 1024; N <= (1024 * 64); N += 1024)
	//for(turn=0;turn<8;turn++)
	{
		//N=1024*1;
		//printf("**************%d**************\n",N);
		//for(tpb=32;tpb<=1024;tpb*=2)
		//{
		tpb = 128;
		dim3 grid(tpb, 1, 1);
		//N=1024*1;
		//{
		//printf("**************%f**************\n",100*pow(2,turn-9));
		//printf("<<<%d>>>\n",tpb);
		dim3 threads(N / tpb, 1, 1);//  N/256 must be an int
		//Host_Memory_Allocation(Host_Cell, Host_Geo, Host_Geo2, Host_PosRandx, Host_PosRandy, Host_PosRandz, Host_VRandx, Host_VRandy, Host_VRandz, Host_Node);

		///Cell host memory allocation///
		cudaMallocHost((void**) &Host_Cell, B * sizeof(struct Cell));
		checkCUDAError("cudaCellMalloc");

		///Geo host memory allocation///
		cudaMallocHost((void**) &Host_Geo, N * sizeof(struct Geo));
		checkCUDAError("cudaGeoMalloc");
		///Geo2 host memory allocation///
		cudaMallocHost((void**) &Host_Geo2, N * sizeof(struct Geo2));
		checkCUDAError("cudaGeo2Malloc");
		///Random position and speed parameters (needed for Geo) host memory allocation///
		cudaMallocHost((void **) &Host_PosRandx, N * sizeof(float));
		checkCUDAError("cudaPosRandxMalloc");
		cudaMallocHost((void **) &Host_PosRandy, N * sizeof(float));
		checkCUDAError("cudaPosRandyMalloc");
		cudaMallocHost((void **) &Host_PosRandz, N * sizeof(float));
		checkCUDAError("cudaPosRandzMalloc");
		cudaMallocHost((void **) &Host_VRandx, N * sizeof(float));
		checkCUDAError("cudaVRandxMalloc");
		cudaMallocHost((void **) &Host_VRandy, N * sizeof(float));
		checkCUDAError("cudaVRandyMalloc");
		cudaMallocHost((void **) &Host_VRandz, N * sizeof(float));
		checkCUDAError("cudaVRandzMalloc");
		///Node host memory allocation///
		cudaMallocHost((void **) &Host_RouterBuffer,
				1 * sizeof(struct RouterBuffer));
		checkCUDAError("cudaRouterBufferMalloc");
		cudaMallocHost((void**) &Host_Node, N * sizeof(struct Node));
		checkCUDAError("cudaNodeMalloc");
		Init_Host(Host_Cell, Host_Geo, Host_Geo2, Host_PosRandx, Host_PosRandy,
				Host_PosRandz, Host_VRandx, Host_VRandy, Host_VRandz,
				Host_Node, Host_RouterBuffer, N);
		//Init_Control(Host_Cell, Host_Geo, Host_Geo2, Host_Node);
		/*for(int k=0;k<B;k++)printf("Cell %d[ %d %d %d]\r",k,Host_Cell[k].Center.x,Host_Cell[k].Center.y,Host_Cell[k].Center.z);
		 int num=0;
		 for(int k=0;k<B;k++)
		 {
		 int s=Host_Cell[k].size;

		 if(s!=0)
		 {
		 num+=1;
		 printf("Cell %d, %d nodes\n",k,s);
		 }


		 for(int l=0;l<s;l++)
		 {
		 printf("Node %d ",Host_Cell[k].member[l]);
		 }
		 printf("\n");
		 }
		 printf("CellsWithNodes %d\n",num);*/
		/*for(int k=0;k<N;k++)
		 {
		 printf("%d %d %d\n",Host_Geo[k].P.x,Host_Geo[k].P.y,Host_Geo[k].P.z);
		 }
		 printf("\n");*/
		/*
		 * Device memory allocation
		 */

		///Cell Device memory allocation///
		cudaMalloc((void**) &Device_Cell, B * sizeof(struct Cell));
		checkCUDAError("cudaDeviceCellMalloc");

		///Geo Device memory allocation///
		cudaMalloc((void**) &Device_Geo, N * sizeof(struct Geo));
		checkCUDAError("cudaDeviceGeoMalloc");

		///Geo2 Device memory allocation///
		cudaMalloc((void**) &Device_Geo2, N * sizeof(struct Geo2));
		checkCUDAError("cudaDeviceGeo2Malloc");

		///Random speed parameters (needed for Mobility) Device memory allocation///
		cudaMalloc((void **) &Device_Randx, N * sizeof(float));
		checkCUDAError("cudaDeviceRandxMalloc");
		cudaMalloc((void **) &Device_Randy, N * sizeof(float));
		checkCUDAError("cudaDeviceRandyMalloc");
		cudaMalloc((void **) &Device_Randz, N * sizeof(float));
		checkCUDAError("cudaDeviceRandzMalloc");
		cudaMalloc((void **) &Device_Randv, N * sizeof(float));
		checkCUDAError("cudaDeviceRandzMalloc");

		///Device Router Transmission Probability Allocation///


		///Node Device memory allocation///
		cudaMalloc((void**) &Device_Node, N * sizeof(struct Node));
		checkCUDAError("cudaMalloc");
		/*
		 * Device initialization
		 */
		Init_Device(Host_Cell, Host_Geo, Host_Geo2, Host_Node,
				Host_RouterBuffer, Device_Cell, Device_Geo, Device_Geo2,
				Device_Node, Device_RouterBuffer, N);
		//Initialisation du generateur de nombres aleatoires
		checkCUDAError("cuda creator5");
		checkCUDAError("cuda creator3");
		//Printposition(Host_Geo);
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventCreate(&start2);
		cudaEventCreate(&stop2);
		cudaEventRecord(start2, 0);
		checkCUDAError("cuda creator2");
//		int tempFullNode = 0;
//		float nodeConn = 0;
		
		for (i = 0; i < 10; i++) {
			j = 0;
			//creator<<<threads,grid>>>(Device_OutPhy,1,3);
#ifdef __timing__
			cudaEventRecord(start2, 0);
#endif
			curandGenerateUniform(gen, Device_Randx, N);
			curandGenerateUniform(gen, Device_Randy, N);
			curandGenerateUniform(gen, Device_Randz, N);
			curandGenerateUniform(gen, Device_Randv, N);
#ifdef __timing__
			cudaEventRecord(stop2, 0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&T[i][j][0], start2, stop2);
			cudaEventRecord(start2, 0);
#endif
			Mobility<<<threads,grid>>>(Device_Cell,Device_Geo,Device_Randx,Device_Randy,Device_Randz,Device_Randv,N,&simulation_parameters);
#ifdef __timing__
			cudaEventRecord(stop2, 0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&T[i][j][1], start2, stop2);
			cudaEventRecord(start2, 0);
#endif
			checkCUDAError("cuda kernel error mob");
			Mobility_Control<<<threads,grid>>>(Device_Cell,Device_Geo,N,Device_simulation_parameters);
#ifdef __timing__
			cudaEventRecord(stop2, 0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&T[i][j][2], start2, stop2);
			cudaEventRecord(start2, 0);
#endif
			checkCUDAError("cuda kernel error controle2");
			updateCell<<<32,(B/32)>>>(Visibility,Device_Cell, N);
			checkCUDAError("cuda kernel error controle3");
#ifdef __timing__
			cudaEventRecord(stop2, 0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&T[i][j][3], start2, stop2);
			cudaEventRecord(start2, 0);
#endif
			Visible<<<threads,grid>>>(Device_Geo, Device_Cell,Device_Geo2);
			checkCUDAError("cuda kernel error controle4");
#ifdef __timing__
			cudaEventRecord(stop2, 0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&T[i][j][4], start2, stop2);
			cudaEventRecord(start2, 0);
#endif

			//cudaThreadSynchronize();
			checkCUDAError("cuda kernel12");

			//Cell occupancy test
			/*cudaMemcpy(Host_Geo,Device_Geo,  N * sizeof(struct Geo) , cudaMemcpyDeviceToHost) ;
			 cudaMemcpy(Host_Cell,Device_Cell,  B * sizeof(struct Cell) , cudaMemcpyDeviceToHost) ;
			 for(int k=0;k<N;k++)
			 {
			 tempFullNode+=(int)(Host_Geo[k].Neighbors==Maxneighbor);
			 nodeConn+=Host_Geo[k].Neighbors;
			 }*/
			/*num=0;
			 for(int k=0;k<B;k++)
			 {
			 int s=Host_Cell[k].size;

			 if(s!=0)
			 {
			 num+=1;
			 printf("Cell %d, %d nodes [%d %d %d]\n",k,s,Host_Cell[k].Center.x,Host_Cell[k].Center.y,Host_Cell[k].Center.z);
			 }

			 for(int l=0;l<s;l++)
			 {
			 if(abs(Host_Geo[Host_Cell[k].member[l]].P.x-Host_Cell[k].Center.x)>100||abs(Host_Geo[Host_Cell[k].member[l]].P.y-Host_Cell[k].Center.y)>100||abs(Host_Geo[Host_Cell[k].member[l]].P.z-Host_Cell[k].Center.z)>100)
			 printf("Node %d [%d %d %d]",Host_Cell[k].member[l],Host_Geo[Host_Cell[k].member[l]].P.x,Host_Geo[Host_Cell[k].member[l]].P.y,Host_Geo[Host_Cell[k].member[l]].P.z);
			 }
			 printf("\n");
			 }
			 printf("CellsWithNodes %d\n",num);*/
		}

		/*cudaMemcpy(Host_Geo,Device_Geo,  N * sizeof(struct Geo) , cudaMemcpyDeviceToHost) ;
		 cudaMemcpy(Host_Cell,Device_Cell,  B * sizeof(struct Cell) , cudaMemcpyDeviceToHost) ;
		 for(int k=0;k<N;k++)
		 {
		 tempFullNode+=(int)(Host_Geo[k].Neighbors==Maxneighbor);
		 nodeConn+=Host_Geo[k].Neighbors;
		 }
		 num=0;
		 for(int k=0;k<B;k++)
		 {
		 int s=Host_Cell[k].size;

		 if(s!=0)
		 {
		 num+=1;
		 printf("Cell %d, %d nodes [%d %d %d]\n",k,s,Host_Cell[k].Center.x,Host_Cell[k].Center.y,Host_Cell[k].Center.z);
		 }

		 for(int l=0;l<s;l++)
		 {
		 if(abs(Host_Geo[Host_Cell[k].member[l]].P.x-Host_Cell[k].Center.x)>100||abs(Host_Geo[Host_Cell[k].member[l]].P.y-Host_Cell[k].Center.y)>100||abs(Host_Geo[Host_Cell[k].member[l]].P.z-Host_Cell[k].Center.z)>100)
		 printf("Node %d [%d %d %d]",Host_Cell[k].member[l],Host_Geo[Host_Cell[k].member[l]].P.x,Host_Geo[Host_Cell[k].member[l]].P.y,Host_Geo[Host_Cell[k].member[l]].P.z);
		 }
		 printf("\n");
		 }
		 printf("CellsWithNodes %d\n",num);*/
		cudaMemcpy(Host_Geo,Device_Geo, N * sizeof(struct Geo) , cudaMemcpyDeviceToHost);
		/***
		 for(int k=0;k<N;k++)
		 {
		 printf("%d %d %d\n",Host_Geo[k].P.x,Host_Geo[k].P.y,Host_Geo[k].P.z);
		 }

		 */

		/*float tempNodeComp=0;
		 float tempCellComp=0;
		 for(int k=0;k<N;k++)
		 {
		 tempCellComp+=Host_Geo[k].Viewed;
		 tempNodeComp+=Host_Geo[k].MobMode;
		 }
		 printf("NodeCompByNode : %f\n",tempNodeComp/(10*N));
		 printf("CellCompByNode : %f\n",tempCellComp/(10*N));
		 printf("Nodes by Cell : %d\n",N/(B));
		 printf("FullNodes : %d\n",tempFullNode/10);
		 printf("NodeConn : %f\n",nodeConn/(10*N));*/

		cudaFree(Device_Geo);
		cudaFree(Device_Geo2);
		cudaFree(Device_Cell);
		cudaFree(Device_Node);
		cudaFree(Device_Randx);
		cudaFree(Device_Randy);
		cudaFree(Device_Randz);
		cudaFree(Device_simulation_parameters);
		checkCUDAError("cudaFreeDevice");
		cudaFreeHost(Host_Geo);
		cudaFreeHost(Host_Geo2);
		cudaFreeHost(Host_Cell);
		cudaFreeHost(Host_Node);
		cudaFreeHost(Host_PosRandx);
		cudaFreeHost(Host_PosRandy);
		cudaFreeHost(Host_PosRandz);
		cudaFreeHost(Host_VRandx);
		cudaFreeHost(Host_VRandy);
		cudaFreeHost(Host_VRandz);
		//Free_Host(Host_Geo,Host_Geo2,Host_Cell,Host_Node,Host_InPhy,Host_OutPhy,Host_RouterBuffer,Host_PosRandx,Host_PosRandy,Host_PosRandz,Host_VRandx,Host_VRandy,Host_VRandz);
		checkCUDAError("cudaFreeHost");
#ifdef __timing2__
		cudaEventRecord(stop2, 0);
		cudaEventSynchronize(stop2);
		cudaEventElapsedTime(&time2, start2, stop2);
		//printf("total time for %d Nodes during %d mobility and %d slots is %f \n\r",N,i,j,time2/10);
		//printf("Total time : %f\n",time2);
		//printf("asdfasdf\n");
		//Printconnectivity(Host_Geo);
		//	Printposition(Host_Geo);
#endif
		printf("%d\n",N);
		checkCUDAError("cudaMemcpy4");
#ifdef __timing__
		for(j=0;j<5;j++)
		{
			avg[j]=0;
		}
		for(i=0;i<10;i++)
		{
			for(j=0;j<5;j++)
			{
				avg[j]+=T[i][0][j];
			}
		}
		for(j=0;j<5;j++)
		{
			avg[j]=avg[j]/10;
		}
		for(j=0;j<5;j++) {avrage[(N/256)][j]=avg[j];}
		//printf("%d %f %f %f %f\n",N,avg[0],avg[1],avg[2],avg[3]);
#endif
#ifdef __timing22__
		for(i=0;i<10;i++)
		{
			//printf("%d =>",i);
			for(j=0;j<5;j++)
			{
				//	printf("=>%d",j);
				for(k=0;k<7;k++)
				{
					// if((i==0)||((i>0)&&(j>3)))printf("[%f]",T[i][j][k]);
					if((j==0)||((k>3)))printf("%f|",T[i][j][k]); else {printf("%f|",T[i][0][k]);}
				}
				printf("\n");
			}
			//printf("\n\r");
		}
#endif
		//printf(" %d %f \n",N,(avrage[N/256][4]));
	}
	/*for(j=0;j<4;j++)
	 {
	 printf("*** %d***\n",j);
	 for(i=1;i<257;i++)
	 printf(" %d %f\n",i*256,avrage[i][j]);

	 }
	 //}*/
	printf("*** GPU context***\n",j);
	for(i=1;i<257;i++)
	{
		//printf(" %d %f \n",i*256,(avrage[i][0]+avrage[i][1]+avrage[i][2]));
		printf(" %d %f \n",i*256,(avrage[i][4]));
	}
	return 0;
}
/*------------------------------------------------*/
int main4(int argc, char** argv) {

	// Host variables
	int i, j, k, /*l, nBytes, turn, */ N, tpb;
	//unsigned int num_threads;
	//char *cpu_odata;
	//char *string;
	curandGenerator_t gen;
	cudaEvent_t start, stop, start2, stop2;
	//float time1 = 0.0;
	//float time2 = 0.0;
	float T[1000][5][7];
	float /*maxx, minn, */avg[7], avrage[500][7];
	/*------------------------------------*/
	struct Geo *Host_Geo, *Device_Geo;
	struct Geo2 *Host_Geo2, *Device_Geo2;
	//struct Phy *Host_Phy, *Device_Phy;
	struct Cell *Host_Cell, *Device_Cell;
	struct Node *Host_Node, *Device_Node;
	//struct Buffer *Host_InPhy, *Host_OutPhy, *Device_InPhy, *Device_OutPhy;
	struct RouterBuffer *Host_RouterBuffer, *Device_RouterBuffer;
	float *Device_Randx, *Device_Randy, *Device_Randz, //Device_RouterProb,
			*Host_PosRandx, *Host_PosRandy, *Host_PosRandz, *Host_VRandx,
			*Host_VRandy, *Host_VRandz;
	/*------------------------------------*/
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	checkCUDAError("cuda creator4");
	curandSetPseudoRandomGeneratorSeed(gen, time(NULL));
	srand(time(NULL));

	for (N = 1024; N <= (1024 * 64); N += 1024)

	{

		tpb = 256;
		dim3 grid(tpb, 1, 1);

		dim3 threads(N / tpb, 1, 1);//  N/256 must be an int
		cudaMallocHost((void**) &Host_Cell, B * sizeof(struct Cell));
		checkCUDAError("cudaCellMalloc");
		cudaMallocHost((void**) &Host_Geo, N * sizeof(struct Geo));
		checkCUDAError("cudaGeoMalloc");
		cudaMallocHost((void**) &Host_Geo2, N * sizeof(struct Geo2));
		checkCUDAError("cudaGeo2Malloc");
		cudaMallocHost((void **) &Host_PosRandx, N * sizeof(float));
		checkCUDAError("cudaPosRandxMalloc");
		cudaMallocHost((void **) &Host_PosRandy, N * sizeof(float));
		checkCUDAError("cudaPosRandyMalloc");
		cudaMallocHost((void **) &Host_PosRandz, N * sizeof(float));
		checkCUDAError("cudaPosRandzMalloc");
		cudaMallocHost((void **) &Host_VRandx, N * sizeof(float));
		checkCUDAError("cudaVRandxMalloc");
		cudaMallocHost((void **) &Host_VRandy, N * sizeof(float));
		checkCUDAError("cudaVRandyMalloc");
		cudaMallocHost((void **) &Host_VRandz, N * sizeof(float));
		checkCUDAError("cudaVRandzMalloc");
		cudaMallocHost((void **) &Host_RouterBuffer,
				1 * sizeof(struct RouterBuffer));
		checkCUDAError("cudaRouterBufferMalloc");
		cudaMallocHost((void**) &Host_Node, N * sizeof(struct Node));
		checkCUDAError("cudaNodeMalloc");
		Init_Host(Host_Cell, Host_Geo, Host_Geo2, Host_PosRandx, Host_PosRandy,
				Host_PosRandz, Host_VRandx, Host_VRandy, Host_VRandz,
				Host_Node, Host_RouterBuffer, N);
		/*
		 * Device memory allocation
		 */
		cudaMalloc((void**) &Device_Cell, B * sizeof(struct Cell));
		checkCUDAError("cudaDeviceCellMalloc");
		cudaMalloc((void**) &Device_Geo, N * sizeof(struct Geo));
		checkCUDAError("cudaDeviceGeoMalloc");
		cudaMalloc((void**) &Device_Geo2, N * sizeof(struct Geo2));
		checkCUDAError("cudaDeviceGeo2Malloc");
		cudaMalloc((void **) &Device_Randx, N * sizeof(float));
		checkCUDAError("cudaDeviceRandxMalloc");
		cudaMalloc((void **) &Device_Randy, N * sizeof(float));
		checkCUDAError("cudaDeviceRandyMalloc");
		cudaMalloc((void **) &Device_Randz, N * sizeof(float));
		checkCUDAError("cudaDeviceRandzMalloc");
		///Node Device memory allocation///
		cudaMalloc((void**) &Device_Node, N * sizeof(struct Node));
		checkCUDAError("cudaMalloc");
		Init_Device(Host_Cell, Host_Geo, Host_Geo2, Host_Node,
				Host_RouterBuffer, Device_Cell, Device_Geo, Device_Geo2,
				Device_Node, Device_RouterBuffer, N);
		checkCUDAError("cuda creator5");
		checkCUDAError("cuda creator3");
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventCreate(&start2);
		cudaEventCreate(&stop2);
		cudaEventRecord(start2, 0);
		checkCUDAError("cuda creator2");
		//printf("here 001\n\r");
		for (i = 0; i < 10; i++) {
			j = 0;
#ifdef __timing__
			cudaEventRecord(start2, 0);
#endif
			for (k = 0; k < N; k++)
				Host_PosRandx[k] = ((float) rand()) / ((float) RAND_MAX);
			for (k = 0; k < N; k++)
				Host_PosRandy[k] = ((float) rand()) / ((float) RAND_MAX);
			for (k = 0; k < N; k++)
				Host_PosRandz[k] = ((float) rand()) / ((float) RAND_MAX);

			//printf("here 002\n\r");
#ifdef __timing__
			cudaEventRecord(stop2, 0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&T[i][j][0], start2, stop2);
			cudaEventRecord(start2, 0);
#endif
			//printf("here 003\n\r");
			for (k = 0; k < N; k++)
				Mobility(Host_Cell, Host_Geo, 4, Host_PosRandx, Host_PosRandy,
						Host_PosRandz, k,N);
#ifdef __timing__
			cudaEventRecord(stop2, 0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&T[i][j][1], start2, stop2);
			cudaEventRecord(start2, 0);
#endif
			checkCUDAError("cuda kernel error mob");
			//	Mobility_Control<<<threads,grid>>>(Device_Cell,Device_Geo,2);
			for (k = 0; k < N; k++)
				Mobility_Control(Host_Cell, Host_Geo, 2, k, N);
#ifdef __timing__
			cudaEventRecord(stop2, 0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&T[i][j][2], start2, stop2);
			cudaEventRecord(start2, 0);
#endif
			checkCUDAError("cuda kernel error controle2");
			//updateCell<<<32,(B/32)>>>(Device_Cell, N);
			//printf("here 004\n\r");
			for (k = 0; k < B; k++)
				updateCell(Host_Cell, N, k);
			checkCUDAError("cuda kernel error controle3");
#ifdef __timing__
			cudaEventRecord(stop2, 0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&T[i][j][3], start2, stop2);
			cudaEventRecord(start2, 0);
#endif
			for (k = 0; k < N; k++)
				VisibleCPU(Host_Geo, Host_Cell, Host_Geo2, k);
			//cudaThreadSynchronize();
			checkCUDAError("cuda kernel12");
#ifdef __timing__
			cudaEventRecord(stop2, 0);
			cudaEventSynchronize(stop2);
			cudaEventElapsedTime(&T[i][j][4], start2, stop2);
			cudaEventRecord(start2, 0);
#endif
		}
		cudaFree(Device_Geo);
		cudaFree(Device_Geo2);
		cudaFree(Device_Cell);
		cudaFree(Device_Node);
		cudaFree(Device_Randx);
		cudaFree(Device_Randy);
		cudaFree(Device_Randz);
		checkCUDAError("cudaFreeDevice");
		cudaFreeHost(Host_Geo);
		cudaFreeHost(Host_Geo2);
		cudaFreeHost(Host_Cell);
		cudaFreeHost(Host_Node);
		cudaFreeHost(Host_PosRandx);
		cudaFreeHost(Host_PosRandy);
		cudaFreeHost(Host_PosRandz);
		cudaFreeHost(Host_VRandx);
		cudaFreeHost(Host_VRandy);
		cudaFreeHost(Host_VRandz);
		checkCUDAError("cudaFreeHost");
#ifdef __timing2__
		cudaEventRecord(stop2, 0);
		cudaEventSynchronize(stop2);
		cudaEventElapsedTime(&time2, start2, stop2);
		//printf("total time for %d Nodes during %d mobility and %d slots is %f \n\r",N,i,j,time2/10);
#endif

		checkCUDAError("cudaMemcpy4");
#ifdef __timing__

		for (i = 0; i < 10; i++) {
			for (j = 0; j < 5; j++) {
				avg[j] += T[i][0][j];
			}
		}
		for (j = 0; j < 5; j++) {
			avg[j] = avg[j] / 10;
		}
		for (j = 0; j < 5; j++)
			avrage[(N / 256)][j] = avg[j];
#endif
#ifdef __timing22__
		for(i=0;i<10;i++)
		{
			for(j=0;j<5;j++)
			{
				for(k=0;k<7;k++)
				{
					if((j==0)||((k>3)))printf("%f|",T[i][j][k]); else {printf("%f|",T[i][0][k]);}
				}
				printf("OK %d\n",N);
			}

		}
#endif
		printf(" %d %f\n", N, avrage[N / 256][4]);
	}
	/*for(j=0;j<4;j++)
	 {
	 printf("*** %d***\n",j);
	 for(i=1;i<257;i++)
	 printf(" %d %f\n",i*256,avrage[i][j]);

	 }*/
	//printf("*** CPU context***\n",j);
	for (i = 1; i < 257; i++) {
		//printf(" %d %f\n",i*256,(avrage[i][0]+avrage[i][1]+avrage[i][2]));
	}
	return 0;
}

