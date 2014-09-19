/*
 * StatticNetwork.cu
 *
 *  Created on: Dec 15, 2011
 *      Author: benromdh
 */

#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include "structures.h"
#include "/usr/local/cuda/include/curand.h"
#endif /* STRUCTURES_H_ */
#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "interfaces.h"
#endif /* INTERFACES_H_ */


void Init_Static_Buffer(struct Buffer *buf, int node_number)
{
	for(int i=0;i<node_number;i++)
	{
		buf[i].read_index=-1;
		buf[i].write_index=0;
	}
}

void BorderGeo(struct Geo *geo, int id, int posx, int posy, int supnode, int prev, int next)
{
	geo[id].p.x=posx;
	geo[id].p.y=posy;
	geo[id].neighbor_vector[0]=supnode;
	geo[id].neighbor_vector[1]=prev;
	geo[id].neighbor_vector[2]=next;
	geo[id].neighbor_vector[3]=id+1;
	geo[id].neighbor_vector[4]=id+2;
	geo[id].neighbor_vector[5]=id+3;
	geo[id].neighbor_vector[6]=id+4;
	geo[id].neighbor_vector[7]=id+5;
	geo[id].neighbor_vector[8]=id+6;
	geo[id].neighbor_vector[9]=id+7;
	geo[id].neighbor_vector[10]=id+8;
	geo[id].neighbor_number=11;

	geo[id+1].p.x=posx;
	geo[id+1].p.y=posy-50;
	geo[id+1].neighbor_vector[0]=id;
	geo[id+1].neighbor_vector[1]=id+2;
	geo[id+1].neighbor_vector[2]=id+8;
	geo[id+1].neighbor_number=3;

	geo[id+2].p.x=posx-5*sqrt(50);
	geo[id+2].p.y=posy-5*sqrt(50);
	geo[id+2].neighbor_vector[0]=id;
	geo[id+2].neighbor_vector[1]=id+1;
	geo[id+2].neighbor_vector[2]=id+3;
	geo[id+2].neighbor_number=3;

	geo[id+3].p.x=posx-50;
	geo[id+3].p.y=posy;
	geo[id+3].neighbor_vector[0]=id;
	geo[id+3].neighbor_vector[1]=id+2;
	geo[id+3].neighbor_vector[2]=id+4;
	geo[id+3].neighbor_number=3;

	geo[id+4].p.x=posx-5*sqrt(50);
	geo[id+4].p.y=posy+5*sqrt(50);
	geo[id+4].neighbor_vector[0]=id;
	geo[id+4].neighbor_vector[1]=id+3;
	geo[id+4].neighbor_vector[2]=id+5;
	geo[id+4].neighbor_number=3;

	geo[id+5].p.x=posx;
	geo[id+5].p.y=posy+50;
	geo[id+5].neighbor_vector[0]=id;
	geo[id+5].neighbor_vector[1]=id+4;
	geo[id+5].neighbor_vector[2]=id+6;
	geo[id+5].neighbor_number=3;

	geo[id+6].p.x=posx+5*sqrt(50);
	geo[id+6].p.y=posy+5*sqrt(50);
	geo[id+6].neighbor_vector[0]=id;
	geo[id+6].neighbor_vector[1]=id+5;
	geo[id+6].neighbor_vector[2]=id+7;
	geo[id+6].neighbor_number=3;

	geo[id+7].p.x=posx+50;
	geo[id+7].p.y=posy;
	geo[id+7].neighbor_vector[0]=id;
	geo[id+7].neighbor_vector[1]=id+6;
	geo[id+7].neighbor_vector[2]=id+8;
	geo[id+7].neighbor_number=3;

	geo[id+8].p.x=posx+5*sqrt(50);
	geo[id+8].p.y=posy-5*sqrt(50);
	geo[id+8].neighbor_vector[0]=id;
	geo[id+8].neighbor_vector[1]=id+7;
	geo[id+8].neighbor_vector[2]=id+1;
	geo[id+8].neighbor_number=3;
}



void Init_Static_Geo(struct Geo *geo, int node_number)
{
	for(int i=0;i<node_number;i++)
	{
		geo[i].speedx=0;
		geo[i].speedy=0;
		geo[i].speedz=0;
		geo[i].CellPosition=0;
		geo[i].neighbor_number=0;
		geo[i].MobMode=0;
		geo[i].Viewed=0;
		geo[i].p.z=0;
	}

	///Central BackBone///
	//Node 0
	geo[0].p.x=800-100;
	geo[0].p.y=800-100;
	geo[0].neighbor_vector[0]=1;
	geo[0].neighbor_vector[1]=3;
	geo[0].neighbor_vector[2]=4;
	geo[0].neighbor_vector[3]=5;
	geo[0].neighbor_vector[4]=6;
	geo[0].neighbor_number=5;
	//Node1
	geo[1].p.x=800-100;
	geo[1].p.y=800+100;
	geo[1].neighbor_vector[0]=0;
	geo[1].neighbor_vector[1]=2;
	geo[1].neighbor_vector[2]=7;
	geo[1].neighbor_vector[3]=8;
	geo[1].neighbor_vector[4]=9;
	geo[1].neighbor_number=5;
	//Node 2
	geo[2].p.x=800+100;
	geo[2].p.y=800+100;
	geo[2].neighbor_vector[0]=1;
	geo[2].neighbor_vector[1]=3;
	geo[2].neighbor_vector[2]=10;
	geo[2].neighbor_vector[3]=11;
	geo[2].neighbor_vector[4]=12;
	geo[2].neighbor_number=5;
	//Node 3
	geo[3].p.x=800+100;
	geo[3].p.y=800-100;
	geo[3].neighbor_vector[0]=0;
	geo[3].neighbor_vector[1]=2;
	geo[3].neighbor_vector[2]=13;
	geo[3].neighbor_vector[3]=14;
	geo[3].neighbor_vector[4]=15;
	geo[3].neighbor_number=5;

	///Secondary BackBone///
	//Node 4
	geo[4].p.x=600+50;
	geo[4].p.y=200+100;
	geo[4].neighbor_vector[0]=0;
	geo[4].neighbor_vector[1]=5;
	geo[4].neighbor_vector[2]=15;
	geo[4].neighbor_vector[3]=16;
	geo[4].neighbor_vector[4]=25;
	geo[4].neighbor_number=5;
	//Node 5
	geo[5].p.x=400;
	geo[5].p.y=400;
	geo[5].neighbor_vector[0]=0;
	geo[5].neighbor_vector[1]=4;
	geo[5].neighbor_vector[2]=6;
	geo[5].neighbor_vector[3]=34;
	geo[5].neighbor_vector[4]=43;
	geo[5].neighbor_vector[5]=52;
	geo[5].neighbor_number=6;
	//Node 6
	geo[6].p.x=200+100;
	geo[6].p.y=600+50;
	geo[6].neighbor_vector[0]=0;
	geo[6].neighbor_vector[1]=5;
	geo[6].neighbor_vector[2]=7;
	geo[6].neighbor_vector[3]=61;
	geo[6].neighbor_vector[4]=70;
	geo[6].neighbor_number=5;
	//Node 7
	geo[7].p.x=200+100;
	geo[7].p.y=1000-50;
	geo[7].neighbor_vector[0]=1;
	geo[7].neighbor_vector[1]=6;
	geo[7].neighbor_vector[2]=8;
	geo[7].neighbor_vector[3]=79;
	geo[7].neighbor_vector[4]=88;
	geo[7].neighbor_number=5;
	//Node 8
	geo[8].p.x=400;
	geo[8].p.y=1200;
	geo[8].neighbor_vector[0]=1;
	geo[8].neighbor_vector[1]=7;
	geo[8].neighbor_vector[2]=9;
	geo[8].neighbor_vector[3]=97;
	geo[8].neighbor_vector[4]=106;
	geo[8].neighbor_vector[5]=115;
	geo[8].neighbor_number=6;
	//Node 9
	geo[9].p.x=600+50;
	geo[9].p.y=1200+100;
	geo[9].neighbor_vector[0]=1;
	geo[9].neighbor_vector[1]=8;
	geo[9].neighbor_vector[2]=10;
	geo[9].neighbor_vector[3]=124;
	geo[9].neighbor_vector[4]=133;
	geo[9].neighbor_number=5;
	//Node 10
	geo[10].p.x=1000-50;
	geo[10].p.y=1200+100;
	geo[10].neighbor_vector[0]=2;
	geo[10].neighbor_vector[1]=9;
	geo[10].neighbor_vector[2]=11;
	geo[10].neighbor_vector[3]=142;
	geo[10].neighbor_vector[4]=151;
	geo[10].neighbor_number=5;
	//Node 11
	geo[11].p.x=1200;
	geo[11].p.y=1200;
	geo[11].neighbor_vector[0]=2;
	geo[11].neighbor_vector[1]=10;
	geo[11].neighbor_vector[2]=12;
	geo[11].neighbor_vector[3]=160;
	geo[11].neighbor_vector[4]=169;
	geo[11].neighbor_vector[5]=178;
	geo[11].neighbor_number=6;
	//Node 12
	geo[12].p.x=1200+100;
	geo[12].p.y=1000-50;
	geo[12].neighbor_vector[0]=2;
	geo[12].neighbor_vector[1]=11;
	geo[12].neighbor_vector[2]=13;
	geo[12].neighbor_vector[3]=187;
	geo[12].neighbor_vector[4]=196;
	geo[12].neighbor_number=5;
	//Node 13
	geo[13].p.x=1200+100;
	geo[13].p.y=600+50;
	geo[13].neighbor_vector[0]=3;
	geo[13].neighbor_vector[1]=12;
	geo[13].neighbor_vector[2]=14;
	geo[13].neighbor_vector[3]=205;
	geo[13].neighbor_vector[4]=214;
	geo[13].neighbor_number=5;
	//Node 14
	geo[14].p.x=1200;
	geo[14].p.y=400;
	geo[14].neighbor_vector[0]=3;
	geo[14].neighbor_vector[1]=13;
	geo[14].neighbor_vector[2]=15;
	geo[14].neighbor_vector[3]=223;
	geo[14].neighbor_vector[4]=232;
	geo[14].neighbor_vector[5]=241;
	geo[14].neighbor_number=6;
	//Node 15
	geo[15].p.x=1000-50;
	geo[15].p.y=200+100;
	geo[15].neighbor_vector[0]=3;
	geo[15].neighbor_vector[1]=14;
	geo[15].neighbor_vector[2]=4;
	geo[15].neighbor_vector[3]=250;
	geo[15].neighbor_vector[4]=259;
	geo[15].neighbor_number=5;


	BorderGeo(geo,16,700,100,4,259,25);
	BorderGeo(geo,25,500,100,4,16,34);
	BorderGeo(geo,34,300,100,5,25,43);
	BorderGeo(geo,43,100,100,5,34,52);
	BorderGeo(geo,52,100,300,5,43,61);
	BorderGeo(geo,61,100,500,6,52,70);
	BorderGeo(geo,70,100,700,6,61,79);
	BorderGeo(geo,79,100,900,7,70,88);
	BorderGeo(geo,88,100,1100,7,79,97);
	BorderGeo(geo,97,100,1300,8,88,106);
	BorderGeo(geo,106,100,1500,8,97,115);
	BorderGeo(geo,115,300,1500,8,106,124);
	BorderGeo(geo,124,500,1500,9,115,133);
	BorderGeo(geo,133,700,1500,9,124,142);
	BorderGeo(geo,142,900,1500,10,133,151);
	BorderGeo(geo,151,1100,1500,10,142,160);
	BorderGeo(geo,160,1300,1500,11,151,169);
	BorderGeo(geo,169,1500,1500,11,160,178);
	BorderGeo(geo,178,1500,1300,11,169,187);
	BorderGeo(geo,187,1500,1100,12,178,196);
	BorderGeo(geo,196,1500,900,12,187,205);
	BorderGeo(geo,205,1500,700,13,196,214);
	BorderGeo(geo,214,1500,500,13,205,223);
	BorderGeo(geo,223,1500,300,14,214,232);
	BorderGeo(geo,232,1500,100,14,223,241);
	BorderGeo(geo,241,1300,100,14,232,250);
	BorderGeo(geo,250,1100,100,15,241,259);
	BorderGeo(geo,259,900,100,15,250,16);
}


void Static_Network( int argc, char** argv)
{/*
	int N=9*32;
	struct Cell *Host_Cell, *Device_Cell;
	struct Geo *Host_Geo,*Host_Geo2,*Device_Geo, *Device_Geo2;
	struct Buffer *Host_InPhy,*Host_OutPhy, *Device_InPhy, *Device_OutPhy;
	struct Node *Host_Node, *Device_Node;
	struct RouterBuffer *Host_RouterBuffer, *Device_RouterBuffer;
	float *Device_RouterProb;
	dim3 threads(32,1,1);
	dim3 grid(9,1,1);
	curandGenerator_t gen ;

		///Cell host memory allocation///
		cudaMallocHost((void**)&Host_Cell, B * sizeof(struct Cell)) ;
				    checkCUDAError("cudaCellMalloc");


		///Geo host memory allocation///
		cudaMallocHost((void**)&Host_Geo, N * sizeof(struct Geo)) ;
					checkCUDAError("cudaGeoMalloc");


		///Geo2 host memory allocation///
		cudaMallocHost((void**)&Host_Geo2, N * sizeof(struct Geo2)) ;
					checkCUDAError("cudaGeo2Malloc");

		/// Host Buffer allocation
		cudaMallocHost((void**)&Host_InPhy, N * sizeof(struct Buffer)) ;
					checkCUDAError("cudaBufferInMalloc");
		cudaMallocHost((void**)&Host_OutPhy, N * sizeof(struct Buffer)) ;
					checkCUDAError("cudaBufferOut2Malloc");

		///Node host memory allocation///
		cudaMallocHost((void**)&Host_Node, N * sizeof(struct Node)) ;
					checkCUDAError("cudaNodeMalloc");

		///Host Router Buffer Allocation///
		cudaMallocHost((void **)&Host_RouterBuffer, N * sizeof(struct RouterBuffer));
					checkCUDAError("cudaRouterBufferMalloc");


		Init_Cell_Gen2D(Host_Cell,Step,Step, N);
		Init_Static_Geo(Host_Geo,N);
		Init_Static_Buffer(Host_InPhy,N);
		Init_Static_Buffer(Host_OutPhy,N);
		Init_Node(Host_Node, N);
		Init_RouterBuffer(Host_RouterBuffer, N);

		//Printconnectivity(Host_Geo,N);

		///Cell Device memory allocation///
		cudaMalloc((void**)&Device_Cell, B * sizeof(struct Cell)) ;
				    checkCUDAError("cudaDeviceCellMalloc");


		///Geo Device memory allocation///
		cudaMalloc((void**)&Device_Geo, N * sizeof(struct Geo)) ;
					checkCUDAError("cudaDeviceGeoMalloc");


		///Geo2 Device memory allocation///
		cudaMalloc((void**)&Device_Geo2, N * sizeof(struct Geo2)) ;
						checkCUDAError("cudaDeviceGeo2Malloc");


		///Buffer Device memory allocation///
		cudaMalloc((void**)&Device_InPhy, N * sizeof(struct Buffer)) ;
					checkCUDAError("cudaDeviceBufferINMalloc");
		cudaMalloc((void**)&Device_OutPhy, N * sizeof(struct Buffer)) ;
					checkCUDAError("cudaDeviceBufferOutMalloc");


		///Node Device memory allocation///
		cudaMalloc((void**)&Device_Node, N * sizeof(struct Node)) ;
						checkCUDAError("cudaMalloc");


		///Device Router Buffer Allocation///
		cudaMalloc((void **)&Device_RouterBuffer, N * sizeof(struct RouterBuffer));
						checkCUDAError("cudaDeviceRouterBufferMalloc");


		///Device Router Transmission Probability Allocation///
		cudaMalloc (( void **) & Device_RouterProb , N * sizeof ( float ) );
						checkCUDAError("cudaDeviceBufferProbMalloc");


		cudaMemcpy(Device_Cell,Host_Cell,  B * sizeof(struct Cell) , cudaMemcpyHostToDevice) ;
		cudaMemcpy(Device_Geo,Host_Geo,  N * sizeof(struct Geo) , cudaMemcpyHostToDevice) ;
		cudaMemcpy(Device_InPhy,Host_InPhy,  N * sizeof(struct Buffer) , cudaMemcpyHostToDevice) ;
		cudaMemcpy(Device_OutPhy,Host_OutPhy,  N * sizeof(struct Buffer) , cudaMemcpyHostToDevice) ;
		cudaMemcpy(Device_RouterBuffer,Host_RouterBuffer,  N * sizeof(struct RouterBuffer) , cudaMemcpyHostToDevice) ;

		curandCreateGenerator (& gen ,CURAND_RNG_PSEUDO_DEFAULT );
		curandSetPseudoRandomGeneratorSeed( gen , time(NULL)) ;

		creator<<<grid,threads>>>(Device_OutPhy,1,2);
		for(int i=0;i<100;i++)
		{
			cudaMemcpy(Host_Geo,Device_Geo,  N * sizeof(struct Geo) , cudaMemcpyDeviceToHost) ;
			cudaMemcpy(Host_OutPhy,Device_OutPhy,  N* sizeof(struct Buffer) , cudaMemcpyDeviceToHost) ;
			printpropagation(Host_Geo,Host_OutPhy,0,i,268);
			Sender<<<grid,threads>>>(Device_Geo,Device_InPhy,Device_OutPhy);
			checkCUDAError("cuda bla1");
			Receiver<<<grid,threads>>>(Device_InPhy);
			checkCUDAError("cuda bla2");
			curandGenerateUniform ( gen , Device_RouterProb , N );
			checkCUDAError("cuda bla3");
			router<<<grid,threads>>>(Device_InPhy,Device_OutPhy,Device_RouterBuffer,Device_RouterProb);
			checkCUDAError("cuda bla4");

		}
*/

}
