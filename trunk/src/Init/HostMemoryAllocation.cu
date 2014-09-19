/*
 * HostMemoryAllocation.cu
 *
 *  Created on: Nov 23, 2011
 *      Author: benromdh
 */

#ifndef STRUCTURES_H_
#include "../structures.h"
#define STRUCTURES_H_
#endif /* STRUCTURES_H_ */

#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "../interfaces.h"
#endif /* INTERFACES_H_ */

//extern inline void Host_Memory_Allocation(struct Cell *Host_Cell, struct Geo *Host_Geo, struct Geo2 *Host_Geo2, float *Host_PosRandx, float *Host_PosRandy, float *Host_PosRandz, float *Host_VRandx, float *Host_VRandy, float *Host_VRandz, struct Node *Host_Node)
/*{

	///Cell host memory allocation///
	cudaMallocHost((void**)&Host_Cell, B * sizeof(struct Cell)) ;
			    checkCUDAError("cudaCellMalloc");


	///Geo host memory allocation///
	cudaMallocHost((void**)&Host_Geo, N * sizeof(struct Geo)) ;
				checkCUDAError("cudaGeoMalloc");


	///Geo2 host memory allocation///
	cudaMallocHost((void**)&Host_Geo2, N * sizeof(struct Geo2)) ;
					checkCUDAError("cudaGeo2Malloc");


	///Random position and speed parameters (needed for Geo) host memory allocation///
	cudaMallocHost (( void **) & Host_PosRandx , N * sizeof ( float ) );
					checkCUDAError("cudaPosRandxMalloc");
	cudaMallocHost (( void **) & Host_PosRandy , N * sizeof ( float ) );
					checkCUDAError("cudaPosRandyMalloc");
	cudaMallocHost (( void **) & Host_PosRandz , N * sizeof ( float ) );
					checkCUDAError("cudaPosRandzMalloc");
	cudaMallocHost (( void **) & Host_VRandx , N * sizeof ( float ) );
					checkCUDAError("cudaVRandxMalloc");
	cudaMallocHost (( void **) & Host_VRandy , N * sizeof ( float ) );
					checkCUDAError("cudaVRandyMalloc");
	cudaMallocHost (( void **) & Host_VRandz , N * sizeof ( float ) );
					checkCUDAError("cudaVRandzMalloc");


	///Node host memory allocation///
	cudaMallocHost((void**)&Host_Node, N * sizeof(struct Node)) ;
					checkCUDAError("cudaMalloc");
}
*/
