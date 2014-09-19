/*
 * DeviceMemoryAllocation.cu
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

/*static inline void Device_Memory_Allocation(struct Cell *Device_Cell, struct Geo *Device_Geo, struct Geo2 *Device_Geo2, float *Device_Randx, float *Device_Randy, float *Device_Randz, struct Node *Device_Node)
{

	///Cell Device memory allocation///
	cudaMalloc((void**)&Device_Cell, B * sizeof(struct Cell)) ;
			    checkCUDAError("cudaDeviceCellMalloc");


	///Geo Device memory allocation///
	cudaMalloc((void**)&Device_Geo, N * sizeof(struct Geo)) ;
				checkCUDAError("cudaDeviceGeoMalloc");


	///Geo2 Device memory allocation///
	cudaMalloc((void**)&Device_Geo2, N * sizeof(struct Geo2)) ;
					checkCUDAError("cudaDeviceGeo2Malloc");


	///Random speed parameters (needed for Mobility) Device memory allocation///
	cudaMalloc (( void **) & Device_Randx , N * sizeof ( float ) );
					checkCUDAError("cudaDeviceRandxMalloc");
	cudaMalloc (( void **) & Device_Randy , N * sizeof ( float ) );
					checkCUDAError("cudaDeviceRandyMalloc");
	cudaMalloc (( void **) & Device_Randz , N * sizeof ( float ) );
					checkCUDAError("cudaDeviceRandzMalloc");


	///Node Device memory allocation///
	cudaMalloc((void**)&Device_Node, N * sizeof(struct Node)) ;
					checkCUDAError("cudaMalloc");
}*/
