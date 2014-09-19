/*
 * FreeHost.cu
 *
 *  Created on: Dec 7, 2011
 *      Author: benromdh
 */


#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include "../structures.h"
#endif /* STRUCTURES_H_ */


__host__ void Free_Host(struct Geo *Host_Geo, struct Geo2 *Host_Geo2, struct Cell *Host_Cell, struct Node *Host_Node, struct Buffer *Host_InPhy, struct Buffer *Host_OutPhy, struct RouterBuffer *Host_RouterBuffer, float *Host_PosRandx, float  *Host_PosRandy, float *Host_PosRandz, float *Host_VRandx, float  *Host_VRandy, float *Host_VRandz)
{
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
}
