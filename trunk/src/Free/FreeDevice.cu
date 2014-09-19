/*
 * FreeDevice.cu
 *
 *  Created on: Dec 7, 2011
 *      Author: benromdh
 */


#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include "../structures.h"
#endif /* STRUCTURES_H_ */


__host__ void Free_Device(struct Geo *Device_Geo, struct Geo2 *Device_Geo2, struct Cell *Device_Cell, struct Node *Device_Node, struct Buffer *Device_InPhy, struct Buffer *Device_OutPhy, struct RouterBuffer *Device_RouterBuffer, float *Device_Randx, float  *Device_Randy, float *Device_Randz)
{
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
}
