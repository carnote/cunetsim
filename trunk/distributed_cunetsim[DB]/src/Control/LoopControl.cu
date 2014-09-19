/*
 * LoopControl.cu
 *
 *  Created on: Nov 23, 2011
 *      Author: benromdh
 */

#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include "../structures.h"
#endif /* STRUCTURES_H_ */

__host__ void Buffer_control(struct Buffer *host_in_phy, struct Buffer *host_out_phy, int node_number)
{
	for(int i=0;i<node_number;i++)
	{
		printf("Node %d: InBuffer (R:%d,W:%d) OutBuffer (R:%d,W:%d)\n",i,host_in_phy[i].read_index,host_in_phy[i].write_index,host_out_phy[i].read_index,host_out_phy[i].write_index);
	}
}
