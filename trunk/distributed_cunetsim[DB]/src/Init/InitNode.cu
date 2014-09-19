/*
 * InitNode.cu
 *
 *  Created on: Nov 10, 2011
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

/*
 * This function initializes the non geometrical attributes of a node.
 */

//not to the point yet
__host__ void Init_node(struct Node *node, int N)
{
	for(int i=0;i<N;i++)
		{
			node[i].Id=i;
			node[i].GeoId=i;
			node[i].PhyId=i;
			node[i].InBuffId=0;
			node[i].NumBuffer=0;
			node[i].OutBuffId=0;
			node[i].OutLook=0;
			node[i].RoutingId=0;
			node[i].TcId=0;
			//printf("Node %d: GeoId %d",node[i].Id,node[i].GeoId);
		}
}
