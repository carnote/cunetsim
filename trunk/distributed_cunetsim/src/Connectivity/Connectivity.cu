/*******************************************************************************

  Eurecom Cunetsim2
  Copyright(c) 2011 - 2012 Eurecom

  This program is free software; you can redistribute it and/or modify it
  under the terms and conditions of the GNU General Public License,
  version 2, as published by the Free Software Foundation.

  This program is distributed in the hope it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
  more details.

  You should have received a copy of the GNU General Public License along with
  this program; if not, write to the Free Software Foundation, Inc.,
  51 Franklin St - Fifth Floor, Boston, MA 02110-1301 USA.

  The full GNU General Public License is included in this distribution in
  the file called "COPYING".

  Contact Information
  Cunetsim Admin: cunetsim@eurecom.fr
  Cunetsim Tech : cunetsim_tech@eurecom.fr
  Forums       : TODO
  Address      : Eurecom, 2229, route des crêtes, 06560 Valbonne Sophia Antipolis, France

*******************************************************************************/

/**
 * \file Connectivity.cu
 * \brief Function that processes the grid scenario
 * \author Bilel BR
 * \version 0.0.2
 * \date Nov 22, 2011
 */
 
#ifndef STRUCTURES_H_
#include "../structures.h"
#define STRUCTURES_H_
#endif /* STRUCTURES_H_ */
#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "../interfaces.h"
#endif /* INTERFACES_H_ */
#define dist 100
 
 /**
 * \fn void Visible(struct Geo *geo, struct Cell *cell, int node_number, int cell_size)
 * \brief looks for the neighbors of each node
 *
 * \param geo pointer to the space data needed to update a node's neighbors
 * \param cell pointer to the cell distribution data needed to select nodes contained in a particular cell
 * \param node_number is node number in the simulation
 * \param cell_size is the length of the side of the cubic or square cell
 * \return void
 */
__global__ void Visible(struct Geo *geo, struct Cell *cell, int node_number, int cell_size)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int i,j,current_cell;

	if (tid < node_number)
	{

	    geo[tid].neighbor_number=0;

	    if (geo[tid].energy > 0) {
		int number_of_neighboring_cells=cell[geo[tid].cell_id].neighbor_number;

		for(i=0;i<number_of_neighboring_cells;i++)
		{	
			current_cell=cell[geo[tid].cell_id].neighbor_vector[i];
			int number_of_nodes=cell[current_cell].size;
		
			for(j=0;j<number_of_nodes;j++)
			{
				geo[tid].MobMode++;
				int node=cell[current_cell].member[j];
			
				if(Distance(geo[tid].p,geo[node].p)<= ((float)cell_size/2.0) +0.0001)
				{
					if(node!=tid)
					{
						geo[tid].neighbor_vector[geo[tid].neighbor_number]=node;
						geo[tid].neighbor_number++;
					
						if(geo[tid].neighbor_number==Maxneighbor)
						{
							break;
						}
					}
				}
			}
			if(geo[tid].neighbor_number==Maxneighbor) 
			{
				break;
			}
		}
	    }
	}
}

/**
 * \fn __host__ __device__ bool Parse_Cell(Geo *g_node, struct Cell *c_in, struct Cell *c_test)
 * \brief verifies whether a node in a particular cell could find neighbors in an other particular cell
 *
 * \param g_node pointer to the node
 * \param c_in pointer to the cell which contains the node
 * \param c_test pointer to the cell which is tested to know if is could contain neighbors for g_node
 * \return true if c_test could not contain neighbors for g_node
 */
__host__ __device__ bool Parse_Cell(Geo *g_node, struct Cell *c_in, struct Cell *c_test)
{
	if(g_node[0].p.x>=c_in[0].center.x)
	{
		if(c_test[0].center.x<c_in[0].center.x)
			return true;
	}
	else
	{
		if(c_test[0].center.x>c_in[0].center.x)
			return true;
	}
	if(g_node[0].p.y>=c_in[0].center.y)
	{
		if(c_test[0].center.y<c_in[0].center.y)
			return true;
	}
	else
	{
		if(c_test[0].center.y>c_in[0].center.y)
			return true;
	}
	if(g_node[0].p.z>=c_in[0].center.z)
	{
		if(c_test[0].center.z<c_in[0].center.z)
			return true;
	}
	else
	{
		if(c_test[0].center.z>c_in[0].center.z)
			return true;
	}
	return false;
}

 /**
 * \fn void Visible_Opt(struct Geo *geo, struct Cell *cell, int node_number, int cell_size)
 * \brief looks for the neighbors of each node - optimized version that takes into account a reduced number of cells
 *
 * \param geo pointer to the space data needed to update a node's neighbors
 * \param cell pointer to the cell distribution data needed to select nodes contained in a particular cell
 * \param node_number is node number in the simulation
 * \param cell_size is the length of the side of the cubic or square cell
 * \return void
 */
__global__ void Visible_Opt(struct Geo *geo, struct Cell *cell, int node_number, int cell_size, 
		int offset, struct Connection *ext_connections, int nb_ext_conn)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int i,j,current_cell;
	float distance;
	
	if (tid < node_number){
	
	    geo[tid].neighbor_number=0;
	
	    if (geo[tid].energy > 0) {

		int number_of_neighboring_cells = cell[geo[tid].cell_id].neighbor_number;
		
		for(i = 0; i < number_of_neighboring_cells; i++)
		{
			current_cell = cell[geo[tid].cell_id].neighbor_vector[i];
		
			if( Parse_Cell(&geo[tid],&cell[geo[tid].cell_id], &cell[current_cell]))
				continue;
		
			int number_of_nodes = cell[current_cell].size;
		
			for(j = 0; j < number_of_nodes; j++)
			{
				int node = cell[current_cell].member[j];
				distance = Distance(geo[tid].p,geo[node].p);

				if(distance <= ((float)cell_size/2.0) +0.0001)
				{
					if(node!=tid)
					{
						geo[tid].neighbor_vector[geo[tid].neighbor_number]=node + offset;
						geo[tid].neighbor_number++;
					
						if(geo[tid].neighbor_number==Maxneighbor)
						{
							break;
						}
					}
				}
			}
		
			if(geo[tid].neighbor_number==Maxneighbor)  
			{
				break;
			}
		}
		
		if (geo[tid].external_conn == 1) { //external neighbors given by the coordinator
			//printf("I am an external connection %d and I have %d neighbors\n", tid + offset, geo[tid].neighbor_number);
			for (int ext_conn = 0; ext_conn < nb_ext_conn; ext_conn++) {
				int my_node, other_node;
				my_node = ext_connections[ext_conn].node1;
				other_node = ext_connections[ext_conn].node2;
				//printf("[%d] my = %d & oth = %d\n", tid + offset, my_node, other_node);
			
				if (my_node == tid+offset) {
					if(geo[tid].neighbor_number==Maxneighbor) {
						geo[tid].neighbor_vector[0] = other_node; //In case we have already reached the Maxneighbor value, we prefer 
						                                          //the external neighbor to any other internal one, foe example the
						                                          //first assigned neighbor (geo[tid].neighbor_vector[0])
																	 	
					} else {
						geo[tid].neighbor_vector[geo[tid].neighbor_number] = other_node;
						geo[tid].neighbor_number++;
					}
					//printf("I found it %d now I have %d neighbors\n", other_node, geo[tid].neighbor_number);
					break;
				}
				//printf("%d is an ext conn with %d\n", my_node, other_node);
			}
		}

	    }
		
		//printf("EXT CONN = %d\n", nb_ext_conn);
		
		
		/*if (tid + offset == 99) {
			printf("\nI am %d and my neighbors are: ", tid + offset);
			for (int ne = 0; ne < geo[tid].neighbor_number; ne++)
				printf("%d ", geo[tid].neighbor_vector[ne]);
		}*/
	}
}

/////////////////////////////////////////
//The cpu version is not to the point yet
/////////////////////////////////////////

void Visible_cpu(struct Geo *G, struct Cell *C, struct Geo2 *G2, int tid)
{
	int i,j,current_cell;
	float D;
	G[tid].neighbor_number=0;

	int numnumber_of_neighboring_cells=C[G[tid].cell_id].neighbor_number;
	for(i=0;i<numnumber_of_neighboring_cells;i++)
	{
		current_cell=C[G[tid].cell_id].neighbor_vector[i];
		if(Parse_Cell(&G[tid],&C[G[tid].cell_id],&C[current_cell]))
			continue;
			
		int number_of_nodes=C[current_cell].size;
		for(j=0;j<number_of_nodes;j++)
		{

			int node=C[current_cell].member[j];
			D=Distance(G[tid].p,G[node].p);

			
			if(D<=100.001)
			{
				if(node!=tid)
				{
					G[tid].neighbor_vector[G[tid].neighbor_number]=node;
					G[tid].neighbor_number++;
					if(G[tid].neighbor_number==Maxneighbor)
						{
							break;
						}
				}
			}
			G[tid].MobMode++;
		}
		if(G[tid].neighbor_number==Maxneighbor)
		{
			break;
		}
	}
}



__host__ __device__ float Distance(struct Position P1 , struct Position P2)
{
	return sqrtf(powf(P1.x-P2.x,2)+powf(P1.y-P2.y,2)+powf(P1.z-P2.z,2));

}


