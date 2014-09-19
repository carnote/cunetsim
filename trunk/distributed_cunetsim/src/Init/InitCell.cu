/*
 * InitCell.cu
 *
 *  Created on: Oct 25, 2011
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

#include "../vars.h"

/*
 *  This function initializes the cells.
 */

__host__ void Init_cell(struct Cell *cell, int node_number)
{
	int i,j,k,l,id;
	int distance;
	int cell_size = simulation_parameters.topology_config.area.geo_cell.cell_size_m;
	int step_x = simulation_parameters.topology_config.area.geo_cell.step_x;
	int step_y = simulation_parameters.topology_config.area.geo_cell.step_y;
	int step_z = simulation_parameters.topology_config.area.geo_cell.step_z;

	int cell_number = simulation_parameters.topology_config.area.geo_cell.cell_number;

	for(i=0;i<step_x;i++)//x
	{
		for(j=0;j<step_y;j++)//y
		{
			for(k=0;k<step_z;k++)//z
			{
				id=i+j*step_x+k*step_x*step_y; // each cell has a unique id
				//printf("je m'appel %d\n\r",id);

				cell[id].id=id;
				cell[id].size=0;
				cell[id].node=0;
				cell[id].p1.x=i*cell_size;
				cell[id].p1.y=j*cell_size;
				cell[id].p1.z=k*cell_size;
				cell[id].p2.x=(i+1)*cell_size;
				cell[id].p2.y=(j+1)*cell_size;
				cell[id].p2.z=(k+1)*cell_size;
				///cell[id].p2.x=cell[id].p1.x + Visibility; Just to control BBR14/11/2011
				///cell[id].p2.y=cell[id].p1.y + Visibility;
				///cell[id].p2.z=cell[id].p1.z + Visibility;
				cell[id].center.x=(cell[id].p1.x+cell[id].p2.x)/2;
				cell[id].center.y=(cell[id].p1.y+cell[id].p2.y)/2;
				cell[id].center.z=(cell[id].p1.z+cell[id].p2.z)/2;
				for(l=0;l<node_number;l++)
				{
					cell[id].passage[l]=0;
					cell[id].member[l]=0;
				}
			}
		}
	}

	// Then we build the array of neighboring cells (two cells are neighbors if their distance < 2*Visibility)
	//cell[i].v=0;
	for(i=0;i<(cell_number);i++)
	{
		cell[i].neighbor_vector[0]=i;
		cell[i].neighbor_number=1;
		for(j=0;j<(cell_number);j++)
			{///BBR14/11/2011
				if(j!=i)
				{
					distance=(int)sqrt(pow((cell[j].center.x-cell[i].center.x),2)+pow((cell[j].center.y-cell[i].center.y),2)+pow((cell[j].center.z-cell[i].center.z),2));
					if (distance<(2*cell_size))
					{
						cell[i].neighbor_vector[cell[i].neighbor_number]=j;
						cell[i].neighbor_number++;
					}
				}
			}
	}
//return(0);
}
