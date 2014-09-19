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
 * \file InitCellGen.cu
 * \brief Functions necessary to initialize the cells
 * \author Bilel BR
 * \version 0.0.2
 * \date Oct 25, 2011
 */

#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include "../structures.h"
#endif /* STRUCTURES_H_ */
#include "../vars.h"

/*
 *  This function initializes a cell by taking in parameters Stepx, Stepy and Stepz in order to generate all possible situations in 3D
 */

 /**
 * \fn __host__ void Init_Cell_Gen(struct Cell *cell, int step_x, int step_y, int step_z, int node_number)
 * \brief initializes the cell data contained in the host
 *
 * \param cell pointer to the cell data that will be initialized
 * \param step_x number of cells following the x dimension
 * \param step_y number of cells following the y dimension
 * \param step_z number of cells following the z dimension
 * \param node_number is the node number in the simulation
 * \return void
 */
__host__ void Init_cell_gen(struct Cell *cell, int step_x, int step_y, int step_z, int node_number, Simulation_Parameters simulation_parameters)
{
    int i,j,k,l,id,distance;	
    int cell_size = simulation_parameters.topology_config.area.geo_cell.cell_size_m;

    for(i=0;i<step_x;i++) // x
    {
        for(j=0;j<step_y;j++) // y
        {
            for(k=0;k<step_z;k++) // z
            {
                id=i+j*step_x+k*step_y*step_x;
                cell[id].id=id;
                cell[id].size=0;
                cell[id].node=0;
                cell[id].p1.x=i*cell_size;
                cell[id].p1.y=j*cell_size;
                cell[id].p1.z=k*cell_size;
                cell[id].p2.x=(i+1)*cell_size;
                cell[id].p2.y=(j+1)*cell_size;
                cell[id].p2.z=(k+1)*cell_size;
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

    for(id=0;id<(step_x*step_y*step_z);id++)
    {

	cell[id].neighbor_vector[0]=id;
	cell[id].neighbor_number=1;
        for(i=-1;i<2;i++)//x
        {
            for(j=-1;j<2;j++)//y
            {
                for(k=-1;k<2;k++)//z
                {
                    l=id+i+j*step_x+k*step_x*step_y;
                    if ((l>=0)&&(l<step_x*step_y*step_z)&&l!=id)
                    {
                    distance=(int)sqrt(pow((cell[id].center.x-cell[l].center.x),2)+pow((cell[id].center.y-cell[l].center.y),2)+pow((cell[id].center.z-cell[l].center.z),2));
                    if (distance<(2*cell_size))
                    {
                        cell[id].neighbor_vector[cell[id].neighbor_number]=id+i+j*step_x+k*step_x*step_y;
                        cell[id].neighbor_number++;
                    }
                    }
                }
            }
        }
    }
}
