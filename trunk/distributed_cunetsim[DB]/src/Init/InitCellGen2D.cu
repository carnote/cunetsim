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
 * \file InitCellGen2D.cu
 * \brief Functions necessary to initialize the cells in the 2D-mode
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
 * This function initializes a cell in 2D with two parameters in order to generate all different possibilities
 */

 /**
 * \fn __host__ void Init_Cell_Gen2D(struct Cell *cell, int step_x, int step_y, int node_number)
 * \brief initializes the cell data contained in the host (2D case)
 *
 * \param cell pointer to the cell data that will be initialized
 * \param step_x number of cells following the x dimension
 * \param step_y number of cells following the y dimension
 * \param node_number is the node number in the simulation
 * \return void
 */
__host__ void Init_cell_gen_2D(struct Cell *cell, int step_x, int step_y, int node_number, Simulation_Parameters simulation_parameters)
{
    int i,j,l,cid,distance, visibility;
    visibility = simulation_parameters.topology_config.area.geo_cell.cell_size_m;

    for(i=0;i<step_x;i++) // x
    {
        for(j=0;j<step_y;j++) // y
        {
            cid=i+j*step_x;
            cell[cid].id=cid;
            cell[cid].size=0;
            cell[cid].node=0;
            cell[cid].p1.x=i*visibility;
            cell[cid].p1.y=j*visibility;
            cell[cid].p1.z=0;
            cell[cid].p2.x=(i+1)*visibility;
            cell[cid].p2.y=(j+1)*visibility;
            cell[cid].p2.z=0;
            cell[cid].center.x=(cell[cid].p1.x+cell[cid].p2.x)/2;
            cell[cid].center.y=(cell[cid].p1.y+cell[cid].p2.y)/2;
            cell[cid].center.z=0;
            for(l=0;l<node_number;l++)
                {
                cell[cid].passage[l]=0;
                cell[cid].member[l]=0;
                }
        }
    }

    // Here we build the array of neighboring cells by comparing the distance between a cell and the cells located in (x+1), (x-1), (y+1) or (y-1)
    cell[cid].neighbor_number=0;
    for(cid=0;cid<(step_x*step_y);cid++)

    {
        for(i=-1;i<2;i++)//x
        {
            for(j=-1;j<2;j++)//y
            {
                l=cid+i+j*step_x;
                if((l>=0)&&(l<step_x*step_y))
                {
                    distance=(int)sqrt(pow((cell[cid].center.x-cell[l].center.x),2)+pow((cell[cid].center.y-cell[l].center.y),2));
                    if (distance<(2*visibility))
                    {
			int cell_neighbor;
			for (cell_neighbor = 0; cell_neighbor < cell[cid].neighbor_number; cell_neighbor++)
				if (cell[cid].neighbor_vector[cell_neighbor] == cid+i+j*step_x)
					break;
			if (cell_neighbor == cell[cid].neighbor_number) {
       	                 	cell[cid].neighbor_vector[cell[cid].neighbor_number]=cid+i+j*step_x;
                        	cell[cid].neighbor_number++;
			}
                    }
                }
            }
        }

    }
//return(0);
}
