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
 * \file MobilityControl.cu
 * \brief Functions necessary for the mobility control process
 * \author Bilel BR
 * \version 0.0.2
 * \date Oct 25, 2011
 */

#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include "../structures.h"
#endif /* STRUCTURES_H_ */

#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "../interfaces.h"
#endif /* INTERFACES_H_ */

#define xmax 1600
#define ymax 1600
#define zmax 1600


/*
 * This function controls the mobility of the nodes when they hit the boundary of the space
 */


/**
 * \fn __global__ void Mobility_Control(struct Cell *cell,struct Geo *geo, int node_number, Simulation_Parameters* device_simulation_parameters)
 * \brief does the mobility control process which means modifying wrong positions produced by the mobility process
 *
 * \param cell pointer to the cell distribution data needed to notify cells of the nodes they contain after the last movement process
 * \param geo pointer to the space data that will be updated following mobility type
 * \param node_number is the node number in the simulation
 * \param device_simulation_parameters pointer to a device copy of the global structure containing simulation parameters
 * \return void
 */
__global__ void Mobility_Control(struct Cell *cell,struct Geo *geo, int node_number, Simulation_Parameters* device_simulation_parameters)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int x_min, y_min, x_max, y_max, zmin, z_max;
	int cell_nb, cell_size, step_x, step_y;
	enum Boundary_Policy mode;
	int _3D = device_simulation_parameters->simulation_config._3D_is_activated;
	
		
	//printf("XMAX = %d et YMAX = %d\n",x_max,y_max);

	if (tid < node_number) 
	{

	cell_nb = device_simulation_parameters->topology_config.area.geo_cell.cell_number;
	cell_size = device_simulation_parameters->topology_config.area.geo_cell.cell_size_m;
	step_x = device_simulation_parameters->topology_config.area.geo_cell.step_x;
	step_y = device_simulation_parameters->topology_config.area.geo_cell.step_y;

	mode = device_simulation_parameters->topology_config.mobility_parameters.boundary_policy;
	x_min = y_min = zmin = 0;
	x_max = (int) (device_simulation_parameters->topology_config.area.x_km * 1000);
	y_max = (int) (device_simulation_parameters->topology_config.area.y_km * 1000);
	z_max = (int) (device_simulation_parameters->topology_config.area.z_km * 1000);
	
	switch (mode)
	{
		case STOP:
		// Stop approach: the node is reset at the boundary of the space
		{
			if(geo[tid].p.x>=x_max)
				geo[tid].p.x=x_max-1;
			if(geo[tid].p.y>=y_max)
				geo[tid].p.y=y_max-1;
			if ( _3D )
				if(geo[tid].p.z>=z_max)
					geo[tid].p.z=z_max-1;
			break;
		}
		case BOUNCE:
		// Bounce approach: the node bounces back from the boundary of the space
		{
			if(geo[tid].p.x>=x_max)
				geo[tid].p.x=2*x_max-geo[tid].p.x-1;
			if(geo[tid].p.y>=y_max)
				geo[tid].p.y=2*y_max-geo[tid].p.y-1;
			if ( _3D )
				if(geo[tid].p.z>=z_max)
					geo[tid].p.z=2*z_max-geo[tid].p.z-1;
			if(geo[tid].p.x<x_min)
				geo[tid].p.x=-geo[tid].p.x;
			if(geo[tid].p.y<y_min)
				geo[tid].p.y=-geo[tid].p.y;
			if ( _3D )
				if(geo[tid].p.z<zmin)
					geo[tid].p.z=-geo[tid].p.z;
			break;
		}
		case MODULAR:
		// Modular approach: the node appears on the other side of the space
		{
			if(geo[tid].p.x>=x_max)
				geo[tid].p.x = geo[tid].p.x%x_max;
			if(geo[tid].p.x<x_min)
				geo[tid].p.x += x_max;
			if(geo[tid].p.y>=y_max)
				geo[tid].p.y=geo[tid].p.y%y_max;
			if(geo[tid].p.y<y_min)
				geo[tid].p.y += y_max;
			if ( _3D ){
				if(geo[tid].p.z>=z_max)
					geo[tid].p.z=geo[tid].p.z%z_max;
				if(geo[tid].p.z<zmin)
					geo[tid].p.z += z_max;
			}
			break;
		}
		case RANDOM:
		// Random approach: the node reappears randomly in the space
		{
			/*srand(time(NULL));
			if(geo[tid].p.x>xmax)
				geo[tid].p.x=(int) (rand()/RAND_MAX)*xmax;
			if(geo[tid].p.y>ymax)
				geo[tid].p.y=(int) (rand()/RAND_MAX)*ymax;
			if(geo[tid].p.z>zmax)
				geo[tid].p.z=(int) (rand()/RAND_MAX)*zmax;
			break;
			*/
		}
		default:
		// Default is set to bounce approach
		{
			if(geo[tid].p.x>x_max)
				geo[tid].p.x=2*x_max-geo[tid].p.x;
			if(geo[tid].p.y>y_max)
				geo[tid].p.y=2*y_max-geo[tid].p.y;
			if ( _3D )
				if(geo[tid].p.z>z_max)
					geo[tid].p.z=2*z_max-geo[tid].p.z;
			break;
		}

	}

	// Here we run the passage of the nodes through cells
	geo[tid].cell_id = geo[tid].p.x/cell_size + geo[tid].p.y/cell_size*step_x + geo[tid].p.z/cell_size*step_x*step_y;

	if((geo[tid].cell_id<(cell_nb)))//&&(geo[tid].cell_id>(-1)))
	{
		if(geo[tid].cell_id!=geo[tid].old_cell_id) // if the node moves toward an other cell
		{
			cell[geo[tid].old_cell_id].passage[tid]=(char)0; // the node leaves OldCellId
			cell[geo[tid].cell_id].passage[tid]=(char)1; // the node passes by CellId
			geo[tid].old_cell_id=geo[tid].cell_id;
		}
	}
	
	//this is for debugging
	//printf("(%d,%d,%d,%d,%d)\n",tid,geo[tid].cell_id,geo[tid].p.x,geo[tid].p.y,geo[tid].p.z);
	//if(tid==0)printf("%d %d %d \n",geo[tid].p.x,geo[tid].p.y,geo[tid].p.z);

	}
}

//not to the point yet
 void Mobility_control(struct Cell *cell,struct Geo *geo, int mode, int rank, int N)
{
	int tid = rank;
	int xmin,zmin,ymin;
	xmin=ymin=zmin=0;
	switch (mode)
	{
		case 1:
			// Stop approach: the node is reset at the boundary of the space
		{
			if(geo[tid].p.x>=xmax)
				geo[tid].p.x=xmax-1;
			if(geo[tid].p.y>=ymax)
				geo[tid].p.y=ymax-1;
			if(geo[tid].p.z>=zmax)
				geo[tid].p.z=zmax-1;
			break;
		}
		case 2:
			// Bounce approach: the node bounces back from the boundary of the space
		{
			if(geo[tid].p.x>=xmax)
				geo[tid].p.x=2*xmax-geo[tid].p.x-1;
			if(geo[tid].p.y>=ymax)
				geo[tid].p.y=2*ymax-geo[tid].p.y-1;
			if(geo[tid].p.z>=zmax)
				geo[tid].p.z=2*zmax-geo[tid].p.z-1;
			if(geo[tid].p.x<xmin)
				geo[tid].p.x=-geo[tid].p.x;
			if(geo[tid].p.y<ymin)
				geo[tid].p.y=-geo[tid].p.y;
			if(geo[tid].p.z<zmin)
				geo[tid].p.z=-geo[tid].p.z;
			break;
		}
		case 3:
			// Modular approach: the node appears on the other side of the space
		{
			if(geo[tid].p.x>=xmax)
				geo[tid].p.x=geo[tid].p.x%xmax;
			if(geo[tid].p.y>=ymax)
				geo[tid].p.y=geo[tid].p.y%ymax;
			if(geo[tid].p.x>=zmax)
				geo[tid].p.z=geo[tid].p.z%zmax;
			break;
		}
		case 4:
		// Random approach: the node reappears randomly in the space
		{
			/*srand(time(NULL));
			if(geo[tid].p.x>xmax)
				geo[tid].p.x=(int) (rand()/RAND_MAX)*xmax;
			if(geo[tid].p.y>ymax)
				geo[tid].p.y=(int) (rand()/RAND_MAX)*ymax;
			if(geo[tid].p.z>zmax)
				geo[tid].p.z=(int) (rand()/RAND_MAX)*zmax;
			break;
			*/
		}
		default:
			// Default is set to bounce approach
		{
			if(geo[tid].p.x>xmax)
				geo[tid].p.x=2*xmax-geo[tid].p.x;
			if(geo[tid].p.y>ymax)
				geo[tid].p.y=2*ymax-geo[tid].p.y;
			if(geo[tid].p.z>zmax)
				geo[tid].p.z=2*zmax-geo[tid].p.z;
			break;
		}

	}

	// Here we runs the passage of the nodes through cells
	geo[tid].cell_id=geo[tid].p.x/Visibility+geo[tid].p.y/Visibility*Step+geo[tid].p.z/Visibility*Step*Step;
	if((geo[tid].cell_id<(B+1)))//&&(geo[tid].cell_id>(-1)))
	{
		if(geo[tid].cell_id!=geo[tid].old_cell_id) // if the node moves toward an other cell
		{
			cell[geo[tid].old_cell_id].passage[tid]=(char)0; // the node leaves OldCellId
			cell[geo[tid].cell_id].passage[tid]=(char)1; // the node passes by CellId
			geo[tid].old_cell_id=geo[tid].cell_id;
		}
	}
	//if(tid==0)printf("%d %d %d \n",geo[tid].p.x,geo[tid].p.y,geo[tid].p.z);
}

/*
 * This function updates the attributes of a cell after the nodes movement
 */

/**
 * \fn __global__ void updateCell(int cell_number, struct Cell *cell, int N)
 * \brief finishes the cell updating process by making each cell take into account the notifications emitted in mobility control
 *
 * \param cell_number is the total cell number in the simulation
 * \param cell pointer to the cell distribution data needed to access cell internal data to update contained nodes
 * \param N is the node number in the simulation
 * \return void
 */
__global__ void Update_Cell(int cell_number, struct Cell *cell, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int i=0;
    
    if(tid<cell_number)
    {
        cell[tid].size=0;

        for(i=0;i<N;i++)
        {
        	if(cell[tid].passage[i]==1)
        	// if the node i is passed to a cell we write it as a member of the cell and increase the number of nodes in the cell
            {
            	cell[tid].member[cell[tid].size]=i;
            	cell[tid].size++;
            }
         }
     }
}

//not to the point yet
 void Update_cell(struct Cell *cell, int N,int rank)
{
    int tid = rank;
    int i=0;
    //if(tid<B) ??????
    {
        cell[tid].size=0;
        for(i=0;i<N;i++)
        {
                if(cell[tid].passage[i]==1)
                // if the node i is passed to a cell we write it as a member of the cell and increase the number of nodes in the cell
                {
                	cell[tid].member[cell[tid].size]=i;
                	cell[tid].size++;
                }
         }
     }
}
