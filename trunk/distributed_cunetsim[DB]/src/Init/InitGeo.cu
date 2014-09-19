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
 * \file InitGeo.cu
 * \brief Functions necessary to initialize the space data
 * \author Bilel BR
 * \version 0.0.2
 * \date Nov 10, 2011
 */

#include <time.h>
#include <unistd.h>
#include <math.h>
#ifndef STRUCTURES_H_
#include "../structures.h"
#define STRUCTURES_H_
#endif /* STRUCTURES_H_ */
#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "../interfaces.h"
#endif /* INTERFACES_H_ */
#include "../vars.h"



//#define _3D_


 /**
 * \fn __host__ void Init_geo(struct Cell *cell, struct Geo *geo, float *posrandx,
		float *posrandy, float *posrandz, float *vrandx, float *vrandy,
		float *vrandz, int node_number)
 * \brief initializes geographical properties of the nodes
 *
 * \param cell pointer to the cell distribution data
 * \param geo pointer to the space data
 * \param posrandx pointer to the table providing a random value for the x component of each node's position
 * \param posrandy pointer to the table providing a random value for the y component of each node's position
 * \param posrandz pointer to the table providing a random value for the z component of each node's position
 * \param vrandx pointer to the table providing a random value for the x component of each node's speed
 * \param vrandy pointer to the table providing a random value for the y component of each node's speed
 * \param vrandz pointer to the table providing a random value for the z component of each node's speed
 * \param node_number is the number of nodes in the simulation
 * \return void
 */
__host__ void Init_geo(struct Cell *cell, struct Geo *geo, float *posrandx,
		float *posrandy, float *posrandz, float *vrandx, float *vrandy,
		float *vrandz, int node_number, Simulation_Parameters _simulation_parameters,
		int offset, struct Connection *ext_connections, int nb_ext_conn) {

	int step_x = _simulation_parameters.topology_config.area.geo_cell.step_x;
	int step_y = _simulation_parameters.topology_config.area.geo_cell.step_y;
	int step_z = _simulation_parameters.topology_config.area.geo_cell.step_z;

	int cell_size = _simulation_parameters.topology_config.area.geo_cell.cell_size_m;
	int v_min = _simulation_parameters.topology_config.mobility_parameters.moving_dynamics.min_speed_mps;
	int v_max = _simulation_parameters.topology_config.mobility_parameters.moving_dynamics.max_speed_mps;
	int _3D = _simulation_parameters.simulation_config._3D_is_activated;

	float energy = simulation_parameters.environment_config.init_energy;

	for (int i = 0; i < node_number; i++) {
		// We initialize the energy level for the node
		geo[i].energy = energy;
	
		// The absolute iD
		geo[i].id = i + offset;
		
		// Whether the node has an external connection. The default option is no external connection
		geo[i].external_conn = 0;
		
		// We initialize a node with a random position (but in the space because each coordinate is bounded by Step*Visibility)
		geo[i].p.x = (int) ((1 - posrandx[i]) * step_x * cell_size);
		geo[i].p.y = (int) ((1 - posrandy[i]) * step_y * cell_size);
		if ( _3D )
			geo[i].p.z = (int) ((1 - posrandz[i]) * step_z * cell_size);
		else
			geo[i].p.z = 0;

		// We initialize the speeds randomly but upper bounded by vmax and lower bounded by vmin
		geo[i].speedx = (int) ((v_max - v_min) * vrandx[i] + v_min);
		geo[i].speedy = (int) ((v_max - v_min) * vrandy[i] + v_min);///BBR14/11/2011
		if ( _3D )
			geo[i].speedz = (int) ((v_max - v_min) * vrandz[i] + v_min);
		else
			geo[i].speedz = 0;

		// The model of mobility is not the same for all the nodes, it is intialized randomly
		geo[i].cell_id = geo[i].p.x / cell_size + geo[i].p.y / cell_size
				* step_x + geo[i].p.z / cell_size * step_y * step_x;
		//printf("(%d,%d,%d,%d,%d)\n",i,geo[i].cell_id,geo[i].p.x,geo[i].p.y,geo[i].p.z);
		
		geo[i].CellPosition = 0; // ?
		geo[i].old_cell_id = geo[i].cell_id;
		cell[geo[i].cell_id].passage[i] = 1;
		cell[geo[i].cell_id].member[cell[geo[i].cell_id].size] = i;
		cell[geo[i].cell_id].size += 1;

		geo[i].neighbor_number = 0;
		//printf("Node: %d - GeoPos (x,y): (%d,%d) so I am in cell: %d\n",i,geo[i].p.x,geo[i].p.y,geo[i].cell_id);
	}
	
	//external neighbors given by the coordinator
	for (int ext_conn = 0; ext_conn < nb_ext_conn; ext_conn++) {
		int my_node, other_node;
		
		my_node = ext_connections[ext_conn].node1;
		other_node = ext_connections[ext_conn].node2;
		//printf("Before Seg Fault ! my_node - offset = %d offset = %d geo[my_node - offset].neighbor_number = %d\n", my_node - offset, offset, geo[my_node - offset].neighbor_number);	
		geo[my_node - offset].neighbor_vector[geo[my_node - offset].neighbor_number] = other_node;
		geo[my_node - offset].neighbor_number++;
		geo[my_node - offset].external_conn = 1; 
		
		//printf("%d is an ext conn with %d\n", my_node, other_node);
	}
}

/*************************/
/*** Static grid model ***/
/*************************/


 /**
 * \fn __host__ void Init_geo_static_grid(struct Geo *geo, int node_number)
 * \brief initializes space data for the grid scenario
 *
 * \param geo pointer to the space data needed to set the initial position and the neighbors of each node according to the grid model
 * \param node_number is the number of nodes in the simulation
 * \return void
 */
__host__ void Init_geo_static_grid(struct Geo *geo, int node_number, int offset, struct Connection *ext_connections, int nb_ext_conn) {

	/*
	 *   The grid width and height are equal to the square root of node_number
	 */

	/* Verification of the value of node_number */
	if (pow(sqrt(node_number), 2) != node_number) {
		printf("Error: %d is not the square of an integer!\n",node_number);
		exit(0);
	}

	int grid_dimension = (int) sqrt(node_number);
	float energy = simulation_parameters.environment_config.init_energy;

	//printf("The grid dimension is: %d", grid_dimension);

	for (int i = 0; i < node_number; i++) {
		// We initialize the energy level for the node
		geo[i].energy = energy;
	
		// The absolute iD
		geo[i].id = i + offset;
		
		// Whether the node has an external connection. The default option is no external connection
		geo[i].external_conn = 0;

		// We initialize a node with a predefined position, but id does not have effect on the simulation
		geo[i].p.x = (i % grid_dimension) * 35;
		geo[i].p.y = (i / grid_dimension) * 35;
		geo[i].p.z = 0;

		// Looking for the neighbor_number
		geo[i].neighbor_number = 0;

		// Ids that are put in the neighbor_vector table are the absolute iDs !
		if (i - grid_dimension >= 0 && i - grid_dimension <= node_number - 1) {
			geo[i].neighbor_vector[geo[i].neighbor_number] = i - grid_dimension + offset;
			geo[i].neighbor_number++;
		}

		if (i + grid_dimension >= 0 && i + grid_dimension <= node_number - 1) {
			geo[i].neighbor_vector[geo[i].neighbor_number] = i + grid_dimension + offset;
			geo[i].neighbor_number++;
		}

		if (i - 1 >= 0 && i - 1 <= node_number - 1 && (i % grid_dimension != 0)) {
			geo[i].neighbor_vector[geo[i].neighbor_number] = i - 1 + offset;
			geo[i].neighbor_number++;
		}

		if (i + 1 >= 0 && i + 1 <= node_number - 1 && ((i + 1) % grid_dimension != 0)) {
			geo[i].neighbor_vector[geo[i].neighbor_number] = i + 1 + offset;
			geo[i].neighbor_number++;
		}
	}
	
	//external neighbors given by the coordinator
	for (int ext_conn = 0; ext_conn < nb_ext_conn; ext_conn++) {
		int my_node, other_node;
		my_node = ext_connections[ext_conn].node1;
		other_node = ext_connections[ext_conn].node2;
		
		geo[my_node - offset].neighbor_vector[geo[my_node - offset].neighbor_number] = other_node;
		geo[my_node - offset].neighbor_number++;
		geo[my_node - offset].external_conn = 1; 
		
		//printf("%d is an ext conn with %d\n", my_node, other_node);
	}
	/*
	for (int i = 0; i < node_number; i++) {
		 printf("\nNode: %d - GeoPos (x,y,z): (%d,%d,%d) - neighbor_number: %d - neighbors: ", i + offset,
		 geo[i].p.x, geo[i].p.y, geo[i].p.z, geo[i].neighbor_number);

		 for (int j = 0; j < geo[i].neighbor_number; j++)
		 printf("%d ", geo[i].neighbor_vector[j]);
	}*/
}
