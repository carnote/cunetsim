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
 * \file Init.cu
 * \brief Functions necessary to initialize the global structure, the host and the device data
 * \author Bilel BR
 * \version 0.0.2
 * \date Nov 23, 2011
 */

#ifndef STRUCTURES_H_
#include "../structures.h"
#define STRUCTURES_H_
#endif /* STRUCTURES_H_ */

#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "../interfaces.h"
#endif /* INTERFACES_H_ */
//#define _3D_

Simulation_Parameters simulation_parameters;

 /**
 * \fn void Init_simulation_parameters()
 * \brief initializes simulation (most important) parameters with default values
 *
 * \return void
 */
void Init_simulation_parameters(){

	float area_x, area_y, area_z;
	int reach, step_x, step_y, step_z, cell_size;
	int node_nb;
	struct Connection connection, connection2, connection3;

	/* Environment Config */
	simulation_parameters.environment_config.m_send = 1.9;
	simulation_parameters.environment_config.b_send = 420;
	simulation_parameters.environment_config.m_recv = 0.42;
	simulation_parameters.environment_config.b_recv = 330;
	simulation_parameters.environment_config.init_energy = 100000.0;

	/* Simulation Config */ 
	//simulation_parameters.simulation_config.node_number = 200;
	node_nb = simulation_parameters.simulation_config.node_number;
	simulation_parameters.simulation_config.drop_probability = 0.1;	
	simulation_parameters.simulation_config.simulation_time = NB_TOURS;
	simulation_parameters.simulation_config._3D_is_activated = 1;

	/* Topology Config */
	simulation_parameters.topology_config.connectivity_model = UDG;
	simulation_parameters.topology_config.distribution.initial_distribution = RANDOM_DISTRIBUTION; //GRID / RANDOM_DISTRIBUTION
	simulation_parameters.topology_config.mobility_parameters.moving_dynamics.min_speed_mps = vmin;
	simulation_parameters.topology_config.mobility_parameters.moving_dynamics.max_speed_mps = vmax;
	simulation_parameters.topology_config.mobility_parameters.moving_dynamics.min_sleep_ms = 0;
	simulation_parameters.topology_config.mobility_parameters.moving_dynamics.max_sleep_ms = 2000;
	simulation_parameters.topology_config.mobility_parameters.mobility_type = RANDOM_UNIFORM_MOVEMENT_MAX_AND_MIN_BOUNDED; 
	simulation_parameters.topology_config.mobility_parameters.boundary_policy = MODULAR;
	simulation_parameters.topology_config.area.geo_cell.reach = Reach;
	reach = simulation_parameters.topology_config.area.geo_cell.reach;	
	simulation_parameters.topology_config.area.geo_cell.cell_size_m = 2 * reach;
	cell_size = simulation_parameters.topology_config.area.geo_cell.cell_size_m;

	/* Application_Config */
	simulation_parameters.application_config.customized = 0;
	simulation_parameters.application_config.predefined_traffic.application_type = DEFAULT;
	
	/* Distributed Simulation Config (Default) */
	simulation_parameters.distributed_simulation_config.partition_number = 4;
	
	//common values for the 4 partitions
	for (int part = 0; part < 4; part++) {
		simulation_parameters.distributed_simulation_config.partitions[part].number_of_external_connections = 1;
		simulation_parameters.distributed_simulation_config.partitions[part].conversion_table[0] = 0;
		simulation_parameters.distributed_simulation_config.partitions[part].conversion_table[1] = node_nb / 4;
		simulation_parameters.distributed_simulation_config.partitions[part].conversion_table[2] = node_nb / 2;
		simulation_parameters.distributed_simulation_config.partitions[part].conversion_table[3] = (node_nb / 4) * 3;
		
		// for the moment node_number and area dimensions are the same for all the partitions
		simulation_parameters.distributed_simulation_config.partitions[part].area_x = 10; 
		simulation_parameters.distributed_simulation_config.partitions[part].area_y = 1.0;
		simulation_parameters.distributed_simulation_config.partitions[part].area_z = 0.2;
		simulation_parameters.distributed_simulation_config.partitions[part].node_number = node_nb / 4;
		simulation_parameters.distributed_simulation_config.partitions[part].masters_number = 4;
		simulation_parameters.distributed_simulation_config.partitions[part].additional_time = 2 * (50 + 50 + 50 + 50);
		simulation_parameters.distributed_simulation_config.partitions[part].total_node_number = node_nb;
		
	}
	
	//specific values:
	
	//offsets
	simulation_parameters.distributed_simulation_config.partitions[0].offset = 0;
	simulation_parameters.distributed_simulation_config.partitions[1].offset = node_nb / 4;
	simulation_parameters.distributed_simulation_config.partitions[2].offset = node_nb / 2;
	simulation_parameters.distributed_simulation_config.partitions[3].offset = (node_nb / 4) * 3;
	
	//external connections (and their numbers)
	simulation_parameters.distributed_simulation_config.partitions[0].number_of_external_connections = 1;
	simulation_parameters.distributed_simulation_config.partitions[1].number_of_external_connections = 2;
	simulation_parameters.distributed_simulation_config.partitions[2].number_of_external_connections = 2;
	simulation_parameters.distributed_simulation_config.partitions[3].number_of_external_connections = 1;
	
	connection.node1 = (node_nb / 4) - 1;
 	connection.node2 = node_nb / 4;
	simulation_parameters.distributed_simulation_config.partitions[0].external_connections[0] = connection;
	
	connection.node1 = node_nb / 4;
 	connection.node2 = (node_nb / 4) - 1;
	simulation_parameters.distributed_simulation_config.partitions[1].external_connections[0] = connection;
	
	connection.node1 = (node_nb / 2) - 1;
 	connection.node2 = node_nb / 2;
	simulation_parameters.distributed_simulation_config.partitions[1].external_connections[1] = connection;
	
	connection.node1 = node_nb / 2;
 	connection.node2 = (node_nb / 2) - 1;
	simulation_parameters.distributed_simulation_config.partitions[2].external_connections[0] = connection;

	connection.node1 = ((node_nb / 4) * 3) - 1;
 	connection.node2 = (node_nb / 4) * 3;
	simulation_parameters.distributed_simulation_config.partitions[2].external_connections[1] = connection;
	
	connection.node1 = (node_nb / 4) * 3;
 	connection.node2 = ((node_nb / 4) * 3) - 1;
	simulation_parameters.distributed_simulation_config.partitions[3].external_connections[0] = connection;
	
}


 /**
 * \fn __host__ void Init_Host_Static_Grid(struct Geo *Host_Geo, int N)
 * \brief initializes the space data contained in the host
 *
 * \param Host_Geo pointer to the space data that will be passed to the function which role is to initialize space data
 * \param node_number is the node number in the simulation
 * \return void
 */
__host__ void Init_host_static_grid(struct Geo *host_geo, int node_number, int offset, struct Connection *ext_connections, int nb_ext_conn) {

	///Geo initialization///
	Init_geo_static_grid(host_geo, node_number, offset, ext_connections, nb_ext_conn);

}

 /**
 * \fn __host__ void Init_Device_Static_Grid(struct Geo *Host_Geo, struct Geo *Device_Geo, int N)
 * \brief initializes the space data contained in the device
 *
 * \param Host_Geo pointer to the space data that is contained in the host and will be copied to the device
 * \param Device_Geo pointer to the space data that is contained in the device and that will be updated
 * \param N is the node number in the simulation
 * \return void
 */
__host__ void Init_device_static_grid(struct Geo *host_geo,
		struct Geo *device_geo, int node_number) {

	cudaMemcpy(device_geo, host_geo, node_number * sizeof(struct Geo),
			cudaMemcpyHostToDevice);


}

__host__ void Init_device(struct Cell *Host_Cell, struct Geo *Host_Geo,
		struct Geo2* Host_Geo2, struct Node *Host_Node,
		struct RouterBuffer *Host_RouterBuffer, struct Cell *Device_Cell,
		struct Geo *Device_Geo, struct Geo2 *Device_Geo2,
		struct Node *Device_Node, struct RouterBuffer *Device_RouterBuffer,
		int N) {

	cudaMemcpy(Device_Cell, Host_Cell, B * sizeof(struct Cell),
			cudaMemcpyHostToDevice);
	cudaMemcpy(Device_Geo, Host_Geo, N * sizeof(struct Geo),
			cudaMemcpyHostToDevice);
	cudaMemcpy(Device_Geo2, Host_Geo2, N * sizeof(struct Geo2),
			cudaMemcpyHostToDevice);
#ifndef __mobonly__
	//cudaMemcpy(Device_RouterBuffer,Host_RouterBuffer,  N * sizeof(struct RouterBuffer) , cudaMemcpyHostToDevice) ;
#endif
}

/*************************/
/*** Random model ***/
/*************************/

/**
 * \fn __host__ void Init_Host_Random(struct Cell *host_cell, struct Geo *host_geo,
		float *host_pos_randx, float *host_pos_randy,
		float *host_pos_randz, float *host_v_randx, float *host_v_randy,
		float *host_v_randz, int node_number)
 * \brief initializes the space data contained in the host (random scenario)
 *
 * \param host_cell pointer to the cell distribution data will be pass to the function which role is to initialize cell	data
 * \param host_geo pointer to the space data that will be passed to the function which role is to initialize space data
 * \param host_pos_randx pointer to the float that will contain a random value for the x coordinate of the node's position
 * \param host_pos_randy pointer to the float that will contain a random value for the y coordinate of the node's position
 * \param host_pos_randz pointer to the float that will contain a random value for the z coordinate of the node's position
 * \param host_v_randx pointer to the float that will contain a random value for the x coordinate of the node's speed
 * \param host_v_randy pointer to the float that will contain a random value for the y coordinate of the node's speed
 * \param host_v_randz pointer to the float that will contain a random value for the z coordinate of the node's speed
 * \param node_number is the node number in the simulation
 * \return void
 */
__host__ void Init_host_random(struct Cell *host_cell, struct Geo *host_geo,
		float *host_pos_randx, float *host_pos_randy,
		float *host_pos_randz, float *host_v_randx, float *host_v_randy,
		float *host_v_randz, int node_number, Simulation_Parameters _simulation_parameters,
		int offset, struct Connection *ext_connections, int nb_ext_conn) {
	
	int step_x = _simulation_parameters.topology_config.area.geo_cell.step_x;
	int step_y = _simulation_parameters.topology_config.area.geo_cell.step_y;
	int step_z = _simulation_parameters.topology_config.area.geo_cell.step_z;
	int _3D = _simulation_parameters.simulation_config._3D_is_activated;

	/* Cell initialization */

    if ( _3D )
	Init_cell_gen(host_cell, step_x, step_y, step_z, node_number, _simulation_parameters); //3D version
    else
	Init_cell_gen_2D(host_cell,step_x,step_y, node_number, _simulation_parameters); //2D version

	/* Random position and speed parameters (needed for Geo) initialization */
	srand(time(NULL));

    if ( _3D )
	for (int i = 0; i < node_number; i++)
	{
		host_pos_randx[i] = rand() / (double) RAND_MAX;
		host_pos_randy[i] = rand() / (double) RAND_MAX;
		host_pos_randz[i] = rand() / (double) RAND_MAX;
		host_v_randx[i] = rand() / (double) RAND_MAX;
		host_v_randy[i] = rand() / (double) RAND_MAX;
		host_v_randz[i] = rand() / (double) RAND_MAX;
	}
    else
	for(int i=0; i < node_number; i++) 
	{
		host_pos_randx[i] = rand() / (double) RAND_MAX;
		host_pos_randy[i] = rand() / (double) RAND_MAX;
		host_pos_randz[i]=0;
		host_v_randx[i] = rand() / (double) RAND_MAX;
		host_v_randy[i] = rand() / (double) RAND_MAX;
		host_v_randy[i]=0;
	}
	
	///Geo initialization///
	Init_geo(host_cell, host_geo, host_pos_randx, host_pos_randy, host_pos_randz,
			host_v_randx, host_v_randy, host_v_randz, node_number, _simulation_parameters, 
			offset, ext_connections, nb_ext_conn);

}

/**
 * \fn __host__ void Init_Device_Random(struct Cell *Host_Cell, struct Geo *Host_Geo, struct Cell *Device_Cell, struct Geo *Device_Geo, int N)
 * \brief initializes the space data contained in the device (random scenario)
 *
 * \param host_cell pointer to the cell distribution data that will be copied to the device
 * \param host_geo pointer to the space data that is contained in the host and will be copied to the device
 * \param device_cell pointer to the cell distribution data that is contained in the device and that will be updated
 * \param device_geo pointer to the space data that is contained in the device and that will be updated
 * \param node_number is the node number in the simulation
 * \return void
 */
__host__ void Init_device_random(struct Cell *host_cell, struct Geo *host_geo,
		struct Cell *device_cell, struct Geo *device_geo, int node_number, Simulation_Parameters _simulation_parameters) {

	int cell_number = _simulation_parameters.topology_config.area.geo_cell.cell_number;

	cudaMemcpy(device_cell, host_cell, cell_number * sizeof(struct Cell),
			cudaMemcpyHostToDevice);
	cudaMemcpy(device_geo, host_geo, node_number * sizeof(struct Geo),
			cudaMemcpyHostToDevice);

}
