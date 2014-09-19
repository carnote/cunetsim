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

	/* Environment Config */
	simulation_parameters.environment_config.m_send = 1.9;
	simulation_parameters.environment_config.b_send = 420;
	simulation_parameters.environment_config.m_recv = 0.42;
	simulation_parameters.environment_config.b_recv = 330;
	simulation_parameters.environment_config.init_energy = 1000.0;

	/* Simulation Config */ 
	simulation_parameters.simulation_config.node_number = 1500;
	simulation_parameters.simulation_config.drop_probability = 0.1;	
	simulation_parameters.simulation_config.simulation_time = NB_TOURS;
	simulation_parameters.simulation_config._3D_is_activated = 1;

	/* Topology Config */
	simulation_parameters.topology_config.connectivity_model = UDG;
	simulation_parameters.topology_config.distribution.initial_distribution = GRID; //GRID-RANDOM_DISTRIBUTION
	simulation_parameters.topology_config.mobility_parameters.moving_dynamics.min_speed_mps = vmin;
	simulation_parameters.topology_config.mobility_parameters.moving_dynamics.max_speed_mps = vmax;
	simulation_parameters.topology_config.mobility_parameters.moving_dynamics.min_sleep_ms = 0;
	simulation_parameters.topology_config.mobility_parameters.moving_dynamics.max_sleep_ms = 2000;
	simulation_parameters.topology_config.mobility_parameters.mobility_type = RANDOM_UNIFORM_MOVEMENT_MAX_AND_MIN_BOUNDED;
	simulation_parameters.topology_config.mobility_parameters.boundary_policy = MODULAR;
	simulation_parameters.topology_config.area.geo_cell.reach = Reach;
	reach = simulation_parameters.topology_config.area.geo_cell.reach;

	simulation_parameters.topology_config.area.x_km = 5.0;
	simulation_parameters.topology_config.area.y_km = 5.0;
	simulation_parameters.topology_config.area.z_km = 0.2;

	area_x = simulation_parameters.topology_config.area.x_km;
	area_y = simulation_parameters.topology_config.area.y_km;
	area_z = simulation_parameters.topology_config.area.z_km;
	
	simulation_parameters.topology_config.area.geo_cell.cell_size_m = 2 * reach;
	cell_size = simulation_parameters.topology_config.area.geo_cell.cell_size_m;
	
	simulation_parameters.topology_config.area.geo_cell.step_x = (int)((area_x * 1000) / cell_size);
	step_x = simulation_parameters.topology_config.area.geo_cell.step_x;
	if (((int)(area_x * 1000)) % cell_size != 0) {
		simulation_parameters.topology_config.area.geo_cell.step_x += 1;
		step_x = simulation_parameters.topology_config.area.geo_cell.step_x;
		simulation_parameters.topology_config.area.x_km = ((float)(step_x * cell_size))/1000.0;
		area_x = simulation_parameters.topology_config.area.x_km;
	}
		
	simulation_parameters.topology_config.area.geo_cell.step_y = (int)((area_y * 1000) / cell_size);
	step_y = simulation_parameters.topology_config.area.geo_cell.step_y;
	if (((int)(area_y * 1000)) % cell_size != 0) {
		simulation_parameters.topology_config.area.geo_cell.step_y += 1;
		step_y = simulation_parameters.topology_config.area.geo_cell.step_y;
		simulation_parameters.topology_config.area.y_km = ((float)(step_y * cell_size))/1000.0;
		area_y = simulation_parameters.topology_config.area.y_km;
	}

	simulation_parameters.topology_config.area.geo_cell.step_z = (int)((area_z * 1000) / cell_size);
	step_z = simulation_parameters.topology_config.area.geo_cell.step_z;
	if (((int)(area_z * 1000)) % cell_size != 0) {
		simulation_parameters.topology_config.area.geo_cell.step_z += 1;
		step_z = simulation_parameters.topology_config.area.geo_cell.step_z;
		simulation_parameters.topology_config.area.z_km = ((float)(step_z * cell_size))/1000.0;
		area_z = simulation_parameters.topology_config.area.z_km;
	}
	
	simulation_parameters.topology_config.area.geo_cell.cell_number = step_x * step_y;

	if (simulation_parameters.simulation_config._3D_is_activated)
		simulation_parameters.topology_config.area.geo_cell.cell_number *= step_z;

	//This is only for the debug phase
	//printf("[%d,%d,%d,%f,%f,%f]\n",step_x,step_y,step_z, area_x,area_y,area_z);

	/* Application_Config */
	simulation_parameters.application_config.customized = 0;
	simulation_parameters.application_config.predefined_traffic.application_type = DEFAULT;
}

__host__ void Init_host(struct Cell *host_cell, struct Geo *host_geo,
		struct Geo2 *host_geo2, float *host_pos_randx, float *host_pos_randy,
		float *host_pos_randz, float *host_v_randx, float *host_v_randy,
		float *host_v_randz, struct Node *host_node,
		struct RouterBuffer *host_router_buffer, int node_number) {

	int step_x = simulation_parameters.topology_config.area.geo_cell.step_x;
	int step_y = simulation_parameters.topology_config.area.geo_cell.step_y;

	///Cell initialization///
#ifdef _3D_
	Init_cell(host_cell, node_number); //3D version
#else
	Init_cell_gen_2D(host_cell,step_x,step_y, node_number); //2D version
#endif

	///Random position and speed parameters (needed for Geo) initialization///
	srand(time(NULL));
#ifdef _3D_
	for (int i = 0; i < node_number; i++) //3D version
	{
		host_pos_randx[i] = rand() / (double) RAND_MAX;
		host_pos_randy[i] = rand() / (double) RAND_MAX;
		host_pos_randz[i] = rand() / (double) RAND_MAX;
		host_v_randx[i] = rand() / (double) RAND_MAX;
		host_v_randy[i] = rand() / (double) RAND_MAX;
		host_v_randz[i] = rand() / (double) RAND_MAX;
		//printf("%f\n", 2*Host_Randx[i]-1);
	}
#else
	for(int i=0;i<node_number;i++) //2D version
	{
		host_pos_randx[i]=rand()/(double)RAND_MAX;
		host_pos_randy[i]=rand()/(double)RAND_MAX;
		host_pos_randz[i]=1;
		host_v_randx[i]=rand()/(double)RAND_MAX;
		host_v_randy[i]=rand()/(double)RAND_MAX;
		host_v_randz[i]=0;
	}

#endif
	///Geo initialization///
	Init_geo(host_cell, host_geo, host_pos_randx, host_pos_randy, host_pos_randz,
			host_v_randx, host_v_randy, host_v_randz, node_number);

	///Geo2 initilization///
	//Todo

	///Node initialization///
	Init_node(host_node, node_number);

#ifndef __mobonly__
	///RouterBuffer initialization///
	//Init_RouterBuffer(Host_RouterBuffer, N);
#endif
}


 /**
 * \fn __host__ void Init_Host_Static_Grid(struct Geo *Host_Geo, int N)
 * \brief initializes the space data contained in the host
 *
 * \param Host_Geo pointer to the space data that will be passed to the function which role is to initialize space data
 * \param node_number is the node number in the simulation
 * \return void
 */
__host__ void Init_host_static_grid(struct Geo *host_geo, int node_number) {

	///Geo initialization///
	Init_geo_static_grid(host_geo, node_number);

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
		float *host_v_randz, int node_number) {
	
	int step_x = simulation_parameters.topology_config.area.geo_cell.step_x;
	int step_y = simulation_parameters.topology_config.area.geo_cell.step_y;
	int step_z = simulation_parameters.topology_config.area.geo_cell.step_z;
	int _3D = simulation_parameters.simulation_config._3D_is_activated;

	/* Cell initialization */

    if ( _3D )
	Init_cell_gen(host_cell, step_x, step_y, step_z, node_number); //3D version
    else
	Init_cell_gen_2D(host_cell,step_x,step_y, node_number); //2D version

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
	for(int i=0; i<node_number; i++) 
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
			host_v_randx, host_v_randy, host_v_randz, node_number);

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
		struct Cell *device_cell, struct Geo *device_geo, int node_number) {

	int cell_number = simulation_parameters.topology_config.area.geo_cell.cell_number;

	cudaMemcpy(device_cell, host_cell, cell_number * sizeof(struct Cell),
			cudaMemcpyHostToDevice);
	cudaMemcpy(device_geo, host_geo, node_number * sizeof(struct Geo),
			cudaMemcpyHostToDevice);

}
