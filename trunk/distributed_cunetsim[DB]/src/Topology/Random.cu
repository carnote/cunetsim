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

#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include "../structures.h"
#include "/usr/local/cuda/include/curand.h"
#endif /* STRUCTURES_H_ */
#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "../interfaces.h"
#endif /* INTERFACES_H_ */
#include "../vars.h"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <mpi.h>

struct Performances Random_sched(struct Partition partition, int f, float drop_prob,
		void *genV, int gui, MPI_Comm comm_masters, MPI_data_types mpi_types, MYSQL *connection) {
	
	struct Cell *host_cell, *device_cell;
	struct Connection *device_ext_connections;
	float *host_pos_randx, *host_pos_randy, *host_pos_randz, *host_v_randx, *host_v_randy, *host_v_randz;
	float *device_randx, *device_randy, *device_randz, *device_randv;
	struct Performances performance;
	struct Geo *host_geo, *device_geo;
	struct Buffer *host_out_phy, *host_in_phy, *device_out_phy, *device_in_phy;
	struct RemoteCommBuffer *host_remote_comm_out, *host_remote_comm_in, *device_remote_comm_out, *device_remote_comm_in;
	struct MasterBuffer *master_in, *master_out;
	int *host_total_dest, *device_total_dest, *device_forwarded_per_node,
			*host_forwarded_per_node, *host_new_message, *device_new_message, *host_recv_new_message, *device_recv_new_message;
	struct RouterBuffer *host_router_buffer, *device_router_buffer;
	struct MessageBuffer *host_out_app, *host_in_app, *device_out_app,
			*device_in_app;
	dim3 threads, grid;
	int *device_traffic_table, *host_traffic_table;
	int node_number = partition.node_number;
	int offset = partition.offset;
	int total_node_number = partition.total_node_number;
	int ext_conn_nb = partition.number_of_external_connections;
	int commRank;

	// Database variables
	MYSQL *conn = connection;

	char query[4096];
	
	
	Simulation_Parameters *device_simulation_parameters;
	Simulation_Parameters simulation_custom_parameters;

	// variables for MPI	
	int ierr;
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
	int masters_number = partition.masters_number;
	MPI_Status status;	 
	
	float area_x = partition.area_x;
	float area_y = partition.area_y;
	float area_z = partition.area_z;
	int step_x, step_y, step_z;

	
	// These are values that are common to all of the masters. We don't need to change them
	int nb_tours = simulation_parameters.simulation_config.simulation_time;
	int cell_size = simulation_parameters.topology_config.area.geo_cell.cell_size_m;

	float m_send = simulation_parameters.environment_config.m_send;
	float m_recv = simulation_parameters.environment_config.m_recv;
	float b_send = simulation_parameters.environment_config.b_send;
	float b_recv = simulation_parameters.environment_config.b_recv;

	
	
	/* This part is used to specify particular area parameters for each master depending on data read from "partition" */
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	simulation_custom_parameters = simulation_parameters;
	
	simulation_custom_parameters.topology_config.area.geo_cell.step_x = (int)((area_x * 1000) / cell_size);
	step_x = simulation_custom_parameters.topology_config.area.geo_cell.step_x;
	if (((int)(area_x * 1000)) % cell_size != 0) {
		simulation_custom_parameters.topology_config.area.geo_cell.step_x += 1;
		step_x = simulation_custom_parameters.topology_config.area.geo_cell.step_x;
		area_x = ((float)(step_x * cell_size))/1000.0;
	}
	simulation_custom_parameters.topology_config.area.x_km = area_x;
		
	simulation_custom_parameters.topology_config.area.geo_cell.step_y = (int)((area_y * 1000) / cell_size);
	step_y = simulation_custom_parameters.topology_config.area.geo_cell.step_y;
	if (((int)(area_y * 1000)) % cell_size != 0) {
		simulation_custom_parameters.topology_config.area.geo_cell.step_y += 1;
		step_y = simulation_custom_parameters.topology_config.area.geo_cell.step_y;
		area_y = ((float)(step_y * cell_size))/1000.0;
	}
	simulation_custom_parameters.topology_config.area.y_km = area_y;

	simulation_custom_parameters.topology_config.area.geo_cell.step_z = (int)((area_z * 1000) / cell_size);
	step_z = simulation_custom_parameters.topology_config.area.geo_cell.step_z;
	if (((int)(area_z * 1000)) % cell_size != 0) {
		simulation_custom_parameters.topology_config.area.geo_cell.step_z += 1;
		step_z = simulation_custom_parameters.topology_config.area.geo_cell.step_z;
		area_z = ((float)(step_z * cell_size))/1000.0;
	}
	simulation_custom_parameters.topology_config.area.z_km = area_z;
		
	simulation_custom_parameters.topology_config.area.geo_cell.cell_number = step_x * step_y; 
	if (simulation_parameters.simulation_config._3D_is_activated)
		simulation_custom_parameters.topology_config.area.geo_cell.cell_number *= step_z;
	int cell_number = simulation_custom_parameters.topology_config.area.geo_cell.cell_number;
	
	//printf("[Proc %d] - cell_nb %d, area_x %f, area_y %f, area_z %f\n", commRank, cell_number, area_x, area_y, area_z);
	
	/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
	//Get back MPI types
	MPI_Datatype mpi_master_buff = mpi_types.mpi_master_buff;
	MPI_Datatype mpi_data_flow = mpi_types.mpi_data_flow;
	
	curandGenerator_t *gen = (curandGenerator_t *)genV;
	
	grid.x = 32;

	/* Test of the node number to decide of the number and dimension of thread blocks */
	if (node_number % 32 == 0 && node_number > 0) {

		threads.x = node_number / 32;
	} else {
		threads.x = (node_number / 32) + 1;
	}

	/* Will contain different probabilities that are going to be used for the routing of the messages */
	float *device_router_prob, *host_router_prob;

	cudaMalloc((void**) &device_simulation_parameters, sizeof(Simulation_Parameters));
	checkCUDAError("cudaDeviceSimParamMalloc");

	cudaMemcpy(device_simulation_parameters, &simulation_custom_parameters, sizeof(Simulation_Parameters),
			cudaMemcpyHostToDevice);

	/* These parameters will allow us to analyze some features (comment to be changed) */
	cudaMallocHost((void**) &host_total_dest, sizeof(int));
	checkCUDAError("cudaBufferInMalloc");

	cudaMallocHost((void**) &host_new_message, node_number * sizeof(int));
	checkCUDAError("cudaNewMessageMalloc");

	cudaMallocHost((void**) &host_recv_new_message, node_number * sizeof(int));
	checkCUDAError("cudaRecvNewMessageMalloc");

	cudaMallocHost((void**) &host_forwarded_per_node, node_number * sizeof(int));
	checkCUDAError("cudaBufferOutMalloc");

	cudaMalloc((void**) &device_total_dest, sizeof(int));
	checkCUDAError("cudaDeviceTotalDestMalloc");

	cudaMalloc((void**) &device_forwarded_per_node, node_number * sizeof(int));
	checkCUDAError("cudaDeviceForwardedPerNodeMalloc");

	cudaMalloc((void**) &device_new_message, node_number * sizeof(int));
	checkCUDAError("cudaDeviceNewMessageMalloc");

	cudaMalloc((void**) &device_recv_new_message, node_number * sizeof(int));
	checkCUDAError("cudaDeviceRecvNewMessageMalloc");

	/* Initialization of the parameters */
	*host_total_dest = 0;

	cudaMemcpy(device_total_dest, host_total_dest, sizeof(int),
			cudaMemcpyHostToDevice);

	for (int i = 0; i < node_number; i++) {
		host_forwarded_per_node[i] = 0;
		host_new_message[i] = -1;
		host_recv_new_message[i] = -1;
	}
	cudaMemcpy(device_forwarded_per_node, host_forwarded_per_node,
			node_number * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(device_new_message, host_new_message,
			node_number * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(device_recv_new_message, host_recv_new_message,
			node_number * sizeof(int), cudaMemcpyHostToDevice);

	/* Receivers Host memory allocation */
	cudaMallocHost((void**) &host_traffic_table, node_number * sizeof(int));
	checkCUDAError("cudaRecMalloc");

	/* Receivers Device memory allocation */
	cudaMalloc((void**) &device_traffic_table, node_number * sizeof(int));
	checkCUDAError("cudaDeviceRecMalloc");

	/* Cell host memory allocation */
	cudaMallocHost((void**) &host_cell, cell_number * sizeof(struct Cell));
	checkCUDAError("cudaCellMalloc");

	/* Cell Device memory allocation */
 	cudaMalloc((void**) &device_cell, cell_number * sizeof(struct Cell));
 	checkCUDAError("cudaDeviceCellMalloc");

	/* Random position and speed parameters (needed for Geo) host memory allocation */
 	cudaMallocHost((void **) &host_pos_randx, node_number * sizeof(float));
 	checkCUDAError("cudaPosRandxMalloc");
 
	cudaMallocHost((void **) &host_pos_randy, node_number * sizeof(float));
	checkCUDAError("cudaPosRandyMalloc");
 
	cudaMallocHost((void **) &host_pos_randz, node_number * sizeof(float));
	checkCUDAError("cudaPosRandzMalloc");
 
	cudaMallocHost((void **) &host_v_randx, node_number * sizeof(float));
 	checkCUDAError("cudaVRandxMalloc");
 
	cudaMallocHost((void **) &host_v_randy, node_number * sizeof(float));
 	checkCUDAError("cudaVRandyMalloc");
 
	cudaMallocHost((void **) &host_v_randz, node_number * sizeof(float));
 	checkCUDAError("cudaVRandzMalloc");

	/* Random speed parameters (needed for Mobility) Device memory allocation */
 	cudaMalloc((void **) &device_randx, node_number * sizeof(float));
 	checkCUDAError("cudaDeviceRandxMalloc");
 
	cudaMalloc((void **) &device_randy, node_number * sizeof(float));
 	checkCUDAError("cudaDeviceRandyMalloc");
 
	cudaMalloc((void **) &device_randz, node_number * sizeof(float));
 	checkCUDAError("cudaDeviceRandzMalloc");
 
	cudaMalloc((void **) &device_randv, node_number * sizeof(float));
 	checkCUDAError("cudaDeviceRandvMalloc");
	
	/* Geo host memory allocation */
	cudaMallocHost((void**) &host_geo, node_number * sizeof(struct Geo));
	checkCUDAError("cudaGeoMalloc");

	/* Geo Device memory allocation */
	cudaMalloc((void**) &device_geo, node_number * sizeof(struct Geo));
	checkCUDAError("cudaDeviceGeoMalloc");

	/* Host Buffer allocation */
	cudaMallocHost((void**) &host_in_app, node_number * sizeof(struct MessageBuffer));
	checkCUDAError("cudaBufferInMalloc");

	cudaMallocHost((void**) &host_out_app, node_number * sizeof(struct MessageBuffer));
	checkCUDAError("cudaBufferOutMalloc");

	cudaMallocHost((void**) &host_in_phy, node_number * sizeof(struct Buffer));
	checkCUDAError("cudaBufferInMalloc");

	cudaMallocHost((void**) &host_out_phy, node_number * sizeof(struct Buffer));
	checkCUDAError("cudaBufferOutMalloc");

	/* Device Buffer memory allocation */
	cudaMalloc((void**) &device_in_phy, node_number * sizeof(struct Buffer));
	checkCUDAError("cudaDeviceBufferInMalloc");

	cudaMalloc((void**) &device_out_phy, node_number * sizeof(struct Buffer));
	checkCUDAError("cudaDeviceOutPhyMalloc");
	
	cudaMalloc((void**) &device_ext_connections, ext_conn_nb * sizeof(struct Connection));
	cudaMemcpy(device_ext_connections, partition.external_connections, ext_conn_nb * sizeof(struct Connection),
			cudaMemcpyHostToDevice);

	cudaMallocHost((void**) &host_remote_comm_out, ext_conn_nb * sizeof(struct RemoteCommBuffer));
	checkCUDAError("cudaRemCommOutMalloc");
	
	cudaMallocHost((void**) &host_remote_comm_in, ext_conn_nb * sizeof(struct RemoteCommBuffer));
	checkCUDAError("cudaRemCommInMalloc");

	/* Device Buffer memory allocation */
	cudaMalloc((void**) &device_remote_comm_out, ext_conn_nb * sizeof(struct RemoteCommBuffer));
	checkCUDAError("cudaDeviceRemCommOutMalloc");
	
	cudaMalloc((void**) &device_remote_comm_in, ext_conn_nb * sizeof(struct RemoteCommBuffer));
	checkCUDAError("cudaDeviceRemCommInMalloc");

	cudaMalloc((void**) &device_in_app, node_number * sizeof(struct MessageBuffer));
	checkCUDAError("cudaDeviceInPhyMalloc");

	cudaMalloc((void**) &device_out_app, node_number * sizeof(struct MessageBuffer));
	checkCUDAError("cudaDeviceOutAppMalloc");

	cudaMallocHost((void **) &host_router_buffer,
			node_number * sizeof(struct RouterBuffer));
	checkCUDAError("cudaRouterBufferMalloc");

	cudaMalloc((void**) &device_router_buffer, node_number * sizeof(struct RouterBuffer));
	checkCUDAError("cudaDeviceRouterBufferMalloc");

	cudaMalloc((void **) &device_router_prob, node_number * sizeof(float));
	checkCUDAError("cudaDeviceBufferProbMalloc");

	cudaMallocHost((void **) &host_router_prob, node_number * sizeof(float));
	checkCUDAError("cudaBufferProbMalloc");
	
	/* These buffers are used for the inter-master communication */
	/* Instead of sending the remote messages in the order in which we read them, we prefer to regroup them by master_addressee id */
	/* Here we allocate memory for both emission and reception of remote messages and initialize the e */
	master_in = (MasterBuffer *) malloc (masters_number * sizeof(MasterBuffer));
	master_out = (MasterBuffer *) malloc (masters_number * sizeof(MasterBuffer));
	for (int master = 0; master < masters_number; master++) {
		master_in[master].read_index = -1;
		master_in[master].write_index = 0;
		master_out[master].read_index = -1;
		master_out[master].write_index = 0;
		for (int elt = 0; elt < 2 * Maxelement; elt ++) {
			master_in[master].element[elt].header[0] = -1;
			master_in[master].element[elt].header[1] = -1;
			master_in[master].element[elt].header[2] = -1;
			master_in[master].element[elt].header[3] = -1;
			master_out[master].element[elt].header[0] = -1;
			master_out[master].element[elt].header[1] = -1;
			master_out[master].element[elt].header[2] = -1;
			master_out[master].element[elt].header[3] = -1;
		}
	}
	
	/* Host initialization */
	Init_host_random(host_cell, host_geo, host_pos_randx, host_pos_randy,
 	host_pos_randz, host_v_randx, host_v_randy, host_v_randz, node_number, simulation_custom_parameters,
 	offset, partition.external_connections, partition.number_of_external_connections);

	/****************************/
	/* Testing Geo values 

	printf("N = %d\n",N);
	for (int i = 0; i < N; i++){
		printf("Node = %d - PosX = %d - PosY = %d - PosZ = %d - SpeedX = %d - SpeedY = %d - SpeedZ = %d - CellID = %d\n",i,host_geo[i].P.x,host_geo[i].P.y,host_geo[i].P.z,host_geo[i].Speedx,host_geo[i].Speedy,host_geo[i].Speedz,host_geo[i].CellId);
	} */

	/* Device initialization */
	Init_device_random(host_cell, host_geo, device_cell, device_geo, node_number, simulation_custom_parameters);

	/*
	 * Setting of the different senders and receivers. Here the unique sender is 0 and the unique receiver is N-1
	 * Later, this may be replaced by a function. It depends on the scenario we want to apply
	 */

	/* May be doing it directly in the device could be better for the performance */
	if (simulation_parameters.application_config.predefined_traffic.application_type == DEFAULT){
		if (offset == 0)
			host_traffic_table[0] = total_node_number-1;
		else
			host_traffic_table[0] = -1;
		for (int i = 1; i < node_number; i++) {
			host_traffic_table[i] = -1;
		}
	}

	cudaMemcpy(device_traffic_table, host_traffic_table, node_number * sizeof(int),
			cudaMemcpyHostToDevice);

	/* Device buffers initialization */
	Init_Buffer<<<threads,grid>>>(device_out_phy, node_number);
	checkCUDAError("cuda kernel error Init OutBuff");

	Init_Buffer<<<threads,grid>>>(device_in_phy, node_number);
	checkCUDAError("cuda kernel error Init InBuff");

	//Initialization of remote comm buffers in the CPU since the number of these buffers is low
	for (int external_connection = 0; external_connection < ext_conn_nb; external_connection ++) {
		host_remote_comm_out[external_connection].write_index = 0;
		host_remote_comm_out[external_connection].read_index = -1;
		host_remote_comm_out[external_connection].conn_node_id = partition.external_connections[external_connection].node1;
		host_remote_comm_in[external_connection].write_index = 0;
		host_remote_comm_in[external_connection].read_index = -1;
		host_remote_comm_in[external_connection].conn_node_id = partition.external_connections[external_connection].node1;
		for (int i = 0; i < Maxelement; i++) {
			host_remote_comm_out[external_connection].element[i].header[0] = -1;
			host_remote_comm_out[external_connection].element[i].header[1] = -1;
			host_remote_comm_out[external_connection].element[i].header[2] = -1;
			host_remote_comm_out[external_connection].element[i].header[3] = -1;
			host_remote_comm_in[external_connection].element[i].header[0] = -1;
			host_remote_comm_in[external_connection].element[i].header[1] = -1;
			host_remote_comm_in[external_connection].element[i].header[2] = -1;
			host_remote_comm_in[external_connection].element[i].header[3] = -1;
		}
	}
	
	//Then copy to the GPU
	cudaMemcpy(device_remote_comm_out, host_remote_comm_out, ext_conn_nb * sizeof(struct RemoteCommBuffer),
			cudaMemcpyHostToDevice);
	cudaMemcpy(device_remote_comm_in, host_remote_comm_in, ext_conn_nb * sizeof(struct RemoteCommBuffer),
			cudaMemcpyHostToDevice);
			

	Init_App_Buffer<<<threads,grid>>>(device_in_app, node_number);
	checkCUDAError("cuda kernel error Init InAppBuff");

	Init_App_Buffer<<<threads,grid>>>(device_out_app, node_number);
	checkCUDAError("cuda kernel error Init OutAppBuff");

	/* This may only be useful for debugging */
	//cudaMemcpy(host_out_phy,device_out_phy, N * sizeof(struct Buffer), cudaMemcpyDeviceToHost);

	/* When are we going to use it ? */
	Init_Router<<<threads,grid>>>(device_router_buffer, node_number);

	// ?? Only with memcheck ? why ?
	checkCUDAError("cuda kernel error init router");

	/* Initialization of the random number generator */
	curandSetPseudoRandomGeneratorSeed(*gen, time(NULL));
	
	/* Setting of the random initial positions and the random speed */ 	
	curandGenerateUniform ( *gen , device_randx , node_number );
	curandGenerateUniform ( *gen , device_randy , node_number );
	curandGenerateUniform ( *gen , device_randz , node_number );
 	curandGenerateUniform ( *gen , device_randv , node_number );

	//////////////////////////
	// HERE SCHEDULING STEP //
	//////////////////////////

	Schedule (0, 1, (nb_tours + partition.additional_time) / 5, DISTRIB, 1, MOB, nb_tours + partition.additional_time);
	Schedule (0, 2, (nb_tours + partition.additional_time) / 5, DISTRIB, 1, CON, nb_tours + partition.additional_time);
	//To be scheduled before PROTO_OUT absolutely!
	//Schedule (0, 4, 215, DISTRIB, 1, TC_OUT, nb_tours+100);
	//The last APP-level called kernel that is the one which message is kept because for the moment the two APP kernels share the same
	//message buffer and always write their message at the first place

	Schedule (0, 3, nb_tours + partition.additional_time, CONSEC, 1, APP_OUT, nb_tours + partition.additional_time);
	Schedule (0, 5, nb_tours + partition.additional_time, CONSEC, 1, PROTO_OUT, nb_tours + partition.additional_time);
	Schedule (0, 6, nb_tours + partition.additional_time, CONSEC, 1, PKT_OUT, nb_tours + partition.additional_time);
	Schedule (0, 7, nb_tours + partition.additional_time, CONSEC, 1, PKT_IN, nb_tours + partition.additional_time);
	Schedule (0, 8, nb_tours + partition.additional_time, CONSEC, 1, PROTO_IN, nb_tours + partition.additional_time);
	Schedule (0, 9, nb_tours + partition.additional_time, CONSEC, 1, APP_IN, nb_tours + partition.additional_time);

	Calculate_timestamps(simulation_custom_parameters);
	//Print_event_list();

	//////////////////////////

	//for (int i = 0; i < 100; i++) {
	for (int i = 0; i < nb_tours + partition.additional_time; i++) {
		
		struct Event event;
		curandGenerateUniform ( *gen , device_router_prob , node_number );

		// Database dumping
		if (i>0){
			cudaMemcpy(host_geo, device_geo, node_number * sizeof(struct Geo),cudaMemcpyDeviceToHost);
			cudaMemcpy(host_new_message, device_new_message, node_number * sizeof(int),cudaMemcpyDeviceToHost);
			cudaMemcpy(host_recv_new_message, device_recv_new_message, node_number * sizeof(int),cudaMemcpyDeviceToHost);
		}

		for (int node = 0; node < node_number; node++) {
			sprintf(query,"INSERT INTO geo (round, node_id, x, y, z, cell_id, neighbor_number");
			for (int neighbor = 0; neighbor < host_geo[node].neighbor_number; neighbor++) 
				sprintf(query,"%s, N%d", query, neighbor+1);
			sprintf(query,"%s) VALUES ('%d', '%d', '%d', '%d', '%d', '%d', '%d'", query, i, node+offset, host_geo[node].p.x, 
				host_geo[node].p.y, host_geo[node].p.z, host_geo[node].cell_id, host_geo[node].neighbor_number);
			for (int neighbor = 0; neighbor < host_geo[node].neighbor_number; neighbor++) 
				sprintf(query,"%s, '%d'", query, host_geo[node].neighbor_vector[neighbor]);
			sprintf(query, "%s)", query);

			// Adding the line to the table 'geo'			
			process_query (conn,query);

			// Adding a line to the table 'out_pkt'
			if (host_new_message[node] >= 0)
			  sprintf(query,"INSERT INTO out_pkt (round, node_id, sent) VALUES ('%d', '%d', '%d')", i, node+offset, host_new_message[node]);
			else
			  sprintf(query,"INSERT INTO out_pkt (round, node_id) VALUES ('%d', '%d')", i, node+offset);
			process_query (conn,query);

			// Adding a line to the table 'in_pkt'
			if (host_recv_new_message[node] >= 0)
			  sprintf(query,"INSERT INTO in_pkt (round, node_id, received) VALUES ('%d', '%d', '%d')", i, node+offset, host_recv_new_message[node]);
			else
			  sprintf(query,"INSERT INTO in_pkt (round, node_id) VALUES ('%d', '%d')", i, node+offset);
			process_query (conn,query);

		}
				
		while (Next_event_of_the_round (&event, i)){
			//if (commRank == 2) printf("%d is at round %d doing %d\n", commRank, i, event.type);
			switch (event.type){
			
				case MOB:
				for (int j=0; j < event.frequency; j++){
					Mobility<<<threads,grid>>>(device_geo, device_randx, device_randy, device_randz, device_randv, node_number, device_simulation_parameters);
					
					checkCUDAError("cuda kernel error Mob");
					Mobility_Control<<<threads,grid>>>(device_cell,device_geo, node_number, device_simulation_parameters);	
					checkCUDAError("cuda kernel error Mob ctrl");
				}
				break;
				
				case CON:
				for (int j=0; j < event.frequency; j++){
					Update_Cell<<<(cell_number % 32 == 0 ? cell_number/32 : cell_number/32 + 1),32>>>(cell_number, device_cell, node_number);
					checkCUDAError("cuda kernel error Up cell");
					Visible_Opt<<<threads,grid>>>(device_geo, device_cell,node_number,cell_size,
						offset, device_ext_connections, partition.number_of_external_connections);
					checkCUDAError("cuda kernel error Visib");

					//cudaMemcpy(host_cell, device_cell, node_number * sizeof(struct Cell),cudaMemcpyDeviceToHost);
					//for (int ce = 0; ce < cell_number; ce++){
					//	printf("Master %d: cell %d - size = %d - neighbor_nb %d - neighbors: ", commRank, ce, host_cell[ce].size,
					//		host_cell[ce].neighbor_number);
					//	for (int ne=0; ne < host_cell[ce].neighbor_number; ne++)
					//		printf(" [%d,%d,%d]  ", commRank, ce, host_cell[ce].neighbor_vector[ne] );
					//	printf("\n");
					//}
				/*	
					cudaMemcpy(host_geo, device_geo, node_number * sizeof(struct Geo),cudaMemcpyDeviceToHost);
					if (commRank == 1)
						for (int ne = 0; ne < host_geo[0].neighbor_number; ne++) {
							printf("[rd = %d] nb = %d N: %d\n", i, host_geo[0].neighbor_number, host_geo[0].neighbor_vector[ne]);
						}
						
					
					/* 
			 		 * If we would like to test the connectivity step 
			 		 * cudaMemcpy(host_geo, device_geo, node_number * sizeof(struct Geo),cudaMemcpyDeviceToHost);
			 		 * cudaMemcpy(host_cell, device_cell, cell_number * sizeof(struct Cell),cudaMemcpyDeviceToHost);
			 		 * Connectivity_control(host_geo, node_number, i, host_cell, cell_size);
			 		 */	 
				}
				break;
				
				case APP_OUT:
				for (int j=0; j < event.frequency; j++)
					Application_Out<<<threads,grid>>>(device_out_app, device_traffic_table, 3, node_number, i*f+j, nb_tours, offset);
					checkCUDAError("cuda kernel error App out");
				break;
				
				case APP_IN:
				for (int j=0; j < event.frequency; j++)
					Message_In<<<threads,grid>>>(device_in_app, node_number, device_total_dest);
					checkCUDAError("cuda kernel error Msg In");
				break;
				
				case PROTO_OUT:
				for (int j=0; j < event.frequency; j++){
					Router_Out<<<threads,grid>>>(device_out_phy, device_out_app, i*f+j, node_number, offset);
					checkCUDAError("cuda kernel error Router out");
					Reset_Buffer<<<threads,grid>>>(device_in_phy, node_number);
					checkCUDAError("cuda kernel error Reset Buffer");
				}
				break;
				
				case PROTO_IN:
				for (int j=0; j < event.frequency; j++)
					Router_In<<<threads,grid>>>(device_in_phy,device_out_phy,device_in_app,device_router_buffer,device_router_prob, drop_prob, node_number, offset);
					checkCUDAError("cuda kernel error Router In");
				break;
				
				case PKT_OUT:
				for (int j=0; j < event.frequency; j++) {
					
					for (int ext_conn = 0; ext_conn < ext_conn_nb; ext_conn++) {
						host_remote_comm_out[ext_conn].write_index = 0;
					}
			
					cudaMemcpy(device_remote_comm_out, host_remote_comm_out, ext_conn_nb * sizeof(struct RemoteCommBuffer), cudaMemcpyHostToDevice);
					
					Sender<<<threads,grid>>>(device_geo,device_in_phy,device_out_phy, node_number, device_forwarded_per_node, device_new_message, offset, device_remote_comm_out, m_send, b_send);
					checkCUDAError("cuda kernel error Sender");
					cudaMemcpy(host_remote_comm_out, device_remote_comm_out, ext_conn_nb * sizeof(struct RemoteCommBuffer), cudaMemcpyDeviceToHost);
					
					for (int master = 0; master < masters_number; master++) {
						master_out[master].write_index = 0;
					}
			
			
					/* This loop organizes the remote messages in the "masters" local buffers before beginning the 
			   		effective communication with the other masters */   
					//printf("\nRound[%d] - Master = %d - ", i, commRank);
					for (int ext_conn = 0; ext_conn < ext_conn_nb; ext_conn++) {
					//printf("ext_conn = %d - remote_msgs = %d for: ", partition.external_connections[ext_conn].node1, host_remote_comm_out[ext_conn].write_index);
						if (host_remote_comm_out[ext_conn].write_index > 0) {
							for (int remote_msg = 0; remote_msg < host_remote_comm_out[ext_conn].write_index; remote_msg++) {
								int master_addressee = 0;
								int worker_addressee = host_remote_comm_out[ext_conn].ext_conn[remote_msg];
						
								// Looking for the master responsible of the worker addressee
								while (partition.conversion_table[master_addressee] < worker_addressee && master_addressee < masters_number - 1)
									master_addressee++;
								if (partition.conversion_table[master_addressee] > worker_addressee)
									master_addressee--;
							
								//Writing the message in the master's buffer if there is some place...
								if (master_out[master_addressee].write_index < (2 * Maxelement) - 1) {
									int index = master_out[master_addressee].write_index;
									struct Element elt = host_remote_comm_out[ext_conn].element[remote_msg];
							
									//printf("(mast: %d, worker: %d) ",master_addressee + 1, worker_addressee);
									master_out[master_addressee].ext_conn[index] = host_remote_comm_out[ext_conn].ext_conn[remote_msg];
									master_out[master_addressee].element[index] = elt;
									master_out[master_addressee].write_index++;
								}
							}
						}
					}
			
			
					/* Begginning of the real communication */
			
					//Sending step 	
					for (int master = 0; master < masters_number; master++) {
						if (master_out[master].write_index > 0) {
				
							int msg_nb = master_out[master].write_index;
							//TAG 0 ==> there are messages
							ierr = MPI_Send( &master_out[master], 1, mpi_master_buff, master + 1, 0, MPI_COMM_WORLD ) ;
							//ierr = MPI_Send( master_out[master].element, 1, mpi_element, master + 1, 0, MPI_COMM_WORLD ) ; 
		
						} else {
							//TAG 1 ==> No messages
							ierr = MPI_Send( &master_out[master], 1, mpi_master_buff, master + 1, 1, MPI_COMM_WORLD ) ; 
						}
					}
				
					for (int ext_conn = 0; ext_conn < ext_conn_nb; ext_conn++)
						host_remote_comm_in[ext_conn].write_index = 0;
				
					//Receiving step				
					for (int master = 0; master < masters_number; master++) {
					
						ierr = MPI_Recv(&master_in[master], 1, mpi_master_buff, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
						if (status.MPI_TAG == 0) {
							
							//printf("[Round %d] Process %d received %d messages from process %d: the %dth message \n", i, commRank, master_in[master].write_index, status.MPI_SOURCE, master_in[master].element[0].header[2]);
						
							for (int msg = 0; msg < master_in[master].write_index; msg++) {
								int node = master_in[master].ext_conn[msg];
								int ext_conn = 0;
							
								//if (commRank == 2 && i % 20 == 0) printf("node = %d ",node);
							
								while (host_remote_comm_in[ext_conn].conn_node_id != node && ext_conn < partition.number_of_external_connections)
									ext_conn++;
							
								//if (commRank == 2 && i % 20 == 0) printf("ext_conn %d\n",ext_conn);
							
								if (host_remote_comm_in[ext_conn].write_index < Maxelement) {
									host_remote_comm_in[ext_conn].element[host_remote_comm_in[ext_conn].write_index] = master_in[master].element[msg];
									host_remote_comm_in[ext_conn].write_index++;
								}
							}
						
						} //else if (commRank == 2 && i % 20 == 0) printf("[Round %d] Process %d received nothing from process %d\n", i, commRank, status.MPI_SOURCE);
					}
				
					//printf("Before -- Write Index = %d\n",host_remote_comm_in[0].write_index);
					cudaMemcpy(device_remote_comm_in, host_remote_comm_in, ext_conn_nb*sizeof(struct RemoteCommBuffer), cudaMemcpyHostToDevice);
					
				}
				//cudaMemcpy(host_geo, device_geo, node_number * sizeof(struct Geo), cudaMemcpyDeviceToHost);
					//for (int node = 0; node < node_number; node++)
						//printf("Node 0: Energy level = %f with %d neighbors after Send of round %d\n", host_geo[0].energy, host_geo[0].neighbor_number, i);
				break;
				
				case PKT_IN:
				for (int j=0; j < event.frequency; j++){
					//data[0] = i;
					//ierr = MPI_Send ( data, 2, MPI_INT, 0, 0, MPI_COMM_WORLD ) ;
					// Synchronization point. Before receiving any message, we make sure all processes reached this 
					// step, which means that whatever node intending to send a message in this round has finished 
					/* This would be more coherent if we transform all the 6 event - block into a comm_event ? */
					// This seems to make nodes rather too dependent ! if a node sends, the receiver is immediately
					// called to receive the message (the three levels PKT - PROTO - APP)
					// is it tolerable ?

					ierr = MPI_Barrier ( comm_masters );
					Receiver<<<threads,grid>>>(device_in_phy, node_number, device_remote_comm_in, device_geo, offset, m_recv, b_recv, device_recv_new_message);
					checkCUDAError("cuda kernel error Receiver");
				}
				break;
				
				case TC_OUT:
					TC_Out<<<threads,grid>>>(device_out_app, device_traffic_table, node_number, i*f, nb_tours);
					checkCUDAError("cuda kernel error tc out");
				break;
				
			}//switch
			
			/* This is for the validation of the timestamp ==> OK
			switch (event.type){
				case MOB:
				printf("MOB: freq = %d - ts = %f\n",event.frequency, event.timestamp);
				break;
				case CON:
				printf("CON: freq = %d - ts = %f\n",event.frequency, event.timestamp);
				break;
				case APP_OUT:
				printf("APP_OUT: freq = %d - ts = %f\n",event.frequency, event.timestamp);
				break;
				case APP_IN:
				printf("APP_IN: freq = %d - ts = %f\n",event.frequency, event.timestamp);
				break;				
				case PROTO_OUT:
				printf("PROTO_OUT: freq = %d - ts = %f\n",event.frequency, event.timestamp);
				break;
				case PROTO_IN:
				printf("PROTO_IN: freq = %d - ts = %f\n",event.frequency, event.timestamp);
				break;
				case PKT_OUT:
				printf("PKT_OUT: freq = %d - ts = %f\n",event.frequency, event.timestamp);
				break;
				case PKT_IN:
				printf("PKT_IN: freq = %d - ts = %f\n",event.frequency, event.timestamp);
				break;
			}
			*/

			
			
			
		}//while
		
	}//for

	int end;

	ierr = MPI_Send( &end, 1, MPI_INT, 0, 1, MPI_COMM_WORLD ) ;
	
	cudaMemcpy(host_total_dest, device_total_dest, sizeof(int) , cudaMemcpyDeviceToHost);
	cudaMemcpy(host_forwarded_per_node, device_forwarded_per_node, node_number * sizeof(int), cudaMemcpyDeviceToHost);

	int absolute_total_forwarded = 0;
	for (int i=0; i<node_number; i++) {
		absolute_total_forwarded += host_forwarded_per_node[i];

	}

	performance.total_dest = *host_total_dest;
	performance.total_forwarded = absolute_total_forwarded;

	performance.host_used_memory = 
		node_number * 
		      (	4 * sizeof(int) + 
			sizeof(struct Geo) + 
			2 * sizeof(struct Buffer) + 
			2 * sizeof(struct MessageBuffer) +
			2 * sizeof(struct RemoteCommBuffer) +
			7 * sizeof(float) + 
			sizeof(RouterBuffer)  ) +
			cell_number * sizeof(struct Cell) + 
			sizeof(int);

	performance.device_used_memory = 
		node_number * 
		      (	4 * sizeof(int) + 
			sizeof(struct Geo) + 
			2 * sizeof(struct Buffer) + 
			2 * sizeof(struct MessageBuffer) +
			2 * sizeof(struct RemoteCommBuffer) +
			5 * sizeof(float) + 
			sizeof(RouterBuffer)  ) +
			sizeof(int) +
			cell_number * sizeof(struct Cell) + 
			ext_conn_nb * sizeof(struct Connection) +
			sizeof(Simulation_Parameters);


	performance.host_used_memory = node_number*(3*sizeof(int)+sizeof(struct Geo)+2*sizeof(struct Buffer)+2*sizeof(struct MessageBuffer)+7*sizeof(float)+sizeof(RouterBuffer))+sizeof(int)+cell_number*sizeof(struct Cell)+sizeof(Simulation_Parameters);
	performance.device_used_memory = node_number*(3*sizeof(int)+sizeof(struct Geo)+2*sizeof(struct Buffer)+2*sizeof(struct MessageBuffer)+5*sizeof(float)+sizeof(RouterBuffer))+sizeof(int)+cell_number*sizeof(struct Cell)+sizeof(Simulation_Parameters);
	
	mysql_close (conn);

	/* Freeing the memory at the end of computing */

	cudaFree(device_geo);
	cudaFree(device_ext_connections);
	cudaFree(device_remote_comm_out);
	cudaFree(device_remote_comm_in);
	cudaFree(device_new_message);
	cudaFree(device_recv_new_message);
	cudaFree(device_cell);
	cudaFree(device_in_phy);
	cudaFree(device_out_phy);
	cudaFree(device_router_buffer);
	cudaFree(device_router_prob);
	cudaFree(device_in_app);
	cudaFree(device_out_app);
	cudaFree(device_traffic_table);
	cudaFree(device_total_dest);
	cudaFree(device_forwarded_per_node);
	cudaFree(device_randx);
	cudaFree(device_randy);
	cudaFree(device_randz);
	cudaFree(device_randv);
	cudaFree(device_simulation_parameters);

	cudaFreeHost(host_geo);
	cudaFreeHost(host_remote_comm_out);
	cudaFreeHost(host_remote_comm_in);
	cudaFreeHost(host_new_message);
	cudaFreeHost(host_recv_new_message);
	cudaFreeHost(host_cell);
	cudaFreeHost(host_in_phy);
	cudaFreeHost(host_out_phy);
	cudaFreeHost(host_router_buffer);
	cudaFreeHost(host_router_prob);
	cudaFreeHost(host_in_app);
	cudaFreeHost(host_out_app);
	cudaFreeHost(host_traffic_table);
	cudaFreeHost(host_total_dest);
	cudaFreeHost(host_forwarded_per_node);
	cudaFreeHost(host_pos_randx);
	cudaFreeHost(host_pos_randy);	
	cudaFreeHost(host_pos_randz);
	cudaFreeHost(host_v_randx);
	cudaFreeHost(host_v_randy);	
	cudaFreeHost(host_v_randz);
	
	return performance;
}
