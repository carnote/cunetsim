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
 * \file Grid.cu
 * \brief Function that processes the grid scenario
 * \author Bilel BR
 * \version 0.0.2
 * \date Feb 20, 2012
 */


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

/**
 * \fn struct Performances Static_grid(int node_number, int f, float drop_prob, void *genV, int gui)
 * \brief processes the grid scenario
 *
 * \param node_number is the node number in the simulation
 * \param f is the packet generating frequency = generated packet number per round
 * \param drop_prob probability threshold under which the node drops a message instead of broadcasting it
 * \param genV pointer to the random numbers generator created in the main program
 * \param gui has the value 1 is the visualizor is on and 0 if it is off
 * \return void
 */
struct Performances Static_grid(struct Partition partition, int f, float drop_prob,
		void *genV, int gui, MPI_Comm comm_masters, MPI_data_types mpi_types, MYSQL *connection) {

	struct Performances performance;
	struct Geo *host_geo, *device_geo;
	struct Buffer *host_out_phy, *host_in_phy, *device_out_phy, *device_in_phy;
	struct RemoteCommBuffer *host_remote_comm_out, *host_remote_comm_in, *device_remote_comm_out, *device_remote_comm_in;
	struct MasterBuffer *master_in, *master_out;
	int *host_total_dest, *device_total_dest, *device_forwarded_per_node,
			*host_forwarded_per_node;
	struct RouterBuffer *host_router_buffer, *device_router_buffer;
	struct MessageBuffer *host_out_app, *host_in_app, *device_out_app,
			*device_in_app;
	dim3 threads, grid;
	int *device_traffic_table, *host_traffic_table, *host_new_message, *device_new_message, *host_recv_new_message, *device_recv_new_message;
	int node_number = partition.node_number;
	int offset = partition.offset;
	int total_node_number = partition.total_node_number;
	int grid_dimension = (int) sqrt(node_number);
	int ext_conn_nb = partition.number_of_external_connections;

	curandGenerator_t *gen = (curandGenerator_t *)genV;
	int nb_tours = simulation_parameters.simulation_config.simulation_time;
	int ierr;	
	int masters_number = partition.masters_number;
	MPI_Status status;

	float m_send = simulation_parameters.environment_config.m_send;
	float m_recv = simulation_parameters.environment_config.m_recv;
	float b_send = simulation_parameters.environment_config.b_send;
	float b_recv = simulation_parameters.environment_config.b_recv;

	// Database variables
	MYSQL *conn = connection;

	char query[4096];
	
    //Get back MPI types
	MPI_Datatype mpi_master_buff = mpi_types.mpi_master_buff;
	MPI_Datatype mpi_data_flow = mpi_types.mpi_data_flow;
	
    
	/* Verification of the value of node_number */
	if (pow(sqrt(node_number), 2) != node_number) {
		printf("Error: %d is not the square of an integer!\n",node_number);
		exit(0);
	}

	grid.x = 32;

	/* Test of the node number to decide of the number and dimension of thread blocks */
	if (node_number % 32 == 0 && node_number > 0) {

		threads.x = node_number / 32;
	} else {
		threads.x = (node_number / 32) + 1;
	}

	/* Will contain different probabilities that are going to be used for the routing of the messages */
	float *device_router_prob, *host_router_prob;

	/* These parameters will allow us to analyze some features (comment to be changed) */
	cudaMallocHost((void**) &host_total_dest, sizeof(int));
	checkCUDAError("cudaBufferInMalloc");

	cudaMallocHost((void**) &host_new_message, node_number * sizeof(int));
	checkCUDAError("cudaNewMessageMalloc");

	cudaMallocHost((void**) &host_recv_new_message, node_number * sizeof(int));
	checkCUDAError("cudaRecvNewMessageMalloc");
	
	cudaMallocHost((void**) &host_forwarded_per_node, node_number * sizeof(int));
	checkCUDAError("cudaForwarded_Per_NodeMalloc");

	cudaMalloc((void**) &device_total_dest, sizeof(int));
	checkCUDAError("cudaDeviceBufferInMalloc");

	cudaMalloc((void**) &device_forwarded_per_node, node_number * sizeof(int));
	checkCUDAError("cudaDeviceForwarded_Per_NodeMalloc");

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

	/* Geo host memory allocation */
	cudaMallocHost((void**) &host_geo, node_number * sizeof(struct Geo));
	checkCUDAError("cudaGeoMalloc");

	/* Geo Device memory allocation */
	cudaMalloc((void**) &device_geo, node_number * sizeof(struct Geo));
	checkCUDAError("cudaDeviceGeoMalloc");

	cudaMalloc((void**) &device_new_message, node_number * sizeof(int));
	checkCUDAError("cudaDeviceNewMessageMalloc");

	cudaMalloc((void**) &device_recv_new_message, node_number * sizeof(int));
	checkCUDAError("cudaDeviceRecvNewMessageMalloc");

	/* Host Buffer allocation */
	cudaMallocHost((void**) &host_in_app, node_number * sizeof(struct MessageBuffer));
	checkCUDAError("cudaBufferInMalloc");

	cudaMallocHost((void**) &host_out_app, node_number * sizeof(struct MessageBuffer));
	checkCUDAError("cudaOutAppMalloc");

	cudaMallocHost((void**) &host_in_phy, node_number * sizeof(struct Buffer));
	checkCUDAError("cudaBufferInMalloc");

	cudaMallocHost((void**) &host_out_phy, node_number * sizeof(struct Buffer));
	checkCUDAError("cudaOutPhyMalloc");
	
	cudaMallocHost((void**) &host_remote_comm_out, ext_conn_nb * sizeof(struct RemoteCommBuffer));
	checkCUDAError("cudaRemCommOutMalloc");
	
	cudaMallocHost((void**) &host_remote_comm_in, ext_conn_nb * sizeof(struct RemoteCommBuffer));
	checkCUDAError("cudaRemCommInMalloc");

	/* Device Buffer memory allocation */
	cudaMalloc((void**) &device_remote_comm_out, ext_conn_nb * sizeof(struct RemoteCommBuffer));
	checkCUDAError("cudaDeviceRemCommOutMalloc");
	
	cudaMalloc((void**) &device_remote_comm_in, ext_conn_nb * sizeof(struct RemoteCommBuffer));
	checkCUDAError("cudaDeviceRemCommInMalloc");
	
	cudaMalloc((void**) &device_in_phy, node_number * sizeof(struct Buffer));
	checkCUDAError("cudaDeviceBufferInMalloc");

	cudaMalloc((void**) &device_out_phy, node_number * sizeof(struct Buffer));
	checkCUDAError("cudaDeviceOutPhyMalloc");

	cudaMalloc((void**) &device_in_app, node_number * sizeof(struct MessageBuffer));
	checkCUDAError("cudaDeviceBufferInMalloc");

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
	Init_host_static_grid(host_geo, node_number, offset, partition.external_connections, partition.number_of_external_connections);

	/* Device initialization */
	Init_device_static_grid(host_geo, device_geo, node_number);

	/*for (int k = 0; k < host_geo[N-1].Neighbors; k++)
		printf("Neighbor[%d] = %d\n",k,host_geo[N-1].Neighbor[k]);

	/*
	 * Setting of the different senders and receivers. Here the unique sender is 0 and the unique receiver is N-1
	 * Later, this may be replaced by a function. It depends on the scenario we want to apply
	 */

	// To ensure the traffic is the same as in the seq version : 0 ==> N-1
	if (simulation_parameters.application_config.predefined_traffic.application_type == DEFAULT){
		if (offset == 0)
			host_traffic_table[0] = total_node_number-1;
		else
			host_traffic_table[0] = -1;
		for (int i = 1; i < node_number; i++) {
			host_traffic_table[i] = -1;
		}
	}
	//printf("Le destinataire est: %d\n",host_traffic_table[0]);

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

	//cudaMemcpy(host_out_phy,device_out_phy, node_number * sizeof(struct Buffer), cudaMemcpyDeviceToHost);

	/* When are we going to use it ? */
	Init_Router<<<threads,grid>>>(device_router_buffer, node_number);

	// ?? Only with memcheck ? why ?
	checkCUDAError("cuda kernel error init router");

	/* Initialization of the random nuber generator */
	curandSetPseudoRandomGeneratorSeed(*gen, time(NULL));
	
	// Finding the rank of the process
	int commRank;
	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &commRank);

	
	for (int i = 0; i < nb_tours + partition.additional_time; i++) {


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


		curandGenerateUniform ( *gen , device_router_prob , node_number );

		for (int j = 0; j < f; j++)

		{

			/* Creation of the message to send by the initial senders (Application Layer) */
			Application_Out<<<threads,grid>>>(device_out_app, device_traffic_table, 3, node_number, i*f+j, nb_tours, offset);

			/* Routing step for the outgoing message */
			Router_Out<<<threads,grid>>>(device_out_phy, device_out_app, i*f+j, node_number, offset);
			checkCUDAError("cuda kernel error Router_Out");

			Reset_Buffer<<<threads,grid>>>(device_in_phy, node_number);
			checkCUDAError("cuda kernel error Reset");

			for (int ext_conn = 0; ext_conn < ext_conn_nb; ext_conn++) {
				host_remote_comm_out[ext_conn].write_index = 0;
			}
			
			cudaMemcpy(device_remote_comm_out, host_remote_comm_out, ext_conn_nb * sizeof(struct RemoteCommBuffer), cudaMemcpyHostToDevice);

			/* Physical layer step for the outgoing message */
			Sender<<<threads,grid>>>(device_geo,device_in_phy,device_out_phy, node_number, device_forwarded_per_node, device_new_message, offset, device_remote_comm_out, m_send, b_send);
			cudaMemcpy(host_remote_comm_out, device_remote_comm_out, ext_conn_nb * sizeof(struct RemoteCommBuffer), cudaMemcpyDeviceToHost);
			
			/*if (host_remote_comm_out[0].write_index > 0) {
					printf("\nRound %d - Node %d - Remote messages (%d): ", i, host_remote_comm_out[0].conn_node_id, host_remote_comm_out[0].write_index);
					for (int h = 0; h < host_remote_comm_out[0].write_index; h++) {
						printf("seq_nb = %d ", host_remote_comm_out[0].element[h].header[2]);
					}
			}*/
			
			/* Initialization of the write_index at the master's buffer level */
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
			
			checkCUDAError("cuda kernel error Sender");

			/* Synchronization between masters */ 			
			ierr = MPI_Barrier ( comm_masters );

			/* Physical layer step for the incoming message */
			Receiver<<<threads,grid>>>(device_in_phy, node_number, device_remote_comm_in, device_geo, offset, m_recv, b_recv, device_recv_new_message);
			checkCUDAError("cuda kernel error Receiver");

			/* Routing step for the incoming message */
			Router_In<<<threads,grid>>>(device_in_phy,device_out_phy,device_in_app,device_router_buffer,device_router_prob, drop_prob, node_number, offset);
			checkCUDAError("cuda kernel error Router");

			/* Receiving the messages by tha application layer */
			Message_In<<<threads,grid>>>(device_in_app, node_number,device_total_dest);

		}

	}

	int end;

	ierr = MPI_Send( &end, 1, MPI_INT, 0, 1, MPI_COMM_WORLD ) ;

	/* Testing the results */

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
			sizeof(RouterBuffer)  ) + 
			sizeof(int) + 
			sizeof(Simulation_Parameters);

	performance.device_used_memory = performance.host_used_memory;

	/* Freeing the memory at the end of computing */

	free(master_in);
	free(master_out);

	cudaFree(device_geo);
	cudaFree(device_new_message);
	cudaFree(device_recv_new_message);
	cudaFree(device_in_phy);
	cudaFree(device_out_phy);
	cudaFree(device_remote_comm_out);
	cudaFree(device_remote_comm_in);
	cudaFree(device_router_buffer);
	cudaFree(device_router_prob);
	cudaFree(device_in_app);
	cudaFree(device_out_app);
	cudaFree(device_traffic_table);
	cudaFree(device_total_dest);
	cudaFree(device_forwarded_per_node);

	cudaFreeHost(host_geo);
	cudaFreeHost(host_new_message);
	cudaFreeHost(host_recv_new_message);
	cudaFreeHost(host_in_phy);
	cudaFreeHost(host_out_phy);
	cudaFreeHost(host_remote_comm_out);
	cudaFreeHost(host_remote_comm_in);
	cudaFreeHost(host_router_buffer);
	cudaFreeHost(host_router_prob);
	cudaFreeHost(host_in_app);
	cudaFreeHost(host_out_app);
	cudaFreeHost(host_traffic_table);
	cudaFreeHost(host_total_dest);
	cudaFreeHost(host_forwarded_per_node);

	return performance;
}
