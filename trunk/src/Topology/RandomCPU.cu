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
 * \file RandomCPU.cu
 * \brief Functions necessary to process the CPU random process (under construction)
 * \author Med MBK
 * \version 0.0.2
 * \date Apr 24, 2012
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
/////////////////////
//ajouter la generation de nmbres aleatoire avec rand();




/**
 * \fn struct Performances Random_cpu(int node_number, int f, float drop_prob, int write_descriptor)
 * \brief processes the random scenario when run on the CPU
 *
 * \param node_number is the node number in the simulation
 * \param f is the packet generating frequency = generated packet number per round
 * \param drop_prob probability threshold under which the node drops a message instead of broadcasting it
 * \param write_descriptor the file descriptor of the pipe (if it exists) which is the means of communication between this function and the GUI
 * \return void
 */
struct Performances Random_cpu(int node_number, int f, float drop_prob, int write_descriptor) {

	
	
	struct Cell *host_cell;
	float *host_pos_randx, *host_pos_randy, *host_pos_randz, *host_v_randx, *host_v_randy, *host_v_randz;
	struct Performances performance;
	struct Geo *host_geo;
	struct Buffer *host_out_phy, *host_in_phy;
	int *host_total_dest, *host_forwarded_per_node, *host_new_message;
	struct RouterBuffer *host_router_buffer;
	struct MessageBuffer *host_out_app, *host_in_app;
	int *host_traffic_table;
	
	int cell_number = simulation_parameters.topology_config.area.geo_cell.cell_number;
	int nb_tours = simulation_parameters.simulation_config.simulation_time;
	int cell_size = simulation_parameters.topology_config.area.geo_cell.cell_size_m;
	

	/* Will contain different probabilities that are going to be used for the routing of the messages */
	float *host_router_prob;

	/* These parameters will allow us to analyze some features (comment to be changed) */
	cudaMallocHost((void**) &host_total_dest, sizeof(int));
	checkCUDAError("cudaBufferInMalloc");

	cudaMallocHost((void**) &host_new_message, node_number * sizeof(int));
	checkCUDAError("cudaNewMessageMalloc");

	cudaMallocHost((void**) &host_forwarded_per_node, node_number * sizeof(int));
	checkCUDAError("cudaBufferOutMalloc");

	/* Initialization of the parameters */
	*host_total_dest = 0;

	for (int i = 0; i < node_number; i++) {
		host_forwarded_per_node[i] = 0;
		host_new_message[i] = -1;
	}

	/* Receivers Host memory allocation */
	cudaMallocHost((void**) &host_traffic_table, node_number * sizeof(int));
	checkCUDAError("cudaRecMalloc");

	/* Cell host memory allocation */
	cudaMallocHost((void**) &host_cell, cell_number * sizeof(struct Cell));
	checkCUDAError("cudaCellMalloc");

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
	
	/* Geo host memory allocation */
	cudaMallocHost((void**) &host_geo, node_number * sizeof(struct Geo));
	checkCUDAError("cudaGeoMalloc");

	/* Host Buffer allocation */
	cudaMallocHost((void**) &host_in_app, node_number * sizeof(struct MessageBuffer));
	checkCUDAError("cudaBufferInMalloc");

	cudaMallocHost((void**) &host_out_app, node_number * sizeof(struct MessageBuffer));
	checkCUDAError("cudaBufferOutMalloc");

	cudaMallocHost((void**) &host_in_phy, node_number * sizeof(struct Buffer));
	checkCUDAError("cudaBufferInMalloc");

	cudaMallocHost((void**) &host_out_phy, node_number * sizeof(struct Buffer));
	checkCUDAError("cudaBufferOutMalloc");

	cudaMallocHost((void **) &host_router_buffer,
			node_number * sizeof(struct RouterBuffer));
	checkCUDAError("cudaRouterBufferMalloc");

	cudaMallocHost((void **) &host_router_prob, node_number * sizeof(float));
	checkCUDAError("cudaBufferProbMalloc");

	/* Host initialization */
	Init_host_random(host_cell, host_geo, host_pos_randx, host_pos_randy,
 	host_pos_randz, host_v_randx, host_v_randy, host_v_randz, node_number);

	/****************************/
	/* Testing Geo values 

	printf("N = %d\n",N);
	for (int i = 0; i < N; i++){
		printf("Node = %d - PosX = %d - PosY = %d - PosZ = %d - SpeedX = %d - SpeedY = %d - SpeedZ = %d - CellID = %d\n",i,host_geo[i].P.x,host_geo[i].P.y,host_geo[i].P.z,host_geo[i].Speedx,host_geo[i].Speedy,host_geo[i].Speedz,host_geo[i].CellId);
	} 

	/*
	 * Setting of the different senders and receivers. Here the unique sender is 0 and the unique receiver is N-1
	 * Later, this may be replaced by a function. It depends on the scenario we want to apply
	 */


	/* May be doing it directly in the device could be better for the performance */
	if (simulation_parameters.application_config.predefined_traffic.application_type == DEFAULT){
		host_traffic_table[0] = node_number-1;		
		for (int i = 1; i < node_number; i++) {
			host_traffic_table[i] = -1;
		}
	}

	

	/* Device buffers initialization 
	Init_Buffer<<<threads,grid>>>(device_out_phy, node_number);
	checkCUDAError("cuda kernel error Init OutBuff");

	Init_Buffer<<<threads,grid>>>(device_in_phy, node_number);
	checkCUDAError("cuda kernel error Init InBuff");

	Init_App_Buffer<<<threads,grid>>>(device_in_app, node_number);
	checkCUDAError("cuda kernel error Init InAppBuff");

	Init_App_Buffer<<<threads,grid>>>(device_out_app, node_number);
	checkCUDAError("cuda kernel error Init OutAppBuff");*/


	/* When are we going to use it ? 
	Init_Router<<<threads,grid>>>(device_router_buffer, node_number);*/


	for (int i = 0; i < nb_tours + 100; i++) {

		// If we have to communicate with a GUI
		if (write_descriptor != -3){
			Data_Flow_Unit data;
			
			for (int node = 0; node < NB_DISPLAYED_NODES; node++){
				data.geo[node].neighbor_number = host_geo[node].neighbor_number;
				data.geo[node].p = host_geo[node].p;
				data.new_message[node] = host_new_message[node];
				for(int j = 0; j<host_geo[node].neighbor_number; j++)
                        		data.geo[node].neighbor_vector[j] = host_geo[node].neighbor_vector[j];
			}
			data.tour = i;
			if( write( write_descriptor, &data, sizeof(struct Data_Flow_Unit) ) == -1 )
            		perror( "write random" );
		}
		
		if (i % 5 == 0){
		
			for (int node = 0; node < node_number; node++)
				Mobility(host_geo, host_pos_randx, host_pos_randy, host_pos_randz, node, &simulation_parameters);
			
		/*	Mobility<<<threads,grid>>>(device_geo, device_randx, device_randy, device_randz, device_randv, node_number, device_simulation_parameters);
			Mobility_Control<<<threads,grid>>>(device_cell,device_geo, node_number, device_simulation_parameters);
			Update_Cell<<<(cell_number % 32 == 0 ? cell_number/32 : cell_number/32 + 1),32>>>(cell_number, device_cell, node_number);
			Visible<<<threads,grid>>>(device_geo, device_cell,node_number,cell_size);*/
			
			/* 
			 * If we would like to test the connectivity step 
			 *
			 * cudaMemcpy(host_geo, device_geo, N * sizeof(struct Geo),cudaMemcpyDeviceToHost);
			 * cudaMemcpy(host_cell, device_cell, cell_number * sizeof(struct Cell),cudaMemcpyDeviceToHost);
			 * Connectivity_control(host_geo, N, i, host_cell);
			 *
			 */
		}


		for (int j = 0; j < f; j++)

		{
			/* Creation of the message to send by the initial senders (Application Layer) 
			Application_Out<<<threads,grid>>>(device_out_app, device_traffic_table, 3, node_number, i*f+j, nb_tours);
			
			/* Routing step for the outgoing message 
			Router_Out<<<threads,grid>>>(device_out_phy, device_out_app, i*f+j, node_number);
			checkCUDAError("cuda kernel error Router_Out");
			
			Reset_Buffer<<<threads,grid>>>(device_in_phy, node_number);
			checkCUDAError("cuda kernel error Reset");

			/* Physical layer step for the outgoing message 
			Sender<<<threads,grid>>>(device_geo,device_in_phy,device_out_phy, node_number, device_forwarded_per_node, device_new_message);
			checkCUDAError("cuda kernel error Sender");

			/* Physical layer step for the incoming message 
			Receiver<<<threads,grid>>>(device_in_phy, node_number);
			checkCUDAError("cuda kernel error Receiver");

			/* Routing step for the incoming message 
			Router_In<<<threads,grid>>>(device_in_phy,device_out_phy,device_in_app,device_router_buffer,device_router_prob, drop_prob, node_number);
			checkCUDAError("cuda kernel error Router");

			/* Receiving the messages by tha application layer 
			Application_In<<<threads,grid>>>(device_in_app, node_number, device_total_dest);*/
		}

	}

	/* Testing the results */


	int absolute_total_forwarded = 0;
	for (int i=0; i<node_number; i++) {
		absolute_total_forwarded += host_forwarded_per_node[i];
	}

	performance.total_dest = *host_total_dest;
	performance.total_forwarded = absolute_total_forwarded;
	performance.host_used_memory = node_number*(3*sizeof(int)+sizeof(struct Geo)+2*sizeof(struct Buffer)+2*sizeof(struct MessageBuffer)+7*sizeof(float)+sizeof(RouterBuffer))+sizeof(int)+cell_number*sizeof(struct Cell)+sizeof(Simulation_Parameters);
	performance.device_used_memory = 0;
	

	/* Freeing the memory at the end of computing */
	cudaFreeHost(host_geo);
	cudaFreeHost(host_new_message);
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
