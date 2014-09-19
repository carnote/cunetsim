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

/**
 * \fn struct Performances Static_grid(int node_number, int f, float drop_prob, void *genV, int write_descriptor)
 * \brief processes the grid scenario
 *
 * \param node_number is the node number in the simulation
 * \param f is the packet generating frequency = generated packet number per round
 * \param drop_prob probability threshold under which the node drops a message instead of broadcasting it
 * \param genV pointer to the random numbers generator created in the main program
 * \param write_descriptor the file descriptor of the pipe (if it exists) which is the means of communication between this function and the GUI
 * \return void
 */
struct Performances Static_grid(int node_number, int f, float drop_prob,
		void *genV, int write_descriptor) {

	struct Performances performance;
	struct Geo *host_geo, *device_geo;
	struct Buffer *host_out_phy, *host_in_phy, *device_out_phy, *device_in_phy;
	int *host_total_dest, *device_total_dest, *device_forwarded_per_node,
			*host_forwarded_per_node;
	struct RouterBuffer *host_router_buffer, *device_router_buffer;
	struct MessageBuffer *host_out_app, *host_in_app, *device_out_app,
			*device_in_app;
	dim3 threads, grid;
	int *device_traffic_table, *host_traffic_table, *host_new_message, *device_new_message;
	int grid_dimension = (int) sqrt(node_number);

	curandGenerator_t *gen = (curandGenerator_t *)genV;
	int nb_tours = simulation_parameters.simulation_config.simulation_time;

	float m_send = simulation_parameters.environment_config.m_send;
	float m_recv = simulation_parameters.environment_config.m_recv;
	float b_send = simulation_parameters.environment_config.b_send;
	float b_recv = simulation_parameters.environment_config.b_recv;
	
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
	}
	cudaMemcpy(device_forwarded_per_node, host_forwarded_per_node,
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

	/* Host Buffer allocation */
	cudaMallocHost((void**) &host_in_app, node_number * sizeof(struct MessageBuffer));
	checkCUDAError("cudaBufferInMalloc");

	cudaMallocHost((void**) &host_out_app, node_number * sizeof(struct MessageBuffer));
	checkCUDAError("cudaOutAppMalloc");

	cudaMallocHost((void**) &host_in_phy, node_number * sizeof(struct Buffer));
	checkCUDAError("cudaBufferInMalloc");

	cudaMallocHost((void**) &host_out_phy, node_number * sizeof(struct Buffer));
	checkCUDAError("cudaOutPhyMalloc");

	/* Device Buffer memory allocation */
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

	/* Host initialization */
	Init_host_static_grid(host_geo, node_number);

	/* Device initialization */
	Init_device_static_grid(host_geo, device_geo, node_number);

	/*for (int k = 0; k < host_geo[N-1].Neighbors; k++)
		printf("Neighbor[%d] = %d\n",k,host_geo[N-1].Neighbor[k]);

	/*
	 * Setting of the different senders and receivers. Here the unique sender is 0 and the unique receiver is N-1
	 * Later, this may be replaced by a function. It depends on the scenario we want to apply
	 */

	if (simulation_parameters.application_config.predefined_traffic.application_type == DEFAULT){
		host_traffic_table[0] = node_number-1;
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

	Init_App_Buffer<<<threads,grid>>>(device_in_app, node_number);
	checkCUDAError("cuda kernel error Init InAppBuff");

	Init_App_Buffer<<<threads,grid>>>(device_out_app, node_number);
	checkCUDAError("cuda kernel error Init OutAppBuff");

	cudaMemcpy(host_out_phy,device_out_phy, node_number * sizeof(struct Buffer), cudaMemcpyDeviceToHost);

	/* When are we going to use it ? */
	Init_Router<<<threads,grid>>>(device_router_buffer, node_number);

	// ?? Only with memcheck ? why ?
	checkCUDAError("cuda kernel error init router");

	/* Initialization of the random nuber generator */
	curandSetPseudoRandomGeneratorSeed(*gen, time(NULL));

	
	for (int i = 0; i < nb_tours + (2*grid_dimension > 100 ? 2*grid_dimension : 100); i++) {
			
		// If we have to communicate with a GUI
		if (write_descriptor != -3){
			Data_Flow_Unit data;
			if (i>0){
				cudaMemcpy(host_geo, device_geo, node_number * sizeof(struct Geo),cudaMemcpyDeviceToHost);
				cudaMemcpy(host_new_message, device_new_message, node_number * sizeof(int),cudaMemcpyDeviceToHost);
			}
			for (int node = 0; node < NB_DISPLAYED_NODES; node++){ //momentanement
				data.geo[node].neighbor_number = host_geo[node].neighbor_number;
				data.geo[node].p = host_geo[node].p;
				data.new_message[node] = host_new_message[node];
				for(int j = 0; j<host_geo[node].neighbor_number; j++)
                	data.geo[node].neighbor_vector[j] = host_geo[node].neighbor_vector[j];
			}
			data.tour = i;
			if( write( write_descriptor, &data, sizeof(struct Data_Flow_Unit) ) == -1 )
            		perror( "write grid" );
		}


		curandGenerateUniform ( *gen , device_router_prob , node_number );

		for (int j = 0; j < f; j++)

		{

			/* Creation of the message to send by the initial senders (Application Layer) */
			Application_Out<<<threads,grid>>>(device_out_app, device_traffic_table, 3, node_number, i*f+j, nb_tours);

			/* Routing step for the outgoing message */
			Router_Out<<<threads,grid>>>(device_out_phy, device_out_app, i*f+j, node_number);
			checkCUDAError("cuda kernel error Router_Out");

			Reset_Buffer<<<threads,grid>>>(device_in_phy, node_number);
			checkCUDAError("cuda kernel error Reset");

			/* Physical layer step for the outgoing message */
			Sender<<<threads,grid>>>(device_geo,device_in_phy,device_out_phy, node_number, device_forwarded_per_node, device_new_message, m_send, b_send);

			/* Physical layer step for the incoming message */
			Receiver<<<threads,grid>>>(device_in_phy, node_number, m_recv, b_recv, device_geo);			
			checkCUDAError("cuda kernel error Receiver");

			/* Routing step for the incoming message */
			Router_In<<<threads,grid>>>(device_in_phy,device_out_phy,device_in_app,device_router_buffer,device_router_prob, drop_prob, node_number);
			checkCUDAError("cuda kernel error Router");

			/* Receiving the messages by tha application layer */
			Message_In<<<threads,grid>>>(device_in_app, node_number,device_total_dest);

		}

	}

	/* Testing the results */

	cudaMemcpy(host_total_dest, device_total_dest, sizeof(int) , cudaMemcpyDeviceToHost);
	cudaMemcpy(host_forwarded_per_node, device_forwarded_per_node, node_number * sizeof(int), cudaMemcpyDeviceToHost);

	int absolute_total_forwarded = 0;
	for (int i=0; i<node_number; i++) {
		absolute_total_forwarded += host_forwarded_per_node[i];
	}

	performance.total_dest = *host_total_dest;
	performance.total_forwarded = absolute_total_forwarded;
	performance.host_used_memory = node_number*(3*sizeof(int)+sizeof(struct Geo)+2*sizeof(struct Buffer)+2*sizeof(struct MessageBuffer)+sizeof(float)+sizeof(RouterBuffer))+sizeof(int);
	performance.device_used_memory = performance.host_used_memory;

	/* Freeing the memory at the end of computing */

	cudaFree(device_geo);
	cudaFree(device_new_message);
	cudaFree(device_in_phy);
	cudaFree(device_out_phy);
	cudaFree(device_router_buffer);
	cudaFree(device_router_prob);
	cudaFree(device_in_app);
	cudaFree(device_out_app);
	cudaFree(device_traffic_table);
	cudaFree(device_total_dest);
	cudaFree(device_forwarded_per_node);

	cudaFreeHost(host_geo);
	cudaFreeHost(host_new_message);
	cudaFreeHost(host_in_phy);
	cudaFreeHost(host_out_phy);
	cudaFreeHost(host_router_buffer);
	cudaFreeHost(host_router_prob);
	cudaFreeHost(host_in_app);
	cudaFreeHost(host_out_app);
	cudaFreeHost(host_traffic_table);
	cudaFreeHost(host_total_dest);
	cudaFreeHost(host_forwarded_per_node);

	return performance;
}
