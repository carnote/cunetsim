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

struct Performances Random_sched_try(int node_number, int f, float drop_prob,
		void *genV, int write_descriptor) {
	
	struct Cell *host_cell, *device_cell;
	float *host_pos_randx, *host_pos_randy, *host_pos_randz, *host_v_randx, *host_v_randy, *host_v_randz;
	float *device_randx, *device_randy, *device_randz, *device_randv;
	struct Performances performance;
	struct Cluster *host_cluster, *device_cluster;
	struct Geo *host_geo, *device_geo;
	struct Buffer *host_out_phy, *host_in_phy, *device_out_phy, *device_in_phy;
	int *host_total_dest, *device_total_dest, *device_forwarded_per_node,
			*host_forwarded_per_node, *host_new_message, *device_new_message;
	struct RouterBuffer *host_router_buffer, *device_router_buffer;
	struct MessageBuffer *host_out_app, *host_in_app, *device_out_app,
			*device_in_app;
	dim3 threads, grid;
	int *device_traffic_table, *host_traffic_table;
	Simulation_Parameters *device_simulation_parameters;
	 

	int cell_number = simulation_parameters.topology_config.area.geo_cell.cell_number;
	int nb_tours = simulation_parameters.simulation_config.simulation_time;
	int cell_size = simulation_parameters.topology_config.area.geo_cell.cell_size_m;
	
	float m_send = simulation_parameters.environment_config.m_send;
	float m_recv = simulation_parameters.environment_config.m_recv;
	float b_send = simulation_parameters.environment_config.b_send;
	float b_recv = simulation_parameters.environment_config.b_recv;
	
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

	cudaMemcpy(device_simulation_parameters, &simulation_parameters, sizeof(Simulation_Parameters),
			cudaMemcpyHostToDevice);

	/* These parameters will allow us to analyze some features (comment to be changed) */
	cudaMallocHost((void**) &host_total_dest, sizeof(int));
	checkCUDAError("cudaBufferInMalloc");

	cudaMallocHost((void**) &host_new_message, node_number * sizeof(int));
	checkCUDAError("cudaNewMessageMalloc");

	cudaMallocHost((void**) &host_forwarded_per_node, node_number * sizeof(int));
	checkCUDAError("cudaBufferOutMalloc");

	cudaMalloc((void**) &device_total_dest, sizeof(int));
	checkCUDAError("cudaDeviceTotalDestMalloc");

	cudaMalloc((void**) &device_forwarded_per_node, node_number * sizeof(int));
	checkCUDAError("cudaDeviceForwardedPerNodeMalloc");

	cudaMalloc((void**) &device_new_message, node_number * sizeof(int));
	checkCUDAError("cudaDeviceNewMessageMalloc");

	cudaMalloc((void**) &device_cluster, node_number * sizeof(struct Cluster));
	checkCUDAError("cudaDeviceClusterMalloc");

	cudaMallocHost((void**) &host_cluster, node_number * sizeof(struct Cluster));
	checkCUDAError("cudaClusterMalloc");

	/* Initialization of the parameters */
	*host_total_dest = 0;

	cudaMemcpy(device_total_dest, host_total_dest, sizeof(int),
			cudaMemcpyHostToDevice);

	for (int i = 0; i < node_number; i++) {
		host_forwarded_per_node[i] = 0;
		host_new_message[i] = -1;
	}
	cudaMemcpy(device_forwarded_per_node, host_forwarded_per_node,
			node_number * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(device_new_message, host_new_message,
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

	/* Host initialization */
	Init_host_random(host_cell, host_geo, host_pos_randx, host_pos_randy,
 	host_pos_randz, host_v_randx, host_v_randy, host_v_randz, node_number);

	/****************************/
	/* Testing Geo values 

	printf("N = %d\n",N);
	for (int i = 0; i < N; i++){
		printf("Node = %d - PosX = %d - PosY = %d - PosZ = %d - SpeedX = %d - SpeedY = %d - SpeedZ = %d - CellID = %d\n",i,host_geo[i].P.x,host_geo[i].P.y,host_geo[i].P.z,host_geo[i].Speedx,host_geo[i].Speedy,host_geo[i].Speedz,host_geo[i].CellId);
	} */

	/* Device initialization */
	Init_device_random(host_cell, host_geo, device_cell, device_geo, node_number);

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

	Init_Cluster<<<threads,grid>>>(device_cluster);

//////////////////////////
// HERE SCHEDULING STEP //
//////////////////////////

Schedule (0, 1, 140, DISTRIB, 1, MOB, nb_tours+100);
Schedule (0, 2, 140, DISTRIB, 1, CON, nb_tours+100);
//To be scheduled before PROTO_OUT absolutely!
//Schedule (0, 4, 215, DISTRIB, 1, TC_OUT, nb_tours+100);
//The last APP-level called kernel that is the one which message is kept because for the moment the two APP kernels share the same
//message buffer and always write their message at the first place
//Schedule (0, 3, 700, CONSEC, 1, CLSTR, nb_tours+100);

Schedule (0, 4, 700, CONSEC, 1, APP_OUT, nb_tours+100);
Schedule (0, 5, 700, CONSEC, 1, PROTO_OUT, nb_tours+100);
Schedule (0, 6, 700, CONSEC, 1, PKT_OUT, nb_tours+100);
Schedule (0, 7, 700, CONSEC, 1, PKT_IN, nb_tours+100);
Schedule (0, 8, 700, CONSEC, 1, PROTO_IN, nb_tours+100);
Schedule (0, 9, 700, CONSEC, 1, APP_IN, nb_tours+100);

Calculate_timestamps();
//Print_event_list();

//////////////////////////

	//for (int i = 0; i < 20; i++) {
	for (int i = 0; i < nb_tours + 100; i++) {

		struct Event event;
		curandGenerateUniform ( *gen , device_router_prob , node_number );

		// If we have to communicate with a GUI
		if (write_descriptor != -3){
			Data_Flow_Unit data;
			if (i>0){
				cudaMemcpy(host_geo, device_geo, node_number * sizeof(struct Geo),cudaMemcpyDeviceToHost);
				cudaMemcpy(host_cluster, device_cluster, node_number * sizeof(struct Cluster),cudaMemcpyDeviceToHost);
				cudaMemcpy(host_new_message, device_new_message, node_number * sizeof(int),cudaMemcpyDeviceToHost);
			}
			for (int node = 0; node < NB_DISPLAYED_NODES; node++){
				data.geo[node].neighbor_number = host_geo[node].neighbor_number;
				data.geo[node].p = host_geo[node].p;
				data.new_message[node] = host_new_message[node];
				for(int j = 0; j<host_geo[node].neighbor_number; j++) {
                        		data.geo[node].neighbor_vector[j] = host_geo[node].neighbor_vector[j];
					data.role[node] = host_cluster[node].noderole;
				}	
			}
			data.tour = i;
			if( write( write_descriptor, &data, sizeof(struct Data_Flow_Unit) ) == -1 )
            		perror( "write random" );
		}
		
		while (Next_event_of_the_round (&event, i)){
		
			switch (event.type){
			
				case MOB:
				for (int j=0; j < event.frequency; j++){
					Mobility<<<threads,grid>>>(device_geo, device_randx, device_randy, device_randz, device_randv, node_number, device_simulation_parameters);					

					Mobility_Control<<<threads,grid>>>(device_cell,device_geo, node_number, device_simulation_parameters);
				}
				break;
				
				case CON:
				for (int j=0; j < event.frequency; j++){
					Update_Cell<<<(cell_number % 32 == 0 ? cell_number/32 : cell_number/32 + 1),32>>>(cell_number, device_cell, node_number);
					Visible_Opt<<<threads,grid>>>(device_geo, device_cell,node_number,cell_size);
				}
				break;
				
				case APP_OUT:
				for (int j=0; j < event.frequency; j++)
					Application_Out<<<threads,grid>>>(device_out_app, device_traffic_table, 3, node_number, i*f+j, nb_tours);
				break;
				
				case APP_IN:
				for (int j=0; j < event.frequency; j++)
					Message_In<<<threads,grid>>>(device_in_app, node_number, device_total_dest);
				break;
				
				case PROTO_OUT:
				for (int j=0; j < event.frequency; j++){
					Router_Out<<<threads,grid>>>(device_out_phy, device_out_app, i*f+j, node_number);
					Reset_Buffer<<<threads,grid>>>(device_in_phy, node_number);
				}
				break;
				
				case PROTO_IN:
				for (int j=0; j < event.frequency; j++)
					Router_In<<<threads,grid>>>(device_in_phy,device_out_phy,device_in_app,device_router_buffer,device_router_prob, drop_prob, node_number);
				break;
				
				case PKT_OUT:
				for (int j=0; j < event.frequency; j++)
					Sender<<<threads,grid>>>(device_geo,device_in_phy,device_out_phy, node_number, device_forwarded_per_node, device_new_message, m_send, b_send);
					//cudaMemcpy(host_geo, device_geo, node_number * sizeof(struct Geo), cudaMemcpyDeviceToHost);
					//for (int node = 0; node < node_number; node++)
						//printf("Node 0: Energy level = %f with %d neighbors after Send of round %d\n", host_geo[0].energy, host_geo[0].neighbor_number, i);
				break;
				
				case PKT_IN:
				for (int j=0; j < event.frequency; j++)
					Receiver<<<threads,grid>>>(device_in_phy, node_number, m_recv, b_recv, device_geo);
					//cudaMemcpy(host_geo, device_geo, node_number * sizeof(struct Geo), cudaMemcpyDeviceToHost);
					//for (int node = 0; node < node_number; node++)
						//printf("Node 0: Energy level = %f after Recv of round %d\n", host_geo[0].energy, i);
				break;
				
				case TC_OUT:
				TC_Out<<<threads,grid>>>(device_out_app, device_traffic_table, node_number, i*f, nb_tours);
				break;

				case CLSTR:
				Clustering<<<threads,grid>>>(device_geo, device_cluster, node_number);
				Clustering2<<<threads,grid>>>(device_geo, device_cluster, node_number);
				Clustering3<<<threads,grid>>>(device_geo, device_cluster, node_number);
				cudaMemcpy(host_geo, device_geo, node_number * sizeof(struct Geo), cudaMemcpyDeviceToHost);
				cudaMemcpy(host_cluster, device_cluster, node_number * sizeof(struct Cluster), cudaMemcpyDeviceToHost);
				Print_cluster(host_geo, host_cluster, i, i, node_number);
				break;
				
			}
			
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
		}
		
	}
	
	cudaMemcpy(host_total_dest, device_total_dest, sizeof(int) , cudaMemcpyDeviceToHost);
	cudaMemcpy(host_forwarded_per_node, device_forwarded_per_node, node_number * sizeof(int), cudaMemcpyDeviceToHost);

	int absolute_total_forwarded = 0;
	for (int i=0; i<node_number; i++) {
		absolute_total_forwarded += host_forwarded_per_node[i];

	}

	performance.total_dest = *host_total_dest;
	performance.total_forwarded = absolute_total_forwarded;
	performance.host_used_memory = node_number*(3*sizeof(int)+sizeof(struct Geo)+2*sizeof(struct Buffer)+2*sizeof(struct MessageBuffer)+7*sizeof(float)+sizeof(RouterBuffer))+sizeof(int)+cell_number*sizeof(struct Cell)+sizeof(Simulation_Parameters);
	performance.device_used_memory = node_number*(3*sizeof(int)+sizeof(struct Geo)+2*sizeof(struct Buffer)+2*sizeof(struct MessageBuffer)+5*sizeof(float)+sizeof(RouterBuffer))+sizeof(int)+cell_number*sizeof(struct Cell)+sizeof(Simulation_Parameters);
	
	

	/* Freeing the memory at the end of computing */

	cudaFree(device_geo);
	cudaFree(device_new_message);
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
	cudaFree(device_cluster);

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
	cudaFreeHost(host_cluster);
	
	return performance;
}
