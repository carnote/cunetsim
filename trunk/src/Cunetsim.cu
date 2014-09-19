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
 * \file Cunetsim.cu
 * \brief Program that launches the scenario, displays results and finishes the simulation
 * \author Bilel BR
 * \version 0.0.2
 * \date 
 */
 
 
// Host defines
#define NUM_THREADS 8
#define STR_SIZE 50

// Includes
#include <time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include "structures.h"
#include "/usr/local/cuda/include/curand.h"
#endif /* STRUCTURES_H_ */
#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "interfaces.h"
#endif /* INTERFACES_H_ */

#include "vars.h"

//#define _control_
//#define _printgraph_
//#define __timing2__

//#define __timing__
#define __mobonly__

/* global structure */
extern Simulation_Parameters simulation_parameters;
float kernel_duration[700][14];
float accumulated_measure[14];

void Print_help(){
	printf("\n*****************************************************************************************\n");
	printf("****************************************  HELP  *****************************************\n\n");
	printf("Usage: ./Cunetsim [-h] [-d device] [-g gui_pipe_descriptor]\n");
	printf ("-h provides this help message!\n");
  	printf ("-d [0-device_number] sets the current device\n");	
  	printf ("-g descriptor sets the file descriptor of the communication pipe\n(the communication is between the simulator and the GUI)\n");
  	printf("\n*****************************************************************************************\n");
}

 /**
 * \fn int main(int argc, char** argv)
 * \brief calls the wanted scenario function and gets back the monitoring data
 *
 * \param argc number of arguments passed to the program
 * \param argv vector containing the arguments passed to the program
 * \return 0 if all went well
 */
int main(int argc, char** argv) {
	
	cudaEvent_t start, stop;// timers to compute the runtime
	curandGenerator_t gen;	// seed for the random generator
	float elapsed, total_elapsed = 0.0, total_dest = 0.0; //runtime monitoring var
	float elapsed_min = 10000000.0, elapsed_max = 0.0;
	float total_forwarded = 0.0; // traffic monitoring var
	float drop_probability; // 
	enum Initial_Distribution initial_distribution;
	struct Final_Data final_data;
	char output_format = 0;

	int node_number, nb_tours;// N is the number of node,nb_tours is the number of rounds 
	struct Performances performance;
	int write_descriptor;
	
	//default values in case no options were given
	write_descriptor = -3;
	output_format = 0;
	char c;
	
#ifdef __timing__
	/* initialization of the table that will contain measures */
	for (int k=0; k<700; k++)
		for (int l=0; l < 14; l++){
			kernel_duration[k][l] = 0.0;
			accumulated_measure[l] = 0.0;
		}
#endif 

	while ((c = getopt (argc, argv, "hg:d:n:")) != -1) {
	 	switch (c) {
	 		case 'h':
	 		Print_help();
	 		exit(0);
	 		break;
	 		
	 		case 'g':
	 		output_format = 1;
	 		write_descriptor = atoi(optarg);
	 		break;
	 		
	 		case 'd':
	 		cudaSetDevice(atoi(optarg));
	 		break;
	 		
	 		case 'n':
	 		node_number = atoi(optarg);
	 		break;
	 		
	 		default:
	 		Print_help();
	 		exit (-1);
	 		break;
	 		
	 	}
	 
	}

	cudaSetDevice(1);

	/* Initialization of the random number generator */
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	checkCUDAError("cuda create random");

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	
	/* Setting default values for the simulation parameters */
	Init_simulation_parameters();
		
	/* Getting back values that are necessary to launch the simulation */
	drop_probability = simulation_parameters.simulation_config.drop_probability;
	node_number = simulation_parameters.simulation_config.node_number;
	nb_tours = simulation_parameters.simulation_config.simulation_time;
	initial_distribution = simulation_parameters.topology_config.distribution.initial_distribution;
		
	performance.total_dest = 0;

	for (int H = 496; H <= 706; (H < 65 ? (H < 20 ? H++ : H+= 2) : H += 10))
	//for (int H = 356; H <= 446; (H < 65 ? (H < 20 ? H++ : H+= 2) : H += 10)) 
	{			
		node_number = H*H;

	    if (output_format)			
			final_data.node_number = node_number;
	    else
		    printf("%d ", node_number);

		total_elapsed = 0.0;
		total_dest = 0.0;
		elapsed_max = 0.0;
		elapsed_min = 10000000.0;
		total_forwarded = 0.0;

#ifdef __timing__
			for (int k=0; k<700; k++)
				for (int l=0; l < 14; l++){
					kernel_duration[k][l] = 0.0;
					accumulated_measure[l] = 0.0;
			}
#endif 

		for (int exec_nb = 0; exec_nb < 5; exec_nb++) {
			cudaEventRecord(start, 0);

			switch (initial_distribution) {
			
				case RANDOM_DISTRIBUTION:
				
				//performance = Random(node_number, 1, drop_probability, (curandGenerator_t *) &gen, write_descriptor);
						
				performance = Random_sched_try(node_number, 1, drop_probability, (curandGenerator_t *) &gen, write_descriptor);


				break;

				case GRID:
				
				performance = Static_grid(node_number, 1, drop_probability,
						(curandGenerator_t *) &gen, write_descriptor);

				break;
			}

			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&elapsed, start, stop);

			total_elapsed += elapsed;

			if (elapsed > elapsed_max) {
				elapsed_max = elapsed;
			}

			if (elapsed < elapsed_min) {
				elapsed_min = elapsed;
			}

			total_dest += (float) performance.total_dest;

			total_forwarded += (float) performance.total_forwarded;
			
#ifdef __timing__
	
			for (int k=0; k<700; k++){
				kernel_duration[k][13] = 0;
				for (int i=0; i < 11; i++)
					kernel_duration[k][13] += kernel_duration[k][i];
				for (int l=0; l < 14; l++)
					accumulated_measure[l] += kernel_duration[k][l];
			}
			
#endif
		}
		
#ifdef __timing__
	
		for (int l=0; l < 14; l++)
		{
			accumulated_measure[l] = accumulated_measure[l] / 700.0;
			printf("%.2f ",accumulated_measure[l]);
		}
		printf("\n");		
#endif

		total_elapsed = total_elapsed / 5.0;
			
		total_dest = total_dest / 5.0;
		
		total_forwarded = total_forwarded / 5.0;
		

	    if (output_format)
			final_data.loss = 1.0 - (total_dest / (float) nb_tours);
	   	else
		    printf("%d ", total_dest);
			
	
		if (output_format)		
		    final_data.forwarded = (int)total_forwarded;
	    else
		    printf("%d ", (int)total_forwarded);

			
		if (output_format){		
		    final_data.device_memory = performance.device_used_memory;
		    final_data.host_memory = performance.host_used_memory;
	    }else
	 	    printf("%d %d ", performance.device_used_memory, performance.host_used_memory);
	
		if (output_format)	
		    final_data.average_time = total_elapsed;
	    else
		    printf("%f ", total_elapsed);
			
		if (output_format)		
		    final_data.min_time = elapsed_min;
		else
		    printf("%f ", elapsed_min);

		if (output_format)		
		    final_data.max_time = elapsed_max;
		else
		    printf("%f\n", elapsed_max);
			
		if (write_descriptor != -3){

			if( write( write_descriptor, &final_data, sizeof(struct Final_Data) ) == -1 )
        		perror( "write main" );

			if (::close (write_descriptor) == -1 ) /* we close the write desc. */
	    		perror( "close on write" );
		}
	}	
	
	return 0;
}


void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}
