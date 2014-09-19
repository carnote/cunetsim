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
#include <fcntl.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

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

#define __timing__
#define __mobonly__

/* global structure */
extern Simulation_Parameters simulation_parameters;

float kernel_duration[1000][17];

void process_result_set (MYSQL *conn, MYSQL_RES *res_set){
  MYSQL_ROW	row;
  unsigned int	i;
  
  while ((row = mysql_fetch_row (res_set)) != NULL){
    mysql_field_seek (res_set, 0);
    
    for (i = 0; i < mysql_num_fields (res_set); i++){
      mysql_fetch_field (res_set);
      printf (" %s |",row[i]);
    }
    printf("\n");
  }
  printf ("%lu rows returned\n", (unsigned long) mysql_num_rows (res_set));
}


int process_query (MYSQL *conn, char *query){

  MYSQL_RES *res_set;
  
  if (mysql_real_query (conn, query, strlen(query)) != 0){
    printf("process_query() %s failed => error: %s\n", query, mysql_error(conn));
    return -1; //exit (-1);
  }
  
  res_set = mysql_store_result (conn);
  if (res_set != NULL){
    process_result_set (conn, res_set);
    mysql_free_result (res_set);
    return 0;
  }

  return -1;
}



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
	int commSize, commRank, masters_commRank;
	int ierr ;
	struct Data_Flow_Unit data, data_flow;
	struct MPI_data_types mpi_types;
	float final_result [7];
	MPI_Status status ;
	MPI_Group  group_world, group_masters;
	MPI_Comm   comm_masters, comm_world;
	int manager_id = 0;
	int number_of_executions = 1;
	char host[80];  
  	int length;
  	int offset, total_node_number;
  	float area_x, area_y, area_z;

#ifdef __timing__
	
	// Database variables
	MYSQL *conn = NULL;
	char *host_name = "pyroclaste";
  	char *user_name = "root";
  	char *password  = "database";
 	unsigned int port_num = 0;
  	char *socket_name = NULL;
  	char *db_name = "cunetsim";
  	int flags = 0;
	char query[4096];

#endif

	int node_number, nb_tours; //nb_tours is the number of rounds 
	struct Performances performance;
	struct Partition partition;
	int write_descriptor;
	
	//default values in case no options were given
	write_descriptor = -3;
	output_format = 0;
	char c;
	
	// Initialize MPI state
    	MPI_Init(NULL, NULL);
		
    	ierr = MPI_Comm_size(MPI_COMM_WORLD, &commSize);
 	ierr = MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
	
	while ((c = getopt (argc, argv, "hgd:n:")) != -1) {
	 	switch (c) {
	 		case 'h':
	 		Print_help();
	 		exit(0);
	 		break;
	 		
	 		case 'g':
	 		if (commRank == 0)
	 			// Only the coordinator opens the communication tube with the GUI since both are running on the same machine
	 			write_descriptor = open("../../Debug/comm_tube", O_WRONLY);
	 		output_format = 1; // All the processes use this parameter to know whether the visualizor is on
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

	/* Initialization of the random number generator */
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	checkCUDAError("cuda create random");

	// Select the right GPU
	cudaSetDevice((commRank <= 2 ? 0 : 1));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	performance.total_dest = 0;
	
	for (int H = 1000; H <= 1000; (H < 65 ? H+= 2 : H += 10))
	//for (int H = 2; H <= 256; (H < 65 ? (H < 20 ? H++ : H+= 2) : H += 10))
	//for (int H = 356; H <= 446; (H < 65 ? (H < 20 ? H++ : H+= 2) : H += 10)) 
	{			

		simulation_parameters.simulation_config.node_number = H*H;
	
		/* Setting default values for the simulation parameters */
		Init_simulation_parameters();
		
		/* Getting back values that are necessary to launch the simulation */
		drop_probability = simulation_parameters.simulation_config.drop_probability;
		node_number = simulation_parameters.simulation_config.node_number;
		total_node_number = node_number;
		nb_tours = simulation_parameters.simulation_config.simulation_time;
		initial_distribution = simulation_parameters.topology_config.distribution.initial_distribution;

		total_elapsed = 0.0;
		total_dest = 0.0;
		elapsed_max = 0.0;
		elapsed_min = 10000000.0;
		total_forwarded = 0.0;

#ifdef __timing__
		
		for (int k=0; k<1000; k++)
			for (int l=0; l < 13; l++){
				kernel_duration[k][l] = 0.0;
			}
#endif 		

 	   	MPI_Get_processor_name(host, &length);  /* Get name of this processor */
 	 		
		/* Creating a new communicator excluding the manager (rank 0) for the synchronization */
		//Biiig error
		//if (commRank != 0) {

		comm_world = MPI_COMM_WORLD;

		/* Extract the original group handle */	  		
		MPI_Comm_group(comm_world, &group_world);
  		ierr = MPI_Group_excl(group_world, 1, &manager_id, &group_masters);  /* process 0 not member */
 		ierr = MPI_Comm_create(comm_world, group_masters, &comm_masters);	
		ierr = MPI_Group_rank (group_masters, &masters_commRank);
		
		// Step of MPI types creation //
		/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
	
		struct MasterBuffer master_buff;
		struct Element elt;
	
		// Variables needed to create a new MPI type which will encapsulate our 'struct Element'
		int blocklengths[MPI_STRUCT_ELTS + 11];
		MPI_Datatype types[MPI_STRUCT_ELTS + 11];
		MPI_Aint displacements[MPI_STRUCT_ELTS + 11];
	
		MPI_Datatype mpi_element, mpi_master_buff, mpi_position, mpi_geo, mpi_data_flow, mpi_connection, mpi_partition;
	
		MPI_Aint extent, element_extent, position_extent, geo_extent, connection_extent, float_extent;

		MPI_Type_extent(MPI_INT, &extent);
		MPI_Type_extent(MPI_FLOAT, &float_extent);

		types[0]= MPI_INT;
	   	types[1]= MPI_INT;
	    	types[2]= MPI_INT;

		blocklengths[0]= 1;         
	    	blocklengths[1]= 4;        
	    	blocklengths[2]= PAYLOAD_SIZE;
	    
	    	displacements[0]= 0;
	    	displacements[1]= extent;
	    	displacements[2]= 5 * extent;
   	
   		//Creation of the type mpi_element
   		ierr = MPI_Type_struct(MPI_STRUCT_ELTS, blocklengths, displacements, types, &mpi_element);
    		MPI_Type_commit(&mpi_element);

		//Preparation for the 2nd MPI data type creation, which is going to be transfered between masters
		types[0]= mpi_element;
    		types[1]= MPI_INT;
    		types[2]= MPI_INT;
    		types[3]= MPI_INT;

    		blocklengths[0]= 2 * Maxelement;         
    		blocklengths[1]= 1 * Maxelement;        
    		blocklengths[2]= 1;
    		blocklengths[3]= 1;
    
    		MPI_Type_extent(mpi_element, &element_extent);
    
    		displacements[0]= 0;
    		displacements[1]= 2 * Maxelement * element_extent;
    		displacements[2]= 2 * Maxelement * (extent + element_extent);
    		displacements[3]= 2 * Maxelement * (extent + element_extent) + extent;
   	
   		//Creation of the type mpi_master_buff
    		ierr = MPI_Type_struct(MPI_STRUCT_ELTS + 1, blocklengths, displacements, types, &mpi_master_buff);
    		MPI_Type_commit(&mpi_master_buff);
    
    		//Preparation of a 3rd mpi new type that contains a 3d coordinates ==> struct Position
    		types[0]= MPI_INT;
    		types[1]= MPI_INT;
    		types[2]= MPI_INT;
    
  		blocklengths[0]= 1;         
  	  	blocklengths[1]= 1;        
  	  	blocklengths[2]= 1;
    
    		displacements[0]= 0;
    		displacements[1]= extent;
    		displacements[2]= 2 * extent;
    
    		//Creation of the type mpi_position
    		ierr = MPI_Type_struct(MPI_STRUCT_ELTS, blocklengths, displacements, types, &mpi_position);
    		MPI_Type_commit(&mpi_position);
    
    		//Preparation of a 4th mpi new type that contains geo data for a node ==> struct Geo
    		MPI_Type_extent(mpi_position, &position_extent);
    
    		types[0] = mpi_position;
    		blocklengths[0]= 1;
    		displacements[0]= 0;

		types[13] = MPI_FLOAT;
    		blocklengths[13]= 1;
    		displacements[13]= position_extent + (11 + Maxneighbor) * extent;
    
    		for (int t = 1; t < 13; t++) {
    			types[t] = MPI_INT;
    			blocklengths[t]= 1;
    			displacements[t]= position_extent + (t - 1) * extent;
    		}
    
    		blocklengths[12] = Maxneighbor;
    
    		//Creation of the type mpi_geo
    		ierr = MPI_Type_struct(MPI_STRUCT_ELTS + 11, blocklengths, displacements, types, &mpi_geo);
    		MPI_Type_commit(&mpi_geo);
    
    		//Preparation of a 5th mpi new type that contains geo data for a number of nodes ==> struct Data_Flow_Unit
    		MPI_Type_extent(mpi_geo, &geo_extent);
    
    		types[0] = mpi_geo;
    		types[1] = MPI_INT;
    		types[2] = MPI_INT;
    		types[3] = MPI_INT;
    
    		blocklengths[0] = NB_DISPLAYED_NODES;
    		blocklengths[1] = NB_DISPLAYED_NODES;  
    		blocklengths[2] = 1;
    		blocklengths[3] = 1;
    
   	 	displacements[0] = 0;
   	 	displacements[1] = NB_DISPLAYED_NODES * geo_extent;
    		displacements[2] = NB_DISPLAYED_NODES * (extent + geo_extent);
    		displacements[3] = NB_DISPLAYED_NODES * (extent + geo_extent) + extent;
    	
    		//Creation of the type mpi_data_flow
    		ierr = MPI_Type_struct(MPI_STRUCT_ELTS + 1, blocklengths, displacements, types, &mpi_data_flow);
    		MPI_Type_commit(&mpi_data_flow);
    	
    		//Preparation of a 6th mpi new type that contains the two nodes that form a "remote" connection ==> struct Connection
    
    		types[0] = MPI_INT;
    		types[1] = MPI_INT;
    
    		blocklengths[0] = 1;
    		blocklengths[1] = 1;  
    		
    		displacements[0] = 0;
    		displacements[1] = extent;
    	
    		//Creation of the type mpi_geo
    		ierr = MPI_Type_struct(MPI_STRUCT_ELTS - 1, blocklengths, displacements, types, &mpi_connection);
    		MPI_Type_commit(&mpi_connection);
    	
    		//Preparation of a 7th mpi new type that contains partition details for a master ==> struct Partition	
    		for (int t = 0; t < 7; t++) {
    			types[t] = MPI_INT;
    			displacements[t] = t * extent;
    		}
    	
    		types[6] = MPI_FLOAT;
    		types[7] = MPI_FLOAT;
    		types[8] = MPI_FLOAT;
    		types[9] = MPI_INT;
    		types[10] = mpi_connection;
    	
    		displacements[7] = 6 * extent + float_extent;
    		displacements[8] = 6 * extent + 2 * float_extent;
    		displacements[9] = 6 * extent + 3 * float_extent;
    		displacements[10] = (6 + MAX_PARTITIONS) * extent + 3 * float_extent;
    	
    		for (int t = 0; t < 9; t++)
    			blocklengths[t]= 1;
    		
    	
    		blocklengths[9] = MAX_PARTITIONS;
    		blocklengths[10] = 3;
    	
    
    		//Creation of the type mpi_geo
    		ierr = MPI_Type_struct(MPI_STRUCT_ELTS + 8, blocklengths, displacements, types, &mpi_partition);
    		MPI_Type_commit(&mpi_partition);
    
    		mpi_types.mpi_master_buff = mpi_master_buff;
		mpi_types.mpi_data_flow = mpi_data_flow;
    
    
    		/************************/
    	
		if (commRank == 0) { // The coordinator code

#ifdef __timing__	

			//Database: creation and connection
			conn = mysql_init (NULL);
			
			if (mysql_real_connect (conn,host_name,user_name,password,db_name,port_num,socket_name,flags) == NULL){
  				fprintf(stderr, "Connection to database failed, error : %s\n", mysql_error(conn));
  			} 

			//Create table specific to this simulation (specific network size)
			sprintf(query,"DROP TABLE IF EXISTS %d_128_1_1_8", total_node_number);	
			process_query (conn,query);
	
			sprintf(query,"CREATE TABLE IF NOT EXISTS %d_128_1_1_8 (round SMALLINT UNSIGNED NOT NULL, master TINYINT UNSIGNED NOT NULL, round_len DECIMAL(10,6),MOB DECIMAL(10,6), CON DECIMAL(10,6), APP_OUT DECIMAL(10,6), APP_IN DECIMAL(10,6), PROTO_OUT DECIMAL(10,6), PROTO_IN DECIMAL(10,6), SND_M DECIMAL(10,6), SND_K DECIMAL(10,6), RCV_M DECIMAL(10,6), RCV_K DECIMAL(10,6), SYNC DECIMAL(10,6), FRWD_PKT MEDIUMINT UNSIGNED, FRWD_PKT_OUT MEDIUMINT UNSIGNED, PRIMARY KEY (round, master))", total_node_number);
			process_query (conn,query);

			mysql_close (conn);

#endif 

			int tag = 1 - commSize;		
			
			data_flow.nb_node = total_node_number;
			
			//Sending a part of work to each of the masters
				for (int proc = 1; proc < commSize; proc++) {
					struct Partition partition = simulation_parameters.distributed_simulation_config.partitions[proc - 1];
					
					// Send partitions to masters
					ierr = MPI_Send ( &partition, 1, mpi_partition, proc, 0, MPI_COMM_WORLD ) ;
				}
			
			if (write_descriptor != -3) {
		
				while (tag < 0) {
					for (int proc = 1; proc < commSize; proc++) {
					
						ierr = MPI_Recv( &data, 1, mpi_data_flow, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status ) ;
						tag += status.MPI_TAG;
						int ofst; 
						if (status.MPI_TAG == 0) {
							ofst = simulation_parameters.distributed_simulation_config.partitions[status.MPI_SOURCE-1].offset;
							for (int nd = 0; nd < data.nb_node; nd++) {
								data_flow.geo[ofst + nd] = data.geo[nd];
							
								data_flow.geo[ofst + nd].p.x += 200 * (status.MPI_SOURCE - 1); 							
								data_flow.new_message[ofst + nd] = data.new_message[nd];	
							}
								
							data_flow.tour = data.tour;
						}
								
					}
					
					if (status.MPI_TAG == 0)
						if ( write( write_descriptor, &data_flow, sizeof(struct Data_Flow_Unit) ) < sizeof(struct Data_Flow_Unit) )
        	    			perror( "write random" );
            		
				}
			} else {
			
				for (int proc = 1; proc < commSize; proc++) {
					ierr = MPI_Recv( &data, 1, mpi_data_flow, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status ) ;
				}
			}

		} else {
		
			for (int exec_nb = 0; exec_nb < number_of_executions; exec_nb++) {
				
				// Receiving partitions
				ierr = MPI_Recv( &partition, 1, mpi_partition, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status ) ;
				
				cudaEventRecord(start, 0);

				switch (initial_distribution) {
			
					case RANDOM_DISTRIBUTION:
					
					partition.additional_time = 100 * partition.masters_number;
						
					performance = Random_sched(partition, 1, drop_probability, 
							(curandGenerator_t *) &gen, output_format, comm_masters, mpi_types);

	
					break;

					case GRID:
				
					performance = Static_grid(partition, 1, drop_probability,
							(curandGenerator_t *) &gen, output_format, comm_masters, mpi_types);
	
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

			}
		
			total_elapsed = total_elapsed / (float) number_of_executions; final_result [0] = total_elapsed ;	
			total_dest = total_dest / (float) number_of_executions;	final_result [1] = total_dest ;
			total_forwarded = total_forwarded / (float) number_of_executions; final_result [2] = total_forwarded;
			final_result [3] = elapsed_min; final_result [4] = elapsed_max;
			
			final_result [5] = performance.host_used_memory;
			final_result [6] = performance.device_used_memory;
			
			ierr = MPI_Send( final_result, 7, MPI_FLOAT, 0, 2, MPI_COMM_WORLD ) ;
			
		}
		
		if (commRank == 0){
	
			total_elapsed = 0;
			performance.host_used_memory = 0;
			performance.device_used_memory = 0;
		
			for (int iProc = 0; iProc < commSize - 1; iProc ++) {
				ierr = MPI_Recv( final_result, 7, MPI_FLOAT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status ) ;
				
				if (final_result[4] > elapsed_max) {
					elapsed_max = final_result[4];
				}
	
				if (final_result[3] < elapsed_min) {
					elapsed_min = final_result[3];
				}
				
				total_dest += (float) final_result [1];
				total_forwarded += (float) final_result [2];
				total_elapsed += final_result [0];
				performance.host_used_memory = final_result[5];
				performance.device_used_memory = final_result[6];

				//printf("[C] Prc %d: Avr_elpsd = %f, loss = %f, tot_frwd = %d\n", status.MPI_SOURCE, final_result[0], 1 - (final_result[1]/(float)nb_tours), (int)final_result[2]);
			}
			total_elapsed /= (float) (commSize - 1);
		
	    	if (output_format)			
		    	final_data.node_number = total_node_number;
	    	else
		    	printf("%d ", total_node_number);

	 	   	if (output_format)
				final_data.loss = 1.0 - (total_dest / (float) nb_tours);
		   	else
				printf("%f ", 1.0 - (total_dest / (float) nb_tours));
				
			if (output_format)		
			    final_data.forwarded = (int)total_forwarded;
	    	else
		    	printf("%d ", (int)total_forwarded);

			if (output_format){		
				final_data.device_memory = performance.device_used_memory;
			    final_data.host_memory = performance.host_used_memory;
   			} else
		 	    printf("%f %f ", performance.device_used_memory, performance.host_used_memory);
	
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
			   
			
			if (write_descriptor != -3) {
			//printf("\n*** Final results ***\n\n* Node number: %d\n* Loss rate: %f\n* All forwarded packets: %d\n* Used memory [device, host]: [%d, %d]\n* Elapsed time (average): %f\n* Elapsed time (min): %f\n* Elapsed time (max): %f\n",final_data.node_number, final_data.loss, final_data.forwarded,final_data.device_memory, final_data.host_memory,final_data.average_time, final_data.min_time, final_data.max_time);
				int truc; 
				truc = write( write_descriptor, &final_data, sizeof(struct Final_Data) );
				if( truc < sizeof(struct Final_Data) )
        				perror( "write main" );

				if (::close (write_descriptor) == -1 ) /* we close the write desc. */
	    				perror( "close on write" );
			}
		}
	}
	
	ierr = MPI_Finalize() ;

	return 0;
}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}

