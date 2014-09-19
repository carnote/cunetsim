/*
 * structures.h
 *
 *  Created on: Oct 19, 2011
 *      Author: benromdh
 */

// Includes
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#endif /* STRUCTURES_H_ */


/****/

//#define N 256// number of nodes
#define Nmax 100000
#define Step 8  // division of the step
//#define B Step*Step*Step // number of cells in 3D (Step*Step in 2D)
#define B Step*Step
#define Visibility 200 // size of a cell
#define Reach 100
#define I Step*Visibility // size of the space
#define cycle 1600
#define MAX_buffer 128
#define msg_length 128
#define MAXBLOC 512
#define Maxneighbor 10
#define vmax 24
#define vmin 1
#define Maxelement 4
#define Maxsender 100
#define NB_TOURS 600
#define MAX_NODE_SPEC 200
#define MAX_NODE_NB 100000
#define NB_DISPLAYED_NODES 200
#define PAYLOAD_SIZE 32
#define TC_MSG 1
#define BROADCAST_ID -999
#define MPI_STRUCT_ELTS 3
#define MAX_PARTITIONS 30

typedef struct MPI_data_types {
	MPI_Datatype mpi_data_flow;
	MPI_Datatype mpi_master_buff;
} MPI_data_types;

typedef struct Connection {
	int node1;
	int node2;
} Connection;

typedef struct Partition {
	int node_number;
	int offset;
	int total_node_number;
	int masters_number;
	int additional_time;
	int number_of_external_connections;
	float area_x; 
	float area_y;
	float area_z;
	int conversion_table[MAX_PARTITIONS];
	int gpu_ord;
	struct Connection external_connections[3];	
} Partition;

typedef enum Role {
	ND, RE, CH, GW,FR 
} Role;
/**/


typedef struct Position {
	int x;
	int y;
	int z;
} Position;

typedef struct Cell {
	int id, size, node; // id is the number of the cell, size the number in the cell and node ?
	struct Position p1, p2, center; // P1 is the point at the front bottom left corner, P2 the one at the back up right corner and Center the center of the cell (middle of [P1,P2]
	char passage[Nmax]; // if a node passes by the cell the value of the node in the array passage gets 1
	int member[Nmax]; // list of the nodes passing by the cell (nodes having 1 in the array passage)
	int neighbor_number; // number of neighboring cells
	int neighbor_vector[27]; // list of the neighboring cells
} Cell;

typedef struct Geo {
	struct Position p; // position of the node
	int id;
	int external_conn;
	int speedx, speedy, speedz; // speeds in each of direction
	int MobMode; // model of mobility
	int cell_id; // id of the cell the node is currently visiting
	int old_cell_id; // id of the former cell it visited
	int CellPosition; // ?
	int Viewed; // ?
	int neighbor_number; // number of neighboring nodes (distance between the node and its neighbors < 100)
	int neighbor_vector[Maxneighbor]; // array of its neighbors
	float energy;
} Geo;

typedef struct Geo2 {
	int NodeId[Maxneighbor];
	float Distance[Maxneighbor];
} Geo2;

typedef enum TCM {
	HELLO, RUCH, RCH, ACKCH, NACKCH, ACKRL, RGW, ACKGW, NACKGW
} TCM;

typedef struct Cluster{
 enum Role noderole;
	int ClusterId;
	int MasterId;
	int MasterDg;
	int Degree;
	int Neighbor;
	int Relay;
	int Gw;
	int Hello_size;
	int Hello[Maxneighbor][2];
	int Activen_Neighbor[Maxneighbor];
	int Active_Relay[Maxneighbor/4];
	int Active_Gw[Maxneighbor/4];
} Cluster;

typedef struct Node2 {
	int Id;
	int GeoId;
	int PhyId;
	int TcId;
	int RoutingId;
	int NumBuffer;
	int InBuffId;
	int OutBuffId;
	int OutLook;
} Node2;

typedef struct Node {
	int Id;
	int GeoId;
	int PhyId;
	int TcId;
	int RoutingId;
	int NumBuffer;
	int InBuffId;
	int OutBuffId;
	int OutLook;
} Node;

typedef struct Element {
	int time_stamp;
	int header[4]; //Header[0]=sr; Header[1]=ds, Header[2]=type, Header[3]=TTL// Header[3]=ttl
	int payload[PAYLOAD_SIZE];
} Element;

typedef struct MasterBuffer {
	struct Element element[2 * Maxelement];
	int ext_conn[2 * Maxelement];
	int read_index;
	int write_index;
} MasterBuffer;

typedef struct Buffer {
	struct Element element[Maxelement];
	int read_index;
	int write_index;
	int node_id;
} Buffer;

typedef struct RemoteCommBuffer {
	struct Element element[Maxelement];
	int ext_conn[Maxelement];
	int read_index;
	int write_index;
	int conn_node_id;
} RemoteCommBuffer;

typedef struct RouterBuffer {
	int header_array[Maxelement][4];
	int write_index;
	char full;
} RouterBuffer;

typedef struct MessageBuffer {
	int payload[Maxelement][PAYLOAD_SIZE];
	int dest[Maxelement];
	int write_index;
	int read_index;
}MessageBuffer;

typedef struct Performances {
	int total_dest;
	int total_forwarded;
	int host_used_memory;
	int device_used_memory;
}Performances;

typedef struct Data_Flow_Unit {
	struct Geo geo[NB_DISPLAYED_NODES]; 
	int new_message[NB_DISPLAYED_NODES];
	int tour;
	int nb_node;
}Data_Flow_Unit;

typedef struct Final_Data {
	int node_number;
	float loss;
	int forwarded;
	int device_memory;
	int host_memory;
	float average_time;
	float min_time;
	float max_time;
}Final_Data;

typedef enum Event_Distribution {
	DISTRIB, CONSEC
}Event_Distribution;

typedef enum Event_Type {
	MOB, //0
	CON, //1
	APP_OUT, //2 
	APP_IN, //3
	PROTO_OUT, //4
	PROTO_IN, //5
	PKT_OUT, //6
	PKT_IN, //7
	TC_OUT //8
}Event_Type;

typedef struct Event {
	enum Event_Type type;
	int round;
	int order;
	int frequency;
	int timestamp;
}Event;

/* Normally we are not going to use this structure any more */
/* That is why is is commented now */
/*
typedef struct Phy {
	struct Energy E;
	int Connectivity;
	int Mob;
	int Interference;
	int Reliablity;
} Phy;
*/
	

//int blocId,oldBlocId,blocPosition,vue,Voisin;// vue nombre de noeud vue geometriquement Voisin les voisin entendus par hello
//struct position position __attribute((aligned(32)));
//struct energy energy __attribute((aligned(32)));
//struct message Inbuffer[1] __attribute((aligned(32)));
//struct message Outbuufer[1]__attribute((aligned(32)));
//enum role noderole __attribute((aligned(32)));
//int masterid, masterd __attribute((aligned(32)));// master id, master degree;
//int aporte[Maxneighbor] __attribute((aligned(32)));
//float dist[Maxneighbor ]__attribute((aligned(32)));
//struct voisinagemake clean V[1] __attribute((aligned(32)));
//int out_indice,in_indice,v,test __attribute((aligned(32)));
//}Node;

			typedef enum Distribution{
				UNIFORM, GAUSSIAN, EXPONENTIAL, POISSON, FIXED
			}Distribution;

			typedef enum Transport_Protocol{
				TCP, UDP
			}Transport_Protocol;
	
			typedef enum IP_Version{
				IPV4, IPV6
			}IP_Version;

			typedef enum Application_Type{
				CBR, M2M_AP, M2M_BR, GAMING_OA, GAMING_TF, FULL_BUFFER, DEFAULT
			}Application_Type;

		typedef struct Customized_Traffic{
			enum Application_Type application_type;
			int idt_mim_ms;
			int idt_max_ms;
			int idt_standard_deviation;
			int idt_lambda;
			int size_min_byte;
			int size_max_byte;
			int size_standard_deviation;
			int size_lambda;
			int direction_port;
			enum IP_Version IP_version;
			enum Transport_Protocol transport_protocol;
			enum Distribution idt_distribution;
			enum Distribution size_distribution;
		}Customized_Traffic;	


		typedef struct Predefined_Traffic{
			enum Application_Type application_type;
		}Predefined_Traffic;

	typedef struct Application_Config{
		struct Predefined_Traffic predefined_traffic;
		struct Customized_Traffic customized_traffic;
		int customized;
	}Application_Config;

			typedef struct Geo_Cell{
				int cell_size_m;
				int step_x;
				int step_y;
				int step_z;
				int cell_number;
				int reach;
			}Geo_Cell;

		typedef struct Area{
			struct Geo_Cell geo_cell;
			double x_km;
			double y_km;
			double z_km;
		}Area;

				typedef struct Grid_Map{
					int horizontal_grid;
					int vertical_grid;
				}Grid_Map;

				typedef enum Grid_Trip_Type{
					RANDOM_DESTINATION, RANDOM_TURN
				}Grid_Trip_Type;

			typedef struct Grid_Walk{
				enum Grid_Trip_Type grid_trip_type;
				struct Grid_Map grid_map;
			}Grid_Walk;

			typedef enum Boundary_Policy{
				BOUNCE, MODULAR, RANDOM, STOP
			}Boundary_Policy;

			typedef enum Mobility_Type{
				IMMOBILE,
				RANDOM_WAYPOINT, RANDOM_MOVEMENT, 
				RANDOM_UNIFORM_MOVEMENT_MAX_BOUNDED, 
				RANDOM_UNIFORM_MOVEMENT_MAX_AND_MIN_BOUNDED,
				GRID_WALK, LINEAR_MOVEMENT
			}Mobility_Type;

			typedef struct Moving_Dynamics{
				double min_speed_mps;
				double max_speed_mps;
				double min_sleep_ms;
				double max_sleep_ms;
			}Moving_Dynamics;

		typedef struct Mobility_Parameters{
			struct Moving_Dynamics moving_dynamics;
			enum Mobility_Type mobility_type;
			enum Boundary_Policy boundary_policy;
			struct Grid_Walk grid_walk;
		}Mobility_Parameters;

			typedef enum Grid_Distribution{
				BORDER_GRID, RANDOM_GRID
			}Grid_Distribution;

			typedef enum Initial_Distribution {
				RANDOM_DISTRIBUTION, CONCENTRATED, GRID
			}Initial_Distribution;

		typedef struct Geo_Distribution {
			enum Initial_Distribution initial_distribution;
			enum Grid_Distribution grid_distribution;
		}Geo_Distribution;

		typedef enum Connectivity_Model {
			UDG, QUDG
		}Connectivity_Model;

	typedef struct Topology_Config {
		enum Connectivity_Model connectivity_model;
		struct Geo_Distribution distribution;
		struct Mobility_Parameters mobility_parameters;
		struct Area area;
	}Topology_Config;
	
			typedef struct Shadowing {
				double decorrelation_distance_m;
				double variance_dB;
				double inter_site_correlation;
			}Shadowing;
	
			typedef struct Free_Space_Model_Parameters {
				double pathloss_exponent;
				double pathloss_0_dB;
			}Free_Space_Model_Parameters;

			typedef struct Ricean_8Tap {
				int rice_factor_dB;
			}Ricean_8Tap;

			typedef enum Small_Scale {
				SCM_A, SCM_B, SCM_C, SCM_D, rayleigh_8tap, ricean_8tap, EPA, EVA, ETU
			}Small_Scale;

			typedef enum Large_Scale {
				FREE_SPACE, URBAN, RURAL
			}Large_Scale;

		typedef struct Fading {
			enum Large_Scale large_scale;
			enum Small_Scale small_scale;
			struct Shadowing shadowing;
			struct Free_Space_Model_Parameters free_space_model_parameters;
			struct Ricean_8Tap ricean_8tap;
		}Fading;
				typedef enum Energy_Model {
					DEFAULT_ENERGY_MODEL
				}Energy_Model;

			typedef struct Energy {
				int E;
				enum Energy_Model energy_model;
			}Energy;

			typedef enum Connectivity_Impact_Model {
				DEFAULT_CON_IMPACT_MODEL
			}Connectivity_Impact_Model;

			typedef enum Mobility_Impact_Model {
				DEFAULT_MOB_IMPACT_MODEL
			}Mobility_Impact_Model;

			typedef enum Interference_Model {
				DEFAULT_INTERFERENCE_MODEL
			}Interference_Model;

		typedef struct Physical_Layer {
			enum Connectivity_Impact_Model connectivity_impact_model;
			enum Mobility_Impact_Model mobility_impact_model;
			enum Interference_Model interference_model;
			int reliability;
			struct Energy energy;
		}Physical_Layer;

	typedef struct Environment_Config {
		double system_frequency_GHz;
		double system_bandwidth_MB;
		double wall_penetration_loss_dB;
		struct Physical_Layer physical_layer;
		struct Fading fading;
		float m_send;
		float m_recv;
		float b_send;
		float b_recv;
		float init_energy;
	}Environment_Config;

			typedef struct Layer {
				int phy;
				int mac;
				int rlc;
				int rrc;
				int pdcp;
				int omg;
				int otg;
				int emu;
			}Layer;		

			typedef struct Packet_Trace {
				int node_id;
			}Packet_Trace;

			typedef struct Metrics {
				int throughput;
				int latency;
				int signalling_overhead;
			}Metrics;

		typedef struct Performance {
			struct Metrics metrics;
			struct Packet_Trace packet_trace;
			struct Layer layer;
		}Performance;

			typedef struct Node_Application_Config{
				struct Predefined_Traffic predefined_traffic;
				struct Customized_Traffic customized_traffic;
				int customized;
				int destination; //-1 if it won't send anything
			}Node_Application_Config;

		typedef struct Node_Specification {
			int id;
			struct Mobility_Parameters mobility_parameters;
			enum Connectivity_Model connectivity_model;
			struct Energy energy;
			struct Node_Application_Config node_application_config;
		}Node_Specification;


	typedef struct Simulation_Config {
		int simulation_time;
		int node_number;
		float drop_probability;
		int _3D_is_activated;
		struct Performance performance;
		struct Node_Specification node_specification[MAX_NODE_SPEC];
	} Simulation_Config;
	
	typedef struct Distributed_Simulation_Config {
		int partition_number;
		struct Partition partitions[MAX_PARTITIONS];
	} Distributed_Simulation_Config;

typedef struct Simulation_Parameters {
	struct Distributed_Simulation_Config distributed_simulation_config;
	struct Simulation_Config simulation_config;
	struct Environment_Config environment_config;
	struct Topology_Config topology_config;
	struct Application_Config application_config;
}Simulation_Parameters;




