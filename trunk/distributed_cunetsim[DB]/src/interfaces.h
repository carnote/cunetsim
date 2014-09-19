/*
 * interfaces.h
 *
 *  Created on: Oct 19, 2011
 *      Author: benromdh
 */

#include <mpi.h>
#include <mysql.h>

#ifndef INTERFACES_H_
#define INTERFACES_H_
#endif /* INTERFACES_H_ */
void Print_help();
void Calculate_timestamps (Simulation_Parameters simulation_parameters);
void Print_event_list();
struct Performances Random_sched(struct Partition partition, int f, float drop_prob, void *genV, int gui, MPI_Comm comm_masters, MPI_data_types mpi_types, MYSQL *conn);
void Schedule (int first_round, int order_in_round, int nb_rounds, enum Event_Distribution distribution, int frequency_in_round, 
enum Event_Type event_type, int nb_tours);
bool Next_event_of_the_round (struct Event * event, int round_num);
void Init_simulation_parameters();
struct Performances Static_grid(struct Partition partition, int f, float drop_prob, void *gen, int gui, MPI_Comm comm_masters, MPI_data_types mpi_types, MYSQL *conn);
int main3( int argc, char** argv);
__global__ void Reset_Buffer(struct Buffer *buffer, int node_number);
__global__ void HelloWorld_kernel(int size, char *gpu_odata);
 inline void Host_Memory_Allocation(struct Cell *Host_Cell, struct Geo *Host_Geo, struct Geo2 *Host_Geo2, float *Host_PosRandx, \
		 float *Host_PosRandy, float *Host_PosRandz, float *Host_VRandx, float *Host_VRandy, float *Host_VRandz, struct Node *Host_Node);
extern inline void Device_Memory_Allocation(struct Cell *Device_Cell, struct Geo *Device_Geo, struct Geo2 *Device_Geo2,\
		float *Device_Randx, float *Device_Randy, float *Device_Randz, struct Node *Device_Node);
__host__ void Init_host_static_grid(struct Geo *host_geo, int node_number, int offset, struct Connection *ext_connections, int nb_ext_conn);
__host__ void Init_geo_static_grid(struct Geo *host_geo, int node_number, int offset, struct Connection *ext_connections, int nb_ext_conn);
__host__ void Init_device_static_grid(struct Geo *host_geo, struct Geo *device_geo, int node_number);
__host__ void Init_control(struct Cell *host_cell, struct Geo *host_geo, struct Geo2 * host_g2, struct Node *host_nodes, int node_number);
__host__ void Init_device(struct Cell *host_cell, struct Geo *host_geo, struct Geo2* host_geo2, struct Node *host_node,\
		struct RouterBuffer *host_router_buffer, struct Cell *device_cell, struct Geo *device_geo, struct Geo2 *device_geo2, \
		struct Node *device_node, struct RouterBuffer *device_router_buffer, int node_number);
__host__ void Init_cell(struct Cell *cell, int node_number);
__host__ void Init_node(struct Node *cell, int node_number);
__host__ void Init_router_buffer(struct RouterBuffer *host_router_buffer, int node_number);
__host__ void Cell_control(struct Cell * cell);
__host__ void Geo_control(struct Cell *cell, struct Geo *geo, int node_number);
__host__ void Buffer_control(struct Buffer *host_in_phy, struct Buffer *host_out_phy, int node_number);
__host__ void Init_geo(struct Cell *cell, struct Geo *geo, float *posrandx, float *posrandy,float *posrandz,float *vrandx,\
		float *vrandy,float *vrandz, int node_number, Simulation_Parameters simulation_parameters,
		int offset, struct Connection *ext_connections, int nb_ext_conn);
__global__ void Mobility(struct Geo *geo, float *randx,float *randy,float *randz,float *randv, int node_number, Simulation_Parameters *device_simulation_parameters);
void Mobility(struct Geo *geo, float *randx,float *randy,float *randz, int rank, Simulation_Parameters *simulation_parameters);
__global__ void Mobility_Control(struct Cell *cell, struct Geo *geo, int node_number, Simulation_Parameters *device_simulation_parameters);
void Mobility_control(struct Cell *cell,struct Geo *geo, int mode, int rank, int node_number);
void Update_cell(struct Cell *cell, int node_number,int rank);
__host__ void Print_position(struct Geo *geo, int node_number);
__host__ void Print_mobility(struct Geo *geo,int node, int i,int j);
__host__ void Init_cell_gen_2D(struct Cell *cell, int step_x, int step_y, int node_number, Simulation_Parameters simulation_parameters);
__host__ void Init_cell_gen(struct Cell *cell, int step_x, int step_y, int step_z, int node_number, Simulation_Parameters simulation_parameters);
__host__ void print_trajectory(struct Geo *geo, int i, int j);
__host__ __device__ float Distance(struct Position p1 , struct Position p2);
__global__ void Visible(struct Geo *geo, struct Cell *cell, int node_number, int cell_size);
void Visible_cpu(struct Geo *geo, struct Cell *cell, struct Geo2 *geo2, int tid);
__global__ void Update_Cell(int cell_number, struct Cell *cell, int node_number);
__host__ void Print_connectivity(struct Geo *G, int N, int suffix);
__host__ int Connectivity_control(struct Geo *geo, int node_number, int round, struct Cell *cell, int cell_size);
__global__ void Visible_Opt(struct Geo *geo, struct Cell *cell, int node_number, int cell_size, 
		int offset, struct Connection *ext_connections, int nb_ext_conn);
__global__ void Init_Buffer(struct Buffer *in_buffer, int node_number);
__global__ void Sender(struct Geo *geo, struct Buffer *in_phy_buff, struct Buffer *out_phy_buff,
		int node_number, int* dev_forwarded_per_node, int *device_new_message, int offset, struct RemoteCommBuffer *remote_comm_buff, 
		float m_send, float b_send);
__global__ void Receiver(struct Buffer *in_phy_buff, int node_number, struct RemoteCommBuffer *remote_buff, struct Geo *geo, int offset, float m_recv, float b_recv, int *device_recv_new_message);
__global__ void Router_In(struct Buffer *in_phy_buff, struct Buffer *out_phy_buff,
		struct MessageBuffer *message_buffer, struct RouterBuffer *router_buffer, float *current_probability,
		float drop_prob, int node_number, int offset);
__global__ void Router_Out(Buffer *out_phy_buff, MessageBuffer *app_out_buff,
		int seq_nb, int node_number, int offset);
__global__ void Cleaner(Buffer *OutBuffer, float *rand);
__host__ void Print_propagation(struct Geo *G, struct Buffer *Buff, int N);
__host__ void Print_propagation(struct Geo *G, struct Buffer *Buff,int t1,int t2, int N);
__host__ void Free_device(struct Geo *Device_Geo, struct Geo2 *Device_Geo2, struct Cell *Device_Cell, struct Node *Device_Node, struct Buffer *Device_InPhy, struct Buffer *Device_OutPhy, struct RouterBuffer *Device_RouterBuffer, float *Device_Randx, float  *Device_Randy, float *Device_Randz);
__host__ void Free_host(struct Geo *Host_Geo, struct Geo2 *Host_Geo2, struct Cell *Host_Cell, struct Node *Host_Node, struct Buffer *Host_InPhy, struct Buffer *Host_OutPhy, struct RouterBuffer *Host_RouterBuffer, float *Host_PosRandx, float  *Host_PosRandy, float *Host_PosRandz, float *Host_VRandx, float  *Host_VRandy, float *Host_VRandz);
__host__ void Static_network( int argc, char** argv);
__global__ void Init_Router(struct RouterBuffer *router_buffer, int node_number);
__device__ void Application_In(MessageBuffer *app_in, int tid, int *dev_tot_dest);
__global__ void Application_Out(MessageBuffer *app_out_buff, int *senders_receivers, int message_digit, int node_number, int tour, int nb_tours, int offset);
__global__ void TC_Out(MessageBuffer *app_out_buff, int *senders_receivers, int node_number, int tour, int nb_tours);
__global__ void Message_In(MessageBuffer *app_in, int node_number, int *dev_tot_dest);
__global__ void Init_App_Buffer(struct MessageBuffer *message_buffer, int node_number);
void checkCUDAError(const char *msg);

__host__ void Init_host_random(struct Cell *host_cell, struct Geo *host_geo,
		float *host_pos_randx, float *host_pos_randy,
		float *host_pos_randz, float *host_v_randx, float *host_v_randy,
		float *host_v_randz, int node_number, Simulation_Parameters simulation_parameters, 
		int offset, struct Connection *ext_connections, int nb_ext_conn);
__host__ void Init_device_random(struct Cell *host_cell, struct Geo *host_geo,
		struct Cell *device_cell, struct Geo *device_geo, int node_number, Simulation_Parameters simulation_parameters);
__global__ void Init_Rem_Comm_Buffer(struct RemoteCommBuffer *rem_comm_buffer, int node_number);

int process_query (MYSQL *conn, char *query);

void process_result_set (MYSQL *conn, MYSQL_RES *res_set);

