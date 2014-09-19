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
 * \file Sender.cu
 * \brief Functions necessary to send messages
 * \author Bilel BR
 * \version 0.0.2
 * \date Nov 24, 2011
 */

#ifndef STRUCTURES_H_
#include "../structures.h"
#define STRUCTURES_H_
#endif /* STRUCTURES_H_ */
#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "../interfaces.h"
#endif /* INTERFACES_H_ */

__device__ void Copy_Message(struct Element *sr, struct Element *ds, int index) {
	
	ds[0].header[0] = sr[0].header[0];
	ds[0].header[1] = sr[0].header[1];
	ds[0].header[2] = sr[0].header[2];
	ds[0].header[3] = sr[0].header[3];
	for (int i = 0; i < PAYLOAD_SIZE; i++) {
		ds[0].payload[i] = sr[0].payload[i];
	}
	ds[0].time_stamp = sr[0].time_stamp;
}
/**
 *We write the message from the source to the destination destination is all neighbor
 *Sender is called the following way Sender<<<threads,grid>>>(Device_Geo,Device_InPhy,Device_OutPhy);
 */


__global__ void Reset_Buffer(struct Buffer *buffer, int node_number){

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < node_number) {
	buffer[tid].write_index = 0; //InBuffer of thread
		__syncthreads();
	}
}

 /**
 * \fn void Sender(struct Geo *geo, struct Buffer *in_phy_buff, struct Buffer *out_phy_buff, int node_number, int* dev_forwarded_per_node, int *device_new_message)
 * \brief forwards the messages contained in the out physical-layer buffer to the node neighbors
 *
 * \param geo pointer to the space data needed to identify a node's neighbors
 * \param in_phy_buff pointer to the in physical-layer buffer that will be used to write messages to the neighbors
 * \param out_phy_buff pointer to the out physical-layer buffer that will be used to read messages before sending them
 * \param node_number is the node number in the simulation
 * \param dev_forwarded_per_node pointer to the integer that will contain the total number of messages forwarded by each node
 * \param device_new_message pointer to an integer that will eventually contain a message sequence number. This is used for the
 * interaction with the GUI
 * \return void
 */
__global__ void Sender(struct Geo *geo, struct Buffer *in_phy_buff, struct Buffer *out_phy_buff,
		int node_number, int* dev_forwarded_per_node, int *device_new_message, float m_send, float b_send)
{

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < node_number) {
		int ds_index;
		int sr_index = out_phy_buff[tid].read_index; //Thread OutBuffer message index

		device_new_message[tid] = -1;
		
		//if there is a message to send and enough energy
		if (sr_index >= 0) 
		{
			for (int i = 0; i < geo[tid].neighbor_number; i++) //to all neighboring nodes
			{
				//if (tid == 0) printf("Send !! 0 +++");
				int neighbor = geo[tid].neighbor_vector[i];
				ds_index = atomicAdd((int*) &in_phy_buff[geo[tid].neighbor_vector[i]].write_index,
						1);

				if (ds_index < Maxelement) {

					// Loosing energy !
					geo[tid].energy -= (m_send * 4 * (4 + PAYLOAD_SIZE) + b_send) / 1000.0;

					device_new_message[tid] = out_phy_buff[tid].element[sr_index].header[2];
					
					/* Here we indicate that this message (which number is seq_nb) is going to be forwarded again */
			
					dev_forwarded_per_node[tid]++;

					Copy_Message(&out_phy_buff[tid].element[sr_index],
							&in_phy_buff[neighbor].element[ds_index], ds_index);

				} else
				{
					atomicSub((int*) &in_phy_buff[geo[tid].neighbor_vector[i]].write_index, 1);
				}
			}
		}
//else if (tid == 0) printf("Energy = %f && consum = %f +++", geo[0].energy , (m_send * 4 * (4 + PAYLOAD_SIZE) + b_send) / 1000.0);
		out_phy_buff[tid].read_index = -1;
		out_phy_buff[tid].write_index = 0;

	}
}

 /**
 * \fn void Application_Out(MessageBuffer *app_out_buff, int *senders_receivers, int message_digit, int N, int tour, int nb_tours)
 * \brief prepares the packet to send in the out application-layer buffer 
 *
 * \param app_out_buff pointer to the out application-layer buffer that will be used to put the outgoing packet
 * \param senders_receivers pointer to the traffic table that is used to determine which nodes will generate packets and for which
 * destination nodes 
 * \param message_digit is a digit by which the message is filled
 * \param node_number is the node number in the simulation
 * \param tour number of current round. If it is among the last tours devoted to finish message prpagation, the node won't generate
 * a packet
 * \param nb_tours total number of rounds
 * \return void
 */
__global__ void Application_Out(MessageBuffer *app_out_buff,
		int *senders_receivers, int message_digit, int node_number, int tour, int nb_tours) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < node_number) {
		
		app_out_buff[tid].read_index = -1;
		if (tour < nb_tours) {
			if (senders_receivers[tid] >= 0) {

				app_out_buff[tid].read_index = 0;
				
				//The last number of the payload is set to 0 to note that this is not a Control message
				app_out_buff[tid].payload[0][0] = 0;

				for (int j = 1; j < PAYLOAD_SIZE; j++) {
					app_out_buff[tid].payload[0][j] = message_digit;
				}

				app_out_buff[tid].write_index = 1;
				app_out_buff[tid].read_index = 0;
				app_out_buff[tid].dest[0] = senders_receivers[tid];
				
			}
		}
	}

}

__global__ void TC_Out(MessageBuffer *app_out_buff,
		int *senders_receivers, int node_number, int tour, int nb_tours) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < node_number) {
		
		app_out_buff[tid].read_index = -1;
		if (tour < nb_tours) {
			if (senders_receivers[tid] >= 0) {

				app_out_buff[tid].read_index = 0;
				
				//The first number of the payload is set to 0 to note that this is not a Control message
				app_out_buff[tid].payload[0][0] = powf(2, sizeof(int) * 8 - 1);

				for (int j = 1; j < PAYLOAD_SIZE; j++) {
					app_out_buff[tid].payload[0][j] = 456;
				}

				app_out_buff[tid].write_index = 1;
				app_out_buff[tid].read_index = 0;
				app_out_buff[tid].dest[0] = senders_receivers[tid];
				
			}
		}
	}

}

 /**
 * \fn void Router_Out(Buffer *out_phy_buff, MessageBuffer *app_out_buff, int seq_nb, int node_number)
 * \brief add a header (the message sequence number) to the packet 
 *
 * \param app_out_buff pointer to the out application-layer buffer that will be used to read the outgoing packet
 * \param out_phy_buff pointer to the out physical-layer buffer that will be used to write the messages to send
 * \param seq_nb sequence number of the message. It is added to the packet to form a message.
 * \param node_number is the node number in the simulation
 * \return void
 */
__global__ void Router_Out(Buffer *out_phy_buff, MessageBuffer *app_out_buff,
		int seq_nb, int node_number) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < node_number) {

		if (app_out_buff[tid].dest[0] >= 0 && app_out_buff[tid].read_index >=0) {

			out_phy_buff[tid].write_index = 1;
			out_phy_buff[tid].read_index = 0;

			/* Sender iD */
			out_phy_buff[tid].element[0].header[0] = tid;

			/* Receiver iD */
			out_phy_buff[tid].element[0].header[1] = app_out_buff[tid].dest[0];

			/* Sequence number not used yet*/
			out_phy_buff[tid].element[0].header[2] = seq_nb;


			/* ttl to be changed */
			out_phy_buff[tid].element[0].header[3] = 1000;

			out_phy_buff[tid].element[0].time_stamp = 0;

			for (int i = 0; i < PAYLOAD_SIZE; i++) {
				out_phy_buff[tid].element[0].payload[i] = app_out_buff[tid].payload[0][i];
			}

		}
	}

}


