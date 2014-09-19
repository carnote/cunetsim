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
 * \file Receiver.cu
 * \brief Functions necessary to receive messages
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

 /**
 * \fn void Receiver(struct Buffer *in_phy_buff, int node_number)
 * \brief looks at the messages received during the last round and keeps the most recent one and puts it in the first place
 *
 * \param in_phy_buff pointer to the in physical-layer buffer
 * \param node_number is the node number in the simulation
 * \return void
 */
__global__ void Receiver(struct Buffer *in_phy_buff, int node_number, struct RemoteCommBuffer *remote_buff, struct Geo *geo, int offset, float m_recv, float b_recv) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < node_number) {
		
		if (in_phy_buff[tid].write_index > 0) {

			geo[tid].energy -= (m_recv * 4 * (4 + PAYLOAD_SIZE) + b_recv) / 1000.0;

			for (int i=1; i<4; i++) {
				if (in_phy_buff[tid].element[i].header[2]>in_phy_buff[tid].element[0].header[2]) {
					in_phy_buff[tid].element[0] = in_phy_buff[tid].element[i];
				}
			}
			//if (tid + offset > 99)
			//printf("I am recv %d I received %d\n",tid +offset, in_phy_buff[tid].element[0].header[2]);
			
			in_phy_buff[tid].write_index = 1;// to say that if you want to add a message in this buffer then do it in position 1
			in_phy_buff[tid].read_index = 0;// to say if you want to read then i t is in position 0
		}
		
		//printf("OFFSET = %d\n", offset);
		//if (offset == 0) printf("8 ext conn ? ==> %d\n",geo[8].external_conn); else printf("9 ext conn ? ==> %d\n",geo[0].external_conn);
		//Has this node external connections ?
		if (geo[tid].external_conn == 1) { //Then verify if some remote messages were received
			//printf("Who am I ? ==> %d\n", tid + offset);
				
			int ext_conn = 0;
			while (remote_buff[ext_conn].conn_node_id != tid + offset)
				ext_conn++;
			//printf("I am %d, an external connection ! And my index %d - write_ind %d\n", tid + offset, ext_conn, remote_buff[0].write_index);
			
			if (in_phy_buff[tid].write_index == 0 && remote_buff[ext_conn].write_index > 0) {
				in_phy_buff[tid].element[0] = remote_buff[ext_conn].element[0];
				in_phy_buff[tid].write_index = 1;// to say that if you want to add a message in this buffer then do it in position 1
				in_phy_buff[tid].read_index = 0;// to say if you want to read then i t is in position 0	
			}
				
			for (int msg = 0; msg < remote_buff[ext_conn].write_index; msg ++) { 
				if (remote_buff[ext_conn].element[msg].header[2]>in_phy_buff[tid].element[0].header[2]) 
					in_phy_buff[tid].element[0] = remote_buff[ext_conn].element[msg];
			}
		}
		
		//if (tid + offset > 9) {
			//if (in_phy_buff[tid].write_index == 1) printf("node %d received seq = %d\n", tid+offset, in_phy_buff[tid].element[0].header[2]);
			//else printf("node %d received nothing\n", tid+offset);
		//}
		
		
	}
}

 /**
 * \fn void Init_Router(struct RouterBuffer *router_buffer, int node_number)
 * \brief initializes a router buffer
 *
 * \param router_buffer pointer to the router buffer
 * \param node_number is the node number in the simulation
 * \return void
 */
__global__ void Init_Router(struct RouterBuffer *router_buffer, int node_number) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < node_number){
		router_buffer[tid].write_index = 0;
		router_buffer[tid].full = 0;
	}
}

 /**
 * \fn void Router_In (struct Buffer *in_phy_buff, struct Buffer *out_phy_buff,
 *		struct MessageBuffer *message_buffer, struct RouterBuffer *router_buffer, float *current_probability,
 *		float drop_prob, int node_number)
 * \brief compares the message choosen by Receiver kernel to the viewed messages. If it is seen for the first time it is kept
 *
 * \param in_phy_buff pointer to the in physical-layer buffer
 * \param out_phy_buff pointer to the out physical-layer buffer that is used, when the node is not the owner of the message, to broadcast it
 * \param message_buffer pointer to the in application-layer buffer that is used, when the node is the owner of the message, to keep it
 * \param router_buffer pointer to the router buffer
 * \param current_probability pointer to the current probability used to know whether the node should or not drop the message
 * \param drop_prob probability threshold to be compared with the current probability
 * \param node_number is the node number in the simulation
 * \return void
 */
__global__ void Router_In (struct Buffer *in_phy_buff, struct Buffer *out_phy_buff,
		struct MessageBuffer *message_buffer, struct RouterBuffer *router_buffer, float *current_probability,
		float drop_prob, int node_number, int offset) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < node_number) {

		int message_number = in_phy_buff[tid].write_index;
		int i, j;
		int out_index = out_phy_buff[tid].write_index;
		int actualised_TTL, view;
		float prob;

		view = 0;
	
		if (message_number > 0) {

			for (i = 0; i < message_number; i++) //For every message in the node's InBuffer
			{	
				struct Element message = in_phy_buff[tid].element[i]; //Get the message

				for (j = 0; j < ((router_buffer[tid].full == 1) ? Maxelement : router_buffer[tid].write_index); j++) //For every header in the router buffer
				{
					if ((message.header[0] == router_buffer[tid].header_array[j][0])
							&& (message.header[1] == router_buffer[tid].header_array[j][1])
							&& (message.header[2] == router_buffer[tid].header_array[j][2])) //If a header corresponds
					{
						view = ++router_buffer[tid].header_array[j][3]; //Increase the header count (seen one more time)
						
						break; //Stop comparing to other headers
					}
				}

				//If all headers were compared, ie header not found
				if ((j == router_buffer[tid].write_index && router_buffer[tid].full == 0) 
						|| (router_buffer[tid].full == 1 && j == Maxelement)) 
				{
					router_buffer[tid].write_index++; //increase the number of elements in the router's buffer
					if (router_buffer[tid].write_index == Maxelement){
						router_buffer[tid].write_index = 0;
						router_buffer[tid].full = 1;
					}
					
					if (j == Maxelement) { 
						j = (router_buffer[tid].write_index - 1 + Maxelement) % Maxelement;
					}
			
						
					router_buffer[tid].header_array[j][0] = message.header[0]; //Copy the header's information
					router_buffer[tid].header_array[j][1] = message.header[1];
					router_buffer[tid].header_array[j][2] = message.header[2];
					router_buffer[tid].header_array[j][3] = 1; //TTL is replaced by a counter
					view = 1;
				}

				actualised_TTL = --message.header[3];
				//=message.Header[3];//new TTL
				
				/* If the message is seen before, the node won't relay it */
				if (view == 1) {
					//if (tid + offset == 8999) printf("%d received %d\n", tid + offset, message.header[2]);
					if (message.header[1] == tid + offset) /* If the message is mine, it is passed to the application layer */
					{
						message_buffer[tid].write_index = 1;
						message_buffer[tid].read_index = 0;
						for (int k = 0; k < PAYLOAD_SIZE; k++) {
							message_buffer[tid].payload[0][k] = message.payload[k];
						}

					} else {
	
						prob = current_probability[tid];
									
						if ((prob >= drop_prob) && (actualised_TTL != 0)
								&& (out_index < Maxelement)) //If you can write in the node's OutBuffer and the TTL isn't 0
						{
							out_phy_buff[tid].element[out_index] = message;	//Write the message une copy pas une affectation
							out_phy_buff[tid].element[out_index].header[3] = actualised_TTL; //Change the TTL
							out_index++; //increase the OutBuffer's writeIndex (outIndex is used as writeIndex temporarily)
							out_phy_buff[tid].read_index = 0;
						}
					}
				}
			}
		}
		out_phy_buff[tid].write_index = out_index;//Update the real OutBuffer's writeIndex
		in_phy_buff[tid].write_index = 0; //Clear the InBuffer
	}
}

 /**
 * \fn void Application_In(MessageBuffer *app_in, int N, int *dev_tot_dest)
 * \brief consumes the received message
 *
 * \param app_in pointer to the in application-layer buffer used to read received messages
 * \param node_number is the node number in the simulation
 * \param dev_tot_dest pointer to the monitoring variable that will finally contain the total number of messages received by this node
 * \return void
 */
__device__ void Application_In(MessageBuffer *app_in, int tid, int *dev_tot_dest) {

	int messages_number = app_in[tid].write_index;

	if (messages_number > 0) {

		app_in[tid].payload[0][0] = 10; /* To say that the node read the message, it sets its first number to 10 */

		/* Here we increment the number of messages arrived to our final destination */
		(*dev_tot_dest)++;
			
	}
	app_in[tid].write_index = 0;


}

__global__ void Message_In(MessageBuffer *app_in, int node_number,
		int *dev_tot_dest) {

	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < node_number) {

		int messages_number = app_in[tid].write_index;
		
		if (messages_number > 0) {
			if (app_in[tid].payload[0][0] < 2147483647) 
				Application_In(app_in, tid, dev_tot_dest);
			//else
				//HERE CALL TC_IN
						
		} 
	}

}


/*

__global__ void Message_In(MessageBuffer *app_in, int node_number, int *dev_tot_dest) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < node_number) {
		int messages_number = app_in[tid].write_index;
		
		if (messages_number > 0) {
		printf("I am here\n\r");
			//printf("--- %d -- %d ---\n\r",app_in[tid].payload[0][0], (int)(powf(2, sizeof(int) * 8 - 1)));
			if ((app_in[tid].payload[0][0] & (int)(powf(2, sizeof(int) * 8 - 1))) != 0)
				//TC_In (); //TODO this is to be continued...
				printf ("Here is a TC message !\n\r");
			else
				Application_In (app_in, dev_tot_dest, tid);
		}
	}
}

*/
