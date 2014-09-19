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
 * \file Buffer.cu
 * \brief Functions necessary for buffer initialization
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
 * \fn void Init_buffer(struct Buffer *buffer, int node_number)
 * \brief puts default values in a physical layer buffer to initialize it
 *
 * \param buffer pointer to the buffer to be initialized
 * \param node_number size of Buffer which is also the node number in the simulation
 * \return void
 */
__global__ void Init_Buffer(struct Buffer *buffer, int node_number) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	int i = 0;

	if (tid < node_number) {
		buffer[tid].write_index = 0;
		buffer[tid].read_index = -1;
		for (i = 0; i < Maxelement; i++) {
			buffer[tid].element[i].header[0] = -1;
			buffer[tid].element[i].header[1] = -1;
			buffer[tid].element[i].header[2] = -1;
			buffer[tid].element[i].header[3] = -1;
		}
	}
}

/**
 * \fn void Init_App_buffer(struct MessageBuffer *message_buffer, int node_number)
 * \brief puts default values in an application layer buffer to initialize it
 *
 * \param message_buffer pointer to the buffer to be initialized
 * \param node_number size of Buffer which is also the node number in the simulation
 * \return void
 */
__global__ void Init_App_Buffer(struct MessageBuffer *message_buffer, int node_number) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < node_number) {
		message_buffer[tid].write_index = 0;
		message_buffer[tid].read_index = -1;

		/*
		 * Dest values are used in the routing step to know whether there is a message to send. -1 means no message to send.
		 * Other values represent the rank of the receiver
		 */

		for (int i = 0; i < Maxelement; i++) {
			message_buffer[tid].dest[i] = -1;
			message_buffer[tid].payload[i][0] = -5;
		}
	}
}
