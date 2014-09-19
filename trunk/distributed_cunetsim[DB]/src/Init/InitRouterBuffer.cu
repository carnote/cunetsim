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
 * \file InitRouterBuffer.cu
 * \brief Functions necessary to initialize the router buffer in the CPU
 * \author Bilel BR
 * \version 0.0.2
 * \date Dec 6, 2011
 */

#ifndef STRUCTURES_H_
#include "../structures.h"
#define STRUCTURES_H_
#endif /* STRUCTURES_H_ */


/**
* \fn __host__ void Init_router_buffer(struct RouterBuffer *host_router_buffer, int node_number)
* \brief this function allows to initialize a router buffer in the CPU
*
* \param host_router_buffer pointer to the buffer to be initialized
* \param node_number size of Buffer which is also the node number in the simulation
* \return void
*/
__host__ void Init_router_buffer(struct RouterBuffer *host_router_buffer, int node_number)
{
	for(int i=0;i<node_number;i++)
	{
		host_router_buffer[i].write_index=0;
		for(int j=0;j<Maxelement;j++)
		{
			host_router_buffer[i].header_array[j][0]=0;
			host_router_buffer[i].header_array[j][1]=0;
			host_router_buffer[i].header_array[j][2]=0;
			host_router_buffer[i].header_array[j][3]=0;
		}
	}
}
