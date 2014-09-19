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
 * \file ConnectivityControl.cu
 * \brief Function that verifies the correctness of the connectivity step
 * \author Bilel BR
 * \version 0.0.2
 * \date Oct 25, 2011
 */


#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include "../structures.h"

#endif /* STRUCTURES_H_ */
#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "../interfaces.h"
#endif /* INTERFACES_H_ */
#include "../vars.h"
#define xmax 1600
#define ymax 1600
#define zmax 1600


// When using it, don't forget to use the HOST as parameter after having copied the device_geo in the host one
// This functions produces error messages when a node is not connected to some other close nodes because it has already 
// too many neighbor_number, think of it especially when problems come with a great number of nodes
 
 /**
 * \fn __host__ int Connectivity_control(struct Geo *geo, int node_number, int round, struct Cell *cell, int cell_size)
 * \brief verifies the correctness of the connectivity step
 *
 * \param geo pointer to the space data needed to read a node's neighbors
 * \param cell pointer to the cell distribution data needed to display details in case an error is found
 * \param node_number is node number in the simulation
 * \param cell_size is the length of the side of the cubic or square cell
 * \param round is the current round, displayed in case an error is identified
 * \return 0 in case no errors were detected and 1 when an error is detected
 */
__host__ int Connectivity_control(struct Geo *geo, int node_number, int round, struct Cell *cell, int cell_size)
{
    int i,j,k = 0;
    float dist;
   
    for (i = 0; i < node_number; i++)
    {

        for(j = (i+1); j < node_number; j++)
        {
            dist = Distance(geo[i].p,geo[j].p);
            if(dist <= ((float)cell_size/2.0) + 0.0001)
            {	
            	k=0;
		
                while((geo[i].neighbor_vector[k]!=j)&&(k<geo[i].neighbor_number))
                {
                    k++;
                }
                if(k==geo[i].neighbor_number)
                {
                    printf("connectivity error between %d and %d with distance %f and turn %d \n\r",i,j,dist,round);
                    printf("cell id of %d = %d , cell id of %d= %d\r\n",i,geo[i].cell_id,j,geo[j].cell_id);
		    		printf("Neighbor numbers are %d and %d\n",geo[i].neighbor_number, geo[j].neighbor_number);
		    
		    		/* 
		    		for (int h=0; h < C[G[i].cell_id].size; h++)
					printf("[%d]\n",C[G[i].cell_id].member[h]);
		    		*/
		    
		    		return 1;
                }
            }
        }
    }
    return 0;
}
