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
 * \file Mobility.cu
 * \brief Functions necessary for the mobility process
 * \author Bilel BR
 * \version 0.0.2
 * \date Nov 8, 2011
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
 * \fn __global__ void Mobility(struct Geo *geo, float *randx,float *randy,float *randz,float *randv, int node_number, Simulation_Parameters *device_simulation_parameters)
 * \brief does the nodes movement process
 *
 * \param geo pointer to the space data that will be updated following mobility type
 * \param randx pointer to a random value of x component that will be eventually involved when calculating the new position
 * \param randy pointer to a random value of y component that will be eventually involved when calculating the new position
 * \param randz pointer to a random value of z component that will be eventually involved when calculating the new position
 * \param randv pointer to a random value of speed that will be eventually involved when calculating the new position
 * \param node_number is the node number in the simulation
 * \param device_simulation_parameters pointer to a device copy of the global structure containing simulation parameters
 * \return void
 */
__global__ void Mobility(struct Geo *geo, float *randx,float *randy,float *randz,float *randv, int node_number, Simulation_Parameters *device_simulation_parameters)
{
	int tid=blockDim.x * blockIdx.x + threadIdx.x;
	int v_min, v_max;
	enum Mobility_Type mode;
	int _3D = device_simulation_parameters->simulation_config._3D_is_activated;

	if(tid<node_number)
	{
		
		mode = device_simulation_parameters->topology_config.mobility_parameters.mobility_type;
		v_min = device_simulation_parameters->topology_config.mobility_parameters.moving_dynamics.min_speed_mps;
		v_max = device_simulation_parameters->topology_config.mobility_parameters.moving_dynamics.max_speed_mps;
		
		switch(mode)
		{
			case IMMOBILE:
			{
				break;
			}

			case RANDOM_MOVEMENT:
				///Random Movement (Non Uniform)
			{
				geo[tid].p.x=geo[tid].p.x+(int)((float)geo[tid].speedx*(2.0*randx[tid]-1.0));
				geo[tid].p.y=geo[tid].p.y+(int)((float)geo[tid].speedy*(2.0*randy[tid]-1.0));
			   	if ( _3D )
					geo[tid].p.z=geo[tid].p.z+(int)((float)geo[tid].speedz*(2.0*randz[tid]-1.0));
			   	else
					geo[tid].p.z=0;
				break;
			}
			case LINEAR_MOVEMENT:
				///Linear movement
			{
				geo[tid].p.x=geo[tid].p.x+geo[tid].speedx;
				geo[tid].p.y=geo[tid].p.y+geo[tid].speedy;
			   	if ( _3D )
					geo[tid].p.z=geo[tid].p.z+geo[tid].speedz;
			   	else
					geo[tid].p.z=0;
				break;
			}
			case RANDOM_UNIFORM_MOVEMENT_MAX_BOUNDED:
				///Random Uniform Movement bounded by vmax
			{
				geo[tid].p.x=geo[tid].p.x+(int)((float)v_max*(2.0*randx[tid]-1.0));
				geo[tid].p.y=geo[tid].p.y+(int)((float)v_max*(2.0*randy[tid]-1.0));
			    if ( _3D )				
					geo[tid].p.z=geo[tid].p.z+(int)((float)v_max*(2.0*randz[tid]-1.0));
			    else
					geo[tid].p.z=0;
				break;
			}
			case RANDOM_UNIFORM_MOVEMENT_MAX_AND_MIN_BOUNDED:
				///Random Uniform Movement bounded by vmax and vmin
			{
				geo[tid].p.x=geo[tid].p.x+(int)((float)(randv[tid]*(v_max-v_min)+v_min)*(2.0*randx[tid]-1.0));
				geo[tid].p.y=geo[tid].p.y+(int)((float)(randv[tid]*(v_max-v_min)+v_min)*(2.0*randy[tid]-1.0));
			    if ( _3D )
					geo[tid].p.z=geo[tid].p.z+(int)((float)(randv[tid]*(v_max-v_min)+v_min)*(2.0*randz[tid]-1.0));
			    else
					geo[tid].p.z=0;
				break;
			}

			default:
			// Default is set to linear movement
			{
				geo[tid].p.x=geo[tid].p.x+geo[tid].speedx;
				geo[tid].p.y=geo[tid].p.y+geo[tid].speedy;
			   	if ( _3D )
					geo[tid].p.z=geo[tid].p.z+geo[tid].speedz;
			    else
					geo[tid].p.z=0;
				break;
			}

		}
	}
}

//is not to the point yet
void Mobility(struct Geo *geo, float *randx,float *randy,float *randz, int rank, Simulation_Parameters *simulation_parameters)
{
 	int tid=rank;
 	
 	enum Mobility_Type mode = simulation_parameters->topology_config.mobility_parameters.mobility_type;
	int _3D = simulation_parameters->simulation_config._3D_is_activated;
	int v_min = simulation_parameters->topology_config.mobility_parameters.moving_dynamics.min_speed_mps;
	int v_max = simulation_parameters->topology_config.mobility_parameters.moving_dynamics.max_speed_mps;

 	switch(mode)
 	{
 		case RANDOM_MOVEMENT:
 		///Random Movement (Non Uniform)
 		{
 			geo[tid].p.x=geo[tid].p.x+(int)((float)geo[tid].speedx*(2.0*randx[tid]-1.0));
 			geo[tid].p.y=geo[tid].p.y+(int)((float)geo[tid].speedy*(2.0*randy[tid]-1.0));
			if ( _3D )
 				geo[tid].p.z=geo[tid].p.z+(int)((float)geo[tid].speedz*(2.0*randz[tid]-1.0));
			else
				geo[tid].p.z=0;

 			break;
 		}
 		case LINEAR_MOVEMENT:
 		///Linear movement
 		{
 			geo[tid].p.x=geo[tid].p.x+geo[tid].speedx;
 			geo[tid].p.y=geo[tid].p.y+geo[tid].speedy;
			if ( _3D )
 				geo[tid].p.z=geo[tid].p.z+geo[tid].speedz;
			else
				geo[tid].p.z=0;
 			break;
 		}
 		case RANDOM_UNIFORM_MOVEMENT_MAX_BOUNDED:
 		///Random Uniform Movement bounded by vmax
 		{
 			geo[tid].p.x=geo[tid].p.x+(int)((float)v_max*(2.0*randx[tid]-1.0));
			geo[tid].p.y=geo[tid].p.y+(int)((float)v_max*(2.0*randy[tid]-1.0));
			if ( _3D )				
				geo[tid].p.z=geo[tid].p.z+(int)((float)v_max*(2.0*randz[tid]-1.0));
			else
				geo[tid].p.z=0;
			break;
 		}
 		case RANDOM_UNIFORM_MOVEMENT_MAX_AND_MIN_BOUNDED:
 		///Random Uniform Movement bounded by vmax and vmin
 		{///randZ ou randV ????
 			geo[tid].p.x=geo[tid].p.x+(int)((float)(randz[tid]*(vmax-vmin)+vmin)*(2.0*randx[tid]-1.0));
 			geo[tid].p.y=geo[tid].p.y+(int)((float)(randz[tid]*(vmax-vmin)+vmin)*(2.0*randy[tid]-1.0));
			if ( _3D )
 				geo[tid].p.z=geo[tid].p.y+(int)((float)(randz[tid]*(vmax-vmin)+vmin)*(2.0*randz[tid]-1.0));
			else
				geo[tid].p.z=0;
 			break;
 		}
		default:
		// Default is set to linear movement
 		{
 			geo[tid].p.x=geo[tid].p.x+geo[tid].speedx;
 			geo[tid].p.y=geo[tid].p.y+geo[tid].speedy;
			if ( _3D )
 				geo[tid].p.z=geo[tid].p.z+geo[tid].speedz;
			else
				geo[tid].p.z=0;
			break;
 		}

 	}
 }
