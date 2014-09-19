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
 * \file structures.h
 * \brief file that contains the global constants and data structures used in the communication with the simulator
 * \author MS MOSLI
 * \version 0.0.2
 * \date
 */

#ifndef STRUCTURES_H
#define STRUCTURES_H

#define Maxneighbor 64
#define NB_DISPLAYED_NODES 200

typedef struct Position {
        int x;
        int y;
        int z;
} Position;

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

typedef struct Data_Flow_Unit {
        struct Geo geo[NB_DISPLAYED_NODES];
        int newMessage[NB_DISPLAYED_NODES];
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

#endif // STRUCTURES_H
