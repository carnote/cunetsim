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
 * \file communicationthread.cpp.cpp
 * \brief Program that manages the communication with the simulator
 * \author MS MOSLI
 * \version 0.0.2
 * \date
 */

#include "communicationthread.h"

extern int pfd[2];
extern struct Geo geo[NB_DISPLAYED_NODES];
enum Role role[NB_DISPLAYED_NODES];
extern int nb_tours;
extern int newMessage[NB_DISPLAYED_NODES];
extern int regularDelay;

CommunicationThread::CommunicationThread(MyWindow* window){
    this->window = window;
    QObject::connect(this, SIGNAL(newData(QString)), window, SLOT(writeToConsole(QString)));
    QObject::connect(this, SIGNAL(newPosition()), window->getGL(), SLOT(drawNewPosition()));
    QObject::connect(this, SIGNAL(endOfTheSimulation()), window, SLOT(endOfTheSimulation()));
}

void CommunicationThread::run()
{
    int nread;
    int counter = 100 + nb_tours;
    Data_Flow_Unit data;
    Final_Data final;

    while (counter > 0){
        switch( nread = read(pfd[0], &data, sizeof(Data_Flow_Unit))) {
        case -1 :
            perror( "read" );
            break;
        case 0 :
            perror( "EOF" );
            break;
        default :
            QString information;
            information.sprintf("Tour %d - Node 0 - Position (%d,%d,%d)" , data.tour, data.geo[0].P.x, data.geo[0].P.y, data.geo[0].P.z);
            emit newData(information);
            if (counter % 5 == 0) {
                for (int i = 0; i<NB_DISPLAYED_NODES; i++){
                    geo[i].Neighbors = data.geo[i].Neighbors;
                    geo[i].P = data.geo[i].P;
                    for(int j = 0; j<data.geo[i].Neighbors; j++)
                        geo[i].Neighbor[j] = data.geo[i].Neighbor[j];
                }
            }
            for (int i = 0; i<NB_DISPLAYED_NODES; i++) {
                newMessage[i] = data.newMessage[i];
		role[i] = data.role[i];
	    }
            emit newPosition();
            break;
        }
        counter--;
        if (regularDelay){
            window->makeAPause();
            usleep(regularDelay);
        }
    }

    switch(nread = read(pfd[0], &final, sizeof(Final_Data))){
        case -1 :
            perror( "read" );
            break;
        case 0 :
            perror( "EOF" );
            break;
        default :
            QString final_data;
            final_data.sprintf("\n*** Final results ***\n\n* Node number: %d\n* Loss rate: %f\n* All forwarded packets: %d\n* Used memory [device, host]: [%d, %d]\n* Elapsed time (average): %f\n* Elapsed time (min): %f\n* Elapsed time (max): %f\n",final.node_number, final.loss, final.forwarded,final.device_memory, final.host_memory,final.average_time, final.min_time, final.max_time);
            emit newData(final_data);
            emit endOfTheSimulation();
            break;
    }
}
