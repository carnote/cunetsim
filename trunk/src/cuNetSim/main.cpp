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
 * \file main.cpp
 * \brief Program that manages the GUI. It initializes the parameters and creates the window
 * \author MS MOSLI
 * \version 0.0.2
 * \date
 */

#include "mywindow.h"
#include <stdlib.h>
#include <sys/stat.h>
#include "communicationthread.h"

//All of these variables have to be collected in a same file or structure
pid_t simulator_pid;
int pfd[2];
struct Geo geo[NB_DISPLAYED_NODES];
int newMessage[NB_DISPLAYED_NODES];
Position area;
int nb_tours;
int cell_size;
int regularDelay;
int isPaused;
CommunicationThread* communication_thread;

void init_parameters(){
    cell_size = 200;
    area.x = 600;
    area.y = 600;
    area.z = 200;
    nb_tours = 600;
}

int main(int argc, char *argv[])
{
    init_parameters();

    QApplication app(argc, argv);

    MyWindow window;
    window.move(*(new QPoint(150,100)));
    window.show();

    int end_value = app.exec();

    return end_value;
}
