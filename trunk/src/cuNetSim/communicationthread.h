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
 * \file communicationthread.h
 * \brief Program that manages the communication with the simulator
 * \author MS MOSLI
 * \version 0.0.2
 * \date
 */

#ifndef COMMUNICATIONTHREAD_H
#define COMMUNICATIONTHREAD_H

#include <QThread>
#include <QDebug>
#include <QTextEdit>
#include "mywindow.h"
#include "structures.h"

class CommunicationThread : public QThread
{
    Q_OBJECT

    public:
        CommunicationThread(MyWindow* window);

    signals:
        void newData(QString data);
        void newPosition();
        void endOfTheSimulation();

    private:
        void run();
        MyWindow* window;
};

#endif // COMMUNICATIONTHREAD_H
