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
 * \file openglwidget.h
 * \brief Program that manages the simulation display widget.
 * \author MS MOSLI
 * \version 0.0.2
 * \date
 */

#ifndef OPENGLWIDGET_H
#define OPENGLWIDGET_H

#include <QGLWidget>
#include "qgl.h"
#include "structures.h"

class OpenGLWidget : public QGLWidget
{
    Q_OBJECT

    public:
        OpenGLWidget();
        void drawGrid();
        void drawNodes();
        void drawConnections();
        void drawMessages();
        void setDrawConnections(int draw);
        void switchView();
        void setPattern(int patt);
        ~OpenGLWidget();

    protected:
        void paintGL();

    public slots:
        void drawNewPosition();

    private:
        int camera[9];
        int view;
        int pattern;
        bool draw_connections;
};

#endif // OPENGLWIDGET_H
