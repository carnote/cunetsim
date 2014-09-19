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
 * \file openglwidget.cpp
 * \brief Program that manages the simulation display widget.
 * \author MS MOSLI
 * \version 0.0.2
 * \date
 */

#include "openglwidget.h"
#include <stdio.h>

extern struct Geo geo[NB_DISPLAYED_NODES];
extern enum Role role[NB_DISPLAYED_NODES];
extern int newMessage[NB_DISPLAYED_NODES];
extern Position area;
extern int cell_size;

OpenGLWidget::OpenGLWidget()
{
    geo[0].P.x = -1;
    view = 0;
    pattern = 0;
    draw_connections = true;
    camera[0] = area.x/2;
    camera[1] = area.y/2;
    camera[2] = area.z+((area.x+area.y)*2)/5;
    camera[3] = area.x/2;
    camera[4] = area.y/2;
    camera[5] = area.z/2;
    camera[6] = 0;
    camera[7] = 1;
    camera[8] = 0;
}

void OpenGLWidget::paintGL()
{

  glClearColor(0,0,0,0);
  //glClearColor(255,255,255,0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  gluPerspective(70,(double)600/440,1,1000);

  glMatrixMode( GL_MODELVIEW );
  glLoadIdentity();

  gluLookAt(camera[0], camera[1], camera[2], camera[3], camera[4], camera[5],camera[6],camera[7],camera[8]);

  drawGrid();
  drawNodes();

  if (draw_connections)
    drawConnections();

}

void OpenGLWidget::drawNewPosition(){
    updateGL();
}

void OpenGLWidget::setDrawConnections(int draw){
    this->draw_connections = draw;
    updateGL();
}

void OpenGLWidget::setPattern(int patt){
    this->pattern = patt;
    updateGL();
}

void OpenGLWidget::switchView(){
    if (view == 0){
        view = 1;
        camera[1] = -(area.y/3)*2;
        camera[2] = area.z/2;
        camera[7] = 0;
        camera[8] = 1;
        updateGL();
    }else{
        view = 0;
        camera[1] = area.y/2;
        camera[2] = area.z+((area.x+area.y)*2)/5;
        camera[7] = 1;
        camera[8] = 0;
        updateGL();
    }
}

OpenGLWidget::~OpenGLWidget(){}

void OpenGLWidget::drawConnections(){

    for (int i=0; i<NB_DISPLAYED_NODES; i++){
        for (int j=i+1; j<NB_DISPLAYED_NODES; j++){
            int k=0;

            while((geo[i].Neighbor[k]!=j)&&(k<geo[i].Neighbors)){
                k++;
            }

            if(k < geo[i].Neighbors){

                //The choice of the color is based on the seq number of the message to be sent
                if (newMessage[i] == -1)
                    glColor3d(255,255,255);
                else
                    glColor3d((newMessage[i]%10)*10,((newMessage[i]/10)%10)*10,(newMessage[i]/100)*20);

                //choose it according to the number of displayed nodes
                glLineWidth(1.2);
                glBegin(GL_LINES);
                    glVertex3d(geo[i].P.x,geo[i].P.y,geo[i].P.z);
                    glVertex3d(geo[j].P.x,geo[j].P.y,geo[j].P.z);
                glEnd();
            }
        }
    }


}

void OpenGLWidget::drawNodes(){

    GLUquadric* params;
    params = gluNewQuadric();
    gluQuadricDrawStyle(params,GLU_FILL);

    if (geo[0].P.x != -1){
        glColor3d(255,255,255);

        glTranslated(geo[0].P.x, geo[0].P.y, geo[0].P.z);

	if (role[0] == CH) 
		glColor3d(255,0,0);
	else if (role[0] == ND) 
		glColor3d(0,255,0);
	else
		glColor3d(255,255,255);

        gluSphere(params,5,25,25);

        for (int i=1; i<NB_DISPLAYED_NODES; i++){

            glTranslated(geo[i].P.x - geo[i-1].P.x,geo[i].P.y - geo[i-1].P.y, geo[i].P.z - geo[i-1].P.z);

	    if (role[i] == CH) 
		glColor3d(255,0,0);
	    else if (role[i] == ND) 
		glColor3d(0,255,0);
	    else
		glColor3d(255,255,255);

            gluSphere(params,5,25,25);
        }

        glTranslated(-geo[NB_DISPLAYED_NODES-1].P.x,-geo[NB_DISPLAYED_NODES - 1].P.y,-geo[NB_DISPLAYED_NODES - 1].P.z);
    }
    gluDeleteQuadric(params);
}

void OpenGLWidget::drawGrid(){
    glColor3d(0,0,255);
    glLineWidth(1.0);

    if (pattern){
        glEnable(GL_LINE_STIPPLE);
        glLineStipple (1, 0x1C47);
    }

    glBegin(GL_LINES);

    /* Lines that are parallel to (Ox) */
    for (int j=0; j <= area.z; j+= cell_size)
        for (int i=0; i <= area.y; i+= cell_size){
            glVertex3d(0,i,j);
            glVertex3d(area.x,i,j);
        }

     /* Lines that are parallel to (Oy) */
     for (int j=0; j <= area.z; j+= cell_size)
         for (int i=0; i <= area.x; i+= cell_size){
            glVertex3d(i,0,j);
            glVertex3d(i,area.y,j);
         }

     /* Lines that are parallel to (Oz) with x = 0 */
     for (int j=0; j <= area.x; j+= cell_size)
         for (int i=0; i <= area.y; i+= cell_size){
            glVertex3d(j,i,0);
            glVertex3d(j,i,area.z);
         }

    glEnd();
    if (pattern)
        glDisable(GL_LINE_STIPPLE);
}

