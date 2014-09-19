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
 * \file mywindow.h
 * \brief Program that manages the main window
 * \author MS MOSLI
 * \version 0.0.2
 * \date
 */

#ifndef MYWINDOW_H
#define MYWINDOW_H

#include <QPushButton>
#include <QFrame>
#include <QTextEdit>
#include <QApplication>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
#include <signal.h>
#include "openglwidget.h"

class MyWindow : public QWidget
{
    Q_OBJECT

    public:
        MyWindow();
        QTextEdit* getConsoleField();
        void initSimulator();
        void makeAPause();
        OpenGLWidget* getGL();
        ~MyWindow();

    public slots:
        void runSimulator();
        void pause();
        void writeToConsole(QString data);
        void endOfTheSimulation();
        void switchTheView();
        void switchPattern();
        void setDrawConnections(int draw);
        void changeDelay(int delay);

    private:
        int simulatorIsInitialized;
        int pattern;
        QPushButton *switchView;
        QPushButton *runButton;
        QPushButton *pauseButton;
        QPushButton *exit;
        QPushButton* switchPatt;
        QFrame *control_field;
        QFrame *openGL_field;
        QFrame *pattern_field;
        OpenGLWidget *openGl;
        QFrame *console_field;
        QTextEdit *output;
};

#endif // MYWINDOW_H
