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
 * \file mywindow.cpp
 * \brief Program that manages the main window
 * \author MS MOSLI
 * \version 0.0.2
 * \date
 */

#include "mywindow.h"
#include "communicationthread.h"
#include <QGridLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QCheckBox>
#include <QSlider>

extern pid_t simulator_pid;
extern int tube_descriptor;
extern CommunicationThread* communication_thread;
extern int regularDelay;
extern int isPaused;

MyWindow::MyWindow() : QWidget()
{
    simulatorIsInitialized = 0;
    pattern = 0;
    regularDelay = (NB_DISPLAYED_NODES/400) * 20000;
    isPaused = 1;

    this->setFixedSize(950, 700);

    /* Control widgets */
    runButton = new QPushButton("Run");
    QObject::connect(runButton, SIGNAL(clicked()), this, SLOT(runSimulator()));

    pauseButton = new QPushButton("Pause");
    pauseButton->setEnabled(false);
    QObject::connect(pauseButton, SIGNAL(clicked()), this, SLOT(pause()));

    switchView = new QPushButton("Switch View");
    QObject::connect(switchView, SIGNAL(clicked()), this, SLOT(switchTheView()));

    exit = new QPushButton("Exit");
    exit->setEnabled(false);
    QObject::connect(exit, SIGNAL(clicked()), qApp, SLOT(quit()));

    QLabel *pattern = new QLabel;
    pattern->setPixmap(QPixmap("../../Debug/motif.png"));
    pattern->setFixedSize(120,120);

    QLabel *pattern1 = new QLabel;
    pattern1->setPixmap(QPixmap("../../Debug/motif1.png"));
    pattern1->setFixedSize(120,120);

    switchPatt = new QPushButton("Switch Pattern");
    QObject::connect(switchPatt, SIGNAL(clicked()), this, SLOT(switchPattern()));

    QCheckBox *drawConnections = new QCheckBox("Draw Connections");
    drawConnections->setChecked(true);
    QObject::connect(drawConnections, SIGNAL(stateChanged(int)), this, SLOT(setDrawConnections(int)));


    QFrame *speed_field = new QFrame(this);
    QLabel *min = new QLabel("Fastest");
    QLabel *max = new QLabel("Slowest");
    QSlider* speed = new QSlider;
    speed->setMinimum(NB_DISPLAYED_NODES / 400);
    speed->setMaximum(10);
    speed->setValue(NB_DISPLAYED_NODES / 400);
    speed->setOrientation(Qt::Horizontal);
    QObject::connect(speed, SIGNAL(sliderMoved(int)), this, SLOT(changeDelay(int)));
    QHBoxLayout *speed_layout = new QHBoxLayout;
    speed_layout->addWidget(min);
    speed_layout->addWidget(speed);
    speed_layout->addWidget(max);
    speed_field->setLayout(speed_layout);


    /* Control area */
    control_field = new QFrame(this);
    control_field->setFrameShape(QFrame::StyledPanel);
    control_field->setFrameStyle( QFrame::Sunken | QFrame::Panel );
    control_field->setLineWidth( 2 );

    pattern_field = new QFrame(this);
    pattern_field->setFrameShape(QFrame::StyledPanel);
    pattern_field->setFixedSize(270,140);
    QHBoxLayout *pattern_layout = new QHBoxLayout;
    pattern_layout->addWidget(pattern);
    pattern_layout->addWidget(pattern1);
    pattern_field->setLayout(pattern_layout);

    QVBoxLayout *control_layout = new QVBoxLayout;
    control_layout->addWidget(runButton);
    control_layout->addWidget(pauseButton);
    control_layout->addWidget(switchView);
    control_layout->addWidget(pattern_field);
    control_layout->addWidget(switchPatt);
    control_layout->addWidget(exit);
    control_layout->addWidget(drawConnections);
    control_layout->addWidget(speed_field);
    control_field->setLayout(control_layout);

    /* Drawing area */
    openGL_field = new QFrame(this);
    openGL_field->setFrameStyle( QFrame::Sunken | QFrame::Panel );
    openGL_field->setLineWidth( 2 );
    openGl = new OpenGLWidget();
    openGl->setFixedSize(600,440);
    QVBoxLayout *l1 = new QVBoxLayout;
    l1->addWidget(openGl);
    openGL_field->setLayout(l1);

    /* Console area */
    console_field = new QFrame(this);
    output = new QTextEdit();
    output->setReadOnly(true);
    QVBoxLayout *l2 = new QVBoxLayout;
    l2->addWidget(output);
    console_field->setLayout(l2);

    QGridLayout *layout = new QGridLayout;
    layout->addWidget(openGL_field, 0, 0, 1, 3);
    layout->addWidget(control_field, 0, 3, 3, 1);
    layout->addWidget(console_field, 3, 0, 3, 4);

    this->setLayout(layout);
}

QTextEdit* MyWindow::getConsoleField(){
    return this->output;
}

OpenGLWidget *MyWindow::getGL(){
    return this->openGl;
}

void MyWindow::runSimulator(){

    if (simulatorIsInitialized == 0){

        if ( mkfifo("../../Debug/comm_tube", 0666) == -1)
            perror("mkfifo");

        initSimulator();
        tube_descriptor = open("../../Debug/comm_tube", O_RDONLY);
        communication_thread = new CommunicationThread(this);
        communication_thread->start();
        simulatorIsInitialized = 1;
        exit->setEnabled(false);
        runButton->setEnabled(false);
        isPaused = 0;
        if(!regularDelay)
           pauseButton->setEnabled(true);
        return;
    }
}

void MyWindow::changeDelay(int delay){
    regularDelay = 20000 * delay;
    if (delay == 0){
        pauseButton->setEnabled(true);
    }else if (!isPaused){
        pauseButton->setEnabled(false);
    }
}

void MyWindow::pause(){
    if (!isPaused){
        kill(simulator_pid, SIGSTOP);
        isPaused = 1;
        pauseButton->setText("Rerun");
    }else{
        kill(simulator_pid, SIGCONT);
        isPaused = 0;
        pauseButton->setText("Pause");
        if (regularDelay){
            pauseButton->setEnabled(false);
        }
    }
}

void MyWindow::makeAPause(){
    kill(simulator_pid, SIGSTOP);
    usleep(regularDelay); // can be parametrable
    kill(simulator_pid, SIGCONT);
}

void MyWindow::endOfTheSimulation(){
    simulatorIsInitialized = 0;
    communication_thread->wait();  // do not exit before the thread is completed!
    delete communication_thread;
    runButton->setEnabled(true);
    pauseButton->setEnabled(false);
    exit->setEnabled(true);
    if (::close (tube_descriptor) == -1 ) /* we close the read desc. */
        perror( "close on read" );
    if (unlink("../../Debug/comm_tube") == -1)
        perror("unlink");
}

void MyWindow::setDrawConnections(int draw){
    openGl->setDrawConnections(draw);
}

void MyWindow::switchTheView(){
    openGl->switchView();
}

void MyWindow::switchPattern(){

    pattern = (pattern+1)%2;
    openGl->setPattern(pattern);
}

void MyWindow::writeToConsole(QString data){
    this->output->append(data);
}

void MyWindow::initSimulator(){

    simulator_pid = fork();

    switch(simulator_pid)
    {
        case -1 :
            perror( "fork" );

        case 0 : /* child is going to be the simulator, it is the writer */

            /* execl is used to launch the simulator */
            execl( "/usr/bin/mpiexec.openmpi", "mpiexec.openmpi", "-n", "5", "-hostfile",
                   "../../Debug/hosts", "../../Debug/Cunetsim", "-g", NULL);

            perror( "execl" );

    }

}


MyWindow::~MyWindow() {}
