/*
 * InitControl.cu
 *
 *  Created on: Nov 23, 2011
 *      Author: benromdh
 */

#ifndef STRUCTURES_H_
#define STRUCTURES_H_
#include "../structures.h"
#endif /* STRUCTURES_H_ */
#ifndef INTERFACES_H_
#define INTERFACES_H_
#include "../interfaces.h"
#endif /* INTERFACES_H_ */

#define xmax 1600
#define ymax 1600
#define zmax 1600

__host__ void Init_control(struct Cell *host_cell, struct Geo *host_geo, struct Geo2 * host_g2, struct Node *host_nodes, int node_number)
{
	Cell_control(host_cell);
	Geo_control(host_cell,host_geo,node_number);
	//Node_Control to be created
	//Geo2_Control to be created
}



//to much error in this function TODO
__host__ void Cell_control(struct Cell * cell)
{
    int i,j,k,distance;
    for(i=0;i<B;i++)
    {  printf("%d [",i);
        for(k=0;k<(cell[i].neighbor_number+1);k++)//+1 pour voir un zera a la fin pour verification
        {  printf("%d ",cell[i].neighbor_vector[k]);}
        printf("]\n\r");
        for(j=0;j<B;j++)
        {
            distance=(int) sqrt(pow(cell[i].center.x-cell[j].center.x,2)+pow(cell[i].center.y-cell[j].center.y,2)+pow(cell[i].center.z-cell[j].center.z,2));
            if(distance >= 2*Visibility)
            {
                //for(k=0;k<27;k++) encore une erreur fatale
                for(k=0;k<(cell[i].neighbor_number);k++)
                {
                    //if(cell[i].V[k]=j) erreur catastrophique !!
                	if(cell[i].neighbor_vector[k]==j)
                    {
                    	//printf("error false neighbor");
                    			/*
                    			 * Printf avec information pour debogage
                    			 * */
                				printf("distance between %d %d = %d\n\r", i, j, distance);
                    			printf("False neighbor between /%d [%d %d %d]and %d [%d %d %d]\n\r",\
                    					i,cell[i].center.x,cell[i].center.y,cell[i].center.z,\
                    					j,cell[j].center.x,cell[j].center.y, cell[j].center.z);
                    }
                }
            }
            else
            {
            	k=0;
            	while((cell[i].neighbor_vector[k]!=j)&&(k<27))
            	{
            		k++;
            	}
            	if(k==27)
            	{
            		//printf("error neighbor unfound");
    				printf("distance between %d %d = %d\n\r", i, j, distance);
            		printf("error neighbor unfound between /%d [%d %d %d]and %d [%d %d %d]\n\r",\
            		                    					i,cell[i].center.x,cell[i].center.y,cell[i].center.z,\
            		                    					j,cell[j].center.x,cell[j].center.y, cell[j].center.z);

            	}
           	 }
        }
    }
}



__host__ void Geo_control(struct Cell *cell, struct Geo *geo, int node_number)
{
	int i,currentCell;

	//printf("Moot Moot\n");
	for (i=0;i<node_number;i++)
	{
		//printf("Node %d in Cell %d\n",i,G[i].cell_id);
 		if((geo[i].p.x>xmax)||(geo[i].p.y>ymax)||(geo[i].p.z>zmax)) //Test 3D
 		//if((G[i].p.x>xmax)||(G[i].p.y>ymax)||(G[i].p.z!=0)) //Test 2D
		{
			printf("error node out of space\n");
		}

		currentCell=geo[i].cell_id;
		if((cell[currentCell].p1.x>geo[i].p.x)||(cell[currentCell].p1.y>geo[i].p.y)||(cell[currentCell].p1.z>geo[i].p.z)||(cell[currentCell].p2.x<geo[i].p.x)||(cell[currentCell].p2.y<geo[i].p.y)||(cell[currentCell].p2.z<geo[i].p.z))
		{
			printf("error node %d out of cell %d\n",i,currentCell);
			printf("Node (x,y,z):(%d,%d,%d)\n",geo[i].p.x,geo[i].p.y,geo[i].p.z);
		}
	}
}
