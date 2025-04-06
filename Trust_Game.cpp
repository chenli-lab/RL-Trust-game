//******************Decoding trust: a reinforcement learning perspective*******************// 
#include <iostream>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include<fstream>
#define PI 3.14159265358979
using namespace std;

//-- array creation and destruction --//
//1D
void make1Darray(int *& array,int rows)
{
    array=new int [rows];
}

void delete1Darray(int *array)
{
    delete []array;
    array=0;
}

void make1Darray(double *& array,int rows)
{
    array=new double [rows];
}

void delete1Darray(double *array)
{
    delete []array;
    array=0;
}

//2D
void make2Darray(int **& array,int rows,int cols)
{
    array=new int *[rows];
    for(int i=0;i<rows;i++)  
        array[i]=new int [cols];
}

void delete2Darray(int **array,int rows)
{
    for(int i=0;i<rows;i++)
        delete []array[i];
    delete []array;
    array=0;
}

void make2Darray(double **& array,int rows,int cols)
{
    array=new double *[rows];
    for(int i=0;i<rows;i++)  
        array[i]=new double [cols];
}

void delete2Darray(double **array,int rows)
{
    for(int i=0;i<rows;i++)
        delete []array[i];
    delete []array;
    array=0;
}

//3D
void make3Darray(int *** &array,int high,int row,int col)
{
    int i,j;
    array=new int **[high];
    for(i=0;i<high;i++)
        array[i]=new int*[row];
    
    for(i=0;i<high;i++)
        for(j=0;j<row;j++)
            array[i][j]=new int[col];
}

void delete3Darray(int ***array,int hight, int row)
{
    int i,j;
    for(i=0;i<hight;i++)
        for(j=0;j<row;j++)
            delete []array[i][j] ;
    
    for(i=0;i<hight;i++)
        delete []array[i] ;
    delete []array;
    array=0 ;
}

void make3Darray(double ***&array,int high,int row,int col)
{
    int i,j;
    array=new double **[high];
    for(i=0;i<high;i++)
        array[i]=new double* [row];
    
    for(i=0;i<high;i++)
        for(j=0;j<row;j++)
            array[i][j]=new double[col];
}

void delete3Darray(double ***array,int hight, int row)
{
    int i,j;
    for(i=0;i<hight;i++)
        for(j=0;j<row;j++)
            delete []array[i][j] ;
    for(i=0;i<hight;i++)
        delete []array[i] ;
    delete []array;
    array=0 ;
}

#define size 2
#define tmax 100000000                    // max nr. of time steps

int main()
{
	int i,j,k,t,t_log;
	double C1,C2;
	int it,iNN,N;
    int realization,number;
    int stateA,stateB; 
    int fra_state,num_state,num_action;
    int NB,TB,NR,TR;
    double density_TR[tmax],density_TB[tmax],density_NR[tmax],density_NB[tmax];
    double ave_TR,ave_TB,ave_NR,ave_NB;
	double ***QtableA,***QtableB,*RewardA,*RewardB,Q_maxA,Q_maxB,Q_max_nextA,Q_max_nextB;
    double x,w;
    int *ActionA,*ActionB,*State_newA,*State_newB,*StateA,*StateB,**combin;
    double r,alpha,gamma,epsilon;
    double t_log_exp;
	
	N=size;                                   //the Population size
    
		
	num_action=2;                              //T,N or R,B 
	fra_state=2;
	num_state=4;                              
	
	C1 = 4.0;              //the parameter used to adjust the exploration rate epsilon
	C2 = 6.0;              //the parameter used to adjust the learning rate alpha
	
	x=1.0;            //investment fraction
	w=0.5;          //return fraction 
    

    gamma=0.9;                                //discount factor
	
	t_log=int(pow(10,3));       
    t_log_exp=3;
    
    make1Darray(StateA,N);            					// State array for trustors
	make1Darray(StateB,N);                              // State array for trustees   
    make1Darray(State_newA,N);                         // New state array for trustors
	make1Darray(State_newB,N);                         // New state array for trustees 
    
    make1Darray(ActionA,N);                           // Action array for trustors
    make1Darray(ActionB,N);                           // Action array for trustees
    make1Darray(RewardA,N);                           //the Reward obtained
    make1Darray(RewardB,N);                            //the Reward obtained
    make2Darray(combin,num_action,fra_state);          // final state array
	make3Darray(QtableA,N,num_state,num_action);       //Q table for each trustor
    make3Darray(QtableB,N,num_state,num_action); 	//Q table for each trustee
	
	ofstream outfile1("Trust_TimeSeries.txt");               
    if(!outfile1)
	{
        cout<<"You can't open the pointed file!"<<endl;
        exit(1);
    }	
		//-------------------system initialization------------------//   
	    for(i=0;i<tmax;i++)
	    {
	    	density_TR[i]=0;
	    	density_TB[i]=0;
	    	density_NR[i]=0;
	    	density_NB[i]=0;	
		}
		
		number = 0;
		for(i=0;i<2;i++)
		{
			for(j=0;j<2;j++)
			{
				combin[i][j] = number;
				number++;
			}
		}
		
		for(realization=0;realization<500;realization++)
		{
			//---------------------- System Initialization time=t-1 -----------------//
	        for(i=0;i<N;i++)
	        {
	            for(j=0;j<num_state;j++)
	            {
	                for(k=0;k<num_action;k++)
	                {
	                    QtableA[i][j][k] = drand48();                  //all iterms in Q-table are random assigned
	                	QtableB[i][j][k] = drand48();
					}
	            }
	            ActionA[i] = rand()%num_action;                         //set to T or N randomly
	            ActionB[i] = rand()%num_action;  						//set to R or B randomly
	            
	            RewardA[i] = 0;                                //No reward at the beginning
	            RewardB[i] = 0;
	        }
	        
	        TR=0;
			TB=0;
			NR=0;
			NB=0;
			for(i=0;i<N;i++)
			{
				if(ActionA[i]==0)   //T
				{
					if(ActionB[i]==0)  //R
					{
						TR = TR+1;
					}
        			else //B
					{
						TB = TB+1;	
					}
				}
				else                 //N
				{
					if(ActionB[i]==0)  //R
					{
						NR = NR+1;
					}
        			else          //B
					{
						NB = NB+1;	
					}	
				}
			}
			
			density_TR[0] += (double)TR/N;         //Record the fraction of actions under the initial conditions
			density_TB[0] += (double)TB/N;
			density_NR[0] += (double)NR/N;
			density_NB[0] += (double)NB/N;
	        
	        //----------------Statistical state-----------------//
	        for(i=0;i<N;i++)
	        {
	        	iNN = ((i+1)%size+size)%size;   //Individual i's opponent player
	        
			//--------Record the action taken by the opponent in the previous round---------//	
	        	if(ActionA[iNN]==0) // T
	        	{
	        		stateB = 1;
				}
				else           //N
				{
					stateB = 0;
				}
				
				if(ActionB[iNN]==0) // R
	        	{
	        		stateA = 1;
				}
				else           //B
				{
					stateA = 0;
				}
				
				StateA[i] = combin[ActionA[i]][stateA];   //When acting as the trustor, record the state by considering both the player's and the opponent's actions simultaneously
				StateB[i] = combin[ActionB[i]][stateB];   //When acting as the trustee, record the state by considering both the player's and the opponent's actions simultaneously
			}
			
			//--------- The evolution of the game by Q-learning -----------//
			for(t=1;t<tmax;t++)
			{
				alpha = C2/sqrt(t);    //time-varying form for the learning rate 
				epsilon = C1/sqrt(t);   //time-varying form for the exploration rate 
				
				if(alpha>1.0)
				{
					alpha = 1.0;
				}
				if(alpha<=0.1)
				{
					alpha = 0.1;    //converge to 0.1 over time 
				}
				
				if(epsilon>1.0)
				{
					epsilon = 1.0;
				}
				if(epsilon<=0.01)
				{
					epsilon = 0.01;   //converge to 0.01 over time 
				}
				//--------Choose an action------------//
				for(i=0;i<N;i++)
	            {
	                r = drand48();
	                if(r<epsilon)                         //Exploration with greedy algorithm
	                {
	                    ActionA[i] = rand()%num_action;            
	                    ActionB[i] = rand()%num_action;
	                }
	                else                                  //Exploitation for the optimal action
	                {
	                    Q_maxA = -1000;                     //starting with a very small value
	                    Q_maxB = -1000; 
	                    for(k=0;k<num_action;k++)                                //go through all actions
	                    {
	                        if(QtableA[i][StateA[i]][k] > Q_maxA)                    //Search for the iterm with the largest Q value within the row with given state StateA[i]
	                        {
	                            Q_maxA = QtableA[i][StateA[i]][k];
	                            ActionA[i] = k;                                      
	                        }
	                        
	                        if(QtableB[i][StateB[i]][k] > Q_maxB)                    //Search for the iterm with the largest Q value within the row with given state StateB[i]
	                        {
	                            Q_maxB = QtableB[i][StateB[i]][k];
	                            ActionB[i] = k;                                   
	                        }
	                    }
	                }
	            }
			   //--------Perform action and measure the reward------------//
				for(i=0;i<N;i++)
				{
					RewardA[i] = 0;
					RewardB[i] = 0;
				} 
				
				for(i=0;i<N;i++)   // two players play the role of trustor and trustee in turn
				{
					iNN = ((i+1)%size+size)%size;
					if(ActionA[i]==0)   //T for i (trustor)
					{
						RewardA[i] += 1-x;	
						RewardB[iNN] += 3*x;
						
						if(ActionB[iNN]==0)   //R for iNN (trustee)
						{
							RewardA[i] += w*3*x;
							RewardB[iNN] -= w*3*x; 	
						}
						else              //B for iNN (trustee)
						{
							RewardA[i] += 0;
							RewardB[iNN] += 0;		
						}	
					}
					else           //N for i (trustor)
					{
						RewardA[i] += 1;
						RewardB[iNN] += 0;		
					} 
				}
				//--------------Statistics state-----------//	
				for(i=0;i<N;i++)
		        {
		        	iNN = ((i+1)%size+size)%size; 
		        	
		        	if(ActionA[iNN]==0) // T
		        	{
		        		stateB = 1;
					}
					else           //N
					{
						stateB = 0;
					}
					
					if(ActionB[iNN]==0) // R
		        	{
		        		stateA = 1;
					}
					else           //B
					{
						stateA = 0;
					}
					
					State_newA[i] = combin[ActionA[i]][stateA]; 
					State_newB[i] = combin[ActionB[i]][stateB]; 
				}
				//--------------Update the Q table----------------------//		
				for(i=0;i<N;i++)
		        {
		        	//--------------for the trustor role----------//
		        	
					Q_max_nextA = -1000;		 
					for(k=0;k<num_action;k++)                                   //go through all actions
			        {
			            if(QtableA[i][State_newA[i]][k] > Q_max_nextA)                //Search for the iterm with the largest Q value
			            {
			                Q_max_nextA = QtableA[i][State_newA[i]][k];
			            }
			        }
			        QtableA[i][StateA[i]][ActionA[i]] = (1-alpha)*QtableA[i][StateA[i]][ActionA[i]] + alpha*(RewardA[i] + gamma*Q_max_nextA);
		        	
					//--------------for the trustee role----------//	
		           	Q_max_nextB = -1000; 
					for(k=0;k<num_action;k++)                                   //go through all actions
			        {
			            if(QtableB[i][State_newB[i]][k] > Q_max_nextB) 
			            {
			                Q_max_nextB = QtableB[i][State_newB[i]][k];
						}
			        }  
			        QtableB[i][StateB[i]][ActionB[i]] = (1-alpha)*QtableB[i][StateB[i]][ActionB[i]] + alpha*(RewardB[i] + gamma*Q_max_nextB);
		        }
		        
		        for(i=0;i<N;i++)
	            {
	                StateA[i]=State_newA[i];
	                StateB[i]=State_newB[i];
	            }
	            
	            //---------------- Output ----------------//
	            TR=0;
				TB=0;
				NR=0;
				NB=0;
				for(i=0;i<N;i++)
				{
					if(ActionA[i]==0)   //T
					{
						if(ActionB[i]==0)  //R
						{
							TR = TR+1;
						}
	        			else //B
						{
							TB = TB+1;	
						}
					}
					else                 //N
					{
						if(ActionB[i]==0)  //R
						{
							NR = NR+1;
						}
	        			else          //B
						{
							NB = NB+1;	
						}	
					}
				}	
				density_TR[t] += (double)TR/N;
				density_TB[t] += (double)TB/N;
				density_NR[t] += (double)NR/N;
				density_NB[t] += (double)NB/N;
					
			}
		}
		
		for(t=0;t<tmax;t++)
		{
			if(t>0)
			{
				alpha = C2/sqrt(t);
				epsilon = C1/sqrt(t);
				
				if(alpha>1.0)
				{
					alpha = 1.0;
				}
				if(alpha<=0.1)
				{
					alpha = 0.1;
				}
				
				if(epsilon>1.0)
				{
					epsilon = 1.0;
				}
				if(epsilon<=0.01)
				{
					epsilon = 0.01;
				}
			}

			ave_TR=density_TR[t]/500;
			ave_TB=density_TB[t]/500;
			ave_NR=density_NR[t]/500;
			ave_NB=density_NB[t]/500;
			
			if(t<1000)
		    {
		        outfile1<<t<<"  "<<ave_TR<<"  "<<ave_TB<<"  "<<ave_NR<<"  "<<ave_NB<<"  "<<epsilon<<"  "<<alpha<<endl;   
		    }
		    else if(t==t_log)
		    {
		        outfile1<<t<<"  "<<ave_TR<<"  "<<ave_TB<<"  "<<ave_NR<<"  "<<ave_NB<<"  "<<epsilon<<"  "<<alpha<<endl;
		                
		        t_log_exp += 0.002;
		        t_log=int(pow(10,t_log_exp));         //for visualization within log(t)
		    }
		}
		
		delete1Darray(StateA);
		delete1Darray(StateB);
	    delete1Darray(State_newA);
	    delete1Darray(State_newB);
	    
	    delete1Darray(ActionA);
	    delete1Darray(ActionB);
	    delete1Darray(RewardA);
	    delete1Darray(RewardB);
	    delete3Darray(QtableA,N,num_state);
	    delete3Darray(QtableB,N,num_state);
		    
		return 0;
}
