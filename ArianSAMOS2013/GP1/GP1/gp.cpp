/*
 * pm.c++
 *
 *  Created on: Nov 6, 2012
 *  Author: Arian Maghazeh
 *  Edited by : Unmesh D. Bordoloi
 *  Implements the convolution algorithm
 *  Performance results reported in SAMOS 2013 paper
 *
 *  Check the log.txt that is generated for the performance results
 *  For debugging, you may also put a break point to see the statements printed on the screen
 *
 *  Bug reports and fixes are truly welcome at unmesh.bordoloi@liu.se but has no guarantee of a reply :D
 */ 
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// Include sys/time.h in Linux environments
// #include <sys/time.h>
// else use custom function in Windows environment
#ifdef _WIN32

 #include "gettime.h"
 #include <Winsock2.h>
 #include <windows.h>
 #pragma comment(lib, "Ws2_32.lib")
#else
 #include <unistd.h>
 #include <sys/time.h> // linux machines
#endif

#define NUM_CORES		1
/***************** OpenCL Definitions ******************/
// OpenCL Definitions
// #define FERMI
#define VIVANTE

#define PLATFORM		0
#define DEVICE			1
#define CONTEXT			2
#define CMDQ			3
#define PGM				4
#define KERNEL			5
#define KERNEL_EXEC		6
#define BUFF			7
#define WRDEV			8
#define RDDEV			9
#define CPU				10
#define GPU_SEQ			11

#ifdef VIVANTE
#define CL_GLOBAL_SIZE_0		(32*1024)
#endif

#ifdef VIVANTE
	#define WORK_GROUP_SIZE	32
#else
	#define WORK_GROUP_SIZE	32
#endif

cl_program 			clProgram;
cl_device_id 		clDeviceId;
cl_context 			clContext;
cl_kernel 			clKernel1;
cl_command_queue 	clCommandQueue;
cl_int 				clErr = 0;
cl_mem				clPopulationBuff, cllastofIndBuff, clTrainInBuff, clConstantBuff, clTrainOutBuff, clFitnessBuff, clDebugBuff;
cl_mem				clLengthBuff, clEvaluateBuff;
//cl_event			prof_event;

struct timeval 		start[15];
struct timeval 		stop[15];
float 				timeRes[15] = {0};
FILE 				*fp;
FILE 				*fio;
char 				buff[256];
char 				*kernelSrc;
char 				description[256] = "Genetic Programming: Classification Problem";
size_t 				clLocalSize;
size_t 				clGlobalSize;

#ifdef FERMI
cl_platform_id 		clPlatformId[3];
#else
cl_platform_id		clPlatformId;
#endif

/***************** Application Definitions ******************/
#define IS_VAR(n)		(n == X || n == Y)
#define IS_CONST(n)		(n >= CONST_START && n <= CONST_END)
#define IS_FUNC(n)		(n >= FUNC_START && n <= FUNC_END)

//global parameters
#define NUM_VAR			2
#define NUM_CONST		20
#define MIN_RND			-1
#define MAX_RND			1
#define POP_SIZE		500
#define MAX_DEPTH		10
#define MAX_IND_LEN 	200
#define GENERATION		20
#define TRAIN_SIZE		128
#define TEST_SIZE		(190 - TRAIN_SIZE)

#define CROSSOVER_RATE		0.9
#define MUTATION_RATE		0.0
#define REPRODUCT_RATE		0.1

#define VAR_PROB	0.3
#define CONST_PROB	0.3
#define FUNC_PROB	0.4

//cross-over selection point probability
#define SEL_FUNC	0.8
#define SEL_TERM	0.2

//primitives
#define X			0
#define Y			1

#define CONST_START	(Y+1)
#define CONST_END	(CONST_START + NUM_CONST - 1)

#define FUNC_START	(CONST_END + 1)
#define ADD			FUNC_START
#define MUL			(FUNC_START + 1)
#define SUB			(FUNC_START + 2)
#define DIV 		(FUNC_START + 3)
#define IFLTE		(FUNC_START + 4)
#define COS			(FUNC_START + 5)
#define SIN			(FUNC_START + 6)
#define FUNC_END	SIN


#define DEL		(FUNC_END + 1)	//delimiter
//
#define FROM_ROOT	0
#define FROM_PNT	1
#define INT		0
#define FLOAT	1

char pop[POP_SIZE][MAX_IND_LEN];
float values[NUM_VAR + NUM_CONST];
unsigned char fitness_cpu[POP_SIZE] = {0};
unsigned char fitness_gpu[POP_SIZE] = {0};
int inds_len[POP_SIZE];
float train_set_in[TRAIN_SIZE * 2];
char train_set_out[TRAIN_SIZE];
float test_set_in[TEST_SIZE * 2];
char test_set_out[TEST_SIZE];
int best_inds[POP_SIZE];
int best_ind = -1;
int test_best_fit = 0;
unsigned int gen_best_fit = 0;
int test_best_len = MAX_IND_LEN;
int gen_best_len = MAX_IND_LEN;
char ch;

void start_measure_time(int seg)
{
    gettimeofday(&start[seg], NULL);
}

void stop_measure_time(int seg)
{
    gettimeofday(&stop[seg], NULL);
    timeRes[seg] += 1000 * ((float)(float)(stop[seg].tv_sec  - start[seg].tv_sec) + 1.0e-6 * (stop[seg].tv_usec - start[seg].tv_usec));
}

void oclInit()
{
	/*-----------------------get platform---------------------------*/
	start_measure_time(PLATFORM);
	#ifdef FERMI
		if (clGetPlatformIDs(3, (cl_platform_id *)&clPlatformId, NULL) != CL_SUCCESS)
			printf("Error in clGetPlatformID! \n");
		else
		{
			clGetPlatformInfo(clPlatformId[1], CL_PLATFORM_NAME, 200, buff, NULL);
			printf("Platform ID: %s \n", buff);
		}
	#else
		if (clGetPlatformIDs(1, &clPlatformId, NULL) != CL_SUCCESS)
			printf("Error in clGetPlatformID! \n");
		else
		{
			clGetPlatformInfo(clPlatformId, CL_PLATFORM_NAME, 200, buff, NULL);
			printf("Platform ID: %s \n", buff);
		}
	#endif
	stop_measure_time(PLATFORM);

	/*-----------------------get device-----------------------------*/
	start_measure_time(DEVICE);
	#ifdef FERMI
		if (clGetDeviceIDs(clPlatformId[1], CL_DEVICE_TYPE_GPU, 1, &clDeviceId, NULL) != CL_SUCCESS)
			printf("Error in clGetDeviceIDs! \n");
	#else
		if (clGetDeviceIDs(clPlatformId, CL_DEVICE_TYPE_GPU, 1, &clDeviceId, NULL) != CL_SUCCESS)
			printf("Error in clGetDeviceIDs! \n");
	#endif
		else
		{
			clGetDeviceInfo(clDeviceId, CL_DEVICE_NAME, 200, buff, NULL);
			printf("Device name: %s \n", buff);
		}
	stop_measure_time(DEVICE);

	/*-----------------------create context-------------------------*/
	start_measure_time(CONTEXT);
	clContext = clCreateContext(0, 1, (cl_device_id *) &clDeviceId, NULL, NULL, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating context!, clErr=%i \n", clErr);
	else
		printf("Context created! \n");
	stop_measure_time(CONTEXT);

	/*---------------------create command queue---------------------*/
	start_measure_time(CMDQ);
	clCommandQueue = clCreateCommandQueue(clContext, clDeviceId, CL_QUEUE_PROFILING_ENABLE, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating command queue!, clErr=%i \n", clErr);
	else
		printf("Command queue created! \n");
	stop_measure_time(CMDQ);

	/*------------------create and build program--------------------*/
	start_measure_time(PGM);
	int filelen;
	int readlen;

	fp = fopen("kernel.cl", "r");
	// getting size of the file
	fseek(fp, 0, SEEK_END);  //set position indicator to the end of file
	filelen = ftell(fp);	//return the current value of position indicator
	rewind(fp);	//set position indicator to the beginning

	kernelSrc = (char*) malloc(sizeof(char) * (filelen + 1));
	readlen = fread(kernelSrc, 1, filelen, fp);
	if (readlen < filelen)
		kernelSrc[readlen] = '\0';
	else
		kernelSrc[filelen] = '\0';

	clProgram = clCreateProgramWithSource(clContext, 1, (const char **) &kernelSrc, NULL, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating program!, clErr=%i \n", clErr);
	else
		printf("Program created! \n");

	clErr = clBuildProgram(clProgram, 0, NULL, NULL, NULL, NULL);
	if (clErr != CL_SUCCESS)
	{
		char buff[4096];
		printf("Error in building program 1!, clErr=%i \n", clErr);
		clGetProgramBuildInfo(clProgram, clDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(buff), buff, NULL);
		printf("-----Build log------\n %s\n", buff);
	}
	else printf("Program built! \n");
	stop_measure_time(PGM);

	/*-----------------------create kernel------------------------*/
	start_measure_time(KERNEL);
	clKernel1 = clCreateKernel(clProgram, "fitness", &clErr);
	if (clErr != CL_SUCCESS)
	{
			printf("Error in creating kernel!, clErr=%i \n", clErr);
			exit(1);
	}
	else
		printf("Kernel created! \n");
	fclose(fp);
	free(kernelSrc);
	stop_measure_time(KERNEL);
}

void oclBuffer()
{
	/*-----------------------create buffer------------------------*/
	start_measure_time(BUFF);
	clPopulationBuff = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(char) * MAX_IND_LEN * POP_SIZE, NULL, &clErr);
	if (clErr != CL_SUCCESS)
	{
		printf("Error in creating Pop buffer!, clErr=%i \n", clErr);
		exit(1);
	}
	clLengthBuff = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(int) * POP_SIZE, NULL, &clErr);
	if (clErr != CL_SUCCESS)
	{
		printf("Error in creating Length buffer!, clErr=%i \n", clErr);
		exit(1);
	}
	clEvaluateBuff = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, sizeof(float) * POP_SIZE * TRAIN_SIZE, NULL, &clErr);
	if (clErr != CL_SUCCESS)
	{
		printf("Error in creating TrainIn buffer!, clErr=%i \n", clErr);
		exit(1);
	}
	clTrainInBuff = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(float) * TRAIN_SIZE * 2, NULL, &clErr);
	if (clErr != CL_SUCCESS)
	{
		printf("Error in creating TrainIn buffer!, clErr=%i \n", clErr);
		exit(1);
	}
	clConstantBuff = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(float) * NUM_CONST, NULL, &clErr);
	if (clErr != CL_SUCCESS)
	{
		printf("Error in creating Constant buffer!, clErr=%i \n", clErr);
		exit(1);
	}
	stop_measure_time(BUFF);

	/*-----------------------write into device--------------------*/
	start_measure_time(WRDEV);
	clErr = clEnqueueWriteBuffer(clCommandQueue, clTrainInBuff, CL_TRUE, 0, sizeof(float) * TRAIN_SIZE * 2, train_set_in, 0, NULL, NULL);
	clErr |= clEnqueueWriteBuffer(clCommandQueue, clConstantBuff, CL_TRUE, 0, sizeof(float) * NUM_CONST, (float *)values + NUM_VAR, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
		printf("Error in writing buffer!, clErr=%i \n", clErr);
	clFinish(clCommandQueue);
	stop_measure_time(WRDEV);
}

void oclClean()
{
	clReleaseContext(clContext);
	clReleaseCommandQueue(clCommandQueue);
	clReleaseProgram(clProgram);
	clReleaseMemObject(clPopulationBuff);
//	clReleaseMemObject(cllastofIndBuff);
	clReleaseMemObject(clLengthBuff);
//	clReleaseMemObject(clFitnessBuff);
	clReleaseMemObject(clEvaluateBuff);
	clReleaseMemObject(clTrainInBuff);
//	clReleaseMemObject(clTrainOutBuff);
	clReleaseMemObject(clConstantBuff);
//	clReleaseMemObject(clDebugBuff);
	clReleaseKernel(clKernel1);
}

union fint
{
	float f;
	int i;
};

fint gen_rnd(int type, int floor, int ceil)
{
	fint res;
	if (type == INT)
		res.i = floor + (rand() % ((ceil - floor) + 1));
	else
		res.f = floor + ((float)rand()/(float)(RAND_MAX/(ceil - floor)));
	return res;
}

int length(char *ind)
{
	for(int i=0; i<MAX_IND_LEN; i++)
		if (ind[i] == '*')
			return i;
	return MAX_IND_LEN;
}

char evaluate(char *ind, float *vals, int i)
{
	int p = 0;
	int q = -1;
	char tmp[MAX_IND_LEN] = {0};
	float vals_stack[3*MAX_DEPTH];	// careful!

	p = length(ind) - 1;
	char node;
	while(p != -1)
	{
		node = ind[p--];
		while(!IS_FUNC(node))
		{
			vals_stack[++q] = vals[node];		// put args into the stack
			node = ind[p--];
		}
		switch(node)
		{
			case ADD:
				vals_stack[q-1] = vals_stack[q] + vals_stack[q-1];
				q--;
				break;
			case SUB:
				vals_stack[q-1] = vals_stack[q] - vals_stack[q-1];
				q--;
				break;
			case MUL:
				vals_stack[q-1] = vals_stack[q] * vals_stack[q-1];
				q--;
				break;
			case DIV:
				if (vals_stack[q-1] < 0.001)
					vals_stack[q-1] = vals_stack[q];
				else
					vals_stack[q-1] = vals_stack[q] / vals_stack[q-1];
				q--;
				break;
			case IFLTE:
				if (vals_stack[q] <= vals_stack[q-1])
					vals_stack[q-3] = vals_stack[q-2];	// else vals_stack[q-3] = vals_stack[q-3];
				q -= 3;
				break;
			case COS:
				vals_stack[q] = cos(vals_stack[q]);
				break;
			case SIN:
				vals_stack[q] = sin(vals_stack[q]);
				break;
			default:
				break;
		}
	}
	if (q != 0)
	{
		printf("p or q error! p=%i , q=%i, len=%i \n", p, q, length(ind));
		exit(1);
	}
	if (vals_stack[0] > 0)
		return 1;
	else
		return 0;
}

void fitness_func()
{
	start_measure_time(CPU);
	float vals[NUM_VAR + NUM_CONST];
//	omp_set_num_threads(1);
//	while(1)
//	{
//	#pragma omp parallel private(vals)
	{
		for (int i=0; i<NUM_VAR + NUM_CONST; i++)
			vals[i] = values[i];
//		#pragma omp for
			for (int i=0; i<POP_SIZE; i++)
			{
				fitness_cpu[i] = 0;
				for (int j=0; j<TRAIN_SIZE; j++)
				{
					vals[X] = train_set_in[j];
					vals[Y] = train_set_in[j + TRAIN_SIZE];
					if (evaluate(pop[i], vals, i) == train_set_out[j])
						fitness_cpu[i]++;
				}
			}
//	}
	}
	stop_measure_time(CPU);
}

void gen_ind(char *ind, int max_dep, int max_len)
{
	char stack[MAX_IND_LEN*10];
	int ind_p = -1;
	int stack_p = -1;
	char node;
	int min_len;
	int cur_depth;

	node = (char)gen_rnd(INT, FUNC_START, IFLTE).i; // don't choose sin and cos for root
	ind[++ind_p] = node;

	if (node == IFLTE)
	{
		for(int i=0; i<4; i++)
			stack[++stack_p] = IFLTE;
		min_len = 5;
	}
	else
	{
		for(int i=0; i<2; i++)
			stack[++stack_p] = node;
		min_len = 3;
	}
	cur_depth = 1;

	while(stack_p != -1)
	{
		float prob = gen_rnd(FLOAT, 0, 1).f;
		if (prob < FUNC_PROB)
		{
			if ((min_len + 4) <= max_len && (cur_depth + 1) < max_dep)	// tree length and depth remains in the specified range
			{
				cur_depth++;
				stack[++stack_p] = DEL;	// put delimiter between two instructions
				node = gen_rnd(INT, FUNC_START, FUNC_END).i;
				if (node == IFLTE)
				{
					min_len += 4;
					for(int i=0; i<4; i++)
						stack[++stack_p] = IFLTE;
				}
				else if (node == SIN || node == COS)
				{
					min_len += 1;
					stack[++stack_p] = node;
				}
				else	// node is one of mul, add, div, sub and has 2 args
				{
					min_len += 2;
					for(int i=0; i<2; i++)
						stack[++stack_p] = node;
				}
				ind[++ind_p] = node;
			}
		}
		else
		{
			if (prob < FUNC_PROB + VAR_PROB)
				ind[++ind_p] = gen_rnd(INT, X, Y).i;
			else
				ind[++ind_p] = gen_rnd(INT, CONST_START, CONST_END).i;
			stack_p--;
			if (stack_p != -1)
				while (stack[stack_p] == DEL)
				{
					stack_p -= 2;
					cur_depth--;
					if (stack_p == -1)
						break;
				}
		}
	}
	if (ind_p < MAX_IND_LEN-1)
		ind[++ind_p] = '*';
}

void init_pop()
{
	for (int i=0; i<POP_SIZE; i++)
		gen_ind(pop[i], MAX_DEPTH, MAX_IND_LEN);
}

void init_GP()
{
	// build constant repository
	for(int i=0; i<NUM_CONST; i++)
		values[CONST_START + i] = gen_rnd(FLOAT, MIN_RND, MAX_RND).f;

	// build training set
	FILE * ds = fopen("spiral.txt", "r");
	const char delimiter[] = " ";
	char line[160];

	if (ds)
		for(int l=0; fgets(line, sizeof(line), ds) != NULL; l++)
		{
			if (l<TEST_SIZE)
			{
				test_set_in[l] = atof(strtok(line, delimiter));
				test_set_in[l + TEST_SIZE] = atof(strtok(NULL, delimiter));
				test_set_out[l] = (char)atoi(strtok(NULL, delimiter));
			}
			else
			{
				train_set_in[l - TEST_SIZE] = atof(strtok(line, delimiter));
				train_set_in[l - TEST_SIZE + TRAIN_SIZE] = atof(strtok(NULL, delimiter));
				train_set_out[l - TEST_SIZE] = (char)atoi(strtok(NULL, delimiter));
			}
		}
	fclose(ds);
}

int tournament(int t_size)
{
	int best;
	int best_fitness = 0;
	for (int i=0; i<t_size; i++)
	{
		int next_ind = gen_rnd(INT, 0, POP_SIZE-1).i;
		if (fitness_gpu[next_ind] > best_fitness)
		{
			best = next_ind;
			best_fitness = fitness_gpu[next_ind];
		}
	}
	return best;
}

void push(char *node, char *stack, int *sp, char *cur_dep, char *max_dep)
{
	if (IS_FUNC(*node))
	{
		(*cur_dep)++;
		stack[++*sp] = DEL;	// put delimiter in the beginning of the stack and between two instructions

		if (*max_dep < *cur_dep)
			*max_dep = *cur_dep;

		if (*node == SIN || *node == COS)
			stack[++*sp] = *node;
		else if (*node == IFLTE)
			for(int i=0; i<4; i++)
				stack[++*sp] = IFLTE;
		else
			for(int i=0; i<2; i++)
				stack[++*sp] = *node;
	}
	else
	{
		(*sp)--;
		while(stack[*sp] == DEL)
		{
			if (*sp == 0)
				break;
			*sp -= 2;
			(*cur_dep)--;
		}
	}
}

int traverse(char * ind, bool type, int point, int *depth)
{
	char cur_dep = 0;
	char max_dep = 0;
	char node;
	char stack[MAX_IND_LEN*10];
	int sp = -1;
	int pntr;

	if (type == FROM_ROOT)
		pntr = 0;	//start from root down to the point
	else
		pntr = point; 	//start from the point until valid expression

	node = ind[pntr++];

	if (!IS_FUNC(node))
	{
		if (depth)
			*depth = 1;
		return 1;
	}
	else
	{
		while(1)
		{
			push(&node, stack, &sp, &cur_dep, &max_dep);
			if (type == FROM_ROOT && pntr == point)
			{
				*depth = cur_dep;
				return point;	//which is length if started from root
			}
			else if (sp == 0)
			{
				if (depth)
					*depth = max_dep;
				return pntr - point;
			}
			node = ind[pntr++];
		}
	}
	return -1;	//should never reach here, just to avoid warning
}

void cross_over(char *par1, char *par2, char *offspring)
{
	//future work: replace term with func, func with func, func with term, term with term
	int len1, len2;
	int dep1, dep2;
	int cross_pnt1, cross_pnt2;

	do
	{
		cross_pnt1 = gen_rnd(INT, 1, length(par1)-1).i;	// don't let parent1 be totally removed
		cross_pnt2 = gen_rnd(INT, 0, length(par2)-1).i;

		traverse(par1, FROM_ROOT, cross_pnt1, &dep1);	// depth of parent1 just before the cross over point
		len1 = traverse(par1, FROM_PNT, cross_pnt1, NULL);	// length of parent1's subtree which is to be replaced
		len2 = traverse(par2, FROM_PNT, cross_pnt2, &dep2);	// length of parent2's subtree which replaces
	}
	while((dep1 + dep2 > MAX_DEPTH) || (length(par1) - len1 + len2 > MAX_IND_LEN - 4));	// make sure the size of the new offspring
																				// remains within the range
	memcpy(offspring, par1, cross_pnt1);
	memcpy(offspring + cross_pnt1, par2 + cross_pnt2, len2);
	memcpy(offspring + cross_pnt1 + len2, par1 + cross_pnt1 + len1, MAX_IND_LEN - (cross_pnt1 + len1));
}

void mutate(char *ind, char *offspring)
{
	int ind_len = length(ind);
	int dep, seg_len, sub_len;
	char subtree[MAX_IND_LEN];
	int mutation_pnt = gen_rnd(INT, 1, length(ind) - 1).i;

	if (IS_FUNC(ind[mutation_pnt]))	// if the node selected to be mutated is a function
	{								// grow a random subtree from mutation-point such that the whole individual size remains within the range
		traverse(ind, FROM_ROOT, mutation_pnt, &dep);
		seg_len = traverse(ind, FROM_PNT, mutation_pnt, NULL);
		gen_ind(subtree, MAX_DEPTH - dep, MAX_IND_LEN - 4 - (ind_len - seg_len));
		sub_len = length(subtree);

		memcpy(offspring, ind, mutation_pnt);
		memcpy(offspring + mutation_pnt, subtree, sub_len);
		memcpy(offspring + mutation_pnt + sub_len, ind + mutation_pnt + seg_len, MAX_IND_LEN - (mutation_pnt + seg_len));
		memcpy(offspring, ind, MAX_IND_LEN);
	}
	else // if a terminal is selected as mutation point
	{
		ind[mutation_pnt] = gen_rnd(INT, X, CONST_END).i;	// choose another terminal as replacement
		memcpy(offspring, ind, MAX_IND_LEN);
	}
}

void reproduct(char *ind, char *offspring)
{
	memcpy(offspring, ind, MAX_IND_LEN);
}

void next_gen()
{
	char *parent1, *parent2;
	char new_gen[POP_SIZE][MAX_IND_LEN];
	int cntr = 0;

	for(int j=0; j<POP_SIZE; j++)
		for(int k=0; k<MAX_IND_LEN; k++)
			new_gen[j][k] = '*';

	for(int i=0; i<(int)POP_SIZE * CROSSOVER_RATE; i++)
	{
		parent1 = pop[tournament(2)];
		parent2 = pop[tournament(2)];
		cross_over(parent1, parent2, new_gen[cntr++]);
	}
	for(int i=0; i<(int)POP_SIZE * MUTATION_RATE; i++)
	{
		parent1 = pop[tournament(2)];
		mutate(parent1, new_gen[cntr++]);
	}
	for(int i=0; i<(int)POP_SIZE * REPRODUCT_RATE; i++)
	{
		parent1 = pop[tournament(2)];
		reproduct(parent1, new_gen[cntr++]);
	}
	for (int i=0; i<POP_SIZE; i++)
		for (int j=0; j<MAX_IND_LEN; j++)
			pop[i][j] = new_gen[i][j];
}

void ocl_fitness_func()
{
	float *eval_results = (float *)malloc(sizeof(float) * POP_SIZE * TRAIN_SIZE);
	char *popflat = (char *)malloc(MAX_IND_LEN * POP_SIZE);
	int ind_len;

	for (int i=0; i<POP_SIZE; i++)
	{
		ind_len = length(pop[i]);
		memcpy(popflat + i*MAX_IND_LEN, pop[i], MAX_IND_LEN);
		for (int j=0; j<3; j++) 	// just to be sure we are safe
			popflat[i*MAX_IND_LEN + ind_len + j] = '*';
		inds_len[i] = ceil((float)ind_len / 4);
	}

	// write and transfer new population into the GPU's memory
	start_measure_time(WRDEV);
	clErr = clEnqueueWriteBuffer(clCommandQueue, clPopulationBuff, CL_TRUE, 0, sizeof(char) * MAX_IND_LEN * POP_SIZE, popflat, 0, NULL, NULL);
	clErr |= clEnqueueWriteBuffer(clCommandQueue, clLengthBuff, CL_TRUE, 0, sizeof(int) * POP_SIZE, inds_len, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
		printf("Error in writing buffer!, clErr=%i \n", clErr);
	clFinish(clCommandQueue);
	stop_measure_time(WRDEV);

	clErr = clSetKernelArg(clKernel1, 0, sizeof(cl_mem), &clPopulationBuff);
	clErr |= clSetKernelArg(clKernel1, 1, sizeof(cl_mem), &clLengthBuff);
	clErr |= clSetKernelArg(clKernel1, 2, sizeof(cl_mem), &clEvaluateBuff);
	clErr |= clSetKernelArg(clKernel1, 3, sizeof(cl_mem), &clTrainInBuff);
	clErr |= clSetKernelArg(clKernel1, 4, sizeof(cl_mem), &clConstantBuff);
	if (clErr != CL_SUCCESS)
	{
		printf("Error in clSetKernelArg!, clErr=%i \n", clErr);
		exit(1);
	}

	size_t clLocalSize = WORK_GROUP_SIZE;
	size_t clGlobalSize = POP_SIZE * WORK_GROUP_SIZE;

	start_measure_time(KERNEL_EXEC);
//	while(1)
//	{
	clErr = clEnqueueNDRangeKernel(clCommandQueue, clKernel1, 1, NULL, &clGlobalSize, &clLocalSize, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
	{
		printf("Error in launching kernel!, clErr=%i \n", clErr);
		exit(1);
	}
//	}
	clFinish(clCommandQueue);
	stop_measure_time(KERNEL_EXEC);

	start_measure_time(RDDEV);
	clErr = clEnqueueReadBuffer(clCommandQueue, clEvaluateBuff, CL_TRUE, 0, sizeof(float) * POP_SIZE * TRAIN_SIZE, eval_results, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
	{
		printf("Error in reading buffer!, clErr=%i \n", clErr);
		exit(1);
	}
	clFinish(clCommandQueue);
	stop_measure_time(RDDEV);

	start_measure_time(GPU_SEQ);

	omp_set_num_threads(1);
	#pragma omp parallel
		#pragma omp for
			for (int i=0; i<POP_SIZE; i++)
			{
				fitness_gpu[i] = 0;
				for (int j=0; j<TRAIN_SIZE; j++)
				{
					if (eval_results[i * TRAIN_SIZE + j] > 0 && train_set_out[j] == 1)
						fitness_gpu[i]++;
					else if (eval_results[i * TRAIN_SIZE + j] <= 0 && train_set_out[j] == 0)
						fitness_gpu[i]++;
				}
			}
	stop_measure_time(GPU_SEQ);

	free(eval_results);
	free(popflat);
}

void printResult()
{
	fio = fopen("log.txt", "a+");
	fseek (fio, 0, SEEK_END);
	int appendPos = ftell(fio);

	float total_GPU_time = timeRes[PLATFORM] + timeRes[DEVICE] + timeRes[CONTEXT] + timeRes[CMDQ] + timeRes[PGM] + timeRes[KERNEL] + timeRes[BUFF] + timeRes[WRDEV] + timeRes[KERNEL_EXEC] + timeRes[RDDEV] + timeRes[GPU_SEQ];
	float total_GPU_fair_time = timeRes[KERNEL_EXEC] + timeRes[WRDEV] + timeRes[RDDEV] + timeRes[GPU_SEQ];

	struct tm *local;
	time_t t;
	t = time(NULL);
	local = localtime(&t);

	char hostName[50];
	gethostname(hostName, 50);

	fprintf(fio, "****************************************************\n");
	fprintf(fio, "Created on: %s", asctime(local));
	fprintf(fio, "Host name: %s \n", hostName);
	fprintf(fio, "Description: %s \n", description);
	fprintf(fio, "\n==============GP Parameters=====================\n");
	fprintf(fio,
				"GENERATION = %i, POP_SIZE = %i \n"
				"MAX_IND_LEN = %i, MAX_DEPTH = %i \n"
				"NUM_CONST = %i, MIN_RND = %i, MAX_RND = %i \n"
				"TRAIN_SIZE = %i \n\n"
				"CROSSOVER_RATE = %.2f, MUTATION_RATE = %.2f, REPRODUCT_RATE = %.2f \n"
				"VAR_PROB = %.2f, CONST_PROB = %.2f, FUNC_PROB = %.2f \n"
				"SEL_FUNC = %.2f, SEL_TERM = %.2f \n",
				GENERATION, POP_SIZE, MAX_IND_LEN, MAX_DEPTH, NUM_CONST, MIN_RND, MAX_RND,
				TRAIN_SIZE, CROSSOVER_RATE, MUTATION_RATE, REPRODUCT_RATE, VAR_PROB,
				CONST_PROB, FUNC_PROB, SEL_FUNC, SEL_TERM
			);

	fprintf(fio, "\n===========Performance Measurements=================\n");
	fprintf(fio, "GP Performance: \n"
				"	Training best fitness = %i \n"
				"	Training shortest length = %i \n"
				"	Training hit percentage = %i% \n\n"
				"	Testing best fitness = %i \n"
				"	Testing shortest length = %i \n"
				"	Testing hit percentage = %i% \n",
				(int)gen_best_fit, gen_best_len, ((int)gen_best_fit * 100)/TRAIN_SIZE,
				(int)test_best_fit, test_best_len, ((int)test_best_fit * 100)/TEST_SIZE
			);

	fprintf(fio, "\nExecution times: \n"
				"	PLATFORM = \t%10.2f msecs \n"
				"	DEVICE = \t%10.2f msecs \n"
				"	CONTEXT = \t%10.2f msecs \n"
				"	CMDQ = \t\t%10.2f msecs \n"
				"	PGM = \t\t%10.2f msecs \n"
				"	KERNEL = \t%10.2f msecs \n"
				"	BUFF = \t\t%10.2f msecs \n"
				"	WRDEV = \t%10.2f msecs \n"
				"	KERNEL_EXEC = \t%10.2f msecs \n"
				"	RDDEV = \t%10.2f msecs \n"
				"	GPU_SEQ = \t%10.2f msecs \n",
			   timeRes[PLATFORM],
			   timeRes[DEVICE],
			   timeRes[CONTEXT],
			   timeRes[CMDQ],
			   timeRes[PGM],
			   timeRes[KERNEL],
			   timeRes[BUFF],
			   timeRes[WRDEV],
			   timeRes[KERNEL_EXEC],
			   timeRes[RDDEV],
			   timeRes[GPU_SEQ]
			);
	fprintf(fio, "GPU time: \t\t%10.2f msecs \n"
			"GPU exec time: \t\t%10.2f msecs \n"
			"CPU time: \t\t%10.2f msecs \n\n",
			total_GPU_time, total_GPU_fair_time, timeRes[CPU]);
	fprintf(fio, "Speed UP: \t\t%10.2f \n\n", float(timeRes[CPU])/float(total_GPU_fair_time));

	fseek(fio, appendPos, SEEK_SET);
	while(fgets(buff,sizeof buff,fio))
			printf("%s", buff);
	fclose(fio);
}

void test_gp()
{
	int test_fit;
	test_best_len = 0;
	for(int i=0; best_inds[i] != -1; i++)
	{
		test_fit = 0;
		for(int j=0; j<TEST_SIZE; j++)
		{
			values[X] = test_set_in[j*2];	//j
			values[Y] = test_set_in[j*2 + 1];	//j+TEST_SIZE
//			if (evaluate(pop[best_inds[i]], values) != test_set_out[j])
				test_fit++;
		}
		if (test_fit > test_best_fit)
		{
			test_best_fit = test_fit;
			test_best_len = length(pop[best_inds[i]]);
			best_ind = best_inds[i];
		}
		else if (test_fit == test_best_fit)
		{
			if (length(pop[best_inds[i]]) < test_best_len)
				test_best_len = length(pop[best_inds[i]]);
			best_ind = best_inds[i];
		}
		printf("%i. Test fit of individual(%i) = %i \n", i, best_inds[i], test_fit);
	}
}

void gen_per(int num)
{
	int idx = 0;
	gen_best_fit = 0;
	for (int i=0; i<POP_SIZE; i++)
		if (fitness_gpu[i] > gen_best_fit)
		{
			idx = 0;
			best_inds[idx++] = i;
			best_inds[idx] = -1;
			gen_best_fit = fitness_gpu[i];
			gen_best_len = length(pop[i]);
		}
		else if (fitness_gpu[i] == gen_best_fit)
		{
			if (length(pop[i]) < gen_best_len)
				gen_best_len = length(pop[i]);
			best_inds[idx++] = i;
			best_inds[idx] = -1;	// to show the ending point
		}
	printf("\nGeneration %i completed! \n", num);
	printf("Best fitness = %i \n", fitness_gpu[20]);
	printf("Shortest length = %i \n", gen_best_len);
}

int main()
{
	srand(0);

	init_GP();
	init_pop();

	oclInit();
	oclBuffer();

//	size_t ws, ls;
//	clGetKernelWorkGroupInfo(clKernel1, clDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(ws), (void *) &ws, NULL);
//	printf("CL_KERNEL_WORK_GROUP_SIZE is: %i \n", ws);
//	clGetKernelWorkGroupInfo(clKernel1, clDeviceId, CL_KERNEL_LOCAL_MEM_SIZE , sizeof(ls), (void *) &ls, NULL);
//	printf("CL_KERNEL_LOCAL_MEM_SIZE is: %i \n", ls);

	fitness_func();
	ocl_fitness_func();

	for (int i=0; i<POP_SIZE; i++)
	{
		if (fitness_cpu[i] != fitness_gpu[i])
			printf("mismatch at i = %i \n", i);
		printf("fitness_gpu[%i] = %i, fitness_cpu[%i] = %i \n", i, fitness_gpu[i], i, fitness_cpu[i]);
	}
	for (int i=1; i<GENERATION; i++)
	{
		//printf("hello \n");
		next_gen();
		fitness_func();
		ocl_fitness_func();
		gen_per(i);
	}
	oclClean();
	test_gp();
	printResult();

	return 0;
}
