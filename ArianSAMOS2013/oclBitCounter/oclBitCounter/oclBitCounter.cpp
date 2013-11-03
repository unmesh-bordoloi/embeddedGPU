/*
* Author: Arian Maghazeh
* Edited by : Unmesh D. Bordoloi
* Implements the bitcount algorithm from Mibench automotive suite
* Performance results reported in SAMOS 2013 paper

* If you want to compare local memory versus non-local memory, use #define LOCALMEM
* Check the log.txt that is generated
* You may also put a break point to see the statements printed on the screen
*
* Bug reports and fixes are truly welcome at unmesh.bordoloi@liu.se but has no guarantee of a reply :D
*/

#include <stdio.h>
#include <CL/cl.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Include sys/time.h in Linux environments
// #include <sys/time.h>
// else use custom function in Windows environment
#ifdef _WIN32

 #include "gettime.h"
 #include <windows.h>
#else

 #include <sys/time.h> // linux machines
#endif

// Its funny that OpenCL is touted as platform independent 
// and yet it comes down to the following. What am I missing here?
// #define FERMI

#ifdef FERMI 

 #define PLATFORMS_IDS 3 // on the Fermi machine at IDA
 #define THIS_PLATFORM_ID 1
#else

 #define PLATFORMS_IDS 1 // on Unmesh's local desktop
 #define THIS_PLATFORM_ID 0
#endif


#define PLATFORM		0
#define DEVICE			1
#define CONTEXT			2
#define CMDQ			3
#define PGM1			4
#define PGM2			5
#define KERNEL1			6
#define KERNEL2		7
#define KERNEL1_EXEC	8
#define KERNEL2_EXEC	9
#define BUFF			10
#define WRDEV			11
#define RDDEV			12
#define CPU				13

// #define LOCALMEM    // use this #def if you want to check the version that does not uses local memory    

#define WORK_GROUP_SIZE		32
#define GLOBAL_SIZE_0		32*1024		
struct timeval start[15];
struct timeval stop[15];
float timeRes[15] = {0};


void start_measure_per(int seg)
{ 
    gettimeofday(&start[seg], NULL);
}

void stop_measure_per(int seg)
{
    gettimeofday(&stop[seg], NULL);
    timeRes[seg] = 1000 * ((float)(float)(stop[seg].tv_sec  - start[seg].tv_sec) + 1.0e-6 * (stop[seg].tv_usec - start[seg].tv_usec));
}

int main(void)
{
	cl_context clContext;
	cl_kernel clKernel1;
	cl_kernel clKernel2;
	cl_command_queue clCommandQueue;
	cl_program clProgram;
	cl_platform_id clPlatformId[3];
	cl_device_id clDeviceId;
	cl_mem clSrcBuffer;
	cl_mem clIntermediateBuffer;
	cl_int clErr;
	cl_uint num_platforms;
	size_t clGlobalSize[2];
	size_t clGroupSize[2];

	char version[256] = "BitCounter, optimized, with synchronization";
	int readlen;
	int numofElements = 1*1024*1024;
	int * idata;
	int finalResultGPU;
	int finalResultCPU;
	char buff[256];
	int numofWorkGroups;

	clGroupSize[0] = WORK_GROUP_SIZE;		
	clGroupSize[1] = 1;
	numofWorkGroups = numofElements / WORK_GROUP_SIZE;
	// generate input data
	idata = (int *) malloc(sizeof(cl_int) * numofElements);
	
	srand(time(NULL));
	for (int i=0; i<numofElements; i++)
		idata[i] = rand();

	//===================================CPU=======================================//
	start_measure_per(CPU);
	finalResultCPU = 0;
	cl_int inp;
    
	for(int i=0; i<numofElements; i++)
	{
		inp = idata[i];
		int cntr = 0;
		while(inp != 0)
		{
			cntr++;
			inp = inp & (inp - 1);
		}
		finalResultCPU += cntr;
	}
	stop_measure_per(CPU);
    //===================================CPU=======================================//

	//=================================PLATFORM=======================================//
	start_measure_per(PLATFORM);
	if (clGetPlatformIDs(PLATFORMS_IDS, (cl_platform_id *)&clPlatformId, &num_platforms) != CL_SUCCESS)
		printf("Error in clGetPlatformID! \n");
	else
	{
		clGetPlatformInfo(clPlatformId[THIS_PLATFORM_ID], CL_PLATFORM_NAME, 200, buff, NULL);
		printf("Platform ID: %s \n", buff);
		printf("The number of platforms is/are: %d\n", num_platforms);
		printf("THIS_PLATFORM_ID: %d\n", THIS_PLATFORM_ID);
	}
	stop_measure_per(PLATFORM);
	//==================================DEVICE=======================================//
	start_measure_per(DEVICE);
	clErr = clGetDeviceIDs(clPlatformId[THIS_PLATFORM_ID], CL_DEVICE_TYPE_GPU, 1, &clDeviceId, NULL);
	if (clErr != CL_SUCCESS)
		printf("Error in clGetDeviceIDs! clErr=%i \n", clErr);
	else
	{
		clGetDeviceInfo(clDeviceId, CL_DEVICE_NAME, 200, buff, NULL);
		printf("Device Name: %s \n", buff);
	}
	stop_measure_per(DEVICE);
	//=================================CONTEXT=======================================//
	start_measure_per(CONTEXT);
	clContext = clCreateContext(0, 1, &clDeviceId, NULL, NULL, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating context!, clErr=%i \n", clErr);
	else
		printf("Context created! \n");
	stop_measure_per(CONTEXT);
	//===============================COMMAND QUEUE===================================//
	start_measure_per(CMDQ);
	clCommandQueue = clCreateCommandQueue(clContext, clDeviceId, NULL, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating command queue!, clErr=%i \n", clErr);
	else
		printf("Command queue created! \n");
	stop_measure_per(CMDQ);
	//=========================PROGRAM & BUILD & KERNEL==============================//
	start_measure_per(PGM1);
	char *kernelSrc;
	FILE* fp;
	int filelen;

	fp = fopen("Kernel1.cl", "r");
	// getting size of the file
	fseek(fp, 0, SEEK_END);  //set position indicator to the end of file
	filelen = ftell(fp);	//return the current value of position indicator
	rewind(fp);	//set position indicator to the beginning

	kernelSrc = (char*) malloc(sizeof(char) * (filelen));
	readlen = fread(kernelSrc, 1, filelen, fp);
	if (readlen < filelen)
		kernelSrc[readlen] = '\0';
	else
		kernelSrc[filelen] = '\0';

	clProgram = clCreateProgramWithSource(clContext, 1, (const char **) &kernelSrc, NULL, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating program!, clErr=%i \n", clErr);
	else
		printf("Program 1 created! \n");

	clErr = clBuildProgram(clProgram, 0, NULL, NULL, NULL, NULL);
	if (clErr != CL_SUCCESS)
	{
		char buff[4096];
		printf("Error in building program 1!, clErr=%i \n", clErr);
		clGetProgramBuildInfo(clProgram, clDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(buff), buff, NULL);
		printf("-----Build log------\n %s\n", buff);
	}
	else printf("Program 1 built! \n");
	stop_measure_per(PGM1);

	start_measure_per(KERNEL1);
	clKernel1 = clCreateKernel(clProgram, "BitCounter", &clErr);
	if (clErr != CL_SUCCESS)
			printf("Error in creating kernel 1!, clErr=%i \n", clErr);
	else printf("Kernel 1 created! \n");

	free(kernelSrc);
	fclose(fp);
	stop_measure_per(KERNEL1);
/**************************************************/
	start_measure_per(PGM2);
	cl_program clProgram2;
	char *kernelSrc2;
	FILE* fp2;

	fp2 = fopen("Kernel2.cl", "r");
	fseek(fp2, 0, SEEK_END);
	filelen = ftell(fp2);
	rewind(fp2);

	kernelSrc2 = (char*) malloc(sizeof(char) * (filelen));
	readlen = fread(kernelSrc2, 1, filelen, fp2);
	if (readlen < filelen)
		kernelSrc2[readlen] = '\0';
	else
		kernelSrc2[filelen] = '\0';
	clProgram2 = clCreateProgramWithSource(clContext, 1, (const char **) &kernelSrc2, NULL, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating program 2 (version with no use of local memory)!, clErr=%i \n", clErr);
	else
		printf("Program 2 (version with no use of local memory) Created! \n");

	clErr = clBuildProgram(clProgram2, 0, NULL, NULL, NULL, NULL);
	if (clErr != CL_SUCCESS)
	{
		char buff[4096];
		printf("Error in building program 2 (with no local memory)!, clErr=%i \n", clErr);
		clGetProgramBuildInfo(clProgram2, clDeviceId, CL_PROGRAM_BUILD_LOG, sizeof(buff), buff, NULL);
		printf("-----Build log------\n %s\n", buff);
	}
	else printf("Program 2 Built! \n");
	stop_measure_per(PGM2);

	start_measure_per(KERNEL2);
#ifdef LOCALMEM
	clKernel2 = clCreateKernel(clProgram2, "SumLM", &clErr);
#else
	clKernel2 = clCreateKernel(clProgram2, "SumNoLM", &clErr);
#endif
	if (clErr != CL_SUCCESS)
		printf("Error in creating kernel2-NoLM!, clErr=%i \n", clErr);
	else printf("Kernel2-No-LM Created! \n");
	stop_measure_per(KERNEL2);

	free(kernelSrc2);
	fclose(fp2);
	//==================================BUFFER===================================//
	start_measure_per(BUFF);
	clSrcBuffer = clCreateBuffer(clContext, 0, sizeof(cl_int) * numofElements, NULL, &clErr);
	clIntermediateBuffer = clCreateBuffer(clContext, 0, sizeof(cl_int) * numofWorkGroups, NULL, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating buffer!, clErr=%i \n", clErr);
	else
		printf("Buffer created! \n");
	stop_measure_per(BUFF);
	start_measure_per(WRDEV);
	clErr = clEnqueueWriteBuffer(clCommandQueue, clSrcBuffer, true, 0, sizeof(cl_int) * numofElements, idata, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
		printf("Error in clEnqueueWriteBuffer!, clErr=%i \n", clErr);
	else
		printf("Data transferred into device! \n");

	clFinish(clCommandQueue);
	stop_measure_per(WRDEV);
	//=================================KERNEL1====================================//

	start_measure_per(KERNEL1_EXEC);

	clSetKernelArg(clKernel1, 0, sizeof(cl_mem), (void *) &clSrcBuffer);
	clSetKernelArg(clKernel1, 1, sizeof(cl_mem), (void *) &clIntermediateBuffer);

	clGlobalSize[0] = GLOBAL_SIZE_0;
	clGlobalSize[1] = numofElements/int(GLOBAL_SIZE_0);

	clErr = clEnqueueNDRangeKernel(clCommandQueue, clKernel1, 2, NULL, clGlobalSize, clGroupSize, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
		printf("Error in launching kernel 1!, clErr=%i \n", clErr);
	else
		printf("Kernel 1 launched successfully! \n");

	// finish executing this kernel before starting the other one
	clFinish(clCommandQueue);
	stop_measure_per(KERNEL1_EXEC);
	//=================================KERNEL2====================================//
	// Kernel2 sums up the results of each WorkGroup generated in kernel1
		start_measure_per(KERNEL2_EXEC);

		int numofWorkItems;
		int numofElements_tmp;
		cl_mem Intermediate = clIntermediateBuffer;
		numofElements_tmp = numofWorkGroups;
		numofWorkItems = (numofElements_tmp + 1) / 2;	// numofWorkGroups in kernel1 becomes numofWorkItems in kernel2.
		while (numofElements_tmp > 1)	
		{
			cl_mem tmp;
		
			tmp = clSrcBuffer;
			clSrcBuffer = clIntermediateBuffer;
			clIntermediateBuffer = tmp;

			if (numofWorkItems >= GLOBAL_SIZE_0)
			{
				clGlobalSize[0] = GLOBAL_SIZE_0;
				clGlobalSize[1] = (numofWorkItems % (GLOBAL_SIZE_0) == 0) ? numofWorkItems/(GLOBAL_SIZE_0) : numofWorkItems/(GLOBAL_SIZE_0) + 1;
			}
			else
			{
				clGlobalSize[0] = (numofWorkItems % WORK_GROUP_SIZE == 0) ? numofWorkItems : numofWorkItems + WORK_GROUP_SIZE - (numofWorkItems % WORK_GROUP_SIZE);
				clGlobalSize[1] = 1;
			}

			clSetKernelArg(clKernel2, 0, sizeof(cl_mem), (void *) &clSrcBuffer);
			clSetKernelArg(clKernel2, 1, sizeof(cl_mem), (void *) &clIntermediateBuffer);
			clSetKernelArg(clKernel2, 2, sizeof(int), (void *) &numofElements_tmp);

			clEnqueueNDRangeKernel(clCommandQueue, clKernel2, 2, NULL, clGlobalSize, clGroupSize, 0, NULL, NULL);
			if (clErr != CL_SUCCESS)
					printf("Error in launching kernel2! clErr=%i \n", clErr);
			else
				printf("Kernel2 launched successfully! \n");
			clFinish(clCommandQueue);

			numofElements_tmp = (numofElements_tmp % (2 * WORK_GROUP_SIZE) == 0) ? numofElements_tmp / (2 * WORK_GROUP_SIZE) : numofElements_tmp / (2 * WORK_GROUP_SIZE) + 1;
			numofWorkItems = (numofElements_tmp + 1) / 2;
		}
		stop_measure_per(KERNEL2_EXEC);
	
		//=================================RDDEV_RES====================================//
		start_measure_per(RDDEV);
    	clEnqueueReadBuffer(clCommandQueue, clIntermediateBuffer, CL_TRUE, 0, sizeof(cl_int), (void *) &finalResultGPU, 0, NULL, NULL);
		stop_measure_per(RDDEV);

	clReleaseContext(clContext);
	clReleaseCommandQueue(clCommandQueue);
	clReleaseProgram(clProgram);
	clReleaseMemObject(clSrcBuffer);
	clReleaseMemObject(clIntermediateBuffer);
	clReleaseKernel(clKernel1);
	clReleaseKernel(clKernel2);
	free(idata);

	//================================RESULT PRINT================================//
	FILE * fout;
	fout = fopen("log.txt", "w+");

	float total_GPU_NOLM = timeRes[KERNEL1_EXEC] + timeRes[KERNEL2_EXEC] + timeRes[WRDEV] + timeRes[RDDEV];
	float total_GPU_LM = timeRes[KERNEL1_EXEC] + timeRes[KERNEL2_EXEC] + timeRes[WRDEV] + timeRes[RDDEV];

	struct tm *local;
	time_t t;
	t = time(NULL);
	local = localtime(&t);

	fprintf(fout, "Created on: %s", asctime(local));
	fprintf(fout, "version: %s \n\n", version);
	fprintf(fout, "Result GPU-LM is: %u \n", finalResultGPU);
	fprintf(fout, "Result CPU is: %u \n", finalResultCPU);
	if (finalResultCPU != finalResultGPU)
	{
		fprintf(fout, "ERROR CPU results do not match with GPU no LM!!!!\n");
	}
	if (finalResultCPU != finalResultGPU)
	{
		fprintf(fout, "ERROR CPU results do not match with GPU with LM!!!!\n");
		exit(1);
	}
	fprintf(fout, "\n===========Performance Measurements=================\n");
	fprintf(fout, "Work-Group size is %i \n", WORK_GROUP_SIZE);
	fprintf(fout, "Problem size is %i \n\n", numofElements);
	fprintf(fout, "GPU time breakdown: \n"
			   "	WRDEV (Data Transfer) = \t%10.2f msecs \n"
			   "	KERNEL1_EXEC = \t\t\t%10.2f msecs \n"
			   "	KERNEL2NoLM_EXEC = \t\t%10.2f msecs \n"
			   "	RDDEV_RES (Data Transfer) = \t%10.2f msecs \n\n",
			   timeRes[WRDEV],
			   timeRes[KERNEL1_EXEC],
			   timeRes[KERNEL2_EXEC],
			   timeRes[RDDEV]);
	fprintf(fout, "GPU time: %10.2f msecs \n"
			      "CPU time: %10.2f msecs \n\n",
			total_GPU_NOLM, timeRes[CPU]);
	fprintf(fout, "Speed Up : %10.2f \n\n", float(timeRes[CPU])/float(total_GPU_NOLM));

	rewind(fout);
	while(fgets(buff,sizeof buff,fout))
		printf("%s", buff);
	fclose(fout);
}
