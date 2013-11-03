/*
* Author: Arian Maghazeh, Created on: Oct 18, 2012
* Edited by : Unmesh D. Bordoloi
* Implements the AES algorithm
* Performance results reported in SAMOS 2013 paper
*
* If you want to compare local memory versus non-local memory, use #define LOCALMEM
* Check the log.txt that is generated for the performance results
* You may also put a break point to see the statements printed on the screen
*
* Bug reports and fixes are truly welcome at unmesh.bordoloi@liu.se but has no guarantee of a reply :D
* 
* Known issue: CPU and GPU results are the same except for a mismatch at the end
*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <CL/cl.h>
#include <stdlib.h>
#include <omp.h>


#include "data.h"

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

// OpenCL Definitions
// #define FERMI
// #define VIVANTE

// #define LOCALMEM    // use this #def if you want to check the version that does not uses local memory    

#ifdef FERMI 

 #define PLATFORMS_IDS 3 // on the Fermi machine at IDA
 #define THIS_PLATFORM_ID 1
#else

 #define PLATFORMS_IDS 1 // on Unmesh's local desktop
 #define THIS_PLATFORM_ID 0
#endif


#define NUM_CORES		1

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
	#define WORK_GROUP_SIZE	128
#else
	#define WORK_GROUP_SIZE	256
#endif

cl_program 			clProgram;
cl_device_id 		clDeviceId;
cl_context 			clContext;
cl_kernel 			clKernel1;
cl_command_queue 	clCommandQueue;
cl_int 				clErr;
cl_mem				clPlainTextBuff, clCipherTextBuff, clKeysBuff, clIVBuff;
cl_event			prof_event;

struct timeval 		start[15];
struct timeval 		stop[15];
float 				timeRes[15] = {0};
FILE 				*fp;
FILE 				*fio;
char 				buff[256];
char 				*kernelSrc;
char 				description[256] = "AES";
size_t 				clLocalSize;
size_t 				clGlobalSize;

#ifdef FERMI
cl_platform_id 		clPlatformId[3];
#else
cl_platform_id		clPlatformId;
#endif

//Application Definitions
#define AES_BLOCK_SIZE	16
#define MB				1024 * 1024

void start_measure_time(int seg)
{
    gettimeofday(&start[seg], NULL);
}

void stop_measure_time(int seg)
{
    gettimeofday(&stop[seg], NULL);
    timeRes[seg] = 1000 * ((float)(float)(stop[seg].tv_sec  - start[seg].tv_sec) + 1.0e-6 * (stop[seg].tv_usec - start[seg].tv_usec));
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
//	clKernel1 = clCreateKernel(clProgram, "AES_encryption", &clErr);
	clKernel1 = clCreateKernel(clProgram, "AES_encrypt_local", &clErr);
	if (clErr != CL_SUCCESS)
			printf("Error in creating kernel!, clErr=%i \n", clErr);
	else printf("Kernel created! \n");
	fclose(fp);
	free(kernelSrc);
	stop_measure_time(KERNEL);
}

void oclBuffer(const unsigned char *plainText, const aes_key *eks, size_t filelen)
{
	/*-----------------------create buffer------------------------*/
	start_measure_time(BUFF);
	// filelen = 10205240; 
	clPlainTextBuff = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(unsigned char) * (filelen), NULL, &clErr);
	clCipherTextBuff = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, sizeof(unsigned char) * filelen, NULL, &clErr);
	clKeysBuff = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(unsigned int) * 4 * (eks->rounds + 1), NULL, &clErr);
	stop_measure_time(BUFF);

	/*-----------------------write into device--------------------*/
	start_measure_time(WRDEV);
	clErr  = clEnqueueWriteBuffer(clCommandQueue, clPlainTextBuff, true, 0, sizeof(unsigned char) * (filelen), plainText, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
		printf("Error in writing buffer (clPlainTextBuff)!, clErr=%i \n", clErr);
	clErr = clEnqueueWriteBuffer(clCommandQueue, clKeysBuff, true, 0, sizeof(unsigned int) * 4 * (eks->rounds + 1), eks->rd_key, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
		printf("Error in writing buffer (clKeysBuff)!, clErr=%i \n", clErr);
	clFinish(clCommandQueue);
	stop_measure_time(WRDEV);
}

void oclClean()
{
	clReleaseContext(clContext);
	clReleaseCommandQueue(clCommandQueue);
	clReleaseProgram(clProgram);
	clReleaseMemObject(clPlainTextBuff);
	clReleaseMemObject(clCipherTextBuff);
	clReleaseMemObject(clKeysBuff);
	clReleaseKernel(clKernel1);
}

void ocl_AES_cbc_encryption(const unsigned char *plainText, unsigned char *cipherText, size_t filelen, const aes_key *eks)
{
	oclInit();
	oclBuffer(plainText, eks, filelen);

	int mod = filelen % AES_BLOCK_SIZE;
	int numofWorkItems = (mod == 0 ? filelen/AES_BLOCK_SIZE : (filelen/AES_BLOCK_SIZE)+1);
	mod = numofWorkItems % WORK_GROUP_SIZE;
	if (mod != 0)
		numofWorkItems = numofWorkItems + WORK_GROUP_SIZE - mod;

	clSetKernelArg(clKernel1, 0, sizeof(cl_mem), &clPlainTextBuff);
	clSetKernelArg(clKernel1, 1, sizeof(cl_mem), &clCipherTextBuff);
	clSetKernelArg(clKernel1, 2, sizeof(cl_mem), &clKeysBuff);
	clSetKernelArg(clKernel1, 3, sizeof(unsigned int), &eks->rounds);

	#ifdef VIVANTE
		size_t clGlobalSize[2];
		size_t clLocalSize[2] = {WORK_GROUP_SIZE, 1};
		if (numofWorkItems > CL_GLOBAL_SIZE_0)
		{
			clGlobalSize[0] = CL_GLOBAL_SIZE_0;
			mod = numofWorkItems % CL_GLOBAL_SIZE_0;
			clGlobalSize[1] = (mod == 0 ? numofWorkItems/CL_GLOBAL_SIZE_0 : numofWorkItems/CL_GLOBAL_SIZE_0 + 1);
		}
		else
		{
			clGlobalSize[0] = numofWorkItems;
			clGlobalSize[1] = 1;
		}
		start_measure_time(KERNEL_EXEC);
//		while(1)
//		{
		clErr = clEnqueueNDRangeKernel(clCommandQueue, clKernel1, 2, NULL, clGlobalSize, clLocalSize, 0, NULL, &prof_event);

	#else
		size_t clLocalSize = WORK_GROUP_SIZE;
		size_t clGlobalSize = numofWorkItems;
		start_measure_time(KERNEL_EXEC);
		clErr = clEnqueueNDRangeKernel(clCommandQueue, clKernel1, 1, NULL, &clGlobalSize, &clLocalSize, 0, NULL, &prof_event);
	#endif

	if (clErr != CL_SUCCESS)
		printf("Error in launching kernel!, clErr=%i \n", clErr);
	else
		printf("Kernel launched successfully! \n");

		clFinish(clCommandQueue);
//		}
	stop_measure_time(KERNEL_EXEC);
	start_measure_time(RDDEV);
	clErr = clEnqueueReadBuffer(clCommandQueue, clCipherTextBuff, CL_TRUE, 0, sizeof(unsigned char) * filelen, cipherText, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
		printf("Error in reading buffer!, clErr=%i \n", clErr);

	clFinish(clCommandQueue);
	stop_measure_time(RDDEV);
}

void XorBlock(AESData *a, const AESData *b, const AESData *c)
{
	a->w[0] = b->w[0] ^ c->w[0];
	a->w[1] = b->w[1] ^ c->w[1];
	a->w[2] = b->w[2] ^ c->w[2];
	a->w[3] = b->w[3] ^ c->w[3];
}

void cpu_AES_cbc_encryption(const unsigned char *plainText, unsigned char *cipherText, size_t filelen, const aes_key *eks)
{
	AESData *inp;
	AESData *out;
	AESData *rkey;
	AESData state;
	const Word (*T)[256];

	union word4{ Word w; Byte b[4]; };
	word4 w0, w1, w2, w3;

//	while(1)
//	{
	omp_set_num_threads(NUM_CORES);
	#pragma omp parallel default(none) private(state, rkey, inp, out, T, w0, w1, w2, w3) shared(filelen, plainText, cipherText, eks, AESEncryptTable, AESSubBytesWordTable)
	{
		#pragma omp for
			for(int i=0; i < (filelen / AES_BLOCK_SIZE); i++)
			{
				inp = (AESData *)plainText + i;
				out = (AESData *)cipherText + i;
				rkey = (AESData *)eks->rd_key;
				XorBlock(&state, inp, rkey);
				T = AESEncryptTable;

				for (int round = 1; round < eks->rounds; ++round)
				{
					++rkey;

					w0.w = state.w[0];
					w1.w = state.w[1];
					w2.w = state.w[2];
					w3.w = state.w[3];

					state.w[0] = rkey->w[0] ^ T[0][w0.b[0]] ^ T[1][w1.b[1]] ^ T[2][w2.b[2]] ^ T[3][w3.b[3]];
					state.w[1] = rkey->w[1] ^ T[0][w1.b[0]] ^ T[1][w2.b[1]] ^ T[2][w3.b[2]] ^ T[3][w0.b[3]];
					state.w[2] = rkey->w[2] ^ T[0][w2.b[0]] ^ T[1][w3.b[1]] ^ T[2][w0.b[2]] ^ T[3][w1.b[3]];
					state.w[3] = rkey->w[3] ^ T[0][w3.b[0]] ^ T[1][w0.b[1]] ^ T[2][w1.b[2]] ^ T[3][w2.b[3]];
				}

				T = AESSubBytesWordTable;
				++rkey;

				w0.w = state.w[0];
				w1.w = state.w[1];
				w2.w = state.w[2];
				w3.w = state.w[3];

				state.w[0] = rkey->w[0] ^ T[0][w0.b[0]] ^ T[1][w1.b[1]] ^ T[2][w2.b[2]] ^ T[3][w3.b[3]];
				state.w[1] = rkey->w[1] ^ T[0][w1.b[0]] ^ T[1][w2.b[1]] ^ T[2][w3.b[2]] ^ T[3][w0.b[3]];
				state.w[2] = rkey->w[2] ^ T[0][w2.b[0]] ^ T[1][w3.b[1]] ^ T[2][w0.b[2]] ^ T[3][w1.b[3]];
				state.w[3] = rkey->w[3] ^ T[0][w3.b[0]] ^ T[1][w0.b[1]] ^ T[2][w1.b[2]] ^ T[3][w2.b[3]];

				*out = state;
			}
	}
//	}
}

int main()
{
	char hostName[50];
	unsigned char  *plainText , *cpuCipherText, *gpuCipherText;

	FILE * i_file;
	aes_key eks;

	gethostname(hostName, 50);
	i_file = fopen("input.txt", "r");
	fseek(i_file, 0, SEEK_END);
	size_t filelen = ftell(i_file);
	rewind(i_file);

	plainText = (unsigned char*) malloc(sizeof(unsigned char) * (filelen + 1));
	cpuCipherText = (unsigned char*) malloc(sizeof(unsigned char) * (filelen));
	gpuCipherText = (unsigned char*) malloc(sizeof(unsigned char) * (filelen));

	// read the input text
	unsigned int readlen = fread(plainText, 1, filelen, i_file);
	if (readlen < filelen)
		plainText[readlen] = '\0';
	else
		plainText[filelen] = '\0';
	fclose(i_file);

	memset(cpuCipherText, 0, filelen);
	memset(gpuCipherText, 0, filelen);

	for (int i=0; i<60; i++)
		eks.rd_key[i] = roundKey[i];
	eks.rounds = 14;

	ocl_AES_cbc_encryption(plainText, gpuCipherText, filelen, &eks);

	start_measure_time(CPU);
	cpu_AES_cbc_encryption(plainText, cpuCipherText, filelen, &eks);
	stop_measure_time(CPU);
	/*-------------------------print result-----------------------*/
	fio = fopen("log.txt", "a+");
	fseek (fio, 0, SEEK_END);
	int appendPos = ftell(fio);

	float total_GPU_time = timeRes[PLATFORM] + timeRes[DEVICE] + timeRes[CONTEXT] + timeRes[CMDQ] + timeRes[PGM] + timeRes[KERNEL] + timeRes[BUFF] + timeRes[WRDEV] + timeRes[KERNEL_EXEC] + timeRes[RDDEV] + timeRes[GPU_SEQ];
	float total_GPU_fair_time = timeRes[GPU_SEQ] + timeRes[KERNEL_EXEC] + timeRes[WRDEV] + timeRes[RDDEV];

//	struct tm *local;
//	time_t t;
//	t = time(NULL);
//	local = localtime(&t);

	fprintf(fio, "****************************************************\n");
//	fprintf(fio, "Created on: %s", asctime(local));
	fprintf(fio, "Host name: %s \n", hostName);
	fprintf(fio, "Description: %s \n\n", description);
	fprintf(fio, "Input size: %iMB \n\n", (unsigned int)filelen / (MB));
	/*for (unsigned int i=0; i<filelen; i++)
		if (cpuCipherText[i] != gpuCipherText[i])
		{
			printf("not match at i= %i\n", i);
			return(-1);
		}*/
	fprintf(fio, "\n===========Performance Measurements=================\n");
	fprintf(fio, "Execution times: \n"
			   "	PLATFORM = \t%10.2f msecs \n"
			   "	DEVICE = \t%10.2f msecs \n"
			   "	CONTEXT = \t%10.2f msecs \n"
			   "	CMDQ = \t\t%10.2f msecs \n"
			   "	PGM = \t\t%10.2f msecs \n"
			   "	KERNEL = \t%10.2f msecs \n"
			   "	BUFF = \t\t%10.2f msecs \n"
			   "	WRDEV = \t%10.2f msecs \n"
			   "	KERNEL_EXEC = \t%10.2f msecs \n"
			   "	RDDEV = \t%10.2f msecs \n\n"
			   "    GPU_SEQ = \t%10.2f msec \n\n",
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

	oclClean();
	free(plainText);
	free(cpuCipherText);
	free(gpuCipherText);

	return 0;
}
