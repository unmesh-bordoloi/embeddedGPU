/*
 * pm.c++
 *
 *  Created on: Oct 2, 2012
 *  Author: Arian Maghazeh
 *  Edited by : Unmesh D. Bordoloi
 *  Implements the convolution algorithm
 *  Performance results reported in SAMOS 2013 paper
 *
 *  Check the log.txt that is generated for the performance results
 *  For debugging, you may also put a break point to see the statements printed on the screen
 *
 *  IMPORTANT NOTE: This code is not stable. 2 parts of the compute intensive part have been
 *  implemented but one part has not been implemented yet. See below for hints. #pragma omp 
 *  has been commented but should be enabled for openmp.
 *
 *  Bug reports and fixes are truly welcome at unmesh.bordoloi@liu.se but has no guarantee of a reply :D
 */ 
/******************************************************************************
** File: pm.c
**
** HPEC Challenge Benchmark Suite
** Pattern Match Kernel Benchmark
**
** Contents:
**  This is the ANSI C Pattern Match kernel. It finds the closest
**  match of a pattern from a library of patterns.  The code below
**  serves as a reference implemenation of the pattern match kernel.
**
** Input/Output:
**  The template library is stored in file
**             "./data/<dataSetNum>-pm-lib.dat".
**  The test pattern is stored in file
**             "./data/<dataSetNum>-pm-pattern.dat".
**  The timing will be stored in file
**             "./data/<dataSetNum>-pm-timing.dat".
**  The matched pattern index will be stored in file
**             "./data/<dataSetNum>-pm-patnum.dat"
**
** Command:
**  pm <data set num>
**
** Author: Hector Chan
**         MIT Lincoln Laboratory
**
******************************************************************************/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <omp.h>

#include "PcaCArray.h"
#include "PcaCTimer.h"


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
/*-------------------OpenCL Definitions-----------------------*/
#define FERMI

#ifdef FERMI
cl_platform_id 		clPlatformId[3];
#else
cl_platform_id		clPlatformId;
#endif
#define PLATFORM		0
#define DEVICE			1
#define CONTEXT			2
#define CMDQ			3
#define PGM				4
#define KERNEL			5
#define KERNEL1_EXEC	6
#define KERNEL2_EXEC	7
#define BUFF			8
#define WRDEV			9
#define RDDEV			10
#define CPU				11
#define GPU_SEQ			12

#define GLOBAL_SIZE_0		(32*1024)
#define WORK_GROUP_SIZE		PROFILE_SIZE

cl_program 			clProgram;
cl_device_id 		clDeviceId;
cl_context 			clContext;
cl_kernel 			clKernel1;
cl_kernel 			clKernel2;
cl_command_queue 	clCommandQueue;
cl_int 				clErr;

cl_mem				cl_tmp_pf_db;
cl_mem				cl_inm_tmp_pf_db;
cl_mem				cl_tmp_exc;
cl_mem				cl_tmp_exc_mean;
cl_mem 				cl_noise_shift;
cl_mem				cl_test;

cl_mem				cl_weighted_MSEs;
//cl_mem 				cl_num_tmp_exc;
cl_mem 				cl_test_pf_db;
cl_mem 				cl_test_exc_means;
//cl_mem 				cl_pwr_ratio;
size_t 				clGlobalSize[2];
size_t 				clLocalSize[2];

struct timeval 		start[15];
struct timeval 		stop[15];
float 				timeRes[15] = {0};

FILE 				*fp;
FILE 				*fio;
char 				buff[256];
char 				*kernelSrc;
char 				description[256] = "Pattern Matching from HPEC, data set 1";

/*-------------------Application Definitions-----------------------*/
#define TEMPLATE_SIZE		72
#define SHIFT_SIZE			21
#define PROFILE_SIZE		64

#define LOG10 				2.302585093
#define MIN_NOISE 			1e-10

/* The coefficients for the log and pow functions below */
float pow_coeff[19];
float log_coeff[16];

typedef unsigned char uchar;
typedef unsigned int  uint;

typedef struct {
	float *template_profiles_db; /* the library of patterns */
	float *test_profile_db;      /* the test pattern */

	float *template_copy;        /* temporary storage for a template */
	float *test_noise_db_array;  /* copies of test noise in an array for
								  fast copy */
	float *MSE_scores;           /* the likelihood of the matching between a
								  range shift of the test pattern and the libary */
	float *mag_shift_scores;     /* the likelihood of the matching between a
								  magnitude scaling of the test pattern and the libary */
	float *minimum_MSE_score;    /* the likelihood of the matching between the
								  test pattern and the libary */
	float *all_shifted_test_db;  /* contains the shiftings of the test pattern */

	uchar *template_exceed;      /* marking where a library template
								  exceeds twice the noise level of the test pattern */
	float *test_exceed_means;    /* pixels where test pattern exceeds twice
								  its noise level */

	float shift_ratio;           /* determines the number of range shifts */
	int   shift_size;            /* the actual number of range shifts */
	int   profile_size;          /* the length of the pattern */
	int   num_templates;         /* the number of library templates */
	int   elsize;                /* the size of a single fp number */
} PmData;

PmData			cpuPmdata, gpuPmdata;
PcaCArrayFloat 	lib1, pattern1, lib2, pattern2, rtime;
PcaCArrayInt   	patnum;

float *test1;
float test2[72];
float *GPU_weighted_MSEs = (float *)malloc(sizeof(float) * TEMPLATE_SIZE * SHIFT_SIZE * PROFILE_SIZE);
float *CPU_weighted_MSEs = (float *)malloc(sizeof(float) * TEMPLATE_SIZE * SHIFT_SIZE * PROFILE_SIZE);
float *test = (float *)malloc(sizeof(float) * TEMPLATE_SIZE * SHIFT_SIZE * PROFILE_SIZE);

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
	clKernel1 = clCreateKernel(clProgram, "pm_part1", &clErr);
	clKernel2 = clCreateKernel(clProgram, "pm_part2", &clErr);
	if (clErr != CL_SUCCESS)
			printf("Error in creating kernel!, clErr=%i \n", clErr);
	else printf("Kernel created! \n");
	fclose(fp);
	free(kernelSrc);
	stop_measure_time(KERNEL);
    printf("OpenCL init was successful\n");
}

void oclBuffer(float *template_profiles_db, float *noise_shift, float *test_exc_means, float *test_pf_db)
{
	/*-----------------------create buffer------------------------*/
	printf("OpenCL buffer creation and writing begins now ");
	start_measure_time(BUFF);
	cl_tmp_pf_db = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(cl_float) * TEMPLATE_SIZE * PROFILE_SIZE, NULL, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating buffer cl_tmp_pf_db!, clErr=%i \n", clErr);
	cl_inm_tmp_pf_db = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(cl_float) * TEMPLATE_SIZE * PROFILE_SIZE, NULL, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating image cl_inm_tmp_pf_db!, clErr=%i \n", clErr);
	cl_tmp_exc = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(char) * TEMPLATE_SIZE * PROFILE_SIZE, NULL, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating image cl_tmp_exc!, clErr=%i \n", clErr);
	cl_tmp_exc_mean = clCreateBuffer(clContext, CL_MEM_READ_WRITE, sizeof(cl_float) * TEMPLATE_SIZE, NULL, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating image cl_tmp_exc_mean!, clErr=%i \n", clErr);
	cl_noise_shift = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(float) * TEMPLATE_SIZE, NULL, &clErr);
	if (clErr != CL_SUCCESS)
		printf("Error in creating image cl_noise_shift!, clErr=%i \n", clErr);

	cl_test = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, sizeof(float) * TEMPLATE_SIZE * SHIFT_SIZE * PROFILE_SIZE, NULL, &clErr);
    if (clErr != CL_SUCCESS)
		printf("Error in creating image cl_test!, clErr=%i \n", clErr);
	cl_weighted_MSEs = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, sizeof(float) * TEMPLATE_SIZE * SHIFT_SIZE * PROFILE_SIZE, NULL, &clErr);
    if (clErr != CL_SUCCESS)
		printf("Error in creating image cl_weighted_MSEs!, clErr=%i \n", clErr);
//	cl_num_tmp_exc = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(char) * TEMPLATE_SIZE, NULL, &clErr);
	cl_test_pf_db = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(float) * PROFILE_SIZE, NULL, &clErr);
    if (clErr != CL_SUCCESS)
		printf("Error in creating image cl_test_pf_db!, clErr=%i \n", clErr);
	cl_test_exc_means = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(float) * SHIFT_SIZE, NULL, &clErr);
    if (clErr != CL_SUCCESS)
		printf("Error in creating image cl_test_exc_means!, clErr=%i \n", clErr);
//	cl_pwr_ratio = clCreateBuffer(clContext, CL_MEM_READ_ONLY, sizeof(float) * TEMPLATE_SIZE * SHIFT_SIZE, NULL, &clErr);
	stop_measure_time(BUFF);

	/*-----------------------write into device--------------------*/
	start_measure_time(WRDEV);
	clErr = clEnqueueWriteBuffer(clCommandQueue, cl_tmp_pf_db, true, 0, sizeof(cl_float) * TEMPLATE_SIZE * PROFILE_SIZE, template_profiles_db, 0, NULL, NULL);
    if (clErr != CL_SUCCESS)
		printf("Error in writing image cl_tmp_pf_db!, clErr=%i \n", clErr);
	clErr = clEnqueueWriteBuffer(clCommandQueue, cl_noise_shift, true, 0, sizeof(cl_float) * TEMPLATE_SIZE, noise_shift, 0, NULL, NULL);
    if (clErr != CL_SUCCESS)
		printf("Error in writing image cl_noise_shift!, clErr=%i \n", clErr);
	clErr = clEnqueueWriteBuffer(clCommandQueue, cl_test_exc_means, true, 0, sizeof(cl_float) * SHIFT_SIZE, test_exc_means, 0, NULL, NULL);
    if (clErr != CL_SUCCESS)
		printf("Error in writing image cl_test_exc_means!, clErr=%i \n", clErr);
	clErr = clEnqueueWriteBuffer(clCommandQueue, cl_test_pf_db, true, 0, sizeof(cl_float) * PROFILE_SIZE, test_pf_db, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
		printf("Error in writing image cl_test_pf_db!, clErr=%i \n", clErr);
	clFinish(clCommandQueue);
	stop_measure_time(WRDEV);
	printf("OpenCL buffer creation and writing was successful");
}

void oclClean()
{
	clReleaseContext(clContext);
	clReleaseCommandQueue(clCommandQueue);
	clReleaseProgram(clProgram);
	clReleaseMemObject(cl_inm_tmp_pf_db);
	clReleaseMemObject(cl_tmp_pf_db);
	clReleaseMemObject(cl_tmp_exc);
	clReleaseMemObject(cl_tmp_exc_mean);
	clReleaseKernel(clKernel1);
}
/***********************************************************************/
/* We found out the bottle neck of this kernel was in the pow and log
 * functions. Therefore, we have implemented our own log and pow, instead
 * of using the double fp ones in the standard C math libary. This function
 * sets up the coefficients for the single fp log and pow functions. */
/***********************************************************************/
void setcoeff()
{
	pow_coeff[0]  = 0.5f;             /* 1/2! */
	pow_coeff[1]  = 0.166666667f;     /* 1/3! */
	pow_coeff[2]  = 0.041666666f;     /* 1/4! */
	pow_coeff[3]  = 8.333333333e-3f;
	pow_coeff[4]  = 1.388888889e-3f;
	pow_coeff[5]  = 1.984126984e-4f;
	pow_coeff[6]  = 2.480158730e-5f;
	pow_coeff[7]  = 2.755731922e-6f;
	pow_coeff[8]  = 2.755731922e-7f;
	pow_coeff[9]  = 2.505210839e-8f;
	pow_coeff[10] = 2.087675699e-9f;
	pow_coeff[11] = 1.605904384e-10f;
	pow_coeff[12] = 1.147074560e-11f;
	pow_coeff[13] = 7.647163732e-13f;
	pow_coeff[14] = 4.779477332e-14f;
	pow_coeff[15] = 2.811457254e-15f;
	pow_coeff[16] = 1.561920697e-16f;
	pow_coeff[17] = 8.220635247e-18f;
	pow_coeff[18] = 4.110317623e-19f;

	log_coeff[0]  = 0.333333333f;     /* 1/3 */
	log_coeff[1]  = 0.2f;             /* 1/5 */
	log_coeff[2]  = 0.142857143f;     /* 1/7 */
	log_coeff[3]  = 0.111111111f;     /* 1/9 */
	log_coeff[4]  = 9.090909091e-2f;  /* 1/11 */
	log_coeff[5]  = 7.692307692e-2f;  /* 1/13 */
	log_coeff[6]  = 6.666666667e-2f;  /* 1/15 */
	log_coeff[7]  = 5.882352941e-2f;  /* 1/17 */
	log_coeff[8]  = 5.263157895e-2f;  /* 1/19 */
	log_coeff[9]  = 4.761904762e-2f;  /* 1/21 */
	log_coeff[10] = 4.347826087e-2f;  /* 1/23 */
	log_coeff[11] = 0.04f;            /* 1/25 */
	log_coeff[12] = 3.703703704e-2f;  /* 1/27 */
	log_coeff[13] = 3.448275862e-2f;  /* 1/29 */
	log_coeff[14] = 3.225806452e-2f;  /* 1/31 */
	log_coeff[15] = 3.030303030e-2f;  /* 1/33 */
}

/***********************************************************************/
/* This single fp pow base 10 function implements the corresponding
 * Taylor series.  The loop has been unrolled to save ops. */
/***********************************************************************/
float pow10fpm (float exp)
{
  float mul = exp * LOG10;
  float const term = exp * LOG10;
  float ans = 1.0f;
  float const *fptr = pow_coeff;

  ans += mul;           mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul;

  return ans;
}

/***********************************************************************/
/* This single fp log base 10 function implements the corresponding
 * Taylor series. The loop has been unrolled to save ops. */
/***********************************************************************/
float log10fpm (float exp)
{
  float mul = (exp - 1.0f) / (exp + 1.0f);
  float ans = 0.0f;
  float const *fptr = log_coeff;
  float const term = mul * mul;

  ans  = mul;           mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul; mul *= term;
  ans += *fptr++ * mul;

  ans *= 0.86858896381;  /* ans = ans * 2 / log(10) */
  return ans;
}

/***********************************************************************/
/* Allocate and initailize the test pattern, the template library, and
 * other necessary data structure. */
/***********************************************************************/
void init(PmData *pmdata, PcaCArrayFloat *lib, PcaCArrayFloat *pattern)
{
  int   elsize = sizeof(float);
  float x;

  /* Getting the input parameters from the PCA C array structure */
  pmdata->profile_size  = lib->size[1];
  pmdata->num_templates = lib->size[0];

  pmdata->elsize = elsize;
  pmdata->shift_ratio = 3.0f;

  pmdata->template_profiles_db = lib->data;
  pmdata->test_profile_db = pattern->data;

  /* Equivalent to shift_size = roundf((float)PROFILE_SIZE / shift_ratio) */
  x = (float)(pmdata->profile_size) / pmdata->shift_ratio;
  pmdata->shift_size = ((x - (int)(x)) < 0.5) ? (int)floor(x) : (int)ceil(x);

  pmdata->template_exceed     = (uchar*) malloc(sizeof(char)*pmdata->profile_size);
  pmdata->test_exceed_means   = (float*) malloc(elsize*pmdata->shift_size);

  pmdata->template_copy       = (float*) malloc(elsize*pmdata->profile_size);
  pmdata->test_noise_db_array = (float*) malloc(elsize*pmdata->profile_size);

  pmdata->MSE_scores          = (float*) malloc(elsize*pmdata->shift_size);
  pmdata->mag_shift_scores    = (float*) malloc(elsize*21);

  pmdata->minimum_MSE_score   = (float*) malloc(elsize*pmdata->num_templates);
  pmdata->all_shifted_test_db = (float*) malloc(elsize*((pmdata->shift_size+2)*2+pmdata->profile_size));

  /* Set the coefficients for the log and pow functions */
  setcoeff();
}

/***********************************************************************/
/* Free up memory for all structures */
/***********************************************************************/
void clean(PmData *pmdata)
{
  free(pmdata->test_exceed_means);
  pmdata->test_exceed_means = 0;

  free(pmdata->template_exceed);
  pmdata->template_exceed = 0;

  free(pmdata->template_copy);
  pmdata->template_copy = 0;

  free(pmdata->test_noise_db_array);
  pmdata->test_noise_db_array = 0;

  free(pmdata->MSE_scores);
  pmdata->MSE_scores = 0;

  free(pmdata->mag_shift_scores);
  pmdata->mag_shift_scores = 0;

  free(pmdata->minimum_MSE_score);
  pmdata->minimum_MSE_score = 0;

  free(pmdata->all_shifted_test_db);
  pmdata->all_shifted_test_db = 0;
}

/***********************************************************************/
int pmGPU(PmData *pmdata)
{
	start_measure_time(GPU_SEQ);
	int    elsize               		= pmdata->elsize;
	float *test_profile_db     		 	= pmdata->test_profile_db;
	float *template_profiles_db		 	= pmdata->template_profiles_db;
	float *test_exceed_means 			= pmdata->test_exceed_means;

	float sumWeights_inv = 1.0f / PROFILE_SIZE;
	float test_noise = ( pow10fpm(test_profile_db[0]*0.1f) +              /* noise level of the test pattern */
			       pow10fpm(test_profile_db[PROFILE_SIZE-1]*0.1f) ) * 0.5f;

	float test_peak;
	float template_peak;
	float noise_shift[TEMPLATE_SIZE];

	float power_shift, ave_power_ratio;
	float power_ratio;
	int patsize = PROFILE_SIZE*elsize;
	int half_shift_size = (int)ceil((float)(SHIFT_SIZE) / 2.0f);
	float test_noise_db        = (test_noise == 0.0f) ? -100.0f : 10.0f * log10fpm(fabs(test_noise)); /* test noise in dB */
	float test_noise_db_plus_3 = test_noise_db + 3.0f;
	float *cur_tp, *fptr, *fptr2, *fptr3, *endptr;

	float sum_exceed = 0.0f;
	int num_test_exceed;
	int num_template_exceed[TEMPLATE_SIZE] = {0};

	int i;
	uchar *bptr;
	uchar *template_exceed   = (uchar *)malloc(sizeof(uchar) * PROFILE_SIZE * TEMPLATE_SIZE);
	float *template_exceed_mean = (float *)malloc(sizeof(float) * TEMPLATE_SIZE);

	fptr = test_profile_db;
	test_peak = *fptr++;
	for (i=1; i<PROFILE_SIZE; i++,fptr++)
	{
		if (test_peak < *fptr)
			test_peak = *fptr;
		if (*fptr < test_noise_db)
			*fptr = test_noise_db;
	}
	fptr2 = test_exceed_means;
	for (int current_shift=0; current_shift<SHIFT_SIZE; current_shift++)
	{
		/* Pointer arithmetics to find the start and end pointers */
		if (current_shift < half_shift_size) {
			endptr = test_profile_db + current_shift + PROFILE_SIZE - half_shift_size;
			fptr   = test_profile_db;
		}
		else {
			endptr = test_profile_db + PROFILE_SIZE;
			fptr   = test_profile_db + current_shift - half_shift_size;
		}
		/* Summing the pixels that exceed twice test noise for the current shifts */
		sum_exceed = 0.0f;
		num_test_exceed = 0;
		while (fptr != endptr)
		{
			if (*fptr > test_noise_db_plus_3)
			{
				num_test_exceed++;
				sum_exceed += *fptr;
			}
			fptr++;
		}
		*fptr2++ = num_test_exceed ? sum_exceed / (float)(num_test_exceed) : 0.0f;
	} /* for (current_shift=0; current_shift<shift_size; current_shift++) */

	for (int template_index=0; template_index<TEMPLATE_SIZE; template_index++)
	{
		cur_tp = template_profiles_db+(template_index*PROFILE_SIZE);
		fptr = cur_tp;
		template_peak = *fptr++;
		for (i=1; i<PROFILE_SIZE; i++,fptr++)
		  if (template_peak < *fptr)
			template_peak = *fptr;

		noise_shift[template_index] = test_peak - template_peak;
	}
	stop_measure_time(GPU_SEQ);

	/*--------------------------------OpenCL---------------------------------*/
	oclInit();
	printf("Between OpenCL Init and Buffer creation");
	oclBuffer(template_profiles_db, noise_shift, test_exceed_means, test_profile_db);

 	/*-------------------------launch kernel1--------------------------------*/
 	clSetKernelArg(clKernel1, 0, sizeof(cl_mem), &cl_tmp_pf_db);
 	clSetKernelArg(clKernel1, 1, sizeof(cl_mem), &cl_inm_tmp_pf_db);
 	clSetKernelArg(clKernel1, 2, sizeof(cl_mem), &cl_tmp_exc);
 	clSetKernelArg(clKernel1, 3, sizeof(cl_mem), &cl_tmp_exc_mean);
 	clSetKernelArg(clKernel1, 4, sizeof(cl_mem), &cl_noise_shift);
 	clSetKernelArg(clKernel1, 5, sizeof(float), &test_noise);
 	clSetKernelArg(clKernel1, 6, sizeof(float), &test_noise_db);

 	clGlobalSize[0] = TEMPLATE_SIZE * PROFILE_SIZE;
 	clGlobalSize[1] = 1;
 	clLocalSize[0] = WORK_GROUP_SIZE;
 	clLocalSize[1] = 1;

 	cl_event prof_event;

 	clErr = clEnqueueNDRangeKernel(clCommandQueue, clKernel1, 2, 0, clGlobalSize, clLocalSize, 0, NULL, &prof_event);
 	if (clErr != CL_SUCCESS)
		printf("Error in executing kernel1!, clErr=%i \n", clErr);
	clFinish(clCommandQueue);

	cl_ulong queued_time = (cl_ulong)0;
	cl_ulong submitted_time = (cl_ulong)0;
	cl_ulong start_time = (cl_ulong)0;
	cl_ulong end_time   = (cl_ulong)0;
	size_t return_bytes;

	clErr = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued_time, &return_bytes);
	clErr = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submitted_time, &return_bytes);
	clErr = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
	clErr = clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);
	if (clErr != CL_SUCCESS)
	{
		printf("Error in clGetEvent!, clErr=%i \n", clErr);
		exit(1);
	}
	timeRes[KERNEL1_EXEC] = (end_time-submitted_time) * 1.0e-6;

/*-----------------------read from device---------------------*/
/*
*  This is chunk of code to check that intermediate results after kernel1 are correct 
*  when compared to the intermediate results in the CPU
*/

// 	start_measure_time(RDDEV);
//	clErr = clEnqueueReadBuffer(clCommandQueue, cl_inm_tmp_pf_db, CL_TRUE, 0, sizeof(float) * PROFILE_SIZE * TEMPLATE_SIZE, template_profiles_db, 0, NULL, NULL);
//	clErr = clEnqueueReadBuffer(clCommandQueue, cl_tmp_exc, CL_TRUE, 0, sizeof(char) * PROFILE_SIZE * TEMPLATE_SIZE, template_exceed, 0, NULL, NULL);
//	clErr = clEnqueueReadBuffer(clCommandQueue, cl_tmp_exc_mean, CL_TRUE, 0, sizeof(float) * TEMPLATE_SIZE, template_exceed_mean, 0, NULL, NULL);
//	if (clErr != CL_SUCCESS)
//		printf("Error in reading buffer!, clErr=%i \n", clErr);
//	clFinish(clCommandQueue);
//	stop_measure_time(RDDEV);

//	test1 = template_exceed_mean;
//	start_measure_time(GPU_SEQ);
//	for(int i=0; i<TEMPLATE_SIZE; i++)
//	{
//		sum_exceed = 0.0f;
//		for (int j=0; j<PROFILE_SIZE; j++)
//			if (template_profiles_db[i*PROFILE_SIZE+j] > test_noise_db_plus_3)
//			{
//				template_exceed[i*PROFILE_SIZE+j] = 1;
//				num_template_exceed[i]++;
//				sum_exceed += template_profiles_db[i*PROFILE_SIZE+j];
//			}
//		template_exceed_mean[i] = sum_exceed / (float)num_template_exceed[i];
//	}
//	stop_measure_time(GPU_SEQ);

	/*-------------------------launch kernel2--------------------------------*/
	cl_tmp_pf_db = cl_inm_tmp_pf_db;
	clSetKernelArg(clKernel2, 0, sizeof(cl_mem), &cl_tmp_pf_db);
	clSetKernelArg(clKernel2, 1, sizeof(cl_mem), &cl_weighted_MSEs);
	clSetKernelArg(clKernel2, 2, sizeof(cl_mem), &cl_tmp_exc);
	clSetKernelArg(clKernel2, 3, sizeof(cl_mem), &cl_tmp_exc_mean);
	clSetKernelArg(clKernel2, 4, sizeof(cl_mem), &cl_test_pf_db);
	clSetKernelArg(clKernel2, 5, sizeof(cl_mem), &cl_test_exc_means);
	clSetKernelArg(clKernel2, 6, sizeof(float), &test_noise_db);
	clSetKernelArg(clKernel2, 7, sizeof(cl_mem), &cl_test);

	int numofWorkItems = TEMPLATE_SIZE * SHIFT_SIZE * PROFILE_SIZE;
	clLocalSize[0] = PROFILE_SIZE;
	clLocalSize[1] = 1;

	if (numofWorkItems >= GLOBAL_SIZE_0)
	{
		clGlobalSize[0] = GLOBAL_SIZE_0;
		clGlobalSize[1] = (numofWorkItems % GLOBAL_SIZE_0 == 0) ? numofWorkItems/GLOBAL_SIZE_0 : numofWorkItems/GLOBAL_SIZE_0 + 1;
	}
	else
	{
		clGlobalSize[0] = (numofWorkItems % WORK_GROUP_SIZE == 0) ? numofWorkItems : numofWorkItems + WORK_GROUP_SIZE - (numofWorkItems % WORK_GROUP_SIZE);
		clGlobalSize[1] = 1;
	}

	clErr = clEnqueueNDRangeKernel(clCommandQueue, clKernel2, 2, 0, clGlobalSize, clLocalSize, 0, NULL, &prof_event);
	if (clErr != CL_SUCCESS)
	{
		printf("Error in executing kernel2!, clErr=%i \n", clErr);
		exit(1);
	}
	clFinish(clCommandQueue);

	clErr |= clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued_time, &return_bytes);
	clErr |= clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submitted_time, &return_bytes);
	clErr |= clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
	clErr |= clGetEventProfilingInfo(prof_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);

	start_measure_time(RDDEV);
   
/*
*	The following lines may be commented for the purposes of performance measurement.
*    They have the intermediate results from the second kernel in the GPU 
*/
	clErr = clEnqueueReadBuffer(clCommandQueue, cl_weighted_MSEs, CL_TRUE, 0, sizeof(float) * TEMPLATE_SIZE * SHIFT_SIZE * PROFILE_SIZE, GPU_weighted_MSEs, 0, NULL, NULL);
//	clErr = clEnqueueReadBuffer(clCommandQueue, cl_test, CL_TRUE, 0, sizeof(float) * TEMPLATE_SIZE * SHIFT_SIZE * PROFILE_SIZE, test, 0, NULL, NULL);
//	clErr = clEnqueueReadBuffer(clCommandQueue, cl_tmp_exc, CL_TRUE, 0, sizeof(char) * PROFILE_SIZE * TEMPLATE_SIZE, template_exceed, 0, NULL, NULL);
//	clErr = clEnqueueReadBuffer(clCommandQueue, cl_tmp_exc_mean, CL_TRUE, 0, sizeof(float) * TEMPLATE_SIZE, template_exceed_mean, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
	{
		printf("Error in reading buffer!, clErr=%i \n", clErr);
		exit(1);
	}
	clFinish(clCommandQueue);
	stop_measure_time(RDDEV);

	timeRes[KERNEL2_EXEC] = (end_time-submitted_time) * 1.0e-6;
}
/***********************************************************************/
/* The pattern match kernel overlays two patterns to compute the likelihood
 * that the two vectors match. This process is performed on a library of
 * patterns. */
/***********************************************************************/
int pmCPU(PmData *pmdata)
{
	start_measure_time(CPU);
	int    elsize               = pmdata->elsize;               /* size of a single fp number    */
	int    shift_size           = pmdata->shift_size;           /* number of shifting to the left and right of the test profile */
	int    profile_size         = pmdata->profile_size;         /* number of pixels in a pattern */
	int    num_templates        = pmdata->num_templates;        /* number of library patterns    */
	float *test_profile_db      = pmdata->test_profile_db;      /* the test pattern              */
	float *template_profiles_db = pmdata->template_profiles_db; /* the library of patterns       */
	float *test_noise_db_array  = pmdata->test_noise_db_array;  /* the noise in the test pattern in an array for fast copy */
	float *all_shifted_test_db  = pmdata->all_shifted_test_db;  /* the shifted test pattern      */

	uint  match_index;                          /* the index of the most likely template that matches the test pattern */
	uint  min_MSE_index = shift_size + 1;       /* the index of the range shifts with the lowest mean square error */
	uint  num_template_exceed, num_test_exceed; /* the number of pixels exceeded the test pattern and a library template */

	uchar mag_shift_scores_flag;    /* flag that tells if the magnitude scaling loop has been run (existed just to save ops) */

	float test_peak, template_peak; /* the maximum pixels of the test pattern and a library template pattern */
	float template_noise;           /* the noise level of a library template */

	float noise_shift, noise_shift2; /* temporary storage for calculating the mse for range shifting */

	float min_MSE, match_score; /* temporary storage for finding the minimum mse */

	float sumWeights_inv = 1.0f / profile_size; /* the inverse of the weights used for calculating the mse */
	/* Note: weights for the kernel would be application dependent. They are set to 1 for our purposes */

	float mag_db;                        /* the magnitude shifts in dB */
	float power_shift, ave_power_ratio;  /* the diff of the avg shifted test profile power to the avg template power */
	float power_ratio;                   /* the mean power of the pixels of a template that exceeded twice test noise */

	float test_noise = ( pow10fpm(test_profile_db[0]*0.1f) +              /* noise level of the test pattern */
			   pow10fpm(test_profile_db[profile_size-1]*0.1f) ) * 0.5f;

	int half_shift_size = (int)ceil((float)(shift_size) / 2.0f); /* since "shift_size/2" is used a lot, so we create a var to hold it */
	int template_index, current_shift; /* indices */
	int patsize = profile_size*elsize; /* number of bytes of a pattern */

	float *minimum_MSE_score = pmdata->minimum_MSE_score;
	float *MSE_scores        = pmdata->MSE_scores;
	float *mag_shift_scores  = pmdata->mag_shift_scores;

	float test_noise_db        = (test_noise == 0.0f) ? -100.0f : 10.0f * log10fpm(fabs(test_noise)); /* test noise in dB */
	float test_noise_db_plus_3 = test_noise_db + 3.0f; /* twice test noise in the power domain, approximately +3dB */

	float *template_copy     = pmdata->template_copy;
	uchar *template_exceed   = pmdata->template_exceed;
	float *test_exceed_means = pmdata->test_exceed_means;

	int i, j; /* indices */

	float tmp1, tmp2;                    /* temporary storage for calculating the mse for range shifting */
	float sum_exceed;                    /* the sum of the test pattern pixels exceeded twice test noise */
	float template_exceed_mean=0;        /* the mean of a template pattern pixels exceeded twice test noise */
	float weighted_MSE;                  /* temporary storage for computing the weighted MSE */

	/* These pointers are solely used for fast memory access */
	float *cur_tp, *fptr, *fptr2, *fptr3, *endptr;
	uchar *bptr;

	/* Having an array of test noise for fast copying of noise returns */
	for (i=0; i<profile_size; i++)
		test_noise_db_array[i] = test_noise_db;

	/* Finding the maximum pixels of the test pattern */
	fptr = test_profile_db;
	test_peak = *fptr++;
	for (i=1; i<profile_size; i++,fptr++)
		if (test_peak < *fptr)
			test_peak = *fptr;

	/* Paddle array for all the possible range shifts. Essentially, we are
	* performing the following:
	*
	* Adding these two portions to the beginning and end of the test pattern
	*      |                          |
	*      V                          V
	*  |<------>|                 |<------>|
	*
	*               __       __
	*              |  |     |  |
	*             |    |___|    |
	*            |               |
	*  _________|                 |_________   <- test noise in dB domain
	* ---------------------------------------  <- zero
	*
	*           |<--------------->|
	*           original test pattern
	*
	*
	* The all_shifted_test_db will be accessed in a sliding window manner.
	*/

	memcpy((void*) all_shifted_test_db, (void*) test_noise_db_array, elsize*half_shift_size);
	memcpy((void*) (all_shifted_test_db+half_shift_size), (void*) test_profile_db, elsize*profile_size);
	memcpy((void*) (all_shifted_test_db+half_shift_size+profile_size), (void*) test_noise_db_array, elsize*half_shift_size);

	/* Set the pixels to test noise in dB domain if pixel is less than test noise in dB */
	fptr = all_shifted_test_db + half_shift_size;
	for (i=0; i<profile_size; i++,fptr++)
		if (*fptr < test_noise_db)
			*fptr = test_noise_db;

	/* Calculating the mean of the pixels that exceeded twice test noise for each
	* possible shift of the test profile */
	fptr2 = test_exceed_means;
	for (current_shift=0; current_shift<shift_size; current_shift++)
	{
		/* Pointer arithmetics to find the start and end pointers */
		if (current_shift < half_shift_size) {
			endptr = all_shifted_test_db + current_shift + profile_size;
			fptr   = all_shifted_test_db + half_shift_size;
		}
		else {
			endptr = all_shifted_test_db + half_shift_size + profile_size;
			fptr   = all_shifted_test_db + current_shift;
		}

		/* Summing the pixels that exceed twice test noise for the current shifts */
		sum_exceed = 0.0f;
		num_test_exceed = 0;
		while (fptr != endptr)
		{
			if (*fptr > test_noise_db_plus_3)
			{
				num_test_exceed++;
				sum_exceed += *fptr;
			}
		  fptr++;
		}

		*fptr2++ = num_test_exceed ? sum_exceed / (float)(num_test_exceed) : 0.0f;
	} /* for (current_shift=0; current_shift<shift_size; current_shift++) */

	/* Loop over all the templates. Determine the best shift distance, then
	* the best gain adjustment. */

//	#pragma omp parallel default(none) \
	private(template_index, cur_tp, fptr, template_peak, i, noise_shift, sum_exceed, num_template_exceed, template_noise, template_exceed_mean,\
			noise_shift2, tmp1, template_exceed, current_shift, template_copy, bptr, power_ratio, weighted_MSE, MSE_scores, fptr2, fptr3) \
	shared(template_profiles_db, profile_size, test_peak, test_noise, test_noise_db, CPU_weighted_MSEs, num_templates,\
			test_noise_db_plus_3, test_exceed_means, shift_size, patsize, all_shifted_test_db, sumWeights_inv)
	{
//		#pragma omp for
			for (template_index=0; template_index<num_templates; template_index++)
			{
				cur_tp = template_profiles_db+(template_index*profile_size);

				/* Scale the template profile we're currently working on so that its peak
				 * is equal to the peak of the test profile */

				/* --------------------------------------------------------------------
				 * template_peak = max( template_profile ) */
				fptr = cur_tp;
				template_peak = *fptr++;
				for (i=1; i<profile_size; i++,fptr++)
				  if (template_peak < *fptr)
					template_peak = *fptr;

				/* Additively adjust the noise level of this template profile in the
				 * raw power domain so that its noise level matches the noise level
				 * of the test profile */

				/* --------------------------------------------------------------------
				   Setting up all the constants */

				noise_shift  = test_peak - template_peak;

//				memset ((void*)template_exceed, 0, sizeof(char)*PROFILE_SIZE);
				for (i=0; i<PROFILE_SIZE; i++)
					template_exceed[i] = 0;
				sum_exceed = 0.0f;
				num_template_exceed = 0;

				/* --------------------------------------------------------------------
				 * The following blocks are optimized code that essentially
				 * perform the operations immediately below. The calculation of the
				 * template noise constants is done once the exponentials are complete
				 */

				/* template_profile = template_profile + test_peak - template_peak
				 * template = 10 ^ (template_profile / 10)
				 * template = template + test_noise - template_noise
				 * if (input < fp_epsilon) then clip the input to -100 dB
				 * template = log10( abs(template) )
				 * template_profile = 10 * template + test_noise_db */

				fptr = cur_tp;
				for (i = 0; i < profile_size; i++)
				{
					tmp1 = *fptr + noise_shift;
					*fptr = pow10fpm(tmp1 * 0.1f);
					fptr++;
				}

				/* Calculates noise levels from first and last elements of the current
				   template */

				template_noise = (cur_tp[0] + cur_tp[PROFILE_SIZE - 1]) * 0.5f;
				noise_shift2 = test_noise - template_noise;

				fptr = cur_tp;
				for (i = 0; i < profile_size; i++)
				{
					tmp1 = *fptr + noise_shift2;

					if (tmp1 == 0.0f)
						tmp1 = MIN_NOISE;

					*fptr = 10.0f * log10fpm( fabs(tmp1) ) + test_noise_db;

				  /* Because many of the operations in the search for the best shift
				   * amount depend on knowledge of which pixels in the template
				   * have values exceeding twice test_noise (recall that 3db is roughly
				   * equivalent to a doubling of raw power), we'll put those indices in
				   * template_exceed */

					if (*fptr > test_noise_db_plus_3)
					{
						template_exceed[i] = 1;
						num_template_exceed++;
						sum_exceed += *fptr;
					}
					fptr++;
				}
		//		test2[template_index] = sum_exceed / (float)(num_template_exceed);
		//		stop_measure_time(CPU);

				/* Note: The following block has 4 different branches:
				   1. Both the current template and the test pattern have values exceeded
					  twice test noise.
				   2. Only the current template has values exceeded twice test noise.
				   3. Only the test pattern has values exceeded twice test noise.
				   4. Neither the current template nor the test pattern has values
					  exceeded twice test noise.
				*/
				/* If there is at least one pixel in the template we're
				 * currently working on whose value exceeds twice test_noise */
				if (num_template_exceed)
				{
					template_exceed_mean = sum_exceed / (float)(num_template_exceed);
//					test2[template_index] = template_exceed_mean;
					fptr3 = test_exceed_means;

					for (current_shift=0; current_shift<shift_size; current_shift++,fptr3++)
					{
						/* Work on a copy of the template we're currently working on */
						memcpy ((void*)template_copy, (void*)cur_tp, patsize);

						/* If there is at least one pixel in the shifted test profile
						 * whose value exceeds twice test noise. */
						if (*fptr3 != 0.0f)
						{
							/* CASE 1 */
							/* Considering only those pixels whose powers exceed twice
							 * test noise, compute the difference of the mean power in
							 * template we're currently working on. */
							  power_ratio = *fptr3 - template_exceed_mean;

							  /* Scale template values that exceed twice test noise by power ratio and
							   * set the values that are less than test noise in db to test noise in db */
							  fptr  = template_copy;
							  bptr  = template_exceed;
							  for (i=0; i<profile_size; i++,fptr++)
							  {
								  if (*bptr++)
									  *fptr += power_ratio;

								  if (*fptr < test_noise_db)
									  *fptr = test_noise_db;
							  }
						} /* if (*fptr3 != 0.0f) */
						else
						{
							/* CASE 2 */
							/* Set those pixels in the template we're currently working on
							 * whose values are less than test_noise to test_noise. */
							fptr = cur_tp;
							for (i=0; i<profile_size; i++)
								if (*fptr++ < test_noise_db)
									template_copy[i] = test_noise_db;
						} /* else ... if (num_test_exceed) */

						/* Compute the weighted MSE */
						weighted_MSE = 0.0f;
						fptr  = all_shifted_test_db + current_shift;
						fptr2 = template_copy;
						for (i=0; i<PROFILE_SIZE; i++)
						{
							tmp1 = *fptr++ - *fptr2++;
							weighted_MSE += tmp1 * tmp1;
							CPU_weighted_MSEs[template_index * SHIFT_SIZE * PROFILE_SIZE + current_shift * PROFILE_SIZE + i] = tmp1 * tmp1;
						}

						/* ----------------------------------------------------------------
						 * MSE_scores[current_shift] = weighted_MSE / sumWeights */
						MSE_scores[current_shift] = weighted_MSE * sumWeights_inv;
					} /* for current_shift */
				}
				else /* if (num_template_exceed) */
				{
					fptr3 = test_exceed_means;

					for (current_shift=0; current_shift<shift_size; current_shift++)
					{
						/* CASE 3 */
						/* If there is at least one pixel that exceeds twice test noise */
						if (*fptr3++ != 0.0f)
						{
							fptr2 = cur_tp;
						}
						else
						{
							/* CASE 4 */
							/* Work on a copy of the template we're currently working on. */
							memcpy ((void*)template_copy, (void*)cur_tp, patsize);

							fptr = cur_tp;
							for (i=0; i<profile_size; i++)
								if (*fptr++ < test_noise_db)
									template_copy[i] = test_noise_db;

							fptr2 = template_copy;
						}

						/* Compute the weighted MSE */
						weighted_MSE = 0.0f;
						fptr  = all_shifted_test_db + current_shift;
						for (i=0; i<PROFILE_SIZE; i++)
						{
							tmp1 = *fptr++ - *fptr2++;
							weighted_MSE += tmp1 * tmp1;
							CPU_weighted_MSEs[template_index * SHIFT_SIZE * PROFILE_SIZE + current_shift * PROFILE_SIZE + i] = tmp1 * tmp1;
						}
						MSE_scores[current_shift] = weighted_MSE * sumWeights_inv;
					} /* for current_shift */
				} /* else .. if (num_template_exceed) */
			}
	}
	stop_measure_time(CPU);
/*  
*	The following belongs to the original pm code from HPEC but we have left it commented because we did not implement 
*		the corresponding kernel. If you are the one to write it, please send us an email :D
*     And may we take you for a dinner.
*/
//		/* Finding the minimum MSE for range shifting */
//		fptr = MSE_scores;
//		min_MSE_index = 0;
//		min_MSE = *fptr++;
//		for (i=1; i<shift_size; i++,fptr++)
//		{
//			if (min_MSE > *fptr)
//			{
//				min_MSE = *fptr;
//				min_MSE_index = i;
//			}
//		}

//    /* Work on a copy of the template we're currently working on. */
//    memcpy ((void*)template_copy, (void*)cur_tp, patsize);
//
//    mag_shift_scores_flag = 1;
//
//    if (test_exceed_means[min_MSE_index] != 0.0f)
//    {
//      if (num_template_exceed)
//      {
//        /* Compute the difference of the average shifted test profile
//         * power to the average template power */
//        /* ave_power_ratio = (sum_exceed / (float)(num_test_exceed)) - template_exceed_mean; */
//        ave_power_ratio = test_exceed_means[min_MSE_index] - template_exceed_mean;
//
//        /* Loop over all possible magnitude shifts */
//        for (j=0, mag_db=-5.0f; mag_db<=5.0f; mag_db+=0.5f)
//        {
//          power_shift = ave_power_ratio + mag_db;
//
//          /* --------------------------------------------------------------
//           * template_copy = template_profiles(template_exceed) + ave_power_ratio + mag_db */
//          bptr  = template_exceed;
//          for (i=0; i<profile_size; i++)
//          {
//            if (*bptr++)
//              template_copy[i] = cur_tp[i] + power_shift;
//          }
//
//          /* Compute the weighted MSE */
//          weighted_MSE = 0.0f;
//          fptr  = all_shifted_test_db + min_MSE_index;
//          fptr2 = template_copy;
//          for (i=0; i<profile_size; i++)
//          {
//            tmp1 = *fptr++ - *fptr2++;
//            weighted_MSE += tmp1 * tmp1;
//          }
//
//          mag_shift_scores[j++] = weighted_MSE * sumWeights_inv;
//
//        } /* for mag_db */
//      } /* if (num_template_exceed) */
//
//    }
//    else /* if (num_test_exceed) */
//    {
//      /* Set those pixels in the template we're currently working on
//       * whose values are less than test_noise to test_noise. */
//      fptr = cur_tp;
//      for (i=0; i<profile_size; i++)
//      {
//        if (*fptr++ < test_noise_db)
//          template_copy[i] = test_noise_db;
//      }
//
//      /* Compute the weighted MSE */
//      weighted_MSE = 0.0f;
//      fptr = all_shifted_test_db + min_MSE_index;
//      fptr2 = template_copy;
//      for (i=0; i<profile_size; i++)
//      {
//        tmp1 = *fptr++ - *fptr2++;
//        weighted_MSE += tmp1 * tmp1;
//      }
//
//      minimum_MSE_score[template_index] = weighted_MSE * sumWeights_inv;
//
//      mag_shift_scores_flag = 0;
//    } /* if (num_test_exceed) */
//
//    /* If magnitude shifting has performed above */
//    if (mag_shift_scores_flag)
//    {
//      /* Find the minimum MSE for magnitude scaling */
//      fptr = mag_shift_scores;
//      min_MSE = *fptr++;
//      for (i=1; i<21; i++,fptr++)
//        if (min_MSE > *fptr)
//          min_MSE = *fptr;
//
//      minimum_MSE_score[template_index] = min_MSE;
//    }
//
//  } /* for template_index */
//
//  /* Find the minimum mean square error */
//  fptr = minimum_MSE_score;
//  match_index = 0;
//  match_score = *fptr++;
//  for (i=1; i<num_templates; i++,fptr++)
//  {
//    if (match_score > *fptr)
//    {
//      match_score = *fptr;
//      match_index = i;
//    }
//  }
//  return match_index;
    return 0;
}

void print_result()
{
	fio = fopen("log.txt", "a+");
	fseek (fio, 0, SEEK_END);
	int appendPos = ftell(fio);
	char hostName[50];
	gethostname(hostName, 50);

	float total_GPU_time = timeRes[PLATFORM] + timeRes[DEVICE] + timeRes[CONTEXT] + timeRes[CMDQ] + timeRes[PGM] + timeRes[KERNEL] + timeRes[BUFF] + timeRes[WRDEV]
						   + timeRes[KERNEL1_EXEC] + timeRes[KERNEL2_EXEC] + timeRes[RDDEV] + timeRes[GPU_SEQ];
	float total_GPU_fair_time = timeRes[GPU_SEQ] + timeRes[KERNEL1_EXEC] + timeRes[KERNEL2_EXEC] + timeRes[WRDEV] + timeRes[RDDEV];

	struct tm *local;
	time_t t;
	t = time(NULL);
	local = localtime(&t);

	fprintf(fio, "****************************************************\n");
	fprintf(fio, "Created on: %s", asctime(local));
	fprintf(fio, "Host name: %s \n", hostName);
	fprintf(fio, "Description: %s \n\n", description);
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
			"	KERNEL1_EXEC = \t%10.2f msecs \n"
			"	KERNEL2_EXEC = \t%10.2f msecs \n"
			"	RDDEV = \t%10.2f msecs \n\n"
			"   GPU_SEQ = \t%10.2f msec \n\n",
			   timeRes[PLATFORM],
			   timeRes[DEVICE],
			   timeRes[CONTEXT],
			   timeRes[CMDQ],
			   timeRes[PGM],
			   timeRes[KERNEL],
			   timeRes[BUFF],
			   timeRes[WRDEV],
			   timeRes[KERNEL1_EXEC],
			   timeRes[KERNEL2_EXEC],
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

void clean()
{
	clean(&cpuPmdata);
	clean(&gpuPmdata);
	clean_mem(float, lib1);
	clean_mem(float, pattern1);
	clean_mem(float, lib2);
	clean_mem(float, pattern2);
	clean_mem(int,   patnum);
	clean_mem(float, rtime);
	oclClean();
}

int main(int argc, char **argv)
{
	pca_timer_t    	timer;
	char           	libfile[100], patfile[100], timefile[100], patnumfile[100], buff[256];
	int 			cpuResult, gpuResult;
	FILE*			fio;


 	if (argc != 2) {
 		printf("Usage: %s <data set num>\n", argv[0]);
 	return -1;
 	}

 	/* Build the input file names */
 	sprintf(libfile,    "./data/%s-pm-lib.dat",     argv[1]);
 	sprintf(patfile,    "./data/%s-pm-pattern.dat", argv[1]);
 	sprintf(patnumfile, "./data/%s-pm-patnum.dat",  argv[1]);
 	sprintf(timefile,   "./data/%s-pm-timing.dat",  argv[1]);

 	/* Read the template library and the test pattern from files */
    readFromFile(float, libfile, lib1);
    readFromFile(float, patfile, pattern1);

    readFromFile(float, libfile, lib2);
    readFromFile(float, patfile, pattern2);

	/* Allocate memory for internal arrays and output */
	pca_create_carray_1d(int,  patnum, 1, PCA_REAL);
	pca_create_carray_1d(float, rtime, 1, PCA_REAL);

	init(&gpuPmdata, &lib1, &pattern1);
	init(&cpuPmdata, &lib2, &pattern2);

	/* Run and time the pattern match kernel */
	gpuResult = pmGPU(&gpuPmdata);
	cpuResult = pmCPU(&cpuPmdata);

//	for (int i=0; i<TEMPLATE_SIZE; i++){
//		printf("test2[%i]=%.6f, test[%i]=%.6f \n", i, test2[i], i, test[i]);
//	}

//	for (int i=0; i<TEMPLATE_SIZE * SHIFT_SIZE * PROFILE_SIZE; i++)
//		if (test[i] != 0)
//			printf("test[%i] = %.6f \n", i, test[i]);
	for (int i=0; i<200; i++)
	{
		printf("CPUweighted[%i]=%.6f, GPUweighted[%i]=%.6f \n", i, CPU_weighted_MSEs[i], i, GPU_weighted_MSEs[i]);
		int fromend = TEMPLATE_SIZE * PROFILE_SIZE * SHIFT_SIZE - i;
		printf("CPUweighted[%i]=%.6f, GPUweighted[%i]=%.6f \n", fromend, CPU_weighted_MSEs[fromend],
				fromend, GPU_weighted_MSEs[fromend]);
	}

	print_result();
	clean();
	return 0;
}
