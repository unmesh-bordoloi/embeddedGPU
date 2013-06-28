/*
 * Convolution.cpp
 *
 * Created on: Sep 3, 2012
 * Author: Arian Maghazeh
 * Edited by : Unmesh D. Bordoloi
 * Implements the convolution algorithm
 * Performance results reported in SAMOS 2013 paper
 *
 * Check the log.txt that is generated for the performance results
 * You may also put a break point to see the statements printed on the screen
 *
 * Bug reports and fixes are truly welcome at unmesh.bordoloi@liu.se but has no guarantee of a reply :D
 */
#include <stdio.h>
#include <CL/cl.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "bmp.h"

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

// #define FERMI
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

#define BW				8
#define BH				8

#define WARP_SIZE		16

cl_program 			clProgram;
cl_device_id 		clDeviceId;
cl_context 			clContext;
cl_mem 				clSrcImage;
cl_mem 				clDstImage;
cl_mem				clDstBuff;
cl_mem				clFilterBuff;
cl_mem				clTmpBuff;
cl_sampler			clSampler;
cl_kernel 			clKernel;
cl_command_queue 	clCommandQueue;
cl_int 				clErr;
cl_event 			clEvent;
#ifdef FERMI
cl_platform_id 		clPlatformId[3];
#else
cl_platform_id		clPlatformId;
#endif

FILE *fp;
FILE *fio;
char buff[256];
char *kernelSrc;
char description[256] = "Convolution, using image";
int width;
int height;
const int filterWidth = 3;
int filter[filterWidth * filterWidth] = {0, -1, 0,
										 -1, 5, -1,
										  0, -1, 0};
int gpuResult = 0;
int cpuResult = 0;
char *palette;
char *srcImg;
//int srcImg[512][512*3];
//char **srcImg;
//char **dstImg;
//char dstImg[512][512*3];
char *gpuDstImg;
char *cpuDstImg;
bmp_header bmp;
dib_header dib;
int palette_size;
size_t origin[3] = {0, 0, 0};
size_t region[3];
int lineSize;

struct timeval start[15];
struct timeval stop[15];
float timeRes[15] = {0};

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
	clKernel = clCreateKernel(clProgram, "convolution", &clErr);
	if (clErr != CL_SUCCESS)
			printf("Error in creating kernel!, clErr=%i \n", clErr);
	else printf("Kernel created! \n");
	fclose(fp);
	free(kernelSrc);
	stop_measure_time(KERNEL);
}

void oclBuffer()
{
	/*-----------------------create buffer------------------------*/
	start_measure_time(BUFF);
	cl_image_format format;
	format.image_channel_order = CL_RGBA;
	format.image_channel_data_type = CL_UNSIGNED_INT8;

	clSrcImage = clCreateImage2D(clContext, 0, &format, width, height, 0, NULL, &clErr);
	clDstImage = clCreateImage2D(clContext, 0, &format, width, height, 0, NULL, &clErr);
	clFilterBuff = clCreateBuffer(clContext, 0, sizeof(int) * filterWidth * filterWidth, NULL, &clErr);
	clSampler = clCreateSampler(clContext, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, NULL);
	stop_measure_time(BUFF);

	/*-----------------------write into device--------------------*/
	start_measure_time(WRDEV);
	region[0] = width;
	region[1] = height;
	region[2] = 1;

	clErr = clEnqueueWriteImage(clCommandQueue, clSrcImage, CL_TRUE, origin, region, 0, 0, srcImg, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
		printf("Error in writing image!, clErr=%i \n", clErr);
	clEnqueueWriteBuffer(clCommandQueue, clFilterBuff, CL_TRUE, 0, sizeof(int) * filterWidth * filterWidth, filter, 0, NULL, NULL);
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
	clReleaseMemObject(clSrcImage);
	clReleaseMemObject(clDstImage);
	clReleaseSampler(clSampler);
	clReleaseKernel(clKernel);
	clReleaseMemObject(clDstBuff);
}

int round_up(int value, int multiple)
{
	int ret = (value % multiple == 0) ? value : value + multiple - (value % multiple);
	return ret;
}

char * read_bmp(const char *file, bmp_header *bmp, dib_header *dib, char **palette)
{
	FILE *fimg;
	fimg = fopen(file, "rb");
	if (fimg == NULL)
	{
		printf("Image could not be opened! \n");
		exit(1);
	}
	if (fread(bmp, 1, BMP_HEADER_SIZE, fimg) < BMP_HEADER_SIZE)
	{
		printf("Error in reading bmp header! \n");
		exit(1);
	}
	if (fread(dib, 1, DIB_HEADER_SIZE, fimg) < DIB_HEADER_SIZE)
	{
		printf("Error in reading dib header! \n");
		exit(1);
	}
	palette_size = bmp->offset - (BMP_HEADER_SIZE + DIB_HEADER_SIZE);
	if (palette_size > 0)
	{
		*palette = (char *)malloc(sizeof(char) * palette_size);
		if (fread(palette, 1, palette_size, fimg) < palette_size)
		{
			printf("Error in reading palette! \n");
			exit(1);
		}
	}
	lineSize = dib->width * (dib->bpp / 8);
	char * tmp = (char *)malloc(sizeof(char) * dib->height * lineSize);
	char * image = (char *)malloc(sizeof(char) * dib->height * dib->width * 4);
	if (fread(tmp, 1, dib->image_size, fimg) < dib->image_size)
	{
		printf("Error in reading srcImg! \n");
		exit(1);
	}
	int idx1 = 0;
	int idx2 = 0;
	while (idx1 < dib->height * dib->width * 4)
	{
		image[idx1] = tmp[idx2];
		image[idx1+1] = tmp[idx2+1];
		image[idx1+2] = tmp[idx2+2];
		image[idx1+3] = 0;
		idx1 += 4;
		idx2 += 3;
	}
	fclose(fimg);
	return image;
}

void write_bmp(const char *file, bmp_header *bmp, dib_header *dib, char *palette, char *image_padded)
{
	FILE *fout;
	fout = fopen(file, "w");
	if (fout == NULL)
	{
		printf("Image could not be opened! \n");
		exit(1);
	}
	if (fwrite(bmp, 1, BMP_HEADER_SIZE, fout) < BMP_HEADER_SIZE)
	{
		printf("Error in writing bmp header! \n");
		exit(1);
	}
	if (fwrite(dib, 1, DIB_HEADER_SIZE, fout) < DIB_HEADER_SIZE)
	{
		printf("Error in writing dib header! \n");
		exit(1);
	}
	if (palette)
		fwrite(palette, 1, palette_size, fout);

	char *image = (char *)malloc(sizeof(char) * dib->height * lineSize);
	int idx1 = 0;
	int idx2 = 0;
	while (idx1 < dib->height * dib->width * 4)
	{
		image[idx2] = image_padded[idx1];
		image[idx2+1] = image_padded[idx1+1];
		image[idx2+2] = image_padded[idx1+2];
		idx1 += 4;
		idx2 += 3;
	}
	if (fwrite(image, 1, dib->image_size, fout) < dib->image_size)
	{
		printf("Error in writing dstImg! \n");
		exit(1);
	}
	fclose(fout);
}

struct pixel
{
	unsigned char R;
	unsigned char G;
	unsigned char B;
	unsigned char A;
};

char * make_image(pixel *p, int numofPixels)
{
	int idx = 0;
	char * imageBytes = (char *)malloc(sizeof(pixel)*numofPixels);
	for (int i=0; i<numofPixels; i++)
	{
		imageBytes[idx] = p[i].R;
		imageBytes[idx+1] = p[i].G;
		imageBytes[idx+2] = p[i].B;
		imageBytes[idx+3] = p[i].A;
		idx += 4;
	}
	return imageBytes;
}

int main(void)
{
	char hostName[50];
	gethostname(hostName, 50);

	srcImg = read_bmp("disney.bmp", &bmp, &dib, &palette);
	width = round_up(dib.width, BW);
	height = round_up(dib.height, BH);

	oclInit();
	oclBuffer();

//	size_t ws, ls;
//	clGetKernelWorkGroupInfo(clKernel, clDeviceId, CL_KERNEL_WORK_GROUP_SIZE, sizeof(ws), (void *) &ws, NULL);
//	printf("CL_KERNEL_WORK_GROUP_SIZE is: %i \n", ws);
//	clGetKernelWorkGroupInfo(clKernel, clDeviceId, CL_KERNEL_LOCAL_MEM_SIZE , sizeof(ls), (void *) &ls, NULL);
//	printf("CL_KERNEL_LOCAL_MEM_SIZE is: %i \n", ls);

	/*-----------------------dispatch kernel----------------------*/
	start_measure_time(KERNEL_EXEC);
	size_t clGlobalSize[2] = {width, height};
	size_t clLocalSize[2] = {BW, BH};

	clSetKernelArg(clKernel, 0, sizeof(cl_mem), &clSrcImage);
	clSetKernelArg(clKernel, 1, sizeof(cl_mem), &clDstImage);
	clSetKernelArg(clKernel, 2, sizeof(cl_mem), &clFilterBuff);
	clSetKernelArg(clKernel, 3, sizeof(cl_sampler), &clSampler);
	clSetKernelArg(clKernel, 4, sizeof(int), &width);
	clSetKernelArg(clKernel, 5, sizeof(int), &height);
	clSetKernelArg(clKernel, 6, sizeof(int), &filterWidth);

//	while(1)
//	{
	clErr = clEnqueueNDRangeKernel(clCommandQueue, clKernel, 2, 0, clGlobalSize, clLocalSize, 0, NULL, &clEvent);
	if (clErr != CL_SUCCESS)
		printf("Error in executing kernel!, clErr=%i \n", clErr);
	clFinish(clCommandQueue);
//	}
	stop_measure_time(KERNEL_EXEC);

	cl_ulong queued_time = (cl_ulong)0;
	cl_ulong submitted_time   = (cl_ulong)0;
	cl_ulong start_time = (cl_ulong)0;
	cl_ulong end_time   = (cl_ulong)0;
	size_t return_bytes;

	clErr = clGetEventProfilingInfo(clEvent, CL_PROFILING_COMMAND_QUEUED, sizeof(cl_ulong), &queued_time, &return_bytes);
	clErr = clGetEventProfilingInfo(clEvent, CL_PROFILING_COMMAND_SUBMIT, sizeof(cl_ulong), &submitted_time, &return_bytes);
	clErr = clGetEventProfilingInfo(clEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, &return_bytes);
	clErr = clGetEventProfilingInfo(clEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, &return_bytes);

	printf("Time from queue to submit : %10.3f msecs \n", ((double)(submitted_time-queued_time) * 1.0e-6));
	printf("Time from submit to start : %10.3f msecs \n", ((double)(start_time-submitted_time) * 1.0e-6));
	printf("Time from start to end    : %10.3f msecs \n", ((double)(end_time-start_time) * 1.0e-6));
	printf("Total                     : %10.3f msecs \n", ((double)(end_time-submitted_time) * 1.0e-6));

	/*-----------------------read from device---------------------*/
	start_measure_time(RDDEV);
	gpuDstImg = (char *)malloc(sizeof(char *) * dib.height * dib.width * 4);
	clErr = clEnqueueReadImage(clCommandQueue, clDstImage, CL_TRUE, origin, region, 0, 0, gpuDstImg, 0, NULL, NULL);
	if (clErr != CL_SUCCESS)
		printf("Error in reading image!, clErr=%i \n", clErr);
	clFinish(clCommandQueue);
	stop_measure_time(RDDEV);

	/*-----------------------create final image-------------------*/
	write_bmp("gpuResult.bmp", &bmp, &dib, palette, gpuDstImg);
	free(gpuDstImg);

	/*---------------------convolution on cpu---------------------*/
	pixel * pixels = (pixel *)malloc(sizeof(pixel)*dib.height*dib.width);
	int idx = 0;
	int weight = 0;
	pixel currPix;
	int filterIdx = 0;
	int filterRadious = filterWidth >> 1;

	for (int i=0; i<dib.height*dib.width; i++)
	{
		pixels[i].R = srcImg[idx];
		pixels[i].G = srcImg[idx+1];
		pixels[i].B = srcImg[idx+2];
		pixels[i].A = srcImg[idx+3];
		idx += 4;
	}
	pixel * dstPixels = (pixel *)malloc(sizeof(pixel)*dib.width*dib.height);

	start_measure_time(CPU);
	pixel sum;
	idx = 0;
	int R; int G; int B; int A;

//	while(1)
	{
	omp_set_num_threads(NUM_CORES);
	#pragma omp parallel default(none) shared(filter, pixels, dib, dstPixels) private(idx, sum, currPix, filterIdx, R, G, B, A, weight, filterRadious)
	{
		#pragma omp for
			for (int i=0; i<dib.height; i++)		//for border accesses
			{
				filterRadious = 1;
				for (int j=0; j<dib.width; j++)
				{
					sum.A = 0; sum.B = 0; sum.G = 0; sum.R = 0;
					R = 0; G = 0; B = 0; A = 0;
					weight = 0;
					filterIdx = 0;
					for (int y=-filterRadious; y<=filterRadious; y++)
					{
						for (int x=-filterRadious; x<=filterRadious; x++)
						{
							if (i+y < 0)
								idx = 0;
							else if (i+y >= dib.height)
								idx = (dib.height - 1)*dib.width;
							else
								idx = (i+y)*dib.width;

							if (j+x < 0)
								idx += 0;
							else if (j+x >= dib.width)
								idx += dib.width-1;
							else
								idx += j+x;

							currPix = pixels[idx];
							R += (currPix.R * filter[filterIdx]);
							G += (currPix.G * filter[filterIdx]);
							B += (currPix.B * filter[filterIdx]);
							A += (currPix.A * filter[filterIdx]);
							weight += filter[filterIdx];
							filterIdx++;
						}
					}
					sum.R = (unsigned char)(R / weight);
					sum.G = (unsigned char)(G / weight);
					sum.B = (unsigned char)(B / weight);
					sum.A = (unsigned char)(A / weight);

					dstPixels[i*dib.width + j] = sum;
				}
			}
	}
	}
	char * cpuDstImg = (char *)malloc(sizeof(char *) * dib.height * dib.width * 4);
	cpuDstImg = make_image(dstPixels, dib.height*dib.width);
	stop_measure_time(CPU);

	write_bmp("cpuResult.bmp", &bmp, &dib, palette, cpuDstImg);

	free(dstPixels);
	free(cpuDstImg);
//	}
	free(srcImg);
	/*-------------------------print result-----------------------*/
	fio = fopen("log.txt", "a+");
	fseek (fio, 0, SEEK_END);
	int appendPos = ftell(fio);

	float total_GPU_time = timeRes[PLATFORM] + timeRes[DEVICE] + timeRes[CONTEXT] + timeRes[CMDQ] + timeRes[PGM] + timeRes[KERNEL] + timeRes[BUFF] + timeRes[WRDEV] + timeRes[KERNEL_EXEC] + timeRes[RDDEV];
	float total_GPU_fair_time = timeRes[KERNEL_EXEC] + timeRes[WRDEV] + timeRes[RDDEV];

	struct tm *local;
	time_t t;
	t = time(NULL);
	local = localtime(&t);

	fprintf(fio, "****************************************************\n");
	fprintf(fio, "Created on: %s", asctime(local));
	fprintf(fio, "Host name: %s \n", hostName);
	fprintf(fio, "Description: %s \n\n", description);
	fprintf(fio, "Result GPU is: %i \n", gpuResult);
	fprintf(fio, "Result CPU is: %i \n", cpuResult);
	if (cpuResult != gpuResult)
	{
		fprintf(fio, "ERROR Results do not match!!!!\n");
		printf("ERROR Results do not match!!!!\n");
		exit(1);
	}

	fprintf(fio, "\n===========Performance Measurements=================\n");
	fprintf(fio, "Global size: %i * %i \n", width, height);
	fprintf(fio, "Local size: %i * %i \n", BW, BH);
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
			   "	RDDEV = \t%10.2f msecs \n\n",
			   timeRes[PLATFORM],
			   timeRes[DEVICE],
			   timeRes[CONTEXT],
			   timeRes[CMDQ],
			   timeRes[PGM],
			   timeRes[KERNEL],
			   timeRes[BUFF],
			   timeRes[WRDEV],
			   timeRes[KERNEL_EXEC],
			   timeRes[RDDEV]
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
}


