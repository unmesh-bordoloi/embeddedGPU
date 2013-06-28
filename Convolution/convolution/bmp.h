/*
 * bmp.h
 *
 *  Created on: Sep 10, 2012
 *      Author: arishop
 */

#ifndef BMP_H_
#define BMP_H_

#include <stdint.h>

#define BMP_HEADER_SIZE		14
#define DIB_HEADER_SIZE		40
#define SIGNATURE		0x4D42

#pragma pack(2)
struct bmp_header
{
	int16_t signature;
	int32_t file_size;
	int16_t reserved1;
	int16_t reserved2;
	int32_t offset;
};

struct dib_header
{
	int32_t header_size;
	int32_t width;
	int32_t height;
	int16_t planes;
	int16_t bpp;
	int32_t compression;
	int32_t image_size;
	int32_t xppm;
	int32_t yppm;
	int32_t colors;
	int32_t important_colors;
};
#pragma pack()

#endif /* BMP_H_ */
