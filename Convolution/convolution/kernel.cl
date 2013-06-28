//__kernel void convolution(__read_only image2d_t clSrcImage, __global char * clDstBuff, __constant int * filter, sampler_t sampler, int cols, int rows, int filterWidth)
__kernel void convolution(__read_only image2d_t clSrcImage, __write_only image2d_t clDstImage, __constant int * filter, sampler_t sampler, int cols, int rows, int filterWidth)
{
	int gid_x = get_global_id(0);
	int gid_y = get_global_id(1);
	
	uint4 pix;
	uint4 sum = {0, 0, 0, 0};
	int filterRadius = filterWidth >> 1;
	int2 coord;
	int filterIdx = 0;
	int weight = 0;

	for (int i=-filterRadius; i<=filterRadius; i++)
	{
		coord.y = gid_y + i;
		for (int j=-filterRadius; j<=filterRadius; j++)
		{
			coord.x = gid_x + j;
			pix = read_imageui(clSrcImage, sampler, coord);
			sum += pix * filter[filterIdx];
			weight += filter[filterIdx];
			filterIdx++;
		}
	}
/*	
	pix = read_imageui(clSrcImage, sampler, (int2)(gid_x-1, gid_y-1));
	sum += pix * filter[filterIdx];
	weight += filter[filterIdx++];
	pix = read_imageui(clSrcImage, sampler, (int2)(gid_x, gid_y-1));
	sum += pix * filter[filterIdx];
	weight += filter[filterIdx++];
	pix = read_imageui(clSrcImage, sampler, (int2)(gid_x+1, gid_y-1));
	sum += pix * filter[filterIdx];
	weight += filter[filterIdx++];
	
	pix = read_imageui(clSrcImage, sampler, (int2)(gid_x-1, gid_y));
	sum += pix * filter[filterIdx];
	weight += filter[filterIdx++];
	pix = read_imageui(clSrcImage, sampler, (int2)(gid_x, gid_y));
	sum += pix * filter[filterIdx];
	weight += filter[filterIdx++];
	pix = read_imageui(clSrcImage, sampler, (int2)(gid_x+1, gid_y));
	sum += pix * filter[filterIdx];
	weight += filter[filterIdx++];
	
	pix = read_imageui(clSrcImage, sampler, (int2)(gid_x-1, gid_y+1));
	sum += pix * filter[filterIdx];
	weight += filter[filterIdx++];
	pix = read_imageui(clSrcImage, sampler, (int2)(gid_x, gid_y+1));
	sum += pix * filter[filterIdx];
	weight += filter[filterIdx++];
	pix = read_imageui(clSrcImage, sampler, (int2)(gid_x+1, gid_y+1));
	sum += pix * filter[filterIdx];
	weight += filter[filterIdx++];
	*/
	sum = sum / weight;	
	if (gid_x < cols && gid_y < rows)
	{
		coord.x = gid_x;
		coord.y = gid_y;
		write_imageui(clDstImage, coord, sum);
	}
} 