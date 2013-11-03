__kernel void SumNoLM(__global int * Src, __global int * Tmp, int numofElements)
{
	int gid_x = get_global_id(0);
	int gid_y = get_global_id(1);
	int lid_x = get_local_id(0);
	int gw = get_global_size(0);
	int lw = get_local_size(0);
    int grp_x = get_group_id(0);
    int grp_y = get_group_id(1);
    int ngrp_x = get_num_groups(0);
	
	int index = 2 * (gid_y * gw + gid_x) - lid_x;
	if (index >= numofElements)
		Src[index] = 0;
	if (index + get_local_size(0) >= numofElements)
		Src[index + lw] = 0;
	
	for(int stride = lw; stride > 0; stride >>= 1)
	{
		if (lid_x < stride)
			Src[index] += Src[index + stride];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (lid_x == 0) 
        Tmp[(grp_y * ngrp_x) + grp_x] = Src[index];
}

__kernel void SumLM(__global const int * Src, __global int * Tmp , int numofElements)
{
	int gid_x = get_global_id(0);
	int gid_y = get_global_id(1);
	int lid_x = get_local_id(0);
	int gw = get_global_size(0);
	int lw = get_local_size(0);
	
	__local int partialSum[64];
	
	int index = 2 * (gid_y * gw + gid_x) - lid_x;
	
	if (index < numofElements)
		partialSum[lid_x] = Src[index];
	else 
		partialSum[lid_x] = 0;
	if (index + lw < numofElements)
		partialSum[lid_x + lw] = Src[index + lw];  
	else 
		partialSum[lid_x + lw] = 0;	
	
	for(int stride = lw; stride > 0; stride >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid_x < stride)
			partialSum[lid_x] += partialSum[lid_x + stride];
	}
	if (lid_x == 0) 
		Tmp[get_group_id(1) * (gw/lw) + get_group_id(0)] = partialSum[0];
}
