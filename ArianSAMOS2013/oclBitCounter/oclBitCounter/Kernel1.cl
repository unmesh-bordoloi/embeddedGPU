__kernel void BitCounter(__global const int * Src, __global int * Tmp)
{       
	int gid_x = get_global_id(0);
	int gid_y = get_global_id(1);
	int lid_x = get_local_id(0);
	int gw = get_global_size(0);
	int lw = get_local_size(0);
	
	int input;
	char bit1_cntr;	
	__local int values[32];
	
	input = Src[gid_y * gw + gid_x];
		
	bit1_cntr = 0;
	while(input != 0)
	{
		bit1_cntr++;
		input = input & (input - 1);
	}
	values[lid_x] = bit1_cntr;

	for(int stride = lw/2; stride > 0; stride >>= 1)
	{
		barrier(CLK_LOCAL_MEM_FENCE);
		if (lid_x < stride)
			values[lid_x] += values[lid_x + stride];
	}
	if (lid_x == 0) 
		Tmp[get_group_id(1) * (gw/lw) + get_group_id(0)] = values[0];	
}
