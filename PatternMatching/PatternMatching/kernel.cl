#define SHIFT_SIZE		21
#define GLOBAL_SIZE_0	(32*1024)
#define PROFILE_SIZE	64
#define TEMPLATE_SIZE	72

__kernel void pm_part1(__global float *tmp_pf_db, __global float *inm_tmp_pf_db, __global char *tmp_exc , __global float *tmp_exc_mean,
					 __global float *noise_shift, float test_noise, float test_noise_db)
{
	int gid = get_global_id(0);
	int lid = get_local_id(0);
	int grp = get_group_id(0);
	int gw = get_global_size(0);
	int lw = get_local_size(0);
	int ngrp = get_num_groups(0);
	
	__local float tp1[64];
	__local float tp2[64];	
	
	tp1[lid] = tmp_pf_db[gid];
	
	/* convert db to power --> pow10fpm */
	float db_pwr_val = tmp_pf_db[gid] + noise_shift[grp];	
	float mul = db_pwr_val * 0.2302585093f;	/* mul = db_pwr_val * 0.1 * LOG10 */
  	float term = mul;
	float ans = 1.0f;

 	ans += mul;						mul *= term;
 	ans += 0.5f * mul; 				mul *= term;
  	ans += 0.166666667f * mul; 		mul *= term;
	ans += 0.041666666f * mul; 		mul *= term;
	ans += 8.333333333e-3f * mul; 	mul *= term;
  	ans += 1.388888889e-3f * mul; 	mul *= term;
	ans += 1.984126984e-4f * mul; 	mul *= term;
  	ans += 2.480158730e-5f * mul; 	mul *= term;
  	ans += 2.755731922e-6f * mul; 	mul *= term;
  	ans += 2.755731922e-7f * mul; 	mul *= term;
  	ans += 2.505210839e-8f * mul; 	mul *= term;
  	ans += 2.087675699e-9f * mul; 	mul *= term;
  	ans += 1.605904384e-10f * mul;	mul *= term;
  	ans += 1.147074560e-11f * mul; 	mul *= term;
  	ans += 7.647163732e-13f * mul; 	mul *= term;
  	ans += 4.779477332e-14f * mul; 	mul *= term;
  	ans += 2.811457254e-15f * mul; 	mul *= term;
  	ans += 1.561920697e-16f * mul; 	mul *= term;
  	ans += 8.220635247e-18f * mul; 	mul *= term;
  	ans += 4.110317623e-19f * mul;
  	
  	tp1[lid] = ans;
  	
  	barrier(CLK_LOCAL_MEM_FENCE);
  	if (lid == 0)
  	{
  		float tmp_noise = (tp1[0] + tp1[PROFILE_SIZE - 1]) * 0.5f;		/* template_noise = (tp1[0] + tp[profile_size - 1]) * 0.5f */
  		tp2[0] = test_noise - tmp_noise;	/* noise_shift2 = test_noise - template_noise */
  	}
  	barrier(CLK_LOCAL_MEM_FENCE);	/* note: is it necessary */
  	
  	/* convert power back to db --> log10fpm */
	db_pwr_val = tp1[lid] + tp2[0];		/* db_pwr_val = tmp_val + noise_shift2 */
	if (db_pwr_val == 0.0f)
		db_pwr_val = 1e-10f;		/* 1e-10 is MIN_NOISE */
	
	db_pwr_val = fabs(db_pwr_val);	
	mul = (db_pwr_val - 1.0f) / (db_pwr_val + 1.0f);
  	term = mul * mul;
  	ans = 0.0f;
	
  	ans  = mul;           			mul *= term;
  	ans += 0.333333333f * mul; 		mul *= term;
  	ans += 0.2f * mul; 				mul *= term;
  	ans += 0.142857143f * mul; 		mul *= term;
  	ans += 0.111111111f * mul; 		mul *= term;
  	ans += 9.090909091e-2f * mul; 	mul *= term;
  	ans += 7.692307692e-2f * mul; 	mul *= term;
  	ans += 6.666666667e-2f * mul; 	mul *= term;
  	ans += 5.882352941e-2f * mul; 	mul *= term;
  	ans += 5.263157895e-2f * mul;	mul *= term;
  	ans += 4.761904762e-2f * mul; 	mul *= term;
  	ans += 4.347826087e-2f * mul; 	mul *= term;
  	ans += 0.04f * mul; 			mul *= term;
  	ans += 3.703703704e-2f * mul; 	mul *= term;
  	ans += 3.448275862e-2f * mul; 	mul *= term;
  	ans += 3.225806452e-2f * mul; 	mul *= term;
  	ans += 3.030303030e-2f * mul;
  	ans *= 0.86858896381;  /* ans2 = ans * 2 / log(10) */
  	
  	float res = 10.0f * ans + test_noise_db;
  	inm_tmp_pf_db[gid] = res;		/* store template */
  	
  	if (res > (test_noise_db + 3))
  	{
  		tmp_exc[gid] = 1;
  		tp1[lid] = res;
  		tp2[lid] = 1;
  	}
  	else
  	{
  		tmp_exc[gid] = 0;
  		tp1[lid] = 0;
  		tp2[lid] = 0;
  	}
  	
	for(int stride = lw >> 1; stride > 0; stride >>= 1)
  	{
  		barrier(CLK_LOCAL_MEM_FENCE);
  		if (lid < stride)
  		{
  			tp1[lid] += tp1[lid + stride];
  			tp2[lid] += tp2[lid + stride];
  		}
  	}
  	if (lid == 0)
  	{
  		if (tp2[0] != 0)
  			tmp_exc_mean[grp] = tp1[0] / tp2[0];
  		else 
  			tmp_exc_mean[grp] = 0; 
  	}		 
}

__kernel void pm_part2(__global float *tmp_pf_db, __global float *weighted_MSEs, __global char *tmp_exc, __global float *tmp_exc_mean,
						__global float *test_pf_db, __global float *test_exc_means, float test_noise_db, __global float *test)
{
	int grp_id = get_num_groups(0) * get_global_id(1) + get_group_id(0);
	int glb_id = get_global_id(1) * GLOBAL_SIZE_0 + get_global_id(0);
	int lid = get_local_id(0);

	char cur_tmp = grp_id / 21;
	char cur_shf = grp_id % SHIFT_SIZE;
	int tmp_ntt_id = cur_tmp * PROFILE_SIZE + lid;
	
	float tmp_val = tmp_pf_db[tmp_ntt_id];
	float tem = test_exc_means[cur_shf];
	float tmp1;
	float pwr_ratio = 0.0f;

	if (glb_id < TEMPLATE_SIZE * SHIFT_SIZE * PROFILE_SIZE)
	{
		if (tmp_exc_mean[cur_tmp] > 0)
		{
			if (tem != 0.0f)
			{
				// CASE 1 
				pwr_ratio = tem - tmp_exc_mean[cur_tmp];		//only one work item 
				test[glb_id] = pwr_ratio; 		
				if (tmp_exc[tmp_ntt_id])
					tmp_val += pwr_ratio;
				
			}
			// CASE 2 
			if (tmp_val < test_noise_db)
				tmp_val = test_noise_db;
		} 
		else	// CASE 3: do nothing 
		{
			// CASE 4 
			if (tem == 0.0f)
				if (tmp_val < test_noise_db)
					tmp_val = test_noise_db;
		}
		if (lid + cur_shf < 11 || lid + cur_shf > 74) 
			tmp1 = test_noise_db - tmp_val;
		else
			tmp1 = test_pf_db[lid + cur_shf - 11] - tmp_val;
		weighted_MSEs[glb_id] = tmp1 * tmp1;
	}  
}




