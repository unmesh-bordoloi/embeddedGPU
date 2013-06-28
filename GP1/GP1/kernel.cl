#define MAX_IND_LEN  	200
#define WORK_GROUP_SIZE	32
#define TRAIN_SIZE		128
#define NUM_CONST		20
#define MAX_DEPTH		10

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

__kernel void fitness(__global int *pop, __global int *ind_lens, __global float4 *eval,
			 		__global float4 *trainIn, __global float *constants) 
{	
	//int gid = get_global_id(0);
	//int lid = get_local_id(0);
	//int ls = get_local_size(0);
	//int grp = get_group_id(0);	// which is the individual number
		
	char q = -1;
	int p = ind_lens[get_group_id(0)] - 1;
	int offset = get_group_id(0) * (MAX_IND_LEN / 4);
	float4 vals_stack[3*MAX_DEPTH];
	int node_int;
	char node;
	
	while(p != -1)
	{
		node_int = pop[offset + p];
		p--;
		for (int i=3; i>=0; i--)
		{
			node = (node_int >> (8 * i)) & 0xff;
			if (node < FUNC_START && node >= X)	// node is a terminal
				vals_stack[++q] = (node <= Y ? trainIn[(node * TRAIN_SIZE / 4) + get_local_id(0)] : constants[node - 2]);	// put args into the stack
			else
			{			
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
						/*for (int i=0; i<4; i++)
						{
							if (vals_stack[q-1].s0 < 0.001)
								vals_stack[q-1].s0 = vals_stack[q].s0;
							else
								vals_stack[q-1].s0 = vals_stack[q].s0 / vals_stack[q-1].s0;
							vals_stack[q].s0123 = vals_stack[q].s1230;
							vals_stack[q-1].s0123 = vals_stack[q-1].s1230;
						}
						*/
						if (vals_stack[q-1].x < 0.001)
							vals_stack[q-1].x = vals_stack[q].x;
						else
							vals_stack[q-1].x = vals_stack[q].x / vals_stack[q-1].x;
							
						if (vals_stack[q-1].y < 0.001)
							vals_stack[q-1].y = vals_stack[q].y;
						else
							vals_stack[q-1].y = vals_stack[q].y / vals_stack[q-1].y;
							
						if (vals_stack[q-1].w < 0.001)
							vals_stack[q-1].w = vals_stack[q].w;
						else
							vals_stack[q-1].w = vals_stack[q].w / vals_stack[q-1].w;
							
						if (vals_stack[q-1].z < 0.001)
							vals_stack[q-1].z = vals_stack[q].z;
						else
							vals_stack[q-1].z = vals_stack[q].z / vals_stack[q-1].z;   
						q--;
						break;
					case IFLTE:
						if (vals_stack[q].x <= vals_stack[q-1].x)
							vals_stack[q-3].x = vals_stack[q-2].x;
							
						if (vals_stack[q].y <= vals_stack[q-1].y)
							vals_stack[q-3].y = vals_stack[q-2].y;
							
						if (vals_stack[q].z <= vals_stack[q-1].z)
							vals_stack[q-3].z = vals_stack[q-2].z;
							
						if (vals_stack[q].w <= vals_stack[q-1].w)
							vals_stack[q-3].w = vals_stack[q-2].w;
						q -= 3;
						break;
					case COS:
						vals_stack[q] = native_cos(vals_stack[q]);
						break;
					case SIN:
						vals_stack[q] = native_sin(vals_stack[q]);
						break;
					default:
						break;
				}
			} 
		}
	}
	eval[get_global_id(0)] = vals_stack[0];
}