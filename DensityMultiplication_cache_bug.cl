#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel
__attribute__((reqd_work_group_size(128, 1, 1)))
void multdensity(__global const int *starting_points,__global const double *alpha, __global double *result,const double lambda, const int startid)
{
    int gridindex = startid + get_global_id(0);
    __private int local_id = get_local_id(0);
    __private int point_indices_block0[4];
    __private int point_level_block0[4];
    point_indices_block0[0] = starting_points[(gridindex * 1 + 0) * 4 * 2 + 2 * 0];
    point_level_block0[0] = starting_points[(gridindex * 1 + 0) * 4 * 2 + 2 * 0 + 1];
    point_indices_block0[1] = starting_points[(gridindex * 1 + 0) * 4 * 2 + 2 * 1];
    point_level_block0[1] = starting_points[(gridindex * 1 + 0) * 4 * 2 + 2 * 1 + 1];
    point_indices_block0[2] = starting_points[(gridindex * 1 + 0) * 4 * 2 + 2 * 2];
    point_level_block0[2] = starting_points[(gridindex * 1 + 0) * 4 * 2 + 2 * 2 + 1];
    point_indices_block0[3] = starting_points[(gridindex * 1 + 0) * 4 * 2 + 2 * 3];
    point_level_block0[3] = starting_points[(gridindex * 1 + 0) * 4 * 2 + 2 * 3 + 1];
    __private int teiler = 0;
    __private double h = 1.0 / 3.0;
    __private double umid = 0.0;
    __private double sum = 0.0;
    __private int u= 0;
    double gesamtint_block0 = 0.0;
    __local int indices_local[128];
    __local int level_local[128];
    __local double alpha_local[32];
    for (int group = 0; group < 28; group++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (get_local_id(0) < 32) { // reduced: ok
        for (int j = 0; j <     4 ; j++) {
            indices_local[local_id * 4 + j] = starting_points[group * 1024  + local_id * 8 + 2 * j]; //1024=2*512 = 2*dim*128 -> incorrect, dim=4 -> should be 2*dim*32
            level_local[local_id * 4 + j] = starting_points[group * 1024  + local_id * 8 + 2 * j + 1]; //1024=2*512 = 2*dim*128 -> incorrect, dim=4 -> should be 2*dim*32
        }
        alpha_local[local_id] = alpha[group * 128  + local_id ]; //128 -> incorrect, dim=4 -> should be 32
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0 ; i < 32; i++) {
            __private double zellenintegral = 1.0;
            for(private int dim = 0;dim< 4;dim++) {
                h = 1.0 / (1 << level_local[i* 4+dim]);
                u = (1 << point_level_block0[dim]);
                umid = u * h * (indices_local[i* 4+dim]) - point_indices_block0[dim];
                umid = fabs(umid);
                umid = 1.0-umid;
                umid = (umid + fabs(umid));
                sum = h*(umid);
                h = 1.0 / (1 << point_level_block0[dim]);
                u = (1 << level_local[i* 4+dim]);
                umid = u * h * (point_indices_block0[dim]) - indices_local[i* 4+dim];
                umid = fabs(umid);
                umid = 1.0-umid;
                umid = (umid + fabs(umid));
                sum += h*(umid);
                sum *= point_level_block0[dim] == level_local[i* 4+dim] ? 1.0/3.0 : 1.0;
                zellenintegral*=sum;
            }
            gesamtint_block0 += zellenintegral*alpha_local[i];

        }
    }
    result[get_global_id(0) * 1 + 0] = gesamtint_block0 / 16;
    result[get_global_id(0) * 1 + 0] += alpha[gridindex * 1 + 0]*lambda;
}
