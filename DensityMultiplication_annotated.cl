#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void multdensity(
    __global const int *hs_inverses, __global const double *hs, __global const double *positions,
    __global const double *alpha, __global double *result, const double lambda, const int startid) {
  int gridindex = startid + get_global_id(0);
  __private int local_id = get_local_id(0);
  __private double point_positions_block0[2];
  __private double point_hs_block0[2];
  __private double point_hinverses_block0[2];
  point_hinverses_block0[0] = hs_inverses[gridindex * 2 + 0];  // load grid point informatino
  point_hs_block0[0] = hs[gridindex * 2 + 0];
  point_positions_block0[0] = positions[gridindex * 2 + 0];
  point_hinverses_block0[1] = hs_inverses[gridindex * 2 + 1];
  point_hs_block0[1] = hs[gridindex * 2 + 1];
  point_positions_block0[1] = positions[gridindex * 2 + 1];
  __private int teiler = 0;
  __private double h = 1.0 / 3.0;
  __private double umid = 0.0;
  __private double sum = 0.0;
  __private int u = 0;
  double gesamtint_block0 = 0.0;
  __local double positions_local[256];
  __local double hs_local[256];
  __local int hinverses_local[256];
  __local double alpha_local[128];
  for (int group = 0; group < 1; group++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int j = 0; j < 2; j++) {
      hinverses_local[local_id * 2 + j] = hs_inverses[group * 256 + local_id * 2 + j];
      hs_local[local_id * 2 + j] = hs[group * 256 + local_id * 2 + j];
      positions_local[local_id * 2 + j] = positions[group * 256 + local_id * 2 + j];
    }
    alpha_local[local_id] = alpha[group * 128 + local_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < 128; i++) {
      __private double zellenintegral = 1.0;
      for (private int dim = 0; dim < 2; dim++) { // * d -> 14 *d + 2
        __private double distance =
            fabs(point_positions_block0[dim] - positions_local[i * 2 + dim]);  // 2
        sum = 1.0 - distance * point_hinverses_block0[dim];                    // 2
        sum *= hs_local[i * 2 + dim];                                          // 1
        sum = max(sum, (double)0.0);                                           // 1
        sum += max((double)(point_hs_block0[dim] * (1.0 - hinverses_local[i * 2 + dim] * distance)), (double)0.0);  // 5
        zellenintegral *=
            sum * select((double)1.0, (double)h,
                         (long)(point_hinverses_block0[dim] == hinverses_local[i * 2 + dim]));  // 3
      }
      gesamtint_block0 += zellenintegral * alpha_local[i];  // 2
    }
  }
  result[get_global_id(0) * 1 + 0] = gesamtint_block0;
  result[get_global_id(0) * 1 + 0] += alpha[gridindex * 1 + 0] * lambda; // 2
}
