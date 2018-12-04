__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void multdensity(
    __global const int *hs_inverses, __global const float *hs, __global const float *positions,
    __global const float *alpha, __global float *result, const float lambda, const int startid) {
  int gridindex = startid + get_global_id(0);
  __private int local_id = get_local_id(0);
  __private float point_positions_block0[10];
  __private float point_hs_block0[10];
  __private float point_hinverses_block0[10];
  point_hinverses_block0[0] = hs_inverses[gridindex * 10 + 0];
  point_hs_block0[0] = hs[gridindex * 10 + 0];
  point_positions_block0[0] = positions[gridindex * 10 + 0];
  point_hinverses_block0[1] = hs_inverses[gridindex * 10 + 1];
  point_hs_block0[1] = hs[gridindex * 10 + 1];
  point_positions_block0[1] = positions[gridindex * 10 + 1];
  point_hinverses_block0[2] = hs_inverses[gridindex * 10 + 2];
  point_hs_block0[2] = hs[gridindex * 10 + 2];
  point_positions_block0[2] = positions[gridindex * 10 + 2];
  point_hinverses_block0[3] = hs_inverses[gridindex * 10 + 3];
  point_hs_block0[3] = hs[gridindex * 10 + 3];
  point_positions_block0[3] = positions[gridindex * 10 + 3];
  point_hinverses_block0[4] = hs_inverses[gridindex * 10 + 4];
  point_hs_block0[4] = hs[gridindex * 10 + 4];
  point_positions_block0[4] = positions[gridindex * 10 + 4];
  point_hinverses_block0[5] = hs_inverses[gridindex * 10 + 5];
  point_hs_block0[5] = hs[gridindex * 10 + 5];
  point_positions_block0[5] = positions[gridindex * 10 + 5];
  point_hinverses_block0[6] = hs_inverses[gridindex * 10 + 6];
  point_hs_block0[6] = hs[gridindex * 10 + 6];
  point_positions_block0[6] = positions[gridindex * 10 + 6];
  point_hinverses_block0[7] = hs_inverses[gridindex * 10 + 7];
  point_hs_block0[7] = hs[gridindex * 10 + 7];
  point_positions_block0[7] = positions[gridindex * 10 + 7];
  point_hinverses_block0[8] = hs_inverses[gridindex * 10 + 8];
  point_hs_block0[8] = hs[gridindex * 10 + 8];
  point_positions_block0[8] = positions[gridindex * 10 + 8];
  point_hinverses_block0[9] = hs_inverses[gridindex * 10 + 9];
  point_hs_block0[9] = hs[gridindex * 10 + 9];
  point_positions_block0[9] = positions[gridindex * 10 + 9];
  __private int teiler = 0;
  __private float h = 1.0f / 3.0f;
  __private float umid = 0.0;
  __private float sum = 0.0;
  __private int u = 0;
  float gesamtint_block0 = 0.0;
  __local float positions_local[320];
  __local float hs_local[320];
  __local float hinverses_local[320];  // CHANGED TO FLOAT
  __local float alpha_local[32];
  for (int group = 0; group < 2424; group++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < 32) {
      for (int j = 0; j < 10; j++) {
        hinverses_local[local_id * 10 + j] = (float)hs_inverses[group * 320 + local_id * 10 + j];
        hs_local[local_id * 10 + j] = hs[group * 320 + local_id * 10 + j];
        positions_local[local_id * 10 + j] = positions[group * 320 + local_id * 10 + j];
      }
      alpha_local[local_id] = alpha[group * 32 + local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < 32; i++) {
      __private float zellenintegral = 1.0f;
      for (private int dim = 0; dim < 10; dim++) {
        __private float distance =
            fabs(point_positions_block0[dim] - positions_local[i * 10 + dim]);
        sum = 1.0f - distance * point_hinverses_block0[dim];
        sum *= hs_local[i * 10 + dim];
        sum = max(sum, 0.0f);
        sum += max(point_hs_block0[dim] * (1.0f - hinverses_local[i * 10 + dim] * distance), 0.0f);
        zellenintegral *=
            sum *
            select(1.0f, h, (uint)(point_hinverses_block0[dim] == hinverses_local[i * 10 + dim]));
      }
      gesamtint_block0 += zellenintegral * alpha_local[i];
    }
  }
  result[get_global_id(0) * 1 + 0] = gesamtint_block0;
  result[get_global_id(0) * 1 + 0] += alpha[gridindex * 1 + 0] * lambda;
}
