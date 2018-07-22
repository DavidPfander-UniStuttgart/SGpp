__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void multdensity(
    __global const int *starting_points, __global const ulong *dim_zero_flags_v,
    __global const ulong *level_offsets_v, __global const ulong *level_packed_v,
    __global const ulong *index_packed_v, const unsigned int non_padding_size,
    __global const float *alpha, __global float *result, const float lambda, const int startid) {
  int gridindex = startid + get_global_id(0);
  ulong one_mask = 1;
  __private int local_id = get_local_id(0);
  __private ulong point_dim_zero_flags = dim_zero_flags_v[gridindex];
  __private ulong point_level_offsets = level_offsets_v[gridindex];
  __private ulong point_level_packed = level_packed_v[gridindex];
  __private ulong point_index_packed = index_packed_v[gridindex];
  __private int teiler = 0;
  __private float h = 1.0 / 3.0;
  __private float umid = 0.0;
  __private float sum = 0.0;
  __private int u = 0;
  float gesamtint_block0 = 0.0;
  __local int indices_local[1280];
  __local int level_local[1280];
  __local float alpha_local[32];
  for (int group = 0; group < 2052; group++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if (get_local_id(0) < 32) {
      for (int j = 0; j < 40; j++) {
        indices_local[local_id * 40 + j] = starting_points[group * 2560 + local_id * 80 + 2 * j];
        level_local[local_id * 40 + j] = starting_points[group * 2560 + local_id * 80 + 2 * j + 1];
      }
      alpha_local[local_id] = alpha[group * 32 + local_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
#define EVAL_BLOCKING 8

    for (int local_index = 0; local_index < 32; local_index += EVAL_BLOCKING) {  // CHANGE
      // float zellenintegral = 1.0;
      float zellenintegral_blocked[EVAL_BLOCKING];  // CHANGE
      for (size_t j = 0; j < EVAL_BLOCKING; j++) {  // ADD
        zellenintegral_blocked[j] = 1.0;            // ADD
      }                                             // ADD
      ulong fixed_dim_zero_flags = point_dim_zero_flags;
      ulong fixed_level_offsets = point_level_offsets;
      ulong fixed_level_packed = point_level_packed;
      ulong fixed_index_packed = point_index_packed;
      for (private int dim = 0; dim < 40; dim++) {
        ulong is_dim_implicit = fixed_dim_zero_flags & one_mask;
        fixed_dim_zero_flags >>= 1;
        ulong decompressed_level = 1;
        ulong decompressed_index = 1;
        if (is_dim_implicit != 0) {
          ulong level_bits = 1 + clz(fixed_level_offsets);
          fixed_level_offsets <<= level_bits;
          ulong level_mask = (1 << level_bits) - 1;
          decompressed_level = (fixed_level_packed & level_mask) + 2;
          fixed_level_packed >>= level_bits;
          ulong index_bits = decompressed_level - 1;
          ulong index_mask = (1 << index_bits) - 1;
          decompressed_index = ((fixed_index_packed & index_mask) << 1) + 1;
          fixed_index_packed >>= index_bits;
        }
        float l_2 = (float)(1 << decompressed_level);                  // ADD
        float i = (float)(decompressed_index);                         // ADD
        for (size_t j = 0; j < 8; j++) {                               // ADD
          h = 1.0 / (1 << level_local[(local_index + j) * 40 + dim]);  // CHANGE
          // u = (1 << decompressed_level); // CHANGE
          u = l_2;
          umid = u * h * (indices_local[(local_index + j) * 40 + dim]) - i;  // CHANGE
          umid = fabs(umid);
          umid = 1.0f - umid;  // CHANGE
          umid = (umid + fabs(umid));
          sum = h * (umid);
          // h = 1.0 / (1 << decompressed_level);
          h = 1.0 / l_2;                                         // change
          u = (1 << level_local[(local_index + j) * 40 + dim]);  // CHANGE
          // umid = u * h * (decompressed_index) - indices_local[i* 40+dim];
          umid = u * h * i - indices_local[(local_index + j) * 40 + dim];  // CHANG
          umid = fabs(umid);
          umid = 1.0f - umid;
          umid = (umid + fabs(umid));
          sum += h * (umid);
          sum *= level_local[(local_index + j) * 40 + dim] == decompressed_level ? 1.0f / 3.0f
                                                                                 : 1.0f;  // CHANGE
          zellenintegral_blocked[j] *= sum;                                               // CHANGE
        }
      }
      for (size_t j = 0; j < 8; j++) {                                                 // ADD
        gesamtint_block0 += zellenintegral_blocked[j] * alpha_local[local_index + j];  // CHANGE
      }                                                                                // ADD
    }
  }
  result[get_global_id(0) * 1 + 0] = gesamtint_block0 / 256;
  result[get_global_id(0) * 1 + 0] += alpha[gridindex * 1 + 0] * lambda;
}
