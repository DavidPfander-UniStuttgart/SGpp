__kernel void verify_support(__global double *data, const long data_size,
                             __global long *schedule_level,
                             __global long *schedule_index,
                             __global unsigned char *schedule_support,
                             long min_support) {
  // int idx = get_local_id(0);
  int gid = get_global_id(0);
  int lid = get_local_id(0);

  double bound_left[DIM];
  for (int d = 0; d < DIM; d += 1) {
    bound_left[d] = pow(2.0, (double)(-schedule_level[gid * DIM + d])) *
                    (double)(schedule_index[gid * DIM + d] - 1l);
    // if (gid == 31) {
    //   printf("d: %i, schedule_level: %li, schedule_index: %li, left: %f\n",
    //   d,
    //          schedule_level[gid * DIM + d], schedule_index[gid * DIM + d],
    //          bound_left[d]);
    // }
  }
  double bound_right[DIM];
  for (int d = 0; d < DIM; d += 1) {
    bound_right[d] = pow(2.0, (double)(-schedule_level[gid * DIM + d])) *
                     (double)(schedule_index[gid * DIM + d] + 1l);
    // if (gid == 31) {
    //   printf("d: %i, schedule_level: %li, schedule_index: %li, right: %f\n",
    //   d,
    //          schedule_level[gid * DIM + d], schedule_index[gid * DIM + d],
    //          bound_right[d]);
    // }
  }

  long num_support = 0;

  local double data_local[16 * DIM];

  for (long data_index = 0; data_index < data_size; data_index += 16) {
    barrier(CLK_LOCAL_MEM_FENCE); // otherwise inconsistent across iterations
    if (lid < 16) {
      for (long d = 0; d < DIM; d += 1) {
        data_local[d * 16 + lid] = data[d * data_size + (data_index + lid)];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int local_data_index = 0; local_data_index < 16;
         local_data_index += 1) {
      bool has_support = true;
      for (int d = 0; d < DIM; d += 1) {
        if (data_local[d * 16 + local_data_index] <= bound_left[d] ||
            data_local[d * 16 + local_data_index] >= bound_right[d]) {
          has_support = false;
        }
      }
      if (has_support) {
        num_support += 1;
      }
    }
  }
  if (num_support >= min_support) {
    // if (gid == 31) {
    //   printf("(gid: %i, true)", gid);
    // }
    schedule_support[gid] = true;
  } else {
    schedule_support[gid] = false;
    // if (gid == 31) {
    //   printf("(gid: %i, false)", gid);
    // }
  }
}
