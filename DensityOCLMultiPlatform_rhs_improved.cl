void kernel cscheme(global const int* starting_points, global const float* data_points,
                    global float* C, private int startid) {
  C[get_global_id(0)] = 0.0;
  int grid_index[10];
  int grid_level_2[10];
  for (int d = 0; d < 10; d++) {
    grid_index[d] = starting_points[(startid + get_global_id(0)) * 2 * 10 + 2 * d];
    grid_level_2[d] = 1 << starting_points[(startid + get_global_id(0)) * 2 * 10 + 2 * d + 1];
  }

  __local data_group[128 * 10];

  // for (unsigned int data_index = 0; data_index < 150000; data_index++) {
  for (unsigned int outer_data_index = 0; outer_data_index < 150000; outer_data_index += 128) {
    barrier(CLK_LOCAL_MEM_FENCE);  // remove this?
    for (int d = 0; d < 10; d++) {
      data_group[get_local_id(0) * 10 + d] =
          data_points[(outer_data_index + get_local_id(0)) * 10 + d];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int data_index = 0; data_index < 128; data_index += 1) {
      if (outer_data_index + data_index >= 150000) {
        break;
      }
      float value = 1;
      for (int d = 0; d < 10; d++) {
        float wert = grid_level_2[d];
        // wert *= data_points[data_index * 10 + d];
        wert *= data_group[data_index * 10 + d];
        wert -= grid_index[d];
        wert = fabs(wert);
        wert = 1 - wert;
        if (wert < 0) wert = 0;
        value *= wert;
      }
      C[get_global_id(0)] += value;
    }
  }
  C[get_global_id(0)] /= 150000.0;
}
