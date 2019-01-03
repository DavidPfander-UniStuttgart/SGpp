//TODO: change: converted "int" to "float"
float get_u(const float grenze, const float index, const float level_2) {
  float ret = level_2;
  ret *= grenze;
  ret -= index;
  ret = fabs(ret);
  ret = 1 - ret;
  if (ret < 0.0) ret = 0.0;
  return ret;
}
void kernel __attribute__((reqd_work_group_size(128, 1, 1)))
removeEdges(__global long *nodes, __global const int *starting_points, __global const float *data,
            __global const float *alphas, unsigned long startid, unsigned long chunksize) {
  size_t global_index = startid + get_global_id(0);

  // Potential problem:
  // - Intel compiler lays out variable for each thread mem: eval_locations_0, eval_locations_1, eval_locations_2, ..
  // - cannot load from continguous memory, if first component of each thread is loaded
  // Idea:
  // - Split into individual variables, mem: e_l_0_0, e_l_0_1, e_l_0_2, e_l_0_3, ... e_l_1_0, e_l_0_1, ....
  // - now data is contiguous in memory again
  float eval_locations[(6 + 1) * 10];  // (k + 1) * 10
  float evals[6 + 1];                  // (k + 1)-many results
  for (int cur_eval = 0; cur_eval < 6 + 1; cur_eval += 1) {
    evals[cur_eval] = 0.0f;
  }

  // loading own data into registers
  for (size_t d = 0; d < 10; d += 1) {
    if (global_index < 1020000) {
      eval_locations[6 * 10 + d] = data[global_index * 10 + d];
    } else {
      eval_locations[6 * 10 + d] = 0.0f;
    }
  }
  // loading neighbors into registers
  for (size_t cur_k = 0; cur_k < 6; cur_k += 1) {
    long neighbor_index;
    if (get_global_id(0) < chunksize) {
      neighbor_index = nodes[get_global_id(0) * 6 + cur_k];
    } else {
      neighbor_index = 0;
    }
    for (size_t d = 0; d < 10; d += 1) {
      if (global_index < 1020000 && neighbor_index >= 0) {
        float loc_dim = data[d + neighbor_index * 10];
        eval_locations[cur_k * 10 + d] = loc_dim + 0.5 * (eval_locations[6 * 10 + d] - loc_dim);
      } else {
        eval_locations[cur_k * 10 + d] = 0.0f;
      }
    }
  }

  local float grid_indices[32 * 10];   // local_size * 10
  local float grid_levels_2[32 * 10];  // local_size * 10
  local float grid_alpha[32];          // local_size

  // evaluate in the middle of the edges
  for (int outer_grid_index = 0; outer_grid_index < 77505; outer_grid_index += 32) {
    // cache next set of grid points in local memory
    barrier(CLK_LOCAL_MEM_FENCE);
    int grid_index = outer_grid_index + get_local_id(0);
    if (get_local_id(0) < 32 && grid_index < 77505) {
      for (int d = 0; d < 10; d++) {
        grid_indices[get_local_id(0) * 10 + d] =
            (float)starting_points[grid_index * 2 * 10 + 2 * d];
        grid_levels_2[get_local_id(0) * 10 + d] =
            (float)(1 << starting_points[grid_index * 2 * 10 + 2 * d + 1]);
      }
      grid_alpha[get_local_id(0)] = alphas[grid_index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int inner_grid_index = 0; inner_grid_index < 32; inner_grid_index += 1) {
      if (outer_grid_index + inner_grid_index >= 77505) {
        break;
      }
      float eval_1ds[7];
      for (int cur_eval = 0; cur_eval < 6 + 1; cur_eval += 1) {
        eval_1ds[cur_eval] = 1.0f;
      }
      for (int d = 0; d < 10; d += 1) {
        for (int cur_eval = 0; cur_eval < 6 + 1; cur_eval += 1) {
          float temp =
              grid_levels_2[inner_grid_index * 10 + d] * eval_locations[cur_eval * 10 + d] -
              grid_indices[inner_grid_index * 10 + d];
          temp = fmax(1 - fabs(temp), 0.0f);
          eval_1ds[cur_eval] *= temp;

          // get_u(eval_locations[cur_eval * 10 + d], grid_indices[inner_grid_index * 10 + d],
          //       grid_levels_2[inner_grid_index * 10 + d]);
        }
      }
      for (int cur_eval = 0; cur_eval < 6 + 1; cur_eval += 1) {
        evals[cur_eval] += eval_1ds[cur_eval] * grid_alpha[inner_grid_index];
      }
      // for (int cur_eval = 0; cur_eval < 6 + 1; cur_eval += 1) {
      //     float eval_1d = 1.0f;
      //     for (int d = 0; d < 10; d += 1) {
      //         eval_1d *=
      //                 get_u(eval_locations[cur_eval * 10 + d], grid_indices[inner_grid_index * 10
      //                 + d],
      //                 grid_levels_2[inner_grid_index * 10 + d]);
      //     }
      //     evals[cur_eval] += eval_1d * grid_alpha[inner_grid_index];
      // }
    }
  }

  if (get_global_id(0) < chunksize) {
    // point itself below density?
    if (fmax(evals[6], 0.0f) < 500.000000f) {
      // invalidate all neighbors, point now isolated
      for (int cur_k = 0; cur_k < 6; cur_k += 1) {
        nodes[6 * get_global_id(0) + cur_k] = -1;
      }
    } else {
      // point part of the graph, check individual edges
      for (size_t cur_k = 0; cur_k < 6; cur_k += 1) {
        if (fmax(evals[cur_k], 0.0f) < 500.000000f) {
          nodes[get_global_id(0) * 6 + cur_k] = -2;
        }
      }
    }
  }
}
