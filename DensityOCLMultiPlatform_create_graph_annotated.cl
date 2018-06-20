#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// thread connect one (!) data point
// processing chunk of 128 data elements in this configuration (128 potential neighbors)
// about other candidates?
__kernel __attribute__((reqd_work_group_size(128, 1, 1))) void connectNeighbors(
    __global const double *data, __global int *neighbors, const int startid) {
  __private int global_index = startid + get_global_id(0);
  __private int local_id = get_local_id(0);
  __private int chunk_index = get_global_id(0);
  __private int maxindex = 0;
  __private double datapoint[2];
  datapoint[0] = data[global_index * 2 + 0];
  datapoint[1] = data[global_index * 2 + 1];
  __private double dist = 0.0;
  __local double data_local[256];
  __private double dist_reg[128];
  __private int index_reg[128];
  for (int i = 0; i < 128; i++) {
    // basically infinity in a [0, 1] domain
    dist_reg[i] = 4.0;  // all chunk distances!
  }
  // 2 * 128 elements processed (reduce over 2 groups FOR THIS DATASET!)
  for (int group = 0; group < 2; group++) {
    barrier(CLK_LOCAL_MEM_FENCE);
    // load different dims
    for (int j = 0; j < 2; j++) {
      // load into local memory in non-SoA form (aabbcc)
      data_local[local_id * 2 + j] = data[group * 256 + local_id * 2 + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < 128; i++) {
      // calc dist^2
      dist = 0.0;
      for (int j = 0; j < 2; j++) {  // dims
          dist += (datapoint[j] - data_local[j + i * 2]) * (datapoint[j] - data_local[j + i * 2]); // 4 (or 3 if cached)
      }
      // smaller distance and no reflexive edges
      if (dist < dist_reg[i] && i + group * 128 != global_index) {
        dist_reg[i] = dist;
        index_reg[i] = group;
      }
    }
  }

  // now dist_reg give minimal distance in group

  // reduce 128-element arrays to neighbor size
  for (int neighbor = 0; neighbor < 5; neighbor++) {
    // find the index with the smallest distance
    maxindex = 0;
    for (int i = 1; i < 128; i++) {
      if (dist_reg[i] < dist_reg[maxindex]) maxindex = i;
    }
    // mark the index of the new "nearest" neighbor
    neighbors[chunk_index * 5 + neighbor] = maxindex + index_reg[maxindex] * 128;
    // mark distance of "nearest" neighbor as infinity to ensure that it is now used again
    dist_reg[maxindex] = 4.0;
  }
}
