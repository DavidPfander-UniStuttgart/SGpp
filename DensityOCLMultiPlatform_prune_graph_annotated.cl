#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// eval hat function?
double get_u(__private const double grenze, __private const int index, __private const int level) {
 private
  double ret = (1 << level);
  ret *= grenze;
  ret -= index;
  ret = fabs(ret);
  ret = 1 - ret;
  if (ret < 0.0) ret = 0.0;
  return ret;
} // 4F

void kernel removeEdges(__global int *nodes, __global const int *starting_points,
                        __global const double *data, __global const double *alphas, int startid) {
  size_t index = get_global_id(0);
  size_t global_index = startid + get_global_id(0);
  __private double endwert = 0.0;
  __private double wert = 1.0;
  for (int i = 0; i < 5; i++) { // * k * m, (m = global_size)
    // Calculate density
    endwert = 0;
    int nachbar = nodes[index * 5 + i];
    for (int gridpoint = 0; gridpoint < 5; gridpoint++) { // * N
      wert = 1;
      for (int dimension = 0; dimension < 2; dimension++) { // * d
        double dimension_point = 0;
        dimension_point =
            data[dimension + nachbar * 2] +
            (data[global_index * 2 + dimension] - data[dimension + nachbar * 2]) * 0.5; // 3
        wert *= get_u(dimension_point, starting_points[gridpoint * 2 * 2 + 2 * dimension],
                      starting_points[gridpoint * 2 * 2 + 2 * dimension + 1]); // 4 + 1 = 5
      }
      endwert += wert * alphas[gridpoint]; // 2
    }
    if (endwert < 0.8) {
      nodes[5 * index + i] = -2;
    }
  }
  // if node itself is in low-density region, remove all edges
  endwert = 0;
  for (int gridpoint = 0; gridpoint < 5; gridpoint++) { // * N
    wert = 1;
    for (int dimension = 0; dimension < 2; dimension++) { // * d
      wert *= get_u(data[global_index * 2 + dimension],
                    starting_points[gridpoint * 2 * 2 + 2 * dimension],
                    starting_points[gridpoint * 2 * 2 + 2 * dimension + 1]); // 4 + 1 = 5
    }
    endwert += wert * alphas[gridpoint];
  }
  if (endwert < 0.8) {
    for (int i = 0; i < 5; i++) {
      nodes[5 * index + i] = -1;
    }
  }
}
