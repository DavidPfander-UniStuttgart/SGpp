#pragma OPENCL EXTENSION cl_khr_fp64 : enable

void kernel cscheme(global const int* starting_points, global const double* data_points,
                    global double* C, private int startid) {
  C[get_global_id(0)] = 0.0;
 private
  double value = 1;
 private
  double wert = 1.0;
 for (unsigned int ds = 0; ds < 150000; ds++) { // * m * N (N from global_size)
    value = 1;
    for (int d = 0; d < 10; d++) { // * d
      wert = (1 << starting_points[(startid + get_global_id(0)) * 2 * 10 + 2 * d + 1]);
      wert *= data_points[ds * 10 + d]; // 1
      wert -= starting_points[(startid + get_global_id(0)) * 2 * 10 + 2 * d]; // 2
      wert = fabs(wert); // 3
      wert = 1 - wert; // 4
      if (wert < 0) wert = 0;
      value *= wert; // 5
    }
    C[get_global_id(0)] += value; // 1
  }
 C[get_global_id(0)] /= 150000; // 1
}
// total ops: N * m * (10 * d + 1); final + 1 omitted
