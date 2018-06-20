#pragma OPENCL EXTENSION cl_khr_fp64 : enable

void kernel cscheme(global const int* starting_points, global const double* data_points,
                    global double* C, private int startid) {
  C[get_global_id(0)] = 0.0;
 private
  double value = 1;
 private
  double wert = 1.0;
  for (unsigned int ds = 0; ds < 150000; ds++) {
    value = 1;
    for (private int d = 0; d < 10; d++) {
      wert = (1 << starting_points[(startid + get_global_id(0)) * 2 * 10 + 2 * d + 1]);
      wert *= data_points[ds * 10 + d];
      wert -= starting_points[(startid + get_global_id(0)) * 2 * 10 + 2 * d];
      wert = fabs(wert);
      wert = 1 - wert;
      if (wert < 0) wert = 0;
      value *= wert;
    }
    C[get_global_id(0)] += value;
  }
  C[get_global_id(0)] /= 150000;
}
