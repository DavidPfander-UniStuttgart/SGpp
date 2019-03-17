
#include "support_refinement_iterative.hpp"

#include <algorithm>

namespace sgpp::datadriven {

void support_refinement_iterative::neighbor_pos_dir_blocked(std::vector<int64_t> &schedule_level,
                                                            std::vector<int64_t> &schedule_index,
                                                            std::vector<double> &neighbors,
                                                            direction dir) {
  int64_t num_candidates = static_cast<int64_t>(schedule_level.size() / dim);
  if (dir == direction::left) {
    for (int64_t candidate_index = 0; candidate_index < num_candidates; candidate_index += 1) {
      for (int64_t d = 0; d < dim; d += 1) {
        neighbors[candidate_index * dim + d] =
            std::pow(2.0, static_cast<double>(-schedule_level[candidate_index * dim + d])) *
            static_cast<double>(schedule_index[candidate_index * dim + d] - 1ll);
      }
    }
  } else if (dir == direction::right) {
    for (int64_t candidate_index = 0; candidate_index < num_candidates; candidate_index += 1) {
      for (int64_t d = 0; d < dim; d += 1) {
        neighbors[candidate_index * dim + d] =
            std::pow(2.0, static_cast<double>(-schedule_level[candidate_index * dim + d])) *
            static_cast<double>(schedule_index[candidate_index * dim + d] + 1ll);
      }
    }
  } else {
    throw;  // never
  }
}

void support_refinement_iterative::verify_support_blocked() {
  int64_t num_candidates = static_cast<int64_t>(schedule_level.size() / dim);
  std::vector<double> bound_left(schedule_level.size());
  std::vector<double> bound_right(schedule_level.size());

  neighbor_pos_dir_blocked(schedule_level, schedule_index, bound_left, direction::left);
  neighbor_pos_dir_blocked(schedule_level, schedule_index, bound_right, direction::right);

  for (int64_t candidate_index = 0; candidate_index < num_candidates; candidate_index += 1) {
    schedule_support[candidate_index] = false;
  }

  std::vector<int64_t> num_support(num_candidates, 0);
#pragma omp parallel for
  for (int64_t data_index = 0; data_index < entries; data_index += 1) {
    for (int64_t candidate_index = 0; candidate_index < num_candidates; candidate_index += 1) {
      if (num_support[candidate_index] > min_support) {
        continue;
      }
      bool has_support = true;
      for (int64_t d = 0; d < dim; d += 1) {
        if (data[d * entries + data_index] <= bound_left[candidate_index * dim + d] ||
            data[d * entries + data_index] >= bound_right[candidate_index * dim + d]) {
          has_support = false;
          break;
        }
      }
      if (has_support) {
#pragma omp atomic
        num_support[candidate_index] += 1;
      }
    }
  }

  for (int64_t candidate_index = 0; candidate_index < num_candidates; candidate_index += 1) {
    if (num_support[candidate_index] >= min_support) {
      schedule_support[candidate_index] = true;
    }
    // else {
    //   printf("not enough sup!\n");
    // }
  }

  // std::cout << "schedule_support: ";
  // for (size_t i = 0; i < schedule_support.size(); i += 1) {
  //   std::cout << schedule_support[i];
  // }
  // std::cout << std::endl;
}

// N * m data points accessed (~memory operations)
void support_refinement_iterative::refine_impl() {
  while (schedule_level.size() > 0) {
    // compute support for all points in schedule
    int64_t num_candidates = static_cast<int64_t>(schedule_level.size() / dim);
    num_visited += num_candidates;
    std::cout << "num_candidates: " << num_candidates << std::endl;
    schedule_support.resize(num_candidates);
// for (int64_t candidate_index = 0; candidate_index < num_candidates;
// candidate_index += 1) { std::vector<int64_t> cur_l(dim); for (int64_t d =
// 0; d < dim; d += 1) {
//   cur_l[d] = schedule_level[candidate_index * dim + d];
// }
// std::vector<int64_t> cur_i(dim);
// for (int64_t d = 0; d < dim; d += 1) {
//   cur_i[d] = schedule_index[candidate_index * dim + d];
// }
// /////////////////////////////////
// std::cout << "l: ";
// for (int64_t d = 0; d < dim; d += 1) {
//   if (d > 0) {
//     std::cout << ", ";
//   }
//   std::cout << schedule_level[candidate_index * dim + d];
// }
// std::cout << " i: ";
// for (int64_t d = 0; d < dim; d += 1) {
//   if (d > 0) {
//     std::cout << ", ";
//   }
//   std::cout << schedule_index[candidate_index * dim + d];
// }
// std::cout << std::endl;
// /////////////////////////////////

#if USE_OCL == 1
    if (use_ocl) {
      verify_support_blocked_OCL();
    } else {
      verify_support_blocked();
    }
#else
    verify_support_blocked();
#endif
    // // std::cout << "num_support: " << num_support << std::endl;
    // if (num_support > min_support) {
    //   // std::cout << "ok? true" << std::endl;
    //   schedule_support[candidate_index] = true;
    // } else {
    //   // std::cout << "ok? false" << std::endl;
    //   schedule_support[candidate_index] = false;
    // }
    // }
    std::cout << "candidates with support: "
              << std::count(schedule_support.begin(), schedule_support.end(), true) << std::endl;

    // for (bool b : schedule_support) {
    //   std::cout << b;
    // }
    // std::cout << std::endl;

    // create next schedule
    for (int64_t candidate_index = 0; candidate_index < num_candidates; candidate_index += 1) {
      if (!schedule_support[candidate_index]) {
        // std::cout << "skip!" << std::endl;
        continue;
      }

      levels.insert(levels.end(), schedule_level.begin() + (candidate_index * dim),
                    schedule_level.begin() + (candidate_index + 1) * dim);
      indices.insert(indices.end(), schedule_index.begin() + (candidate_index * dim),
                     schedule_index.begin() + (candidate_index + 1) * dim);

      std::vector<int64_t> cur_l(dim);
      for (int64_t d = 0; d < dim; d += 1) {
        cur_l[d] = schedule_level[candidate_index * dim + d];
      }
      std::vector<int64_t> cur_i(dim);
      for (int64_t d = 0; d < dim; d += 1) {
        cur_i[d] = schedule_index[candidate_index * dim + d];
      }

      int64_t sum_l = 0;
      // std::cout << "l: ";
      for (int64_t d = 0; d < dim; d += 1) {
        sum_l += schedule_level[candidate_index * dim + d];
        // if (d > 0) {
        //   std::cout << ", ";
        // }
        // std::cout << schedule_level[candidate_index * dim + d];
      }
      // std::cout << std::endl;
      // std::cout << "sum_l: " << sum_l << std::endl;

      // recurse children if lower dims are all 1 (but not necessarily the
      // current dim)
      if (sum_l < max_level + dim - 1) {
        for (int64_t d = 0; d < dim; d += 1) {
          std::vector<int64_t> child_l;
          std::vector<int64_t> child_i;
          child_d_dir(cur_l, cur_i, child_l, child_i, d, direction::left);
          schedule_next_level.insert(schedule_next_level.end(), child_l.begin(), child_l.end());
          schedule_next_index.insert(schedule_next_index.end(), child_i.begin(), child_i.end());
          child_d_dir(cur_l, cur_i, child_l, child_i, d, direction::right);
          schedule_next_level.insert(schedule_next_level.end(), child_l.begin(), child_l.end());
          schedule_next_index.insert(schedule_next_index.end(), child_i.begin(), child_i.end());
          if (cur_l[d] != 1) {
            break;
          }
        }
      }
    }  // end candidates
    swap(schedule_level, schedule_next_level);
    schedule_next_level.clear();
    swap(schedule_index, schedule_next_index);
    schedule_next_index.clear();
  }  // end while
}

support_refinement_iterative::support_refinement_iterative(int64_t dim, int64_t max_level,
                                                           int64_t min_support,
                                                           const std::vector<double> &data)
    : dim(dim), max_level(max_level), min_support(min_support), num_visited(0) {
  entries = data.size() / dim;
  this->data.resize(data.size());
  // convert data to SoA
  for (int64_t i = 0; i < entries; i += 1) {
    for (int64_t d = 0; d < dim; d += 1) {
      this->data[d * entries + i] = data[i * dim + d];
    }
  }
}

support_refinement_iterative::~support_refinement_iterative() {
#if USE_OCL == 1
  if (manager) {
    manager->release_kernels(kernels_verify_support);
  }
#endif
}

void support_refinement_iterative::refine() {
  levels.clear();
  indices.clear();
  std::vector<int64_t> l(dim, 1);
  std::vector<int64_t> i(dim, 1);
  schedule_level.insert(schedule_next_level.end(), l.begin(), l.end());
  schedule_index.insert(schedule_next_index.end(), i.begin(), i.end());
  refine_impl();
}

std::vector<int64_t> &support_refinement_iterative::get_levels() { return levels; }
std::vector<int64_t> &support_refinement_iterative::get_indices() { return indices; }

void support_refinement_iterative::print_level(std::vector<int64_t> l) {
  std::cout << "l: (";
  for (int64_t d = 0; d < dim; d += 1) {
    if (d > 0) {
      std::cout << ", ";
    }
    std::cout << l[d];
  }
  std::cout << ")" << std::endl;
}
void support_refinement_iterative::print_index(std::vector<int64_t> i) {
  std::cout << "i: (";
  for (int64_t d = 0; d < dim; d += 1) {
    if (d > 0) {
      std::cout << ", ";
    }
    std::cout << i[d];
  }
  std::cout << ")" << std::endl;
}
void support_refinement_iterative::print_pos(std::vector<double> pos) {
  std::cout << "pos: (";
  for (int64_t d = 0; d < dim; d += 1) {
    if (d > 0) {
      std::cout << ", ";
    }
    std::cout << pos[d];
  }
  std::cout << ")" << std::endl;
}

void support_refinement_iterative::write_grid_positions(int64_t dim,
                                                        const std::string &grid_file_name,
                                                        const std::string &dataset_file_name,
                                                        std::vector<int64_t> &ls,
                                                        std::vector<int64_t> &is) {
  std::ofstream out_file(grid_file_name);

  for (int64_t d = 0; d < dim; d += 1) {
    if (d > 0) {
      out_file << ", ";
    }
    out_file << "x" << std::to_string(d);
  }
  out_file << std::endl;
  out_file << "#filename: " << dataset_file_name << std::endl;
  std::vector<double> pos(dim);
  for (int64_t i = 0; i < static_cast<int64_t>(ls.size() / dim); i += 1) {
    for (int64_t d = 0; d < dim; d += 1) {
      pos[d] = std::pow(2.0, static_cast<double>(-ls[i * dim + d])) *
               static_cast<double>(is[i * dim + d]);
    }
    for (int64_t d = 0; d < dim; d += 1) {
      if (d > 0) {
        out_file << ", ";
      }
      out_file << pos[d];
    }
    out_file << std::endl;
  }
}

}  // namespace sgpp::datadriven
