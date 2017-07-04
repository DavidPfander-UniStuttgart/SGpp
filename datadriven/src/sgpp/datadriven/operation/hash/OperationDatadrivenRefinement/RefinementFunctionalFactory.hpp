/* Copyright (C) 2008-today The SG++ project
 * This file is part of the SG++ project. For conditions of distribution and
 * use, please see the copyright notice provided with SG++ or at
 * sgpp.sparsegrids.org
 *
 * datadriven_refinement.cpp
 *
 *  Created on: 24.04.2017
 *      Author: David Pfander
 */

#pragma once

#define WITH_LOGGING false

namespace sgpp {
namespace datadriven {
namespace datadriven_refinement_factory {
std::function<bool(sgpp::base::DataMatrix &, sgpp::base::GridPoint &)> createMaxLevelFunctor(
    size_t max_level, size_t support_points_for_accept) {
  std::function<bool(sgpp::base::DataMatrix &, sgpp::base::GridPoint &)> f = [=](
      sgpp::base::DataMatrix &data, sgpp::base::GridPoint &p) -> bool {
    for (size_t d = 0; d < p.getDimension(); d++) {
      if (p.getLevel(d) > max_level) {
        return false;
      }
    }
    sgpp::base::SLinearBase base;
    sgpp::base::DataVector data_point(p.getDimension());
    size_t points_in_support = 0;
    for (size_t data_index = 0; data_index < data.getNrows(); data_index++) {
      data.getRow(data_index, data_point);
      bool on_support = true;
#if WITH_LOGGING
      std::cout << "testing ";
#endif

      for (size_t d = 0; d < p.getDimension(); d++) {
        double value = base.eval(p.getLevel(d), p.getIndex(d), data_point[d]);

#if WITH_LOGGING
        if (d > 0) {
          std::cout << ", ";
        }
        std::cout << "data[" << d << "] = " << data_point[d] << " -> " << value;
#endif

        if (value == 0.0) {
          on_support = false;
          break;
        }
      }
      // std::cout << "------------ found" << std::endl;
      // return true;
      if (on_support) {
#if WITH_LOGGING
        std::cout << " accept!" << std::endl;
#endif
        points_in_support += 1;
        if (points_in_support >= support_points_for_accept) {
          break;
        }
      }
#if WITH_LOGGING
      else {
        std::cout << " reject!" << std::endl;
      }
#endif
    }
    if (points_in_support >= support_points_for_accept) {
#if WITH_LOGGING
      std::cout << "-----refined point added-------" << std::endl;
#endif
      return true;
    }
#if WITH_LOGGING
    std::cout << "------point not refined------" << std::endl;
#endif
    return false;
  };
  return f;
}

std::function<bool(sgpp::base::DataMatrix &, sgpp::base::GridPoint &)> createMaxLevelSumFunctor(
    size_t max_level_sum, size_t support_points_for_accept) {
  std::function<bool(sgpp::base::DataMatrix &, sgpp::base::GridPoint &)> f = [=](
      sgpp::base::DataMatrix &data, sgpp::base::GridPoint &p) -> bool {
    size_t sum = 0;
    for (size_t d = 0; d < p.getDimension(); d++) {
      sum += p.getLevel(d);
    }

    if (sum >= max_level_sum) {
      return false;
    }

    sgpp::base::SLinearBase base;
    sgpp::base::DataVector data_point(p.getDimension());

    size_t points_in_support = 0;
    for (size_t data_index = 0; data_index < data.getNrows(); data_index++) {
      data.getRow(data_index, data_point);
      bool on_support = true;
#if WITH_LOGGING
      std::cout << "testing ";
#endif

      for (size_t d = 0; d < p.getDimension(); d++) {
        double value = base.eval(p.getLevel(d), p.getIndex(d), data_point[d]);

#if WITH_LOGGING
        if (d > 0) {
          std::cout << ", ";
        }
        std::cout << "data[" << d << "] = " << data_point[d] << " -> " << value;
#endif

        if (value == 0.0) {
          on_support = false;
          break;
        }
      }
      // std::cout << "------------ found" << std::endl;
      // return true;
      if (on_support) {
#if WITH_LOGGING
        std::cout << " accept!" << std::endl;
#endif
        points_in_support += 1;
        if (points_in_support >= support_points_for_accept) {
          break;
        }
      }
#if WITH_LOGGING
      else {
        std::cout << " reject!" << std::endl;
      }
#endif
    }
    if (points_in_support >= support_points_for_accept) {
#if WITH_LOGGING
      std::cout << "-----refined point added-------" << std::endl;
#endif
      return true;
    }
#if WITH_LOGGING
    std::cout << "------point not refined------" << std::endl;
#endif
    return false;
  };
  return f;
}

std::function<bool(sgpp::base::DataMatrix &, sgpp::base::GridPoint &)>
createMaxLevelAndLevelSumFunctor(size_t max_level, size_t max_level_sum,
                                 size_t support_points_for_accept) {
  std::function<bool(sgpp::base::DataMatrix &, sgpp::base::GridPoint &)> f = [=](
      sgpp::base::DataMatrix &data, sgpp::base::GridPoint &p) -> bool {
    size_t sum = 0;
    for (size_t d = 0; d < p.getDimension(); d++) {
      size_t level = p.getLevel(d);
      if (level >= max_level) {
        return false;
      }
      sum += level;
    }

    if (sum >= max_level_sum) {
      return false;
    }

    sgpp::base::SLinearBase base;
    sgpp::base::DataVector data_point(p.getDimension());

    size_t points_in_support = 0;
    for (size_t data_index = 0; data_index < data.getNrows(); data_index++) {
      data.getRow(data_index, data_point);
      bool on_support = true;
#if WITH_LOGGING
      std::cout << "testing ";
#endif

      for (size_t d = 0; d < p.getDimension(); d++) {
        double value = base.eval(p.getLevel(d), p.getIndex(d), data_point[d]);

#if WITH_LOGGING
        if (d > 0) {
          std::cout << ", ";
        }
        std::cout << "data[" << d << "] = " << data_point[d] << " -> " << value;
#endif

        if (value == 0.0) {
          on_support = false;
          break;
        }
      }
      // std::cout << "------------ found" << std::endl;
      // return true;
      if (on_support) {
#if WITH_LOGGING
        std::cout << " accept!" << std::endl;
#endif
        points_in_support += 1;
        if (points_in_support >= support_points_for_accept) {
          break;
        }
      }
#if WITH_LOGGING
      else {
        std::cout << " reject!" << std::endl;
      }
#endif
    }
    if (points_in_support >= support_points_for_accept) {
#if WITH_LOGGING
      std::cout << "-----refined point added-------" << std::endl;
#endif
      return true;
    }
#if WITH_LOGGING
    std::cout << "------point not refined------" << std::endl;
#endif
    return false;
  };
  return f;
}

}  // namespace datadriven_refinement_factory
}  // namespace datadriven
}  // namespace sgpp
