// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once
#include <sgpp/base/exception/operation_exception.hpp>
#include <sgpp/base/opencl/KernelSourceBuilderBase.hpp>
#include <sgpp/base/opencl/OCLOperationConfiguration.hpp>

#include <fstream>
#include <sstream>
#include <string>

namespace sgpp {
namespace datadriven {
namespace DensityOCLMultiPlatform {

/// OpenCL source builder for density based graph pruning
template <typename real_type>
class SourceBuilderPruneGraph : public base::KernelSourceBuilderBase<real_type> {
 private:
  /// OpenCL configuration containing the building flags
  json::Node &kernelConfiguration;
  /// Dimensions of grid
  size_t dims;

 public:
  SourceBuilderPruneGraph(json::Node &kernelConfiguration, size_t dims)
      : kernelConfiguration(kernelConfiguration), dims(dims) {}

  /// Generates the whole opencl kernel code used for the pruning of a graph
  std::string generateSource(size_t dimensions, size_t data_size, size_t gridSize, size_t k,
                             real_type threshold) {
    if (kernelConfiguration.contains("REUSE_SOURCE")) {
      if (kernelConfiguration["REUSE_SOURCE"].getBool()) {
        return this->reuseSource("DensityOCLMultiPlatform_prune_graph.cl");
      }
    }

    size_t local_size = kernelConfiguration["LOCAL_SIZE"].getUInt();
    uint64_t local_cache_size = kernelConfiguration["KERNEL_LOCAL_CACHE_SIZE"].getUInt();

    std::stringstream sourceStream;

    if (this->floatType().compare("double") == 0) {
      sourceStream << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl << std::endl;
    }
    if (!kernelConfiguration["KERNEL_USE_LOCAL_MEMORY"].getBool()) {
      sourceStream << "" << std::endl
                   << "" << this->floatType() << " get_u(__private const " << this->floatType()
                   << " grenze,__private const int index," << std::endl
                   << "__private const int level)"
                   << " {" << std::endl
                   << this->indent[0] << "private " << this->floatType() << " ret = (1 << level);"
                   << std::endl
                   << this->indent[0] << "ret*=grenze;" << std::endl
                   << this->indent[0] << "ret-=index;" << std::endl
                   << this->indent[0] << "ret = fabs(ret);" << std::endl
                   << this->indent[0] << "ret=1-ret;" << std::endl
                   << this->indent[0] << "if (ret<0.0)" << std::endl
                   << this->indent[1] << "ret=0.0;" << std::endl
                   << this->indent[0] << "return ret;" << std::endl
                   << "}" << std::endl
                   << "" << std::endl
                   << "void kernel __attribute__((reqd_work_group_size(" << local_size
                   << ", 1, 1))) removeEdges(__global int *nodes,"
                   << "__global const int *starting_points,__global const " << this->floatType()
                   << " *data," << std::endl
                   << this->indent[0] << "__global const " << this->floatType()
                   << " *alphas, int startid) {" << std::endl
                   << this->indent[0] << "size_t index = get_global_id(0);" << std::endl
                   << this->indent[0] << "size_t global_index = startid + get_global_id(0);"
                   << std::endl;
      sourceStream << this->indent[0] << "if (global_index < " << data_size << ") {" << std::endl;
      sourceStream << this->indent[0] << "__private " << this->floatType() << " endwert=0.0;"
                   << std::endl
                   << this->indent[0] << "__private " << this->floatType() << " wert=1.0;"
                   << std::endl
                   << this->indent[0] << "for (int i = 0; i <  " << k << " ; i++)" << std::endl
                   << this->indent[0] << "{" << std::endl
                   << this->indent[1] << "//Calculate density" << std::endl
                   << this->indent[1] << "endwert=0;" << std::endl
                   << this->indent[1] << "int nachbar=nodes[index* " << k << " +i];" << std::endl;
      sourceStream << this->indent[1] << "if (nachbar < 0) {" << std::endl;
      sourceStream << this->indent[2] << "continue;" << std::endl;
      sourceStream << this->indent[1] << "}" << std::endl;
      sourceStream << this->indent[1] << "for (int gridpoint=0;gridpoint< " << gridSize
                   << " ;gridpoint++)" << std::endl
                   << this->indent[1] << "{" << std::endl
                   << this->indent[2] << "wert=1;" << std::endl
                   << this->indent[2] << "for (int dimension=0;dimension< " << dimensions
                   << ";dimension++)" << std::endl
                   << this->indent[2] << "{" << std::endl
                   << this->indent[3] << "" << this->floatType() << " dimension_point=0;"
                   << std::endl
                   << this->indent[4] << "dimension_point=data[dimension+nachbar* " << dimensions
                   << "]+" << std::endl
                   << this->indent[4] << "(data[global_index* " << dimensions
                   << "+dimension]-data[dimension+nachbar* " << dimensions << "])*0.5;" << std::endl
                   << this->indent[3] << "wert*=get_u(dimension_point,"
                   << "starting_points[gridpoint*2* " << dimensions << "+2*dimension]," << std::endl
                   << this->indent[3] << "starting_points[gridpoint*2* " << dimensions
                   << "+2*dimension+1]);" << std::endl
                   << this->indent[2] << "}" << std::endl
                   << this->indent[2] << "endwert+=wert*alphas[gridpoint];" << std::endl
                   << this->indent[1] << "}" << std::endl
                   << this->indent[1] << "if (endwert< " << threshold << " )" << std::endl
                   << this->indent[1] << "{" << std::endl
                   << this->indent[2] << "nodes[ " << k << " *index + i] = -2;" << std::endl
                   << this->indent[1] << "}" << std::endl
                   << this->indent[0] << "}" << std::endl
                   << this->indent[0] << "endwert=0;" << std::endl
                   << this->indent[0] << "for (int gridpoint=0;gridpoint< " << gridSize
                   << " ;gridpoint++)" << std::endl
                   << this->indent[0] << "{" << std::endl
                   << this->indent[1] << "wert=1;" << std::endl
                   << this->indent[1] << "for (int dimension=0;dimension< " << dimensions
                   << ";dimension++)" << std::endl
                   << this->indent[1] << "{" << std::endl
                   << this->indent[2] << "wert*=get_u(data[global_index* " << dimensions
                   << "+dimension],starting_points[gridpoint*2* " << dimensions << "+2*dimension],"
                   << std::endl
                   << this->indent[3] << "starting_points[gridpoint*2* " << dimensions
                   << "+2*dimension+1]);" << std::endl
                   << this->indent[1] << "}" << std::endl
                   << this->indent[1] << "endwert+=wert*alphas[gridpoint];" << std::endl
                   << this->indent[0] << "}" << std::endl
                   << this->indent[0] << "if (endwert< " << threshold << " )" << std::endl
                   << this->indent[0] << "{" << std::endl
                   << this->indent[1] << "for (int i = 0; i <  " << k << " ; i++)" << std::endl
                   << this->indent[1] << "{" << std::endl
                   << this->indent[2] << "nodes[ " << k << " *index + i] = -1;" << std::endl
                   << this->indent[1] << "}" << std::endl
                   << this->indent[0] << "}" << std::endl;
      sourceStream << this->indent[0] << "}" << std::endl;
      sourceStream << "}" << std::endl;
    } else {
      sourceStream << "" << this->floatType() << " get_u(const " << this->floatType()
                   << " grenze, const int index, const int level_2) {" << std::endl;
      sourceStream << this->indent[0] << "" << this->floatType() << " ret = level_2;" << std::endl;
      sourceStream << this->indent[0] << "ret *= grenze;" << std::endl;
      sourceStream << this->indent[0] << "ret -= index;" << std::endl;
      sourceStream << this->indent[0] << "ret = fabs(ret);" << std::endl;
      sourceStream << this->indent[0] << "ret = 1 - ret;" << std::endl;
      sourceStream << this->indent[0] << "if (ret < 0.0) ret = 0.0;" << std::endl;
      sourceStream << this->indent[0] << "return ret;" << std::endl;
      sourceStream << "}" << std::endl;

      sourceStream
          << "void kernel __attribute__((reqd_work_group_size(" << local_size
          << ", 1, 1))) removeEdges(__global int *nodes, __global const int *starting_points,"
          << std::endl;
      sourceStream << this->indent[1] << "__global const " << this->floatType()
                   << " *data, __global const " << this->floatType() << " *alphas, int startid) {"
                   << std::endl;
      sourceStream << this->indent[0] << "size_t global_index = startid + get_global_id(0);"
                   << std::endl
                   << std::endl;

      sourceStream << this->indent[0] << "" << this->floatType() << " eval_locations[(" << k
                   << " + 1) * " << dimensions << "];  // (k + 1) * " << dimensions << ""
                   << std::endl;
      sourceStream << this->indent[0] << "" << this->floatType() << " evals[" << k
                   << " + 1];                  // (k + 1)-many results" << std::endl;
      sourceStream << this->indent[0] << "for (int cur_eval = 0; cur_eval < " << k
                   << " + 1; cur_eval += 1) {" << std::endl;
      sourceStream << this->indent[1] << "evals[cur_eval] = 0.0" << this->constSuffix() << ";"
                   << std::endl;
      sourceStream << this->indent[0] << "}" << std::endl << std::endl;

      sourceStream << this->indent[0] << "// loading own data into registers" << std::endl;
      sourceStream << this->indent[0] << "for (size_t d = 0; d < " << dimensions << "; d += 1) {"
                   << std::endl;
      sourceStream << this->indent[1] << "if (global_index < " << data_size << ") {" << std::endl;
      sourceStream << this->indent[2] << "eval_locations[" << k << " * " << dimensions
                   << " + d] = data[global_index * " << dimensions << " + d];" << std::endl;
      sourceStream << this->indent[1] << "} else {" << std::endl;
      sourceStream << this->indent[2] << "eval_locations[" << k << " * " << dimensions
                   << " + d] = 0.0" << this->constSuffix() << ";" << std::endl;
      sourceStream << this->indent[1] << "}" << std::endl;
      sourceStream << this->indent[0] << "}" << std::endl;
      sourceStream << this->indent[0] << "// loading neighbors into registers" << std::endl;
      sourceStream << this->indent[0] << "for (size_t cur_k = 0; cur_k < " << k << "; cur_k += 1) {"
                   << std::endl;
      sourceStream << this->indent[1] << "int neighbor_index;" << std::endl;
      sourceStream << this->indent[1] << "if (global_index < " << data_size << ") {" << std::endl;
      sourceStream << this->indent[2] << "neighbor_index = nodes[get_global_id(0) * " << k
                   << " + cur_k];" << std::endl;
      sourceStream << this->indent[1] << "} else {" << std::endl;
      sourceStream << this->indent[2] << "neighbor_index = 0;" << std::endl;
      sourceStream << this->indent[1] << "}" << std::endl;
      sourceStream << this->indent[1] << "for (size_t d = 0; d < " << dimensions << "; d += 1) {"
                   << std::endl;
      sourceStream << this->indent[2] << "if (global_index < " << data_size
                   << " && neighbor_index >= 0) {" << std::endl;
      sourceStream << this->indent[3] << "" << this->floatType()
                   << " loc_dim = data[d + neighbor_index * " << dimensions << "];" << std::endl;
      sourceStream << this->indent[3] << "eval_locations[cur_k * " << dimensions
                   << " + d] = loc_dim + 0.5 * (eval_locations[" << k << " * " << dimensions
                   << " + d] - loc_dim);" << std::endl;
      sourceStream << this->indent[2] << "} else {" << std::endl;
      sourceStream << this->indent[3] << "eval_locations[cur_k * " << dimensions << " + d] = 0.0"
                   << this->constSuffix() << ";" << std::endl;
      sourceStream << this->indent[2] << "}" << std::endl;
      sourceStream << this->indent[1] << "}" << std::endl;
      sourceStream << this->indent[0] << "}" << std::endl << std::endl;

      sourceStream << this->indent[0] << "local " << this->floatType() << " grid_indices["
                   << local_cache_size << " * " << dimensions << "];   // local_size * "
                   << dimensions << "" << std::endl;
      sourceStream << this->indent[0] << "local " << this->floatType() << " grid_levels_2["
                   << local_cache_size << " * " << dimensions << "];  // local_size * "
                   << dimensions << "" << std::endl;
      sourceStream << this->indent[0] << "local " << this->floatType() << " grid_alpha["
                   << local_cache_size << "]; // local_size" << std::endl
                   << std::endl;

      sourceStream << this->indent[0] << "// evaluate in the middle of the edges" << std::endl;
      sourceStream << this->indent[0] << "for (int outer_grid_index = 0; outer_grid_index < "
                   << gridSize << ";" << std::endl;
      sourceStream << this->indent[2] << "outer_grid_index += " << local_cache_size << ") {"
                   << std::endl;
      sourceStream << this->indent[1] << "// cache next set of grid points in local memory"
                   << std::endl;
      sourceStream << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl;
      sourceStream << this->indent[1] << "int grid_index = outer_grid_index + get_local_id(0);"
                   << std::endl;
      sourceStream << this->indent[1] << "if (get_local_id(0) < " << local_cache_size
                   << " && grid_index < " << gridSize << ") {" << std::endl;
      sourceStream << this->indent[2] << "for (int d = 0; d < " << dimensions << "; d++) {"
                   << std::endl;

      sourceStream << this->indent[3] << "grid_indices[get_local_id(0) * " << dimensions
                   << " + d] =" << std::endl;
      sourceStream << this->indent[5] << "(" << this->floatType()
                   << ")starting_points[grid_index * 2 * " << dimensions << " + 2 * d];"
                   << std::endl;
      sourceStream << this->indent[3] << "grid_levels_2[get_local_id(0) * " << dimensions
                   << " + d] =" << std::endl;
      sourceStream << this->indent[5] << "(" << this->floatType()
                   << ")(1 << starting_points[grid_index * 2 * " << dimensions << " + 2 * d + 1]);"
                   << std::endl;
      sourceStream << this->indent[2] << "}" << std::endl;
      sourceStream << this->indent[2] << "grid_alpha[get_local_id(0)] = alphas[grid_index];"
                   << std::endl;
      sourceStream << this->indent[1] << "}" << std::endl;
      sourceStream << this->indent[1] << "barrier(CLK_LOCAL_MEM_FENCE);" << std::endl << std::endl;

      sourceStream << this->indent[1] << "for (int inner_grid_index = 0; inner_grid_index < "
                   << local_cache_size
                   << "; "
                      "inner_grid_index += 1) {"
                   << std::endl;
      sourceStream << this->indent[2] << "if (outer_grid_index + inner_grid_index >= " << gridSize
                   << ") {" << std::endl;
      sourceStream << this->indent[3] << "break;" << std::endl;
      sourceStream << this->indent[2] << "}" << std::endl;
      sourceStream << this->indent[2] << "for (int cur_eval = 0; cur_eval < " << k
                   << " + 1; cur_eval += 1) {" << std::endl;
      sourceStream << this->indent[3] << "" << this->floatType() << " eval_1d = 1.0"
                   << this->constSuffix() << ";" << std::endl;
      sourceStream << this->indent[3] << "for (int d = 0; d < " << dimensions << "; d += 1) {"
                   << std::endl;
      sourceStream << this->indent[4] << "eval_1d *=" << std::endl;
      sourceStream << this->indent[6] << "get_u(eval_locations[cur_eval * " << dimensions
                   << " + d], grid_indices[inner_grid_index * " << dimensions << " + d],"
                   << std::endl;
      sourceStream << this->indent[6] << "grid_levels_2[inner_grid_index * " << dimensions
                   << " + d]);" << std::endl;
      sourceStream << this->indent[3] << "}" << std::endl;
      sourceStream << this->indent[3]
                   << "evals[cur_eval] += eval_1d * grid_alpha[inner_grid_index];" << std::endl;
      sourceStream << this->indent[2] << "}" << std::endl;
      sourceStream << this->indent[1] << "}" << std::endl;
      sourceStream << this->indent[0] << "}" << std::endl << std::endl;

      sourceStream << this->indent[0] << "if (global_index < " << data_size << ") {" << std::endl;
      sourceStream << this->indent[0] << "// point itself below density?" << std::endl;
      sourceStream << this->indent[0] << "if (evals[" << k << "] < " << threshold
                   << this->constSuffix() << ") {" << std::endl;
      sourceStream << this->indent[1] << "// invalidate all neighbors, point now isolated"
                   << std::endl;
      sourceStream << this->indent[1] << "for (int cur_k = 0; cur_k < " << k << "; cur_k += 1) {"
                   << std::endl;
      sourceStream << this->indent[2] << "nodes[" << k << " * get_global_id(0) + cur_k] = -1;"
                   << std::endl;
      sourceStream << this->indent[1] << "}" << std::endl;
      sourceStream << this->indent[0] << "} else {" << std::endl;
      sourceStream << this->indent[1] << "// point part of the graph, check individual edges"
                   << std::endl;
      sourceStream << this->indent[1] << "for (size_t cur_k = 0; cur_k < " << k << "; cur_k += 1) {"
                   << std::endl;
      sourceStream << this->indent[2] << "if (evals[cur_k] < " << threshold << this->constSuffix()
                   << ") {" << std::endl;
      sourceStream << this->indent[3] << "nodes[get_global_id(0) * " << k << " + cur_k] = -2;"
                   << std::endl;
      sourceStream << this->indent[2] << "}" << std::endl;
      sourceStream << this->indent[1] << "}" << std::endl;
      sourceStream << this->indent[0] << "}" << std::endl;
      sourceStream << this->indent[0] << "}" << std::endl;
      sourceStream << "}" << std::endl;
    }

    if (kernelConfiguration["WRITE_SOURCE"].getBool() &&
        !kernelConfiguration["REUSE_SOURCE"].getBool()) {
      this->writeSource("DensityOCLMultiPlatform_prune_graph.cl", sourceStream.str());
    }

    return sourceStream.str();
  }
};

}  // namespace DensityOCLMultiPlatform
}  // namespace datadriven
}  // namespace sgpp
