// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#include <unistd.h>

#include "sgpp/base/grid/generation/GridGenerator.hpp"
#include "sgpp/base/grid/generation/functors/SurplusCoarseningFunctor.hpp"
#include "sgpp/base/grid/generation/functors/SurplusRefinementFunctor.hpp"
#include <sgpp/datadriven/operation/hash/OperationMPI/MPIEnviroment.hpp>
#include <sgpp/base/datatypes/DataMatrix.hpp>
#include <sgpp/base/datatypes/DataVector.hpp>
#include <sgpp/base/grid/Grid.hpp>
#include <sgpp/globaldef.hpp>
#include <sgpp/solver/sle/ConjugateGradients.hpp>
#include "sgpp/datadriven/operation/hash/OperationMPI/OperationDensityMultMPI.hpp"
#include "sgpp/datadriven/operation/hash/OperationMPI/OperationDensityRhsMPI.hpp"
#include "sgpp/datadriven/operation/hash/OperationMPI/OperationPrunedGraphCreationMPI.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"

#include <experimental/filesystem>
#include <boost/program_options.hpp>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {

  // Init MPI enviroment - always has to be done first - capture slaves
  sgpp::datadriven::clusteringmpi::MPIEnviroment::init(argc, argv, true);
  {
    // Only rank 0 in here

    std::string datasetFileName;
    size_t level;
    double lambda;
    std::string configFileName;
    std::string MPIconfigFileName;
    std::string cluster_file;
    uint64_t k;
    double threshold;

    std::string scenario_name;
    size_t refinement_steps;
    size_t refinement_points;
    size_t coarsening_points;
    double coarsening_threshold;
    double epsilon;

    std::string rhs_erg_filename = "";
    std::string density_coefficients_filename = "";
    std::string pruned_knn_filename = "";

    bool verbose_mult = false;

    boost::program_options::options_description description("Allowed options");
    description.add_options()("help", "display help")(
      "datasetFileName", boost::program_options::value<std::string>(&datasetFileName),
      "training data set as an arff file")(
      "level", boost::program_options::value<size_t>(&level)->default_value(4),
      "level of the sparse grid used for density estimation")(
      "lambda", boost::program_options::value<double>(&lambda)->default_value(0.000001),
      "regularization for density estimation")(
      "config", boost::program_options::value<std::string>(&configFileName),
      "OpenCL and kernel configuration file")(
      "MPIconfig", boost::program_options::value<std::string>(&MPIconfigFileName),
      "MPI configuration file. Should be a json file, specifying the connections of the network")(
      "k", boost::program_options::value<uint64_t>(&k)->default_value(5),
      "specifies number of neighbors for kNN algorithm")(
      "cluster_file",
      boost::program_options::value<std::string>(&cluster_file)->default_value(""),
      "Output file for the detected clusters. None if empty.")(
      "write_graphs", boost::program_options::value<std::string>(&scenario_name),
      "output the clustering steps into files")(
      "refinement_steps",
      boost::program_options::value<uint64_t>(&refinement_steps)->default_value(0),
      "number of refinment steps for density estimation")(
      "refinement_points",
      boost::program_options::value<uint64_t>(&refinement_points)->default_value(0),
      "number of points to refinement during density estimation")(
      "coarsen_points",
      boost::program_options::value<uint64_t>(&coarsening_points)->default_value(0),
      "number of points to coarsen during density estimation")(
      "rhs_erg_file",
      boost::program_options::value<std::string>(&rhs_erg_filename),
      "Filename where the final rhs values will be written.")(
      "density_coefficients_file",
      boost::program_options::value<std::string>(&density_coefficients_filename),
      "Filename where the final grid coefficients for the density function will be written.")(
      "pruned_knn_file",
      boost::program_options::value<std::string>(&pruned_knn_filename),
      "Filename for the pruned knn graph")(
      "epsilon",
      boost::program_options::value<double>(&epsilon)->default_value(0.0001),
      "Exit criteria for the solver. Usually ranges from 0.001 to 0.0001.")(
      "threshold", boost::program_options::value<double>(&threshold)->default_value(0.0),
      "threshold for sparse grid function for removing edges")(
      "coarsen_threshold",
      boost::program_options::value<double>(&coarsening_threshold)->default_value(1000.0),
      "for density estimation, only surpluses below threshold are coarsened")(
      "verbose_mult",
      boost::program_options::value<bool>(&verbose_mult)->default_value(false),
      "Prints times per multiplication");

    boost::program_options::variables_map variables_map;
    boost::program_options::parsed_options options = parse_command_line(argc, argv,
    description);
    boost::program_options::store(options, variables_map);
    boost::program_options::notify(variables_map);

    std::cout << std::endl << std::endl;
    std::cout << "Arguments (Scenario):" << std::endl;
    std::cout << "--------------------- " << std::endl;
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return 0;
    }

    if (variables_map.count("datasetFileName") == 0) {
      std::cerr << "error: option \"datasetFileName\" not specified" << std::endl;
      return 1;
    } else {
      std::experimental::filesystem::path datasetFilePath(datasetFileName);
      if (!std::experimental::filesystem::exists(datasetFilePath)) {
        std::cerr << "error: dataset file does not exist: " << datasetFileName << std::endl;
        return 1;
      }
      std::cout << "datasetFileName: " << datasetFileName << std::endl;
    }

    if (variables_map.count("MPIconfig") == 0) {
      std::cerr << "error: option \"MPIconfig\" not specified" << std::endl;
      return 1;
    } else {
      std::experimental::filesystem::path configFilePath(MPIconfigFileName);
      if (!std::experimental::filesystem::exists(configFilePath)) {
        std::cerr << "error: MPI config file does not exist: " << MPIconfigFileName << std::endl;
        return 1;
      }
      std::cout << "MPI configuration file: " << MPIconfigFileName << std::endl;
    }

    if (variables_map.count("config") == 0) {
      std::cerr << "error: option \"config\" not specified" << std::endl;
      return 1;
    } else {
      std::experimental::filesystem::path configFilePath(configFileName);
      if (!std::experimental::filesystem::exists(configFilePath)) {
        std::cerr << "error: config file does not exist: " << configFileName << std::endl;
        return 1;
      }
      std::cout << "OpenCL configuration file: " << configFileName << std::endl;
    }

    if (variables_map.count("level") == 0) {
      std::cerr << "error: option \"level\" not specified" << std::endl;
      return 1;
    } else {
      std::cout << "level: " << level << std::endl;
    }

    if (variables_map.count("lambda") == 0) {
      std::cerr << "error: option \"lambda\" not specified" << std::endl;
      return 1;
    } else {
      std::cout << "lambda: " << lambda << std::endl;
    }

    if (variables_map.count("k") == 0) {
      std::cerr << "error: option \"k\" not specified" << std::endl;
      return 1;
    } else {
      std::cout << "k: " << k << std::endl;
    }

    if (variables_map.count("threshold") == 0) {
      std::cerr << "error: option \"threshold\" not specified" << std::endl;
      return 1;
    } else {
      std::cout << "threshold: " << threshold << std::endl;
    }
    std::cout << std::endl << std::endl;

    // Measure times
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::system_clock::now();

    // setup MPI network according to config file
    std::cout << "Setup:" << std::endl;
    std::cout << "------ " << std::endl;
    sgpp::base::OperationConfiguration network_conf(MPIconfigFileName);
    sgpp::datadriven::clusteringmpi::MPIEnviroment::connect_nodes(network_conf);

    // Loading dataset
    long offset = 0;
    sgpp::datadriven::ARFFTools::readARFFHeader(datasetFileName, offset);
    sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(datasetFileName);
    sgpp::base::DataMatrix &dataset = data.getData();
    size_t dim = data.getDimension();

    // Create Grid
    std::chrono::time_point<std::chrono::high_resolution_clock> grid_creation_start,
        grid_creation_end;
    sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(dim);
    sgpp::base::GridGenerator &gridGen = grid->getGenerator();
    gridGen.regular(level);
    size_t gridsize = grid->getStorage().getSize();
    sgpp::base::DataVector alpha(gridsize);
    sgpp::base::DataVector result(gridsize);
    sgpp::base::DataVector rhs(gridsize);
    alpha.setAll(1.0);
    std::cerr << "Grid created! Number of grid points:     " << gridsize << std::endl;
    std::cout << std::endl << std::endl;

    std::chrono::time_point<std::chrono::high_resolution_clock> rhs_start, rhs_end;
    std::chrono::time_point<std::chrono::high_resolution_clock> solver_start, solver_end;
    {
      // Create right hand side vector
      std::chrono::time_point<std::chrono::high_resolution_clock> rhs_start, rhs_end;
      rhs_start = std::chrono::system_clock::now();
      std::cout << "Create right-hand side of density equation: " << std::endl;
      std::cout << "-------------------------------------------- " << std::endl;
      sgpp::datadriven::clusteringmpi::OperationDensityRhsMPI rhs_op(*grid, datasetFileName,
                                                                     configFileName);
      rhs_op.generate_b(rhs);
      rhs_end = std::chrono::system_clock::now();
      std::cout << std::endl << std::endl;

      // Solve for alpha vector via CG solver
      std::cout << "Solve for alpha: " << std::endl;
      std::cout << "--------------- " << std::endl;
      sgpp::datadriven::clusteringmpi::OperationDensityMultMPI mult_op(*grid, lambda,
                                                                       configFileName, verbose_mult);
      solver_start = std::chrono::system_clock::now();
      alpha.setAll(1.0);
      sgpp::solver::ConjugateGradients solver(1000, epsilon);
      solver.solve(mult_op, alpha, rhs, false, true);
      solver_end = std::chrono::system_clock::now();
    }
    for (size_t i = 0; i < refinement_steps; i++) {
      if (refinement_points > 0) {
        sgpp::base::SurplusRefinementFunctor refine_func(alpha, refinement_points);
        gridGen.refine(refine_func);
        size_t old_size = alpha.getSize();

        // adjust alpha to refined grid
        alpha.resize(grid->getSize());
        for (size_t j = old_size; j < alpha.getSize(); j++) {
          alpha[j] = 0.0;
        }

        // regenerate b with refined grid
        rhs.resize(grid->getSize());
        for (size_t j = old_size; j < alpha.getSize(); j++) {
          rhs[j] = 0.0;
        }

        // Create right hand side vector
        std::chrono::time_point<std::chrono::high_resolution_clock> rhs_start, rhs_end;
        rhs_start = std::chrono::system_clock::now();
        std::cout << "Create right-hand side of density equation: " << std::endl;
        std::cout << "-------------------------------------------- " << std::endl;
        sgpp::datadriven::clusteringmpi::OperationDensityRhsMPI rhs_op(*grid, datasetFileName,
                                                                       configFileName);
        rhs_op.generate_b(rhs);
        rhs_end = std::chrono::system_clock::now();
        std::cout << std::endl << std::endl;

        // Solve for alpha vector via CG solver
        std::cout << "Solve for alpha: " << std::endl;
        std::cout << "--------------- " << std::endl;
        sgpp::datadriven::clusteringmpi::OperationDensityMultMPI mult_op(*grid, lambda,
                                                                         configFileName, verbose_mult);
        std::chrono::time_point<std::chrono::high_resolution_clock> solver_start, solver_end;
        solver_start = std::chrono::system_clock::now();
        alpha.setAll(1.0);
        sgpp::solver::ConjugateGradients solver(1000, epsilon);
        solver.solve(mult_op, alpha, rhs, false, true);
        solver_end = std::chrono::system_clock::now();
      }
      if (coarsening_points > 0) {
        size_t grid_size_before_coarsen = grid->getSize();
        sgpp::base::SurplusCoarseningFunctor coarsen_func(alpha, coarsening_points,
                                                          coarsening_threshold);
        gridGen.coarsen(coarsen_func, alpha);

        size_t grid_size_after_coarsen = grid->getSize();
        std::cout << "coarsen: removed " << (grid_size_before_coarsen - grid_size_after_coarsen)
                  << " grid points" << std::endl;

        // adjust alpha to coarsen grid
        alpha.resize(grid->getSize());

        // regenerate b with coarsen grid
        rhs.resize(grid->getSize());

        // Create right hand side vector
        rhs_start = std::chrono::system_clock::now();
        std::cout << "Create right-hand side of density equation: " << std::endl;
        std::cout << "-------------------------------------------- " << std::endl;
        sgpp::datadriven::clusteringmpi::OperationDensityRhsMPI rhs_op(*grid, datasetFileName,
                                                                       configFileName);
        rhs_op.generate_b(rhs);
        rhs_end = std::chrono::system_clock::now();
        std::cout << std::endl << std::endl;

        // Solve for alpha vector via CG solver
        std::cout << "Solve for alpha: " << std::endl;
        std::cout << "--------------- " << std::endl;
        sgpp::datadriven::clusteringmpi::OperationDensityMultMPI mult_op(*grid, lambda,
                                                                         configFileName, verbose_mult);
        std::chrono::time_point<std::chrono::high_resolution_clock> solver_start, solver_end;
        solver_start = std::chrono::system_clock::now();
        alpha.setAll(1.0);
        sgpp::solver::ConjugateGradients solver(1000, epsilon);
        solver.solve(mult_op, alpha, rhs, false, true);
        solver_end = std::chrono::system_clock::now();
      }
    }
    gridsize = grid->getSize();
    std::cerr << "Number of grid points:     " << gridsize << std::endl;



    double max = alpha.max();
    double min = alpha.min();
    for (size_t i = 0; i < gridsize; i++) alpha[i] = alpha[i] * 1.0 / (max - min);
    std::cout << std::endl << std::endl;


    // Output final rhs values
    if (rhs_erg_filename != "") {
      std::ofstream out_rhs(rhs_erg_filename);
      for (size_t i = 0; i < grid->getSize(); ++i) {
        out_rhs << rhs[i] << std::endl;
      }
      out_rhs.close();
    }
    // Output final coefficients
    if (density_coefficients_filename != "") {
      std::ofstream out_coefficients(density_coefficients_filename);
      for (size_t i = 0; i < grid->getSize(); ++i) {
        out_coefficients << alpha[i] << std::endl;
      }
      out_coefficients.close();
    }

    // Create and prune knn graph
    std::cout << "Create and prune graph: " << std::endl;
    std::cout << "----------------------- " << std::endl;
    std::chrono::time_point<std::chrono::high_resolution_clock> create_knn_start, create_knn_end;
    create_knn_start = std::chrono::system_clock::now();
    sgpp::datadriven::clusteringmpi::OperationPrunedGraphCreationMPI graph_op(
        *grid, alpha, datasetFileName, k, threshold, configFileName);
    std::vector<int64_t> knn_graph;
    graph_op.create_graph(knn_graph);
    create_knn_end = std::chrono::system_clock::now();
    std::cout << std::endl << std::endl;
    auto print_knn_graph = [&dataset, k](std::string filename, std::vector<int64_t> &graph) {
      std::ofstream out_graph(filename);
      for (size_t i = 0; i < dataset.getNrows(); ++i) {
        bool first = true;
        for (size_t j = 0; j < k; ++j) {
          if (graph[i * k + j] == -1) {
            continue;
          }
          if (!first) {
            out_graph << ", ";
          } else {
            first = false;
          }
          out_graph << graph[i * k + j];
        }
        out_graph << std::endl;
      }
      out_graph.close();
    };
    // output for opencl/mpi comparison script
    if (pruned_knn_filename != "") {
      print_knn_graph(pruned_knn_filename, knn_graph);
    }

    std::cout << "Find clusters in pruned graph: " << std::endl;
    std::cout << "------------------------------ " << std::endl;
    std::chrono::time_point<std::chrono::high_resolution_clock> find_clusters_start,
        find_clusters_end;
    find_clusters_start = std::chrono::system_clock::now();
    std::vector<int64_t> node_cluster_map;
    sgpp::datadriven::DensityOCLMultiPlatform::
        OperationCreateGraphOCL::neighborhood_list_t clusters;
    sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::find_clusters(
        knn_graph, k, node_cluster_map, clusters);
    find_clusters_end = std::chrono::system_clock::now();
    // Output ergs
    std::cout << "detected clusters: " << clusters.size() << std::endl;
    if (cluster_file != "") {
      std::ofstream out_cluster_map(cluster_file);
      for (size_t i = 0; i < dataset.getNrows(); ++i) {
        out_cluster_map << node_cluster_map[i] << std::endl;
      }
      out_cluster_map.close();
    }
    std::cout << std::endl << std::endl;


    // Calc time
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);

    // Output timings
    std::cout << "Runtimes: " << std::endl;
    std::cout << "--------- " << std::endl;
    std::cout << "rhs creation duration: "
              << std::chrono::duration_cast<std::chrono::seconds>(rhs_end - rhs_start).count()
              << "s" << std::endl;
    std::cout << "solver duration: "
              << std::chrono::duration_cast<std::chrono::seconds>(solver_end - solver_start).count()
              << "s" << std::endl;
    std::cout << "create knn operation duration: "
              << std::chrono::duration_cast<std::chrono::seconds>(create_knn_end - create_knn_start)
        .count()
              << "s" << std::endl;
    std::cout << "find clusters duration: "
              << std::chrono::duration_cast<std::chrono::seconds>(find_clusters_end -
                                                                  find_clusters_start).count()
              << "s" << std::endl;
    std::cout << "finished computation at " << std::ctime(&end_time)
              << "elapsed time: " << elapsed_seconds.count() << "s\n";
    std::cout << std::endl << std::endl;

    // Cleanup MPI enviroment
    std::cout << "Finishing: " << std::endl;
    std::cout << "---------- " << std::endl;
  }
  sgpp::datadriven::clusteringmpi::MPIEnviroment::release();
  return 0;
}
