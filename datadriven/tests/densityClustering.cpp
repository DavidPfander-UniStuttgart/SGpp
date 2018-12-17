// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#if USE_OCL == 1

#define BOOST_TEST_DYN_LINK
#include <zlib.h>
#include <boost/test/unit_test.hpp>

#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>
#include "sgpp/datadriven/operation/hash/OperationCreateGraphOCL/OperationCreateGraphOCLSingleDevice.hpp"
#include "sgpp/datadriven/operation/hash/OperationDensityOCLMultiPlatform/OpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationPruneGraphOCL/OpFactory.hpp"
#include "sgpp/solver/sle/ConjugateGradients.hpp"

#include "sgpp/base/grid/generation/functors/SurplusRefinementFunctor.hpp"
#include "sgpp/base/operation/BaseOpFactory.hpp"
#include "sgpp/base/tools/ConfigurationParameters.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"
#include "sgpp/globaldef.hpp"
#include "test_datadrivenCommon.hpp"
void multiply_and_test(std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters,
                       std::vector<double> &mult_optimal_result,
                       std::shared_ptr<sgpp::base::OCLManagerMultiPlatform> manager,
                       sgpp::base::Grid &grid) {
  size_t gridsize = grid.getStorage().getSize();
  // Create vectors for multiplication
  sgpp::base::DataVector alpha(gridsize);
  sgpp::base::DataVector result(gridsize);
  alpha.setAll(1.0);
  // Create operation
  auto mult_operation = std::make_unique<
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
        grid, 2, manager, parameters, 0.001);
  // Execute multiplication
  mult_operation->mult(alpha, result);
  // Compare results with optimal results
  for (size_t i = 0; i < gridsize; ++i) {
    BOOST_CHECK_CLOSE(mult_optimal_result[i], result[i], 0.001);
  }
}

BOOST_AUTO_TEST_SUITE(TestClusteringOpenCL)

BOOST_AUTO_TEST_CASE(DensityRHSOpenCL_Default) {
  std::cout << "Starting rhs kernel tests" << std::endl;
  // Load correct results for comparison
  std::vector<double> rhs_optimal_result;
  std::ifstream rhs_in("datadriven/tests/data/clustering_test_data/rhs_erg_dim2_depth11.txt");
  if (rhs_in) {
    double value;
    while (rhs_in >> value) rhs_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density rhs result file is missing!"));
  }
  rhs_in.close();

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  size_t gridsize = grid->getStorage().getSize();

  // Load dataset for test scenario
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(
      "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
  sgpp::base::DataMatrix &dataset = data.getData();

  {
    // Create OCL configuration
    std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
        getConfigurationDefaultsSingleDevice();
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
    std::cout << "Testing default rhs kernel ..." << std::endl;
    for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
      json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::Node &deviceNode = platformNode["DEVICES"][deviceName];
        const std::string &kernelName = "cscheme";
        json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
        kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
        kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", false);
      }
    }
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
    // Create OpenCL Manager
    auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);
    // Create operation
    auto operation_rhs = std::make_unique<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
          *grid, 2, manager, parameters, 0.001);

    sgpp::base::DataVector b(gridsize);
    operation_rhs->generateb(dataset, b);
    for (size_t i = 0; i < gridsize; ++i) {
      BOOST_CHECK_CLOSE(rhs_optimal_result[i], b[i], 0.001);
    }
  }
}

BOOST_AUTO_TEST_CASE(DensityRHSOpenCL_LocalMemory) {
  std::cout << "Starting rhs kernel tests" << std::endl;
  // Load correct results for comparison
  std::vector<double> rhs_optimal_result;
  std::ifstream rhs_in("datadriven/tests/data/clustering_test_data/rhs_erg_dim2_depth11.txt");
  if (rhs_in) {
    double value;
    while (rhs_in >> value) rhs_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density rhs result file is missing!"));
  }
  rhs_in.close();

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  size_t gridsize = grid->getStorage().getSize();

  // Load dataset for test scenario
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(
									 "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
  sgpp::base::DataMatrix &dataset = data.getData();

  {
    // Create OCL configuration
    std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
    std::cout << "Testing rhs kernel with local memory ..." << std::endl;
    for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
      json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::Node &deviceNode = platformNode["DEVICES"][deviceName];
        const std::string &kernelName = "cscheme";
        json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
        kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
        kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", false);
	// kernelNode.replaceIDAttr("WRITE_SOURCE", true);
	// kernelNode.replaceIDAttr("KERNEL_EVAL_BLOCKING", UINT64_C(1));
      }
    }
    // Create OpenCL Manager
    auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);
    // Create operation
    auto operation_rhs = std::make_unique<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
											   *grid, 2, manager, parameters, 0.001);

    sgpp::base::DataVector b(gridsize);
    operation_rhs->generateb(dataset, b);
    for (size_t i = 0; i < gridsize; ++i) {
      BOOST_CHECK_CLOSE(rhs_optimal_result[i], b[i], 0.001);
    }
  }
}

BOOST_AUTO_TEST_CASE(DensityRHSOpenCL_LocalMemoryCompression) {
  std::cout << "Starting rhs kernel tests" << std::endl;
  // Load correct results for comparison
  std::vector<double> rhs_optimal_result;
  std::ifstream rhs_in("datadriven/tests/data/clustering_test_data/rhs_erg_dim2_depth11.txt");
  if (rhs_in) {
    double value;
    while (rhs_in >> value) rhs_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density rhs result file is missing!"));
  }
  rhs_in.close();

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  size_t gridsize = grid->getStorage().getSize();

  // Load dataset for test scenario
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(
      "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
  sgpp::base::DataMatrix &dataset = data.getData();
  {
    // Create OCL configuration
    std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
        getConfigurationDefaultsSingleDevice();
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
    std::cout << "Testing rhs kernel with local memory and compression ..." << std::endl;
    for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
      json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::Node &deviceNode = platformNode["DEVICES"][deviceName];
        const std::string &kernelName = "cscheme";
        json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
        kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
        kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
	kernelNode.replaceIDAttr("WRITE_SOURCE", true);
      }
    }
    // Create OpenCL Manager
    auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);
    // Create operation
    auto operation_rhs = std::make_unique<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
          *grid, 2, manager, parameters, 0.001);

    sgpp::base::DataVector b(gridsize);
    operation_rhs->generateb(dataset, b);
    for (size_t i = 0; i < gridsize; ++i) {
      BOOST_CHECK_CLOSE(rhs_optimal_result[i], b[i], 0.001);
    }
  }
}

// BOOST_AUTO_TEST_CASE(DensityRHSOpenCL_LocalMemoryCompression) {
//   std::cout << "Starting rhs kernel tests" << std::endl;
//   // Load correct results for comparison
//   std::vector<double> rhs_optimal_result;
//   std::ifstream rhs_in("datadriven/tests/data/clustering_test_data/rhs_erg_dim2_depth11.txt");
//   if (rhs_in) {
//     double value;
//     while (rhs_in >> value) rhs_optimal_result.push_back(value);
//   } else {
//     BOOST_THROW_EXCEPTION(std::runtime_error("Density rhs result file is missing!"));
//   }
//   rhs_in.close();

//   // Create grid for test scenario
//   sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
//   sgpp::base::GridGenerator &gridGen = grid->getGenerator();
//   gridGen.regular(11);
//   size_t gridsize = grid->getStorage().getSize();

//   // Load dataset for test scenario
//   sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(
//       "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
//   sgpp::base::DataMatrix &dataset = data.getData();
//   {
//     // Create OCL configuration
//     std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
//         getConfigurationDefaultsSingleDevice();
//     sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
//     std::cout << "Testing rhs kernel with local memory and compression ..." << std::endl;
//     for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
//       json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
//       for (std::string &deviceName : platformNode["DEVICES"].keys()) {
//         json::Node &deviceNode = platformNode["DEVICES"][deviceName];
//         const std::string &kernelName = "cscheme";
//         json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
//         kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
//         kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
//       }
//     }
//     // Create OpenCL Manager
//     auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);
//     // Create operation
//     auto operation_rhs = std::make_unique<
//       sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
//           *grid, 2, manager, parameters, 0.001);

//     sgpp::base::DataVector b(gridsize);
//     operation_rhs->generateb(dataset, b);
//     for (size_t i = 0; i < gridsize; ++i) {
//       BOOST_CHECK_CLOSE(rhs_optimal_result[i], b[i], 0.001);
//     }
//   }
// }

BOOST_AUTO_TEST_CASE(DensityRHSOpenCL_Compression32) {
  std::cout << "Starting rhs kernel tests" << std::endl;
  // Load correct results for comparison
  std::vector<double> rhs_optimal_result;
  std::ifstream rhs_in("datadriven/tests/data/clustering_test_data/rhs_erg_dim2_depth11.txt");
  if (rhs_in) {
    double value;
    while (rhs_in >> value) rhs_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density rhs result file is missing!"));
  }
  rhs_in.close();

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  size_t gridsize = grid->getStorage().getSize();

  // Load dataset for test scenario
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(
									 "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
  sgpp::base::DataMatrix &dataset = data.getData();
  {
    // Create OCL configuration
    std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
    std::cout << "Testing rhs kernel with unsigned compression type unsigned int and without local memory ..." << std::endl;
    for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
      json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::Node &deviceNode = platformNode["DEVICES"][deviceName];
        const std::string &kernelName = "cscheme";
        json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
        kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
        kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
        kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
      }
    }
    // Create OpenCL Manager
    auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);
    // Create operation
    auto operation_rhs = std::make_unique<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
											   *grid, 2, manager, parameters, 0.001);

    sgpp::base::DataVector b(gridsize);
    operation_rhs->generateb(dataset, b);
    for (size_t i = 0; i < gridsize; ++i) {
      BOOST_CHECK_CLOSE(rhs_optimal_result[i], b[i], 0.001);
    }
  }
}

BOOST_AUTO_TEST_CASE(DensityRHSOpenCL_LocalMemoryCompression32) {
  std::cout << "Starting rhs kernel tests" << std::endl;
  // Load correct results for comparison
  std::vector<double> rhs_optimal_result;
  std::ifstream rhs_in("datadriven/tests/data/clustering_test_data/rhs_erg_dim2_depth11.txt");
  if (rhs_in) {
    double value;
    while (rhs_in >> value) rhs_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density rhs result file is missing!"));
  }
  rhs_in.close();

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  size_t gridsize = grid->getStorage().getSize();

  // Load dataset for test scenario
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(
									 "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
  sgpp::base::DataMatrix &dataset = data.getData();
  {
    // Create OCL configuration
    std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
        getConfigurationDefaultsSingleDevice();
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
    std::cout << "Testing rhs kernel with unsigned compression type unsigned int and with local memory ..." << std::endl;
    for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
      json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::Node &deviceNode = platformNode["DEVICES"][deviceName];
        const std::string &kernelName = "cscheme";
        json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
        kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
        kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
        kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
      }
    }
    // Create OpenCL Manager
    auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);
    // Create operation
    auto operation_rhs = std::make_unique<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
          *grid, 2, manager, parameters, 0.001);

    sgpp::base::DataVector b(gridsize);
    operation_rhs->generateb(dataset, b);
    for (size_t i = 0; i < gridsize; ++i) {
      BOOST_CHECK_CLOSE(rhs_optimal_result[i], b[i], 0.001);
    }
  }
}

BOOST_AUTO_TEST_CASE(DensityRHSOpenCL_LocalMemoryCompression32Registers) {
  std::cout << "Starting rhs kernel tests" << std::endl;
  // Load correct results for comparison
  std::vector<double> rhs_optimal_result;
  std::ifstream rhs_in("datadriven/tests/data/clustering_test_data/rhs_erg_dim2_depth11.txt");
  if (rhs_in) {
    double value;
    while (rhs_in >> value) rhs_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density rhs result file is missing!"));
  }
  rhs_in.close();

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  size_t gridsize = grid->getStorage().getSize();

  // Load dataset for test scenario
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(
									 "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
  sgpp::base::DataMatrix &dataset = data.getData();
  {
    // Create OCL configuration
    std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
        getConfigurationDefaultsSingleDevice();
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
    std::cout << "Testing rhs kernel with unsigned compression type unsigned int, without "
              << "local memory and without compression registers ..." << std::endl;
    for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
      json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::Node &deviceNode = platformNode["DEVICES"][deviceName];
        const std::string &kernelName = "cscheme";
        json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
        kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
        kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
        kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
        kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
      }
    }
    // Create OpenCL Manager
    auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);
    // Create operation
    auto operation_rhs = std::make_unique<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
          *grid, 2, manager, parameters, 0.001);

    sgpp::base::DataVector b(gridsize);
    operation_rhs->generateb(dataset, b);
    for (size_t i = 0; i < gridsize; ++i) {
      BOOST_CHECK_CLOSE(rhs_optimal_result[i], b[i], 0.001);
    }
  }
}

BOOST_AUTO_TEST_CASE(DensityRHSOpenCL_LocalMemoryCompression32NoRegisters) {
  std::cout << "Starting rhs kernel tests" << std::endl;
  // Load correct results for comparison
  std::vector<double> rhs_optimal_result;
  std::ifstream rhs_in("datadriven/tests/data/clustering_test_data/rhs_erg_dim2_depth11.txt");
  if (rhs_in) {
    double value;
    while (rhs_in >> value) rhs_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density rhs result file is missing!"));
  }
  rhs_in.close();

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  size_t gridsize = grid->getStorage().getSize();

  // Load dataset for test scenario
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(
									 "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
  sgpp::base::DataMatrix &dataset = data.getData();
  {
    // Create OCL configuration
    std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
        getConfigurationDefaultsSingleDevice();
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
    std::cout << "Testing rhs kernel with unsigned compression type unsigned int, with "
              << "local memory and without compression registers ..." << std::endl;
    for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
      json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::Node &deviceNode = platformNode["DEVICES"][deviceName];
        const std::string &kernelName = "cscheme";
        json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
        kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
        kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
        kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
        kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
      }
    }
    // Create OpenCL Manager
    auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);
    // Create operation
    auto operation_rhs = std::make_unique<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
          *grid, 2, manager, parameters, 0.001);

    sgpp::base::DataVector b(gridsize);
    operation_rhs->generateb(dataset, b);
    for (size_t i = 0; i < gridsize; ++i) {
      BOOST_CHECK_CLOSE(rhs_optimal_result[i], b[i], 0.001);
    }
  }
}

BOOST_AUTO_TEST_CASE(DensityRHSOpenCL_Compression64NoRegisters) {
  std::cout << "Starting rhs kernel tests" << std::endl;
  // Load correct results for comparison
  std::vector<double> rhs_optimal_result;
  std::ifstream rhs_in("datadriven/tests/data/clustering_test_data/rhs_erg_dim2_depth11.txt");
  if (rhs_in) {
    double value;
    while (rhs_in >> value) rhs_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density rhs result file is missing!"));
  }
  rhs_in.close();

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  size_t gridsize = grid->getStorage().getSize();

  // Load dataset for test scenario
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(
									 "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
  sgpp::base::DataMatrix &dataset = data.getData();
  {
    // Create OCL configuration
    std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
        getConfigurationDefaultsSingleDevice();
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
    std::cout << "Testing rhs kernel with long compression type, without "
              << "local memory and without compression registers ..." << std::endl;
    for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
      json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::Node &deviceNode = platformNode["DEVICES"][deviceName];
        const std::string &kernelName = "cscheme";
        json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
        kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
        kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
        kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
        kernelNode.replaceIDAttr("COMPRESSION_TYPE", "uint64_t");
      }
    }
    // Create OpenCL Manager
    auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);
    // Create operation
    auto operation_rhs = std::make_unique<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
          *grid, 2, manager, parameters, 0.001);

    sgpp::base::DataVector b(gridsize);
    operation_rhs->generateb(dataset, b);
    for (size_t i = 0; i < gridsize; ++i) {
      BOOST_CHECK_CLOSE(rhs_optimal_result[i], b[i], 0.001);
    }
  }
}

BOOST_AUTO_TEST_CASE(DensityRHSOpenCL_LocalMemoryCompression64NoRegisters) {
  std::cout << "Starting rhs kernel tests" << std::endl;
  // Load correct results for comparison
  std::vector<double> rhs_optimal_result;
  std::ifstream rhs_in("datadriven/tests/data/clustering_test_data/rhs_erg_dim2_depth11.txt");
  if (rhs_in) {
    double value;
    while (rhs_in >> value) rhs_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density rhs result file is missing!"));
  }
  rhs_in.close();

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  size_t gridsize = grid->getStorage().getSize();

  // Load dataset for test scenario
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(
									 "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
  sgpp::base::DataMatrix &dataset = data.getData();
  {
    // Create OCL configuration
    std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
        getConfigurationDefaultsSingleDevice();
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
    std::cout << "Testing rhs kernel with long compression type, with "
              << "local memory and without compression registers ..." << std::endl;
    for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
      json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::Node &deviceNode = platformNode["DEVICES"][deviceName];
        const std::string &kernelName = "cscheme";
        json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
        kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
        kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
        kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
        kernelNode.replaceIDAttr("COMPRESSION_TYPE", "uint64_t");
      }
    }
    // Create OpenCL Manager
    auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);
    // Create operation
    auto operation_rhs = std::make_unique<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
          *grid, 2, manager, parameters, 0.001);

    sgpp::base::DataVector b(gridsize);
    operation_rhs->generateb(dataset, b);
    for (size_t i = 0; i < gridsize; ++i) {
      BOOST_CHECK_CLOSE(rhs_optimal_result[i], b[i], 0.001);
    }
  }
}

BOOST_AUTO_TEST_CASE(DensityRHSOpenCL_LocalMemoryCompression64NoRegistersEvalBlocked8) {
  std::cout << "Starting rhs kernel tests" << std::endl;
  // Load correct results for comparison
  std::vector<double> rhs_optimal_result;
  std::ifstream rhs_in("datadriven/tests/data/clustering_test_data/rhs_erg_dim2_depth11.txt");
  if (rhs_in) {
    double value;
    while (rhs_in >> value) rhs_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density rhs result file is missing!"));
  }
  rhs_in.close();

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  size_t gridsize = grid->getStorage().getSize();

  // Load dataset for test scenario
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(
									 "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
  sgpp::base::DataMatrix &dataset = data.getData();
  {
    // Create OCL configuration
    std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
        getConfigurationDefaultsSingleDevice();
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);
    std::cout << "Testing rhs kernel with long compression type, with "
              << "local memory and without compression registers ..." << std::endl;
    for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
      json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
      for (std::string &deviceName : platformNode["DEVICES"].keys()) {
        json::Node &deviceNode = platformNode["DEVICES"][deviceName];
        const std::string &kernelName = "cscheme";
        json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
        kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
        kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
        kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
        kernelNode.replaceIDAttr("COMPRESSION_TYPE", "uint64_t");
				kernelNode.replaceIDAttr("KERNEL_EVAL_BLOCKING", UINT64_C(8));
	// kernelNode.replaceIDAttr("WRITE_SOURCE", true);
      }
    }
    // Create OpenCL Manager
    auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);
    // Create operation
    auto operation_rhs = std::make_unique<
      sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
          *grid, 2, manager, parameters, 0.001);

    sgpp::base::DataVector b(gridsize);
    operation_rhs->generateb(dataset, b);
    for (size_t i = 0; i < gridsize; ++i) {
      BOOST_CHECK_CLOSE(rhs_optimal_result[i], b[i], 0.001);
    }
  }
}

BOOST_AUTO_TEST_CASE(DensityMultiplicationOpenCL) {
  // Load correct results for comparison
  std::vector<double> mult_optimal_result;
  std::ifstream mult_in("datadriven/tests/data/clustering_test_data/mult_erg_dim2_depth11.txt");
  if (mult_in) {
    double value;
    while (mult_in >> value) mult_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density multiplication result file is missing!"));
  }
  mult_in.close();

  // Create OCL configuration
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();
  // parameters->serialize(std::cout, 0);
  sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);

  // Create OpenCL Manager
  auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);

  std::cout << "Testing default kernel configuration..." << std::endl;
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing with preprocessed positions..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", true);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing with preprocessed positions and ignored flags..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", false);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", false);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing with branchless mutliplication kernel..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("USE_FABS", false);
      kernelNode.replaceIDAttr("USE_IMPLICIT", false);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", false);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", false);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing with branchless mutliplication kernel with fabs modifications..."
            << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", false);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", false);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", false);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);
  std::cout << "Testing with branchless mutliplication kernel with level cache..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", false);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", false);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", false);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing with branchless mutliplication kernel with implicit modifications..."
            << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("USE_FABS", false);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", false);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", false);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing with branchless mutliplication kernel with "
            << "all modifications..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", false);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing default multiplcation kernel with fabs modifications..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", false);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", false);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing default multiplcation kernel with implicit modifications..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("USE_FABS", false);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", false);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing default multiplcation kernel with level cache..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("USE_FABS", false);
      kernelNode.replaceIDAttr("USE_IMPLICIT", false);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing default multiplcation kernel with all modifications..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression for streaming gridpoints..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", false);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression for fixed gridpoints..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", false);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression for all gridpoints..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression but without implicit..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", false);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression but without optimized operation count..."
            << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", false);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression and without local memory..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"unsigned int\" for streaming gridpoints..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"unsigned int\" for fixed gridpoints..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", false);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"unsigned int\" for all gridpoints..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"unsigned int\" but without implicit..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", false);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"unsigned int\" but without optimized operation count..."
            << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", false);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"unsigned int\" and without local memory..." << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"unsigned int\" for streaming gridpoints..." << std::endl;
  std::cout << "No compression register!" << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", false);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"unsigned int\" for fixed gridpoints..." << std::endl;
  std::cout << "No compression register!" << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", false);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"unsigned int\" for all gridpoints..." << std::endl;
  std::cout << "No compression register!" << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"unsigned int\" but without implicit..." << std::endl;
  std::cout << "No compression register!" << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", false);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"unsigned int\" but without optimized operation count..."
            << std::endl;
  std::cout << "No compression register!" << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", false);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"unsigned int\" and without local memory..." << std::endl;
  std::cout << "No compression register!" << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "unsigned int");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"uint64_t\" for streaming gridpoints..." << std::endl;
  std::cout << "No compression register!" << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", false);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "uint64_t");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"uint64_t\" for fixed gridpoints..." << std::endl;
  std::cout << "No compression register!" << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", false);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "uint64_t");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"uint64_t\" for all gridpoints..." << std::endl;
  std::cout << "No compression register!" << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "uint64_t");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"uint64_t\" but without implicit..." << std::endl;
  std::cout << "No compression register!" << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", false);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "uint64_t");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"uint64_t\" but without optimized operation count..."
            << std::endl;
  std::cout << "No compression register!" << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", false);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "uint64_t");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);

  std::cout << "Testing multiplication kernel with compression_type \"uint64_t\" and without local memory..." << std::endl;
  std::cout << "No compression register!" << std::endl;
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "uint64_t");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);


  std::cout << "Density multiplication test done!" << std::endl;
}

BOOST_AUTO_TEST_CASE(DensityMultiplicationOpenCL_EvalBlocking) {
  // Load correct results for comparison
  std::vector<double> mult_optimal_result;
  std::ifstream mult_in("datadriven/tests/data/clustering_test_data/mult_erg_dim2_depth11.txt");
  if (mult_in) {
    double value;
    while (mult_in >> value) mult_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density multiplication result file is missing!"));
  }
  mult_in.close();

  // Create OCL configuration
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();
  // parameters->serialize(std::cout, 0);
  sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);

  // Create OpenCL Manager
  auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "multdensity";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("PREPROCESS_POSITIONS", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_FABS", true);
      kernelNode.replaceIDAttr("USE_IMPLICIT", true);
      kernelNode.replaceIDAttr("USE_LESS_OPERATIONS", true);
      kernelNode.replaceIDAttr("USE_LEVEL_CACHE", false);
      kernelNode.replaceIDAttr("USE_COMPRESSION_STREAMING", false);
      kernelNode.replaceIDAttr("USE_COMPRESSION_FIXED", true);
      kernelNode.replaceIDAttr("USE_COMPRESSION_REGISTERS", false);
      kernelNode.replaceIDAttr("KERNEL_EVAL_BLOCKING", UINT64_C(8));
      kernelNode.replaceIDAttr("COMPRESSION_TYPE", "uint64_t");
    }
  }
  multiply_and_test(parameters, mult_optimal_result, manager, *grid);
}

BOOST_AUTO_TEST_CASE(DensityAlphaSolver) {
  // Load correct results for comparison
  std::vector<double> alpha_optimal_result;
  std::ifstream alpha_in("datadriven/tests/data/clustering_test_data/alpha_erg_dim2_depth11.txt");
  if (alpha_in) {
    double value;
    while (alpha_in >> value) alpha_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Alpha result file is missing!"));
  }
  alpha_in.close();

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  size_t gridsize = grid->getStorage().getSize();

  // Load rhs vector
  sgpp::base::DataVector b(gridsize);
  std::ifstream rhs_in("datadriven/tests/data/clustering_test_data/rhs_erg_dim2_depth11.txt");
  if (rhs_in) {
    double value;
    int counter = 0;
    while (rhs_in >> value) {
      b[counter] = value;
      counter++;
    }
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("Density rhs result file is missing!"));
  }
  rhs_in.close();

  // Create OCL configuration
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();
  sgpp::datadriven::DensityOCLMultiPlatform::OperationDensity::load_default_parameters(parameters);

  // Create OpenCL Manager
  auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);

  // Create operation
  auto mult_operation = std::make_unique<
    sgpp::datadriven::DensityOCLMultiPlatform::OperationDensityOCLMultiPlatform<double>>(
        *grid, 2, manager, parameters, 0.001);

  // Create solver
  auto solver = std::make_unique<sgpp::solver::ConjugateGradients>(100, 0.001);
  sgpp::base::DataVector alpha(gridsize);
  alpha.setAll(1.0);

  // Solve
  solver->solve(*mult_operation, alpha, b, false, false);
  // Scaling
  double max = alpha.max();
  double min = alpha.min();
  for (size_t i = 0; i < gridsize; i++) alpha[i] = alpha[i] * 1.0 / (max - min);

  // Compare results with correct results
  for (size_t i = 0; i < gridsize; ++i) {
    BOOST_CHECK_CLOSE(alpha_optimal_result[i], alpha[i], 1.0);
  }
}


BOOST_AUTO_TEST_CASE(KNNGraphOpenCL) {
  // Reference results
  std::vector<int64_t> graph_optimal_result_dim2;
  std::vector<int64_t> graph_optimal_result_dim3;
  std::vector<int64_t> graph_optimal_result_dim10;
  std::vector<int64_t> graph_approx_result_dim2;
  std::vector<int64_t> graph_approx_result_dim3;
  std::vector<int64_t> graph_approx_result_dim10;
  std::vector<int64_t> graph_approx32_result_dim2;
  std::vector<int64_t> graph_approx32_result_dim3;
  std::vector<int64_t> graph_approx32_result_dim10;

  // helper loader function
  auto load_reference = [](const std::string reference_filename, std::vector<int64_t> &target_graph) {
    std::ifstream graph_in(reference_filename);
    if (graph_in) {
      int value;
      while (graph_in >> value) target_graph.push_back(value);
    } else {
      std::string error_message = std::string("Error! Reference file ") +
      reference_filename + std::string(" is missing!");
      BOOST_THROW_EXCEPTION(std::runtime_error(error_message));
    }
    graph_in.close();
  };

  // load dim 2 reference result
  load_reference("datadriven/tests/data/clustering_test_data/graph_erg_dim2_depth11.txt", graph_optimal_result_dim2);
  // load dim 3 reference result
  load_reference("datadriven/tests/data/clustering_test_data/knn_dim3.txt", graph_optimal_result_dim3);
  // load dim 10 reference result
  load_reference("datadriven/tests/data/clustering_test_data/knn_dim10.txt", graph_optimal_result_dim10);
  // load dim 2 approximate reference result
  load_reference("datadriven/tests/data/clustering_test_data/approx_knn_dim2.txt", graph_approx_result_dim2);
  // load dim 3 approximate reference result
  load_reference("datadriven/tests/data/clustering_test_data/approx_knn_dim3.txt", graph_approx_result_dim3);
  // load dim 10 approximate reference result
  load_reference("datadriven/tests/data/clustering_test_data/approx_knn_dim10.txt", graph_approx_result_dim10);
  // load dim 2 approximate reference result with full bins
  load_reference("datadriven/tests/data/clustering_test_data/approx32_knn_dim2.txt", graph_approx32_result_dim2);
  // load dim 3 approximate reference result with full bins
  load_reference("datadriven/tests/data/clustering_test_data/approx32_knn_dim3.txt", graph_approx32_result_dim3);
  // load dim 10 approximate reference result with full bins
  load_reference("datadriven/tests/data/clustering_test_data/approx32_knn_dim10.txt", graph_approx32_result_dim10);

  // Load dataset for test scenario
  sgpp::datadriven::Dataset data_dim2 = sgpp::datadriven::ARFFTools::readARFF(
      "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
  sgpp::datadriven::Dataset data_dim3 = sgpp::datadriven::ARFFTools::readARFF(
      "datadriven/tests/data/clustering_test_data/knn_graph_test_dim3.arff");
  sgpp::datadriven::Dataset data_dim10 = sgpp::datadriven::ARFFTools::readARFF(
      "datadriven/tests/data/clustering_test_data/knn_graph_test_dim10.arff");

  // Create OCL configuration,  manager
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();
  sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::load_default_parameters(
      parameters);
  auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);
  size_t k = 8;

  auto run_knn_test_case = [&manager, k](auto parameters, auto dataset, auto reference_graph) {
    auto operation_graph = std::make_unique<
        sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCLSingleDevice<double>>(
            dataset.getData(), dataset.getDimension(), manager, parameters, k);
    // Test graph kernel
    std::vector<int64_t> graph(dataset.getData().getNrows() * k);
    operation_graph->create_graph(graph);
    for (size_t i = 0; i < dataset.getData().getNrows() * k; ++i) {
        BOOST_CHECK(reference_graph[i] == graph[i]);
        if (reference_graph[i] != graph[i]) {
          std::cout << "Actual result at position " << i << " : " << graph[i] << std::endl;
          std::cout << "Reference file at position " << i << " : " << reference_graph[i] << std::endl;
        }
    }
  };

  // Testing default kernel
  std::cout << "Testing default knn graph kernel ..." << std::endl;
  run_knn_test_case(parameters, data_dim2, graph_optimal_result_dim2);


  // Testing kernel with select
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "connectNeighbors";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("USE_SELECT", true);
    }
  }
  std::cout << "Testing default knn graph kernel with select statements..." << std::endl;
  run_knn_test_case(parameters, data_dim2, graph_optimal_result_dim2);

  // Testing approximate KNN kernel with unrolled dist (2) and maximum number of bins
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "connectNeighbors";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("USE_SELECT", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
      kernelNode.replaceIDAttr("USE_APPROX", false);
      kernelNode.replaceIDAttr("KERNEL_DATA_BLOCKING_SIZE", 2l);
    }
  }
  std::cout << "Testing knn graph kernel with unrolled dist calculation [dim 2]..." << std::endl;
  run_knn_test_case(parameters, data_dim2, graph_optimal_result_dim2);
  std::cout << "Testing knn graph kernel with unrolled dist calculation [dim 3]..." << std::endl;
  run_knn_test_case(parameters, data_dim3, graph_optimal_result_dim3);
  std::cout << "Testing knn graph kernel with unrolled dist calculation [dim 10]..." << std::endl;
  run_knn_test_case(parameters, data_dim10, graph_optimal_result_dim10);


  // Testing approximate KNN kernel
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "connectNeighbors";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("USE_SELECT", false);
      kernelNode.replaceIDAttr("LOCAL_SIZE", 128l);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_APPROX", true);
      kernelNode.replaceIDAttr("ALLOW_REFLEXIVE", false);
      kernelNode.replaceIDAttr("APPROX_REG_COUNT", 16l);
      kernelNode.replaceIDAttr("KERNEL_DATA_BLOCKING_SIZE", 1l);
    }
  }
  std::cout << "Testing approx knn graph kernel with local memory [dim 2]..." << std::endl;
  run_knn_test_case(parameters, data_dim2, graph_approx_result_dim2);
  std::cout << "Testing approx knn graph kernel with local memory [dim 3]..." << std::endl;
  run_knn_test_case(parameters, data_dim3, graph_approx_result_dim3);
  std::cout << "Testing approx knn graph kernel with local memory [dim 10]..." << std::endl;
  run_knn_test_case(parameters, data_dim10, graph_approx_result_dim10);

  // Testing approximate KNN kernel with reflexive edges
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "connectNeighbors";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("USE_SELECT", false);
      kernelNode.replaceIDAttr("LOCAL_SIZE", 128l);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_APPROX", true);
      kernelNode.replaceIDAttr("ALLOW_REFLEXIVE", true);
      kernelNode.replaceIDAttr("APPROX_REG_COUNT", 16l);
      kernelNode.replaceIDAttr("KERNEL_DATA_BLOCKING_SIZE", 1l);
    }
  }
  std::cout << "Testing approx knn graph kernel with local memory and reflexive edges [dim 2]..." << std::endl;
  run_knn_test_case(parameters, data_dim2, graph_approx_result_dim2);
  std::cout << "Testing approx knn graph kernel with local memory and reflexive edges [dim 3]..." << std::endl;
  run_knn_test_case(parameters, data_dim3, graph_approx_result_dim3);
  std::cout << "Testing approx knn graph kernel with local memory and reflexive edges [dim 10]..." << std::endl;
  run_knn_test_case(parameters, data_dim10, graph_approx_result_dim10);


  // Testing approximate KNN kernel with unrolled dist (2) and maximum number of bins
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "connectNeighbors";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("USE_SELECT", false);
      kernelNode.replaceIDAttr("LOCAL_SIZE", 128l);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_APPROX", true);
      kernelNode.replaceIDAttr("ALLOW_REFLEXIVE", false);
      kernelNode.replaceIDAttr("APPROX_REG_COUNT", 32l);
      kernelNode.replaceIDAttr("WRITE_SOURCE", true);
      kernelNode.replaceIDAttr("KERNEL_DATA_BLOCKING_SIZE", 2l);
    }
  }
  std::cout << "Testing approx knn graph kernel with unrolled dist calculation and max bins [dim 2]..." << std::endl;
  run_knn_test_case(parameters, data_dim2, graph_approx32_result_dim2);
  std::cout << "Testing approx knn graph kernel with unrolled dist calculation and max bins [dim 3]..." << std::endl;
  run_knn_test_case(parameters, data_dim3, graph_approx32_result_dim3);
  std::cout << "Testing approx knn graph kernel with unrolled dist calculation and max bins [dim 10]..." << std::endl;
  run_knn_test_case(parameters, data_dim10, graph_approx32_result_dim10);

  // Testing approximate KNN kernel with unrolled dist (2)
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "connectNeighbors";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("USE_SELECT", false);
      kernelNode.replaceIDAttr("LOCAL_SIZE", 128l);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_APPROX", true);
      kernelNode.replaceIDAttr("ALLOW_REFLEXIVE", false);
      kernelNode.replaceIDAttr("APPROX_REG_COUNT", 16l);
      kernelNode.replaceIDAttr("WRITE_SOURCE", true);
      kernelNode.replaceIDAttr("KERNEL_DATA_BLOCKING_SIZE", 2l);
    }
  }
  std::cout << "Testing approx knn graph kernel with unrolled dist calculation [dim 2]..." << std::endl;
  run_knn_test_case(parameters, data_dim2, graph_approx_result_dim2);
  std::cout << "Testing approx knn graph kernel with unrolled dist calculation [dim 3]..." << std::endl;
  run_knn_test_case(parameters, data_dim3, graph_approx_result_dim3);
  std::cout << "Testing approx knn graph kernel with unrolled dist calculation [dim 10]..." << std::endl;
  run_knn_test_case(parameters, data_dim10, graph_approx_result_dim10);

  // Testing approximate KNN kernel with unrolled dist (3)
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "connectNeighbors";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("USE_SELECT", false);
      kernelNode.replaceIDAttr("LOCAL_SIZE", 128l);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("USE_APPROX", true);
      kernelNode.replaceIDAttr("ALLOW_REFLEXIVE", false);
      kernelNode.replaceIDAttr("APPROX_REG_COUNT", 16l);
      kernelNode.replaceIDAttr("WRITE_SOURCE", true);
      kernelNode.replaceIDAttr("KERNEL_DATA_BLOCKING_SIZE", 3l);
    }
  }
  std::cout << "Testing approx knn graph kernel with 3 unrolled dist calculation [dim 3]..." << std::endl;
  run_knn_test_case(parameters, data_dim3, graph_approx_result_dim3);
  std::cout << "Testing approx knn graph kernel with 3 unrolled dist calculation [dim 10]..." << std::endl;
  run_knn_test_case(parameters, data_dim10, graph_approx_result_dim10);

  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "connectNeighbors";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("USE_SELECT", false);
      kernelNode.replaceIDAttr("USE_APPROX", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("LOCAL_SIZE", 128l);
      kernelNode.replaceIDAttr("KERNEL_DATA_BLOCKING_SIZE", 1l);
    }
  }
  std::cout << "Testing default knn graph kernel with local memory [dim2]..."
            << std::endl;
  run_knn_test_case(parameters, data_dim2, graph_optimal_result_dim2);
  std::cout << "Testing default knn graph kernel with local memory [dim2]..."
            << std::endl;
  run_knn_test_case(parameters, data_dim3, graph_optimal_result_dim3);
  std::cout << "Testing default knn graph kernel with local memory [dim2]..."
            << std::endl;
  run_knn_test_case(parameters, data_dim10, graph_optimal_result_dim10);

  // testing with unrolled dim loop
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "connectNeighbors";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("USE_SELECT", false);
      kernelNode.replaceIDAttr("USE_APPROX", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("LOCAL_SIZE", 128l);
      kernelNode.replaceIDAttr("KERNEL_DATA_BLOCKING_SIZE", 2l);
    }
  }
  std::cout << "Testing default knn graph kernel with local memory and blocked dimension loop [dim2]..."
            << std::endl;
  run_knn_test_case(parameters, data_dim2, graph_optimal_result_dim2);
  std::cout << "Testing default knn graph kernel with local memory and blocked dimension loop [dim3]..."
            << std::endl;
  run_knn_test_case(parameters, data_dim3, graph_optimal_result_dim3);
  std::cout << "Testing default knn graph kernel with local memory and blocked dimension loop [dim10]..."
            << std::endl;
  run_knn_test_case(parameters, data_dim10, graph_optimal_result_dim10);

  // testing with unrolled dim loop
  for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
    json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
    for (std::string &deviceName : platformNode["DEVICES"].keys()) {
      json::Node &deviceNode = platformNode["DEVICES"][deviceName];
      const std::string &kernelName = "connectNeighbors";
      json::Node &kernelNode = deviceNode["KERNELS"][kernelName];
      kernelNode.replaceIDAttr("USE_SELECT", false);
      kernelNode.replaceIDAttr("USE_APPROX", false);
      kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
      kernelNode.replaceIDAttr("LOCAL_SIZE", 128l);
      kernelNode.replaceIDAttr("KERNEL_DATA_BLOCKING_SIZE", 3l);
    }
  }
  std::cout << "Testing default knn graph kernel with local memory and blocked dimension loop 3 [dim3]..."
            << std::endl;
  run_knn_test_case(parameters, data_dim3, graph_optimal_result_dim3);
  std::cout << "Testing default knn graph kernel with local memory and blocked dimension loop 3 [dim10]..."
            << std::endl;
  run_knn_test_case(parameters, data_dim10, graph_optimal_result_dim10);
}

BOOST_AUTO_TEST_CASE(KNNPruneGraphOpenCL) {
  // Load input
  std::vector<int64_t> graph;
  std::ifstream graph_in("datadriven/tests/data/clustering_test_data/graph_erg_dim2_depth11.txt");
  if (graph_in) {
    int value;
    while (graph_in >> value) graph.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("knn graph result file is missing!"));
  }
  graph_in.close();

  // Load optimal results for comparison
  std::vector<int64_t> graph_optimal_result;
  std::ifstream graph_result(
      "datadriven/tests/data/clustering_test_data/graph_pruned_erg_dim2_depth11.txt");
  if (graph_result) {
    int value;
    while (graph_result >> value) graph_optimal_result.push_back(value);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("knn pruned graph result file is missing!"));
  }
  graph_result.close();

  // Create grid for test scenario
  sgpp::base::Grid *grid = sgpp::base::Grid::createLinearGrid(2);
  sgpp::base::GridGenerator &gridGen = grid->getGenerator();
  gridGen.regular(11);
  size_t gridsize = grid->getStorage().getSize();

  // Load optimal results for comparison
  sgpp::base::DataVector alpha(gridsize);
  std::ifstream alpha_result(
      "datadriven/tests/data/clustering_test_data/alpha_erg_dim2_depth11.txt");
  if (alpha_result) {
    size_t counter = 0;
    double value;
    while (alpha_result >> value) {
      alpha[counter] = value;
      counter++;
    }
    BOOST_CHECK(counter == gridsize);
  } else {
    BOOST_THROW_EXCEPTION(std::runtime_error("knn pruned graph result file is missing!"));
  }
  alpha_result.close();

  // Create OCL configuration
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();
  sgpp::datadriven::DensityOCLMultiPlatform::OperationPruneGraphOCL::load_default_parameters(
      parameters);

  // Create OpenCL Manager
  auto manager = std::make_shared<sgpp::base::OCLManagerMultiPlatform>(parameters);

  // Load dataset for test scenario
  sgpp::datadriven::Dataset data = sgpp::datadriven::ARFFTools::readARFF(
      "datadriven/tests/data/clustering_test_data/clustering_testdataset_dim2.arff");
  sgpp::base::DataMatrix &dataset = data.getData();

  // Create operation
  auto operation_prune = std::make_unique<
    sgpp::datadriven::DensityOCLMultiPlatform::OperationPruneGraphOCLMultiPlatform<double>>(
        *grid, alpha, dataset, 2, manager, parameters, 0.2, 8);

  std::cout << "Testing knn prune kernel ..." << std::endl;
  operation_prune->prune_graph(graph);
  for (size_t i = 0; i < gridsize; ++i) {
    BOOST_CHECK(graph[i] == graph_optimal_result[i]);
  }
}

BOOST_AUTO_TEST_CASE(KNNClusterSearch) {
  // Load input
  // std::vector<int> graph;
  // std::ifstream graph_in(
  //     "datadriven/tests/data/clustering_test_data/graph_pruned_erg_dim2_depth11.txt");
  // if (graph_in) {
  //   int value;
  //   while (graph_in >> value) graph.push_back(value);
  // } else {
  //   BOOST_THROW_EXCEPTION(std::runtime_error("Pruned knn graph result file is missing!"));
  // }
  // graph_in.close();

  // std::vector<int> optimal_cluster_assignement;
  // std::ifstream assignement_in("datadriven/tests/data/clustering_test_data/cluster_erg.txt");
  // if (assignement_in) {
  //   size_t value;
  //   while (assignement_in >> value) optimal_cluster_assignement.push_back(value);
  // } else {
  //   BOOST_THROW_EXCEPTION(std::runtime_error("Pruned knn graph result file is missing!"));
  // }
  // assignement_in.close();

  // std::vector<int> node_cluster_map;
  // sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::neighborhood_list_t clusters;
  // sgpp::datadriven::DensityOCLMultiPlatform::OperationCreateGraphOCL::find_clusters(
  //     graph, 8, node_cluster_map, clusters);
  // BOOST_CHECK(optimal_cluster_assignement.size() == node_cluster_map.size());
  // if (optimal_cluster_assignement.size() == node_cluster_map.size()) {
  //   for (size_t i = 0; i < node_cluster_map.size(); ++i) {
  //     BOOST_CHECK(optimal_cluster_assignement[i] == node_cluster_map[i]);
  //   }
  // }
}

BOOST_AUTO_TEST_SUITE_END()
#endif
