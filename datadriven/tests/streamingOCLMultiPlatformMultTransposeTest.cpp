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

#include "sgpp/base/grid/generation/functors/SurplusRefinementFunctor.hpp"
#include "sgpp/base/operation/BaseOpFactory.hpp"
#include "sgpp/base/operation/hash/OperationMultipleEval.hpp"
#include "sgpp/base/tools/ConfigurationParameters.hpp"
#include "sgpp/datadriven/DatadrivenOpFactory.hpp"
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalStreamingOCLMultiPlatform/Configuration.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"
#include "sgpp/globaldef.hpp"
#include "test_datadrivenCommon.hpp"

namespace TestStreamingOCLMultiPlatformMultTransposeFixture {
struct FilesNamesAndErrorFixture {
  FilesNamesAndErrorFixture() {}
  ~FilesNamesAndErrorFixture() {}

  std::vector<std::tuple<std::string, double>> fileNamesErrorDouble = {
      std::tuple<std::string, double>("datadriven/tests/data/friedman2_4d_10000.arff.gz", 1E-18),
      std::tuple<std::string, double>("datadriven/tests/data/friedman1_10d_2000.arff.gz", 1E-26)};

  std::vector<std::tuple<std::string, double>> fileNamesErrorFloat = {
      std::tuple<std::string, double>("datadriven/tests/data/friedman2_4d_10000.arff.gz", 1E+5),
      std::tuple<std::string, double>("datadriven/tests/data/friedman1_10d_2000.arff.gz", 1E-8)};

  uint32_t level = 5;
};
}  // namespace TestStreamingOCLMultiPlatformMultTransposeFixture

BOOST_FIXTURE_TEST_SUITE(
    TestStreamingOCLMultiPlatformMultTranspose,
    TestStreamingOCLMultiPlatformMultTransposeFixture::FilesNamesAndErrorFixture)

BOOST_AUTO_TEST_CASE(Simple) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "array");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(1));
  }

  std::cout << "Testing simple version... " << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorDouble, sgpp::base::GridType::Linear, level,
                           configuration);
}

BOOST_AUTO_TEST_CASE(SimpleCompression) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "compressed");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(1));
  }
  std::cout << "Testing compressed simple version... " << std::endl;

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorDouble, sgpp::base::GridType::Linear, level,
                           configuration);
}

BOOST_AUTO_TEST_CASE(Blocking) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "register");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing blocking version... " << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorDouble, sgpp::base::GridType::Linear, level,
                           configuration);
}

BOOST_AUTO_TEST_CASE(CompressedBlocking) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceIDAttr("WRITE_SOURCE", true);
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "compressed");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing compressed blocking version... " << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorDouble, sgpp::base::GridType::Linear, level,
                           configuration);
}


BOOST_AUTO_TEST_CASE(MultiDevice) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsMultiDevice();

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "register");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing multi device version... " << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorDouble, sgpp::base::GridType::Linear, level,
                           configuration);
}
BOOST_AUTO_TEST_CASE(CompressedMultiDevice) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsMultiDevice();

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "compressed");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing compressed multi device version... " << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorDouble, sgpp::base::GridType::Linear, level,
                           configuration);
}

BOOST_AUTO_TEST_CASE(MultiPlatform) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsMultiPlatform();

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "register");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing MultiPlatform..." << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorDouble, sgpp::base::GridType::Linear, level,
                           configuration);
}
BOOST_AUTO_TEST_CASE(CompressedMultiPlatform) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsMultiPlatform();

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "compressed");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing compressed MultiPlatform..." << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorDouble, sgpp::base::GridType::Linear, level,
                           configuration);
}

BOOST_AUTO_TEST_CASE(SimpleSinglePrecision) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();

  (*parameters)["INTERNAL_PRECISION"].set("float");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "array");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing single precision..." << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorFloat, sgpp::base::GridType::Linear, level, configuration);
}
BOOST_AUTO_TEST_CASE(CompressedSimpleSinglePrecision) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();

  (*parameters)["INTERNAL_PRECISION"].set("float");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "compressed");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing compressed single precision..." << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorFloat, sgpp::base::GridType::Linear, level, configuration);
}

BOOST_AUTO_TEST_CASE(BlockingSinglePrecision) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();

  (*parameters)["INTERNAL_PRECISION"].set("float");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "array");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing with single precision and blocking..." << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorFloat, sgpp::base::GridType::Linear, level, configuration);
}

BOOST_AUTO_TEST_CASE(CompressedBlockingSinglePrecision) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();

  (*parameters)["INTERNAL_PRECISION"].set("float");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(4));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "compressed");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing compressed with single precision and blocking..." << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorFloat, sgpp::base::GridType::Linear, level, configuration);
}

BOOST_AUTO_TEST_CASE(MultiDeviceSinglePrecision) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsMultiDevice();

  (*parameters)["INTERNAL_PRECISION"].set("float");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "array");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing with single precision multi device..." << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorFloat, sgpp::base::GridType::Linear, level, configuration);
}

BOOST_AUTO_TEST_CASE(CompressedMultiDeviceSinglePrecision) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsMultiDevice();

  (*parameters)["INTERNAL_PRECISION"].set("float");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "compressed");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing compressed with single precision multi device..." << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorFloat, sgpp::base::GridType::Linear, level, configuration);
}

BOOST_AUTO_TEST_CASE(MultiPlatformSinglePrecision) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsMultiPlatform();

  (*parameters)["INTERNAL_PRECISION"].set("float");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "array");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing with single precision multi platform..." << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorFloat, sgpp::base::GridType::Linear, level, configuration);
}
BOOST_AUTO_TEST_CASE(CompressedMultiPlatformSinglePrecision) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsMultiPlatform();

  (*parameters)["INTERNAL_PRECISION"].set("float");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingOCLMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "compressed");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  std::cout << "Testing compressed with single precision multi platform..." << std::endl;
  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorFloat, sgpp::base::GridType::Linear, level, configuration);
}


BOOST_AUTO_TEST_SUITE_END()

#endif
