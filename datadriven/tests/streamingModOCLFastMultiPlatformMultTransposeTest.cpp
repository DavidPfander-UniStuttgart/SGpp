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
#include "sgpp/datadriven/operation/hash/OperationMultipleEvalStreamingModOCLFastMultiPlatform/Configuration.hpp"
#include "sgpp/datadriven/tools/ARFFTools.hpp"
#include "sgpp/globaldef.hpp"
#include "test_datadrivenCommon.hpp"

namespace TestStreamingModOCLFastMultiPlatformMultTransposeFixture {
struct FilesNamesAndErrorFixture {
  FilesNamesAndErrorFixture() {}
  ~FilesNamesAndErrorFixture() {}

  std::vector<std::tuple<std::string, double>> fileNamesErrorDouble = {
      std::tuple<std::string, double>("datadriven/tests/data/friedman2_4d_10000.arff.gz", 1E-15),
      std::tuple<std::string, double>("datadriven/tests/data/friedman1_10d_2000.arff.gz", 1E-18)};

  std::vector<std::tuple<std::string, double>> fileNamesErrorFloat = {
      std::tuple<std::string, double>("datadriven/tests/data/friedman2_4d_10000.arff.gz", 1E1),
      std::tuple<std::string, double>("datadriven/tests/data/friedman1_10d_2000.arff.gz", 1E-2)};

  uint32_t level = 4;
};
}  // namespace TestStreamingModOCLFastMultiPlatformMultTransposeFixture

BOOST_FIXTURE_TEST_SUITE(
    TestStreamingModOCLFastMultiPlatformMultTranspose,
    TestStreamingModOCLFastMultiPlatformMultTransposeFixture::FilesNamesAndErrorFixture)

BOOST_AUTO_TEST_CASE(Simple) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();

  (*parameters)["INTERNAL_PRECISION"].set("double");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();
  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingModOCLFastMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", true);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceIDAttr("KERNEL_TRANS_DATA_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "array");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(1));
  }

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLFASTMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorDouble, sgpp::base::GridType::ModLinear, level,
                           configuration);
}

BOOST_AUTO_TEST_CASE(Blocking) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();

  (*parameters)["INTERNAL_PRECISION"].set("double");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingModOCLFastMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceIDAttr("KERNEL_TRANS_DATA_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "register");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLFASTMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorDouble, sgpp::base::GridType::ModLinear, level,
                           configuration);
}

BOOST_AUTO_TEST_CASE(MultiDevice) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsMultiDevice();

  (*parameters)["INTERNAL_PRECISION"].set("double");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingModOCLFastMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceIDAttr("KERNEL_TRANS_DATA_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "register");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLFASTMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorDouble, sgpp::base::GridType::ModLinear, level,
                           configuration);
}

BOOST_AUTO_TEST_CASE(MultiPlatform) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsMultiPlatform();

  (*parameters)["INTERNAL_PRECISION"].set("double");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingModOCLFastMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceIDAttr("KERNEL_TRANS_DATA_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "register");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLFASTMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorDouble, sgpp::base::GridType::ModLinear, level,
                           configuration);
}

BOOST_AUTO_TEST_CASE(SimpleSinglePrecision) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();

  (*parameters)["INTERNAL_PRECISION"].set("float");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingModOCLFastMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceIDAttr("KERNEL_TRANS_DATA_BLOCK_SIZE", UINT64_C(1));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "array");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(1));
  }

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLFASTMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorFloat, sgpp::base::GridType::ModLinear, level,
                           configuration);
}

BOOST_AUTO_TEST_CASE(BlockingSinglePrecision) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsSingleDevice();

  (*parameters)["INTERNAL_PRECISION"].set("float");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingModOCLFastMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceIDAttr("KERNEL_TRANS_DATA_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "register");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLFASTMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorFloat, sgpp::base::GridType::ModLinear, level,
                           configuration);
}

BOOST_AUTO_TEST_CASE(MultiDeviceSinglePrecision) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsMultiDevice();

  (*parameters)["INTERNAL_PRECISION"].set("float");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingModOCLFastMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceIDAttr("KERNEL_TRANS_DATA_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "register");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLFASTMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorFloat, sgpp::base::GridType::ModLinear, level,
                           configuration);
}

BOOST_AUTO_TEST_CASE(MultiPlatformSinglePrecision) {
  std::shared_ptr<sgpp::base::OCLOperationConfiguration> parameters =
      getConfigurationDefaultsMultiPlatform();

  (*parameters)["INTERNAL_PRECISION"].set("float");

  std::vector<std::reference_wrapper<json::node>> deviceNodes = parameters->getAllDeviceNodes();

  for (json::node &deviceNode : deviceNodes) {
    auto &kernelNode = deviceNode["KERNELS"].replaceDictAttr(
        sgpp::datadriven::StreamingModOCLFastMultiPlatform::Configuration::getKernelName());
    kernelNode.replaceIDAttr("KERNEL_USE_LOCAL_MEMORY", false);
    kernelNode.replaceIDAttr("KERNEL_DATA_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceIDAttr("KERNEL_TRANS_GRID_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceIDAttr("KERNEL_TRANS_DATA_BLOCK_SIZE", UINT64_C(2));
    kernelNode.replaceTextAttr("KERNEL_STORE_DATA", "register");
    kernelNode.replaceIDAttr("KERNEL_MAX_DIM_UNROLL", UINT64_C(10));
  }

  sgpp::datadriven::OperationMultipleEvalConfiguration configuration(
      sgpp::datadriven::OperationMultipleEvalType::STREAMING,
      sgpp::datadriven::OperationMultipleEvalSubType::OCLFASTMP, *parameters);

  compareDatasetsTranspose(fileNamesErrorFloat, sgpp::base::GridType::ModLinear, level,
                           configuration);
}

BOOST_AUTO_TEST_SUITE_END()

#endif
