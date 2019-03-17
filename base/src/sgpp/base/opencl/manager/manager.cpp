#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

#include "manager.hpp"
#include "manager_error.hpp"

namespace opencl {

manager_t::manager_t() : parameters(), verbose(false) {
  parameters.replaceIDAttr("VERBOSE", verbose);
  parameters.replaceIDAttr("OCL_MANAGER_VERBOSE", false);
  parameters.replaceIDAttr("SHOW_BUILD_LOG", false);
  parameters.replaceDictAttr("PLATFORMS");
  parameters.replaceIDAttr("LOAD_BALANCING_VERBOSE", false);
  parameters.replaceTextAttr("INTERNAL_PRECISION", "double");

  overallDeviceCount = 0;

  this->configure(false);
  if (overallDeviceCount == 0) {
    std::stringstream errorString;
    errorString << "OCL Error: either no devices available, or no devices "
                   "match the configuration!"
                << std::endl;
    throw manager_error(errorString.str());
  }
}

manager_t::manager_t(const std::string &configuration_file_name)
    : manager_t(configuration_t(configuration_file_name)) {}

manager_t::manager_t(const configuration_t &parameters_) : parameters(parameters_) {
  if (!parameters.contains("VERBOSE")) {
    parameters.addIDAttr("VERBOSE", false);
  }
  if (!parameters.contains("OCL_MANAGER_VERBOSE")) {
    parameters.replaceIDAttr("OCL_MANAGER_VERBOSE", false);
  }
  if (!parameters.contains("SHOW_BUILD_LOG")) {
    parameters.replaceIDAttr("SHOW_BUILD_LOG", false);
  }
  if (!parameters.contains("PLATFORMS")) {
    parameters.replaceListAttr("PLATFORMS");
  }
  if (!parameters.contains("LOAD_BALANCING_VERBOSE")) {
    parameters.replaceIDAttr("LOAD_BALANCING_VERBOSE", false);
  }
  if (!parameters.contains("INTERNAL_PRECISION")) {
    parameters.replaceTextAttr("INTERNAL_PRECISION", "double");
  }

  this->verbose = parameters["VERBOSE"].getBool();
  this->overallDeviceCount = 0;

  this->configure(true);
  if (overallDeviceCount == 0) {
    std::stringstream errorString;
    errorString << "OCL Error: either no devices available, or no devices "
                   "match the configuration!"
                << std::endl;
    throw manager_error(errorString.str());
  }
}

manager_t::~manager_t() {}

std::vector<cl_kernel> manager_t::build_kernel(const std::string &program_src,
                                               const std::string &kernel_name,
                                               const std::string additional_options) {
  cl_int err;
  if (verbose) {
    std::cout << "building kernel: " << kernel_name << std::endl;
  }

  std::vector<cl_kernel> kernels;

  for (device_t &device : this->devices) {
    // setting the program
    const char *kernel_src = program_src.c_str();
    cl_program program = clCreateProgramWithSource(device.context, 1, &kernel_src, NULL, &err);

    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to create program! Error Code: " << err << std::endl;
      throw manager_error(errorString.str());
    }

    auto &device_kernel_config = parameters["PLATFORMS"][device.platformName]["DEVICES"]
                                           [device.deviceName]["KERNELS"][kernel_name];

    std::string build_opts;
    if (!device_kernel_config.contains("ENABLE_OPTIMIZATIONS") ||
        device_kernel_config["ENABLE_OPTIMIZATIONS"].getBool()) {
      std::string optimizationFlags = "";
      if (device_kernel_config.contains("OPTIMIZATION_FLAGS")) {
        optimizationFlags = device_kernel_config["OPTIMIZATION_FLAGS"].get();
      }
      if (verbose) {
        std::cout << "building with optimization flags: " << optimizationFlags << std::endl;
      }
      build_opts = optimizationFlags;  // -O5  -cl-mad-enable -cl-denorms-are-zero
      // -cl-no-signed-zeros -cl-unsafe-math-optimizations
      // -cl-finite-math-only -cl-fast-relaxed-math
    } else {
      build_opts = "-cl-opt-disable";  // -g
    }
    build_opts += std::string(" ") + additional_options;

    // compiling the program
    err = clBuildProgram(program, 0, NULL, build_opts.c_str(), NULL, NULL);

    if (verbose) {
      // get the build log
      size_t len;
      cl_int err_buildlog =
          clGetProgramBuildInfo(program, device.deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
      check(err_buildlog, "OCL Error: build error, could not retrieve size of build log");
      if (err_buildlog == CL_SUCCESS) {
        std::string buffer(len, '\0');
        err_buildlog = clGetProgramBuildInfo(program, device.deviceId, CL_PROGRAM_BUILD_LOG, len,
                                             &buffer[0], NULL);
        check(err_buildlog, "OCL Error: build error, could not retrieve build log");
        buffer = buffer.substr(0, buffer.find('\0'));
        std::cout << "--- Build Log ---" << std::endl << buffer << std::endl;
      }
    }

    check(err, "OCL Error: OpenCL build error");

    // creating the kernel
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS) {
      std::stringstream errorString;
      errorString << "OCL Error: Failed to create kernel! Error code: " << err << std::endl;
      throw manager_error(errorString.str());
    }
    kernels.push_back(kernel);

    if (program) {
      clReleaseProgram(program);
    }
  }
  return kernels;
}

cl_kernel manager_t::build_kernel(const std::string &source, device_t &device,
                                  json::node &kernel_configuration, const std::string &kernel_name,
                                  const std::string additional_options) {
  cl_int err;
  if (verbose) {
    std::cout << "building kernel: " << kernel_name << std::endl;
  }

  // setting the program
  const char *kernelSourcePtr = source.c_str();
  cl_program program = clCreateProgramWithSource(device.context, 1, &kernelSourcePtr, NULL, &err);

  if (err != CL_SUCCESS) {
    std::stringstream errorString;
    errorString << "OCL Error: Failed to create program! Error Code: " << err << std::endl;
    throw manager_error(errorString.str());
  }

  std::string build_opts;
  if (!kernel_configuration.contains("ENABLE_OPTIMIZATIONS") ||
      kernel_configuration["ENABLE_OPTIMIZATIONS"].getBool()) {
    std::string optimizationFlags = "";
    if (kernel_configuration.contains("OPTIMIZATION_FLAGS")) {
      optimizationFlags = kernel_configuration["OPTIMIZATION_FLAGS"].get();
    }
    if (verbose) {
      std::cout << "building with optimization flags: " << optimizationFlags << std::endl;
    }
    build_opts = optimizationFlags;  // -O5  -cl-mad-enable -cl-denorms-are-zero
    // -cl-no-signed-zeros -cl-unsafe-math-optimizations
    // -cl-finite-math-only -cl-fast-relaxed-math
  } else {
    build_opts = "-cl-opt-disable";  // -g
  }
  build_opts += std::string(" ") + additional_options;

  // compiling the program
  err = clBuildProgram(program, 0, NULL, build_opts.c_str(), NULL, NULL);

  // collect the build log before throwing an exception if necessary

  // get the build log
  size_t len;
  clGetProgramBuildInfo(program, device.deviceId, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
  std::string buffer(len, '\0');
  clGetProgramBuildInfo(program, device.deviceId, CL_PROGRAM_BUILD_LOG, len, &buffer[0], NULL);
  buffer = buffer.substr(0, buffer.find('\0'));

  if (verbose) {
    std::cout << "--- Begin Build Log ---" << std::endl;
    std::cout << buffer << std::endl;
    std::cout << "--- End Build Log ---" << std::endl;
  }

  // report the error if the build failed
  check(err, "OCL Error: OpenCL build error");

  cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
  check(err, "failed to create kernel");

  if (program) {
    clReleaseProgram(program);
  }
  return kernel;
}

configuration_t &manager_t::get_configuration() { return this->parameters; }

void manager_t::configure(bool useConfiguration) {
  cl_int err;

  // determine number of available OpenCL platforms
  cl_uint platformCount;
  err = clGetPlatformIDs(0, nullptr, &platformCount);

  check(err, "OCL Error: Unable to get number of OpenCL platforms");

  if (verbose) {
    std::cout << "OCL Info: " << platformCount << " OpenCL platforms have been found" << std::endl;
  }

  // get available platforms
  std::vector<cl_platform_id> platformIds(platformCount);
  err = clGetPlatformIDs(platformCount, platformIds.data(), nullptr);

  check(err, "OCL Error: Unable to get Platform ID");

  for (size_t i = 0; i < platformCount; i++) {
    this->configure_platform(platformIds[i], parameters, useConfiguration);
  }
}

void manager_t::configure_platform(cl_platform_id platformId, configuration_t &configuration,
                                   bool useConfiguration) {
  cl_int err;

  char platformName[128] = {0};
  err = clGetPlatformInfo(platformId, CL_PLATFORM_NAME, 128 * sizeof(char), platformName, nullptr);

  check(err, "OCL Error: can't get platform name");

  if (verbose) {
    std::cout << "OCL Info: detected platform, name: \"" << platformName << "\"" << std::endl;
  }

  if (verbose) {
    char vendor_name[128] = {0};
    err =
        clGetPlatformInfo(platformId, CL_PLATFORM_VENDOR, 128 * sizeof(char), vendor_name, nullptr);

    check(err, "OCL Error: Can't get platform vendor");
    std::cout << "OCL Info: detected platform vendor name: \"" << vendor_name << "\"" << std::endl;
  }

  if (useConfiguration) {
    if (!parameters["PLATFORMS"].contains(platformName)) {
      if (verbose) {
        std::cout << "OCL Info: platform not configured, skipping..." << std::endl;
      }
      return;
    }
  } else {
    // creating new configuration
    json::node &platformNode = parameters["PLATFORMS"].addDictAttr(platformName);
    platformNode.addDictAttr("DEVICES");
  }

  if (verbose) {
    std::cout << "OCL Info: using platform, name: \"" << platformName << "\"" << std::endl;
  }

  if (std::string(platformName).compare("Clover") == 0) {
    std::cout << "OCL Info: platform skipped due to buggy behavior" << std::endl;
    return;
  }

  json::node &devicesNode = parameters["PLATFORMS"][platformName]["DEVICES"];

  uint32_t currentPlatformDevices;
  // get the number of devices
  err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, nullptr, &currentPlatformDevices);

  check(err, "OCL Error: Unable to get device count");

  if (verbose) {
    std::cout << "OCL Info: platform device count: " << currentPlatformDevices << std::endl;
  }

  std::vector<cl_device_id> deviceIds(currentPlatformDevices);
  err = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, (cl_uint)currentPlatformDevices,
                       deviceIds.data(), nullptr);

  check(err, std::string("OCL Error: Unable to get device id for platform ") + platformName);

  std::vector<cl_device_id> filteredDeviceIds;

  std::vector<std::string> filteredDeviceNames;

  std::map<std::string, size_t> countLimitMap;

  for (cl_device_id deviceId : deviceIds) {
    // filter device ids
    this->configure_device(deviceId, devicesNode, filteredDeviceIds, filteredDeviceNames,
                           countLimitMap, useConfiguration);
  }

  if (filteredDeviceIds.size() > 0) {
    platforms.emplace_back(platformId, platformName, filteredDeviceIds, filteredDeviceNames);
    platform_wrapper_t &platformWrapper = platforms[platforms.size() - 1];

    // create linear device list
    for (size_t deviceIndex = 0; deviceIndex < filteredDeviceIds.size(); deviceIndex++) {
      devices.emplace_back(platformWrapper.platformId, platformWrapper.deviceIds[deviceIndex],
                           platformName, platformWrapper.deviceNames[deviceIndex],
                           platformWrapper.context, platformWrapper.commandQueues[deviceIndex]);
    }
  }
}

void manager_t::configure_device(cl_device_id deviceId, json::node &devicesNode,
                                 std::vector<cl_device_id> &filteredDeviceIds,
                                 std::vector<std::string> &filteredDeviceNames,
                                 std::map<std::string, size_t> &countLimitMap,
                                 bool useConfiguration) {
  cl_int err;

  char deviceName[128] = {0};
  err = clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 128 * sizeof(char), &deviceName, nullptr);

  check(err, "OCL Error: Failed to read the device name");

  if (verbose) {
    std::cout << "OCL Info: detected device, name: \"" << deviceName << "\"" << std::endl;
  }

  // either the device has to be in the configuration or a new configuration is
  // created and every device is selected
  if (useConfiguration) {
    if (!devicesNode.contains(deviceName)) {
      return;
    }
  } else {
    if (!devicesNode.contains(deviceName)) {
      json::node &deviceNode = devicesNode.addDictAttr(deviceName);
      deviceNode.addDictAttr("KERNELS");
    }
  }

  // count the number of identical devices
  if (countLimitMap.count(deviceName) == 0) {
    countLimitMap[deviceName] = 1;
  } else {
    countLimitMap[deviceName] += 1;
  }

  if (useConfiguration && devicesNode[deviceName].contains("COUNT") &&
      devicesNode[deviceName].contains("SELECT")) {
    std::stringstream errorString;
    errorString << "error: manager: \"COUNT\" and \"SELECT\" "
                   "specified both for device : "
                << deviceName << std::endl;
    throw manager_error(errorString.str());
  }

  // limit the number of identical devices used, excludes a device selection
  if (devicesNode[deviceName].contains("COUNT")) {
    if (countLimitMap[deviceName] > devicesNode[deviceName]["COUNT"].getUInt()) {
      return;
    }
  }

  // check whether a specific device is to be selected
  if (devicesNode[deviceName].contains("SELECT")) {
    if (countLimitMap[deviceName] - 1 != devicesNode[deviceName]["SELECT"].getUInt()) {
      return;
    }
  }

  if (verbose) {
    std::cout << "OCL Info: using device, name: \"" << deviceName << "\"";
    if (devicesNode[deviceName].contains("SELECT")) {
      std::cout << " (selected device no.: " << devicesNode[deviceName]["SELECT"].getUInt() << ")";
    }
    std::cout << std::endl;
  }

  filteredDeviceIds.push_back(deviceId);
  filteredDeviceNames.push_back(deviceName);
  overallDeviceCount += 1;
}

std::vector<device_t> &manager_t::get_devices() { return this->devices; }

void manager_t::set_verbose(bool verbose) { this->verbose = verbose; }

std::string manager_t::read_src_file(const std::string &kernel_src_file_name) const {
  std::string kernel_src_string;
  std::ifstream f(kernel_src_file_name);
  if (f.is_open()) {
    f.seekg(0, std::ios::end);
    kernel_src_string.reserve(f.tellg());
    f.seekg(0, std::ios::beg);

    kernel_src_string.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    f.close();
    return kernel_src_string;
  } else {
    throw manager_error(std::string("could not open kernel src file: ") + kernel_src_file_name);
  }
}

void manager_t::release_kernel(cl_kernel kernel) { clReleaseKernel(kernel); }

void manager_t::release_kernels(std::vector<cl_kernel> kernels) {
  for (cl_kernel &kernel : kernels) {
    clReleaseKernel(kernel);
  }
}

}  // namespace opencl