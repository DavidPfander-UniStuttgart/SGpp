#pragma once

#include <CL/cl.h>
#include <string>

namespace opencl {

class device_t {
public:
  cl_platform_id platformId;
  cl_device_id deviceId;
  std::string platformName;
  std::string deviceName;
  cl_context context;
  cl_command_queue commandQueue;

  device_t() {}

  device_t(cl_platform_id platformId, cl_device_id deviceId,
           const std::string &platformName, const std::string &deviceName,
           cl_context context, cl_command_queue commandQueue)
      : platformId(platformId), deviceId(deviceId), platformName(platformName),
        deviceName(deviceName), context(context), commandQueue(commandQueue) {}
};
} // namespace opencl
