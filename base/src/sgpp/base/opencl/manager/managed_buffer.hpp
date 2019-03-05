#pragma once

#include <CL/cl.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "device.hpp"
#include "manager.hpp"

namespace opencl {

template <class T>
class managed_buffer {
 private:
  device_t device;
  cl_mem ptr;
  size_t buffer_size;

 public:
  managed_buffer() : ptr(nullptr), buffer_size(0){};

  managed_buffer(device_t &device, const int buffer_size)
      : device(device), buffer_size(buffer_size) {
    cl_int err;
    ptr = clCreateBuffer(device.context, CL_MEM_READ_WRITE, sizeof(T) * buffer_size, nullptr, &err);
    opencl::check(err, "managed_buffer: clCreateBuffer failed");
  }
  managed_buffer(const managed_buffer &other) = delete;
  ~managed_buffer() {
    if (ptr) {
      clReleaseMemObject(ptr);
      ptr = nullptr;
    }
  }

  managed_buffer &operator=(const managed_buffer &) = delete;

  managed_buffer(managed_buffer &&other) {
    this->device = other.device;
    this->ptr = other.ptr;
    other.ptr = nullptr;
    this->buffer_size = other.buffer_size;
    other.buffer_size = 0;
  }

  managed_buffer &operator=(managed_buffer &&other) {
    this->device = other.device;
    this->ptr = other.ptr;
    other.ptr = nullptr;
    this->buffer_size = other.buffer_size;
    other.buffer_size = 0;
    return *this;
  }

  cl_mem get() { return ptr; }

  void to_device(const std::vector<T> &data) {
    if (!ptr) {
      throw std::runtime_error("managed_buffer: buffer not initialized");
    }
    if (data.size() > buffer_size) {
      throw std::runtime_error("managed_buffer: to_device: device buffer too small");
    }
    cl_int err;
    err = clEnqueueWriteBuffer(device.commandQueue, ptr, CL_TRUE, 0, sizeof(T) * data.size(),
                               data.data(), 0, nullptr, nullptr);
    opencl::check(err, "managed_buffer: clEnqueueWriteBuffer failed");
    clFinish(device.commandQueue);
  }

  void from_device(std::vector<T> &data) {
    if (!ptr) {
      throw std::runtime_error("managed_buffer: buffer not initialized");
    }
    // if (data.size() < buffer_size) {
    //   throw std::runtime_error("managed_buffer: from_device: host-side buffer too small");
    // }
    cl_int err;
    err = clEnqueueReadBuffer(device.commandQueue, ptr, CL_TRUE, 0,
                              sizeof(T) * std::min(buffer_size, data.size()), data.data(), 0,
                              nullptr, nullptr);
    opencl::check(err, "managed_buffer: clEnqueueReadBuffer failed");
    clFinish(device.commandQueue);
  }

  void fill_buffer(T value) {
    if (!ptr) {
      throw std::runtime_error("managed_buffer: buffer not initialized");
    }
    cl_int err;
    err = clEnqueueFillBuffer(device.commandQueue, ptr, &value, sizeof(value), 0,
                              buffer_size * sizeof(T), 0, nullptr, nullptr);
    opencl::check(err, "managed_buffer: clEnqueueFillBuffer failed");
    clFinish(device.commandQueue);
  }

  size_t size() { return buffer_size; }
};

template <>
class managed_buffer<bool> {
 private:
  device_t device;
  cl_mem ptr;
  size_t buffer_size;

 public:
  managed_buffer() : ptr(nullptr), buffer_size(0){};

  managed_buffer(device_t &device, const int buffer_size)
      : device(device), buffer_size(buffer_size) {
    cl_int err;
    ptr = clCreateBuffer(device.context, CL_MEM_READ_WRITE, sizeof(unsigned char) * buffer_size,
                         nullptr, &err);
    opencl::check(err, "managed_buffer: clCreateBuffer failed");
  }
  managed_buffer(const managed_buffer &other) = delete;
  ~managed_buffer() {
    if (ptr) {
      clReleaseMemObject(ptr);
      ptr = nullptr;
    }
  }

  managed_buffer &operator=(const managed_buffer &) = delete;

  managed_buffer(managed_buffer &&other) {
    this->device = other.device;
    this->ptr = other.ptr;
    other.ptr = nullptr;
    this->buffer_size = other.buffer_size;
    other.buffer_size = 0;
  }

  managed_buffer &operator=(managed_buffer &&other) {
    this->device = other.device;
    this->ptr = other.ptr;
    other.ptr = nullptr;
    this->buffer_size = other.buffer_size;
    other.buffer_size = 0;
    return *this;
  }

  cl_mem get() { return ptr; }

  void to_device(const std::vector<bool> &data) {
    if (!ptr) {
      throw std::runtime_error("managed_buffer: buffer not initialized");
    }
    if (data.size() > buffer_size) {
      throw std::runtime_error("managed_buffer: to_device: device buffer too small");
    }

    std::vector<unsigned char> data_temp(data.size());
    for (size_t i = 0; i < data.size(); i += 1) {
      data_temp[i] = data[i];
    }

    cl_int err;
    err = clEnqueueWriteBuffer(device.commandQueue, ptr, CL_TRUE, 0,
                               sizeof(unsigned char) * data_temp.size(), data_temp.data(), 0,
                               nullptr, nullptr);
    opencl::check(err, "managed_buffer: clEnqueueWriteBuffer failed");
    clFinish(device.commandQueue);
  }

  void from_device(std::vector<bool> &data) {
    if (!ptr) {
      throw std::runtime_error("managed_buffer: buffer not initialized");
    }
    // if (data.size() < buffer_size) {
    //   throw std::runtime_error("managed_buffer: from_device: host-side buffer too small");
    // }
    std::vector<unsigned char> data_temp(data.size());
    cl_int err;
    err = clEnqueueReadBuffer(device.commandQueue, ptr, CL_TRUE, 0,
                              sizeof(unsigned char) * std::min(buffer_size, data.size()),
                              data_temp.data(), 0, nullptr, nullptr);
    opencl::check(err, "managed_buffer: clEnqueueReadBuffer failed");
    clFinish(device.commandQueue);
    for (size_t i = 0; i < data.size(); i += 1) {
      data[i] = data_temp[i];
    }
  }

  void fill_buffer(bool value) {
    if (!ptr) {
      throw std::runtime_error("managed_buffer: buffer not initialized");
    }
    unsigned char value_temp = value;
    cl_int err;
    err = clEnqueueFillBuffer(device.commandQueue, ptr, &value_temp, sizeof(unsigned char), 0,
                              buffer_size * sizeof(unsigned char), 0, nullptr, nullptr);
    opencl::check(err, "managed_buffer: clEnqueueFillBuffer failed");
    clFinish(device.commandQueue);
  }

  size_t size() { return buffer_size; }
};

}  // namespace opencl
