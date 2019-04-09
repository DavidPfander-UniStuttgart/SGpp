// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include <vector>

int main(int argc, char **argv) {
  std::string base_file;
  std::string combine_file;
  std::vector<std::string> combine_prefixes;
  std::string kernel_name;
  std::string result_file;

  boost::program_options::options_description description("Allowed options");
  description.add_options()("help", "display help")(
      "base_file", boost::program_options::value<std::string>(&base_file),
      "config file mostly copied")(
      "combine_file", boost::program_options::value<std::string>(&combine_file),
      "config file to merge into the base file")(
      "combine_prefixes",
      boost::program_options::value<std::vector<std::string>>(&combine_prefixes)
          ->multitoken(),
      "name prefixes of the parameters to copy from merge to base file")(
      "kernel_name", boost::program_options::value<std::string>(&kernel_name),
      "name of the kernel to merge")(
      "result_file", boost::program_options::value<std::string>(&result_file),
      "name of the result file");

  boost::program_options::variables_map variables_map;
  boost::program_options::parsed_options options =
      parse_command_line(argc, argv, description);
  boost::program_options::store(options, variables_map);
  boost::program_options::notify(variables_map);

  if (variables_map.count("help")) {
    std::cout << description << std::endl;
    return 0;
  }
  if (variables_map.count("base_file") == 0) {
    std::cerr << "error: base_file not specified" << std::endl;
    return 1;
  }
  if (variables_map.count("combine_file") == 0) {
    std::cerr << "error: base_file not specified" << std::endl;
    return 1;
  }
  if (variables_map.count("combine_prefixes") == 0) {
    std::cerr << "error: combine_prefixes not specified" << std::endl;
    return 1;
  }
  if (variables_map.count("kernel_name") == 0) {
    std::cerr << "error: kernel_name not specified" << std::endl;
    return 1;
  }
  if (variables_map.count("result_file") == 0) {
    std::cerr << "error: result_file not specified" << std::endl;
    return 1;
  }

  sgpp::base::OCLOperationConfiguration base(base_file);
  sgpp::base::OCLOperationConfiguration combine(combine_file);
  for (std::string &platform_name : base["PLATFORMS"].keys()) {
    for (std::string &device_name :
         base["PLATFORMS"][platform_name]["DEVICES"].keys()) {
      auto &device_node =
          base["PLATFORMS"][platform_name]["DEVICES"][device_name];
      if (device_node["KERNELS"].contains(kernel_name)) {
        auto &kernel_node_base = device_node["KERNELS"][kernel_name];
        if (!combine["PLATFORMS"].contains(platform_name)) {
          continue;
        }
        std::cout << "platform found!" << std::endl;
        if (!combine["PLATFORMS"][platform_name]["DEVICES"].contains(
                device_name)) {
          continue;
        }
        std::cout << "device found!" << std::endl;
        if (!combine["PLATFORMS"][platform_name]["DEVICES"][device_name]
                    ["KERNELS"]
                        .contains(kernel_name)) {
          continue;
        }
        std::cout << "platform_name: \"" << platform_name
                  << "\" device_name: \"" << device_name << "\""
                  << " kernel found!" << std::endl;
        auto &device_node_combine =
            combine["PLATFORMS"][platform_name]["DEVICES"][device_name];
        auto &kernel_node_combine = device_node_combine["KERNELS"][kernel_name];
        for (std::string &parameter_name : kernel_node_combine.keys()) {
          for (std::string &combine_prefix : combine_prefixes) {
            if (boost::starts_with(parameter_name, combine_prefix)) {
              std::cout << "parameter: " << parameter_name << " updated!"
                        << std::endl;
              kernel_node_base.replaceTextAttr(
                  parameter_name, kernel_node_combine[parameter_name].get());
            }
          }
        }
      }
    }
  }

  base.serialize(result_file);

  std::cout << "done" << std::endl;

  return 0;
}
