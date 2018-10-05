// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <boost/algorithm/string.hpp>
#include <boost/program_options.hpp>
#include <iostream>
#include "sgpp/base/opencl/OCLManagerMultiPlatform.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"

int main(int argc, char **argv) {
  std::string precision;
  std::string file_name;
  std::string select;

  boost::program_options::options_description description("Allowed options");
  description.add_options()("help", "display help")(
      "precision", boost::program_options::value<std::string>(&precision)->default_value("double"),
      "set precision in generated config file")(
      "file_name",
      boost::program_options::value<std::string>(&file_name)->default_value("detectedPlatform.cfg"),
      "name of the generated config file")("select",
                                           boost::program_options::value<std::string>(&select),
                                           "select a specific platform and device")(
      "remove_unselected", "select a specific platform and device");

  boost::program_options::variables_map variables_map;
  boost::program_options::parsed_options options = parse_command_line(argc, argv, description);
  boost::program_options::store(options, variables_map);
  boost::program_options::notify(variables_map);

  if (variables_map.count("help")) {
    std::cout << description << std::endl;
    return 0;
  }
  if (variables_map.count("precision") != 0) {
    if (precision.compare("float") != 0 && precision.compare("double") != 0) {
      std::cerr << "error: value for precision can only be \"float\" or \"double\"" << std::endl;
      return 1;
    }
  }

  sgpp::base::OCLManagerMultiPlatform manager;
  manager.set_verbose(true);

  auto configuration = manager.getConfiguration();
  if (precision.compare("float") == 0) {
    configuration->replaceTextAttr("INTERNAL_PRECISION", "float");
  }

  // --select platform/name
  if (select.compare("") != 0) {
    std::vector<std::string> platform_device;
    boost::split(platform_device, select, boost::is_any_of("/"));
    if (platform_device.size() != 2) {
      throw;
    }
    std::string platform_name(platform_device[0]);
    std::string device_name(platform_device[1]);
    std::cout << "platform_name: \"" << platform_name << "\"" << std::endl;
    std::cout << "device_name : \"" << device_name << "\"" << std::endl;
    (*configuration)["PLATFORMS"][platform_name]["DEVICES"][device_name].replaceTextAttr("SELECT",
                                                                                         "0");
  }

  if (variables_map.count("remove_unselected")) {
    for (const std::string &p_name : (*configuration)["PLATFORMS"].keys()) {
      auto &p = (*configuration)["PLATFORMS"][p_name];
      bool has_select = false;
      for (const std::string &d_name : p["DEVICES"].keys()) {
        auto &d = p["DEVICES"][d_name];
        if (!d.contains("SELECT")) {
          d.erase();
        } else {
          has_select = true;
        }
      }
      if (!has_select) {
        p.erase();
      }
    }
  }

  configuration->serialize(file_name);

  std::cout << "done" << std::endl;

  return 0;
}
