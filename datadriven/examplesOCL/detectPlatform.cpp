// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <boost/program_options.hpp>
#include <iostream>
#include "sgpp/base/opencl/OCLManagerMultiPlatform.hpp"
#include "sgpp/base/opencl/OCLOperationConfiguration.hpp"

int main(int argc, char** argv) {
  std::string precision;
  std::string file_name;

  boost::program_options::options_description description("Allowed options");
  description.add_options()("help", "display help")(
      "precision", boost::program_options::value<std::string>(&precision)->default_value("double"),
      "set precision in generated config file")(
      "file_name",
      boost::program_options::value<std::string>(&file_name)->default_value("detectedPlatform.cfg"),
      "name of the generated config file");

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

  configuration->serialize(file_name);

  std::cout << "done" << std::endl;

  return 0;
}
