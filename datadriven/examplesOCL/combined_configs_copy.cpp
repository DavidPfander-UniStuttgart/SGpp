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
  if (variables_map.count("result_file") == 0) {
    std::cerr << "error: result_file not specified" << std::endl;
    return 1;
  }

  sgpp::base::OCLOperationConfiguration base(base_file);
  sgpp::base::OCLOperationConfiguration combine(combine_file);
  for (auto &key : combine.keys()) {
    std::unique_ptr<json::node> node =
        std::unique_ptr<json::node>(combine[key].clone());
    base.addAttribute(key, std::move(node));
  }

  base.serialize(result_file);

  std::cout << "done" << std::endl;

  return 0;
}
