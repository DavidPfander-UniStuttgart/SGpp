#include <boost/program_options.hpp>
#include <iostream>
#include "sgpp/datadriven/tools/ARFFTools.hpp"

int main(int argc, char **argv) {
  std::string orig_filename = "";
  std::string output_filename = "";
  bool compressed;
  boost::program_options::options_description description("Allowed options");
  description.add_options()("help", "display help")(
      "input_filename",
      boost::program_options::value<std::string>(&orig_filename),
      "Path and filename to the ARFF dataset that is to be converted to binary.")(
      "output_filename",
      boost::program_options::value<std::string>(&output_filename),
      "Output dataset filename without extension (contains header <output_filename>.txt and binary data <filename_output>_bindata)")(
      "compression",
      boost::program_options::value<bool>(&compressed),
      "Should compression be used? (true false)");
  {
    boost::program_options::variables_map variables_map;
    boost::program_options::parsed_options options = parse_command_line(argc, argv, description);
    boost::program_options::store(options, variables_map);
    boost::program_options::notify(variables_map);
    if (variables_map.count("help")) {
      std::cout << description << std::endl;
      return 0;
    }
    if (variables_map.count("input_filename") == 0) {
      std::cerr << "error: option \"input_filename\" not specified" << std::endl;
      return 1;
    }
    if (variables_map.count("output_filename") == 0) {
      std::cerr << "error: option \"output_filename\" not specified" << std::endl;
      return 1;
    }
    if (variables_map.count("compression") == 0) {
      std::cerr << "error: option \"compression\" not specified (true false)" << std::endl;
      return 1;
    }
    sgpp::datadriven::ARFFTools::convert_into_binary_file(orig_filename, output_filename, compressed);
  }
}
