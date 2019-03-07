#include "sgpp/datadriven/tools/ARFFTools.hpp"

int main(void) {

  std::string binary_header_filename(
      "../../DissertationCodeTesla1/SGpp/datasets_WPDM18/"
      "gaussian_c10_size1000000_dim10_noise");
  sgpp::base::DataMatrix trainingData =
      sgpp::datadriven::ARFFTools::read_binary_converted_ARFF(
          binary_header_filename);
  trainingData.toFile("gaussian_c10_size1000000_dim10_noise.arff");

  return 1;
}
