// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include <sgpp/base/exception/file_exception.hpp>
#include <sgpp/datadriven/tools/ARFFTools.hpp>

#include <sgpp/globaldef.hpp>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <cstring>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace sgpp {
namespace datadriven {

Dataset ARFFTools::readARFF(const std::string& filename, bool hasTargets) {
  // TODO(fuchsgruber): No idea if this arff interface really can handle data without classes
  std::string line;
  std::ifstream myfile(filename.c_str());
  if (!myfile) {
    const auto msg = "Unable to open file: " + filename;
    throw sgpp::base::file_exception(msg.c_str());
  }
  size_t numberInstances;
  size_t dimension;
  bool dataReached = false;
  size_t instanceNo = 0;

  readARFFSize(filename, numberInstances, dimension);
  Dataset dataset(numberInstances, dimension);

  while (!myfile.eof()) {
    std::getline(myfile, line);
    std::transform(line.begin(), line.end(), line.begin(), toupper);

    if (dataReached && !line.empty()) {
      if (hasTargets)
        writeNewClass(line, dataset.getTargets(), instanceNo);
      writeNewTrainingDataEntry(line, dataset.getData(), instanceNo);
      instanceNo++;
    }

    if (line.find("@DATA", 0) != line.npos) {
      dataReached = true;
    }
  }

  myfile.close();

  return dataset;
}
std::string ARFFTools::readARFFHeader(const std::string &filename, long &offset) {
  std::string line;
  std::string result;
  std::ifstream myfile(filename.c_str());
  if (!myfile) {
    const auto msg = "Unable to open file: " + filename;
    throw sgpp::base::file_exception(msg.c_str());
  }
  bool dataReached = false;
  while (!myfile.eof()) {
    std::getline(myfile, line);
    std::transform(line.begin(), line.end(), line.begin(), toupper);
    bool reached_data =
        (line.find("@DATA", 0) != line.npos);
    if (reached_data) {
      dataReached = true;
      offset = result.size() * sizeof(char);
      break;
    } else {
      result.append(line);
      result.append("\n");
    }
  }
  myfile.close();
  return result;
}

#ifdef USE_MPI
Dataset ARFFTools::distributed_readARFF(const std::string& filename, const int offset, const
                                        int number_instances, const int dimensions) {

  Dataset dataset(number_instances, dimensions);
  double *raw_pointer = dataset.getData().data();
  MPI_File fh;
  MPI_Status stat;
  MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  MPI_File_set_view(fh, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
  MPI_File_read_at(fh, offset, raw_pointer, number_instances * dimensions, MPI_DOUBLE, &stat);
  MPI_File_close(&fh);
  return dataset;
}
#endif

void ARFFTools::readARFFSize(const std::string& filename, size_t& numberInstances,
                             size_t& dimension) {
  std::string line;
  std::ifstream myfile(filename.c_str());
  dimension = 0;
  numberInstances = 0;

  if (!myfile) {
    std::string msg = "Unable to open file: " + filename;
    throw sgpp::base::file_exception(msg.c_str());
  }

  while (!myfile.eof()) {
    std::getline(myfile, line);
    std::transform(line.begin(), line.end(), line.begin(), toupper);

    if (line.find("@ATTRIBUTE class", 0) != line.npos) {
    } else if (line.find("@ATTRIBUTE CLASS", 0) != line.npos) {
    } else if (line.find("@ATTRIBUTE", 0) != line.npos) {
      dimension++;
    } else if (line.find("@DATA", 0) != line.npos) {
      numberInstances = 0;
    } else if (line.find("% DATA SET SIZE ", 0) != line.npos) {
      numberInstances = std::stoi(line.substr(strlen("% DATA SET SIZE ")));
      std::cout << "Set number instances from comment to " << numberInstances << std::endl;
      break;
    } else if (!line.empty()) {
      numberInstances++;
    }
  }
  myfile.close();
}

void ARFFTools::convert_into_binary_file(const std::string &orig_filename, const std::string
                                         &header_filename, const std::string &output_filename) {
  std::string line;
  std::ifstream myfile(orig_filename.c_str());
  if (!myfile) {
    const auto msg = "Unable to open file: " + orig_filename;
    throw sgpp::base::file_exception(msg.c_str());
  }
  size_t numberInstances = 0;
  size_t dimension = 0;
  size_t instanceNo = 0;
  readARFFSize(orig_filename, numberInstances, dimension);

  // read header
  std::string headercontent;
  while (!myfile.eof()) {
    std::getline(myfile, line);
    std::transform(line.begin(), line.end(), line.begin(), toupper);
    if (line.find("@ATTRIBUTE class", 0) != line.npos) {
    } else if (line.find("@ATTRIBUTE CLASS", 0) != line.npos) {
    } else if (line.find("@ATTRIBUTE", 0) != line.npos) {
      dimension++;
    } else if (line.find("@DATA", 0) != line.npos) {
      break;
    }
    headercontent.append(line);
    headercontent.append("\n");
  }
  headercontent.append("% DATA SET SIZE " + std::to_string(numberInstances));
  headercontent.append("\n");
  headercontent.append("@DATA " + output_filename);

  // read data
  size_t dataindex = 0;
  std::vector<double> dataset(numberInstances * dimension);
  char tmp;
  while (!myfile.eof()) {
    std::getline(myfile, line);
    std::transform(line.begin(), line.end(), line.begin(), toupper);
    std::stringstream parser(line);
    for (size_t i = 0; i < dimension; i++, dataindex++) {
      parser >> dataset[dataindex];
      if (i != dimension -1 )
        parser >> tmp;
    }
  }
  myfile.close();

  // write binary file
  std::ofstream fout(output_filename, std::ios::out | std::ios::binary);
  fout.write((char*)&dataset[0], dataset.size() * sizeof(double));
  fout.close();

  // write header file
  std::ofstream hout(header_filename, std::ios::out);
  hout << headercontent;
  hout.close();
}

void ARFFTools::readARFFSizeFromString(const std::string& content, size_t& numberInstances,
                                       size_t& dimension) {
  std::string line;
  std::istringstream contentStream(content);

  dimension = 0;
  numberInstances = 0;

  while (!contentStream.eof()) {
    std::getline(contentStream, line);
    std::transform(line.begin(), line.end(), line.begin(), toupper);

    if (line.find("@ATTRIBUTE class", 0) != line.npos) {
    } else if (line.find("@ATTRIBUTE CLASS", 0) != line.npos) {
    } else if (line.find("@ATTRIBUTE", 0) != line.npos) {
      dimension++;
    } else if (line.find("@DATA", 0) != line.npos) {
      numberInstances = 0;
    } else if (!line.empty()) {
      numberInstances++;
    }
  }
}

Dataset ARFFTools::readARFFFromString(const std::string& content, bool hasTargets) {
  std::string line;
  std::stringstream contentStream;
  contentStream << content;
  size_t numberInstances;
  size_t dimension;
  bool dataReached = false;
  size_t instanceNo = 0;

  ARFFTools::readARFFSizeFromString(content, numberInstances, dimension);
  Dataset dataset(numberInstances, dimension);

  while (!contentStream.eof()) {
    std::getline(contentStream, line);
    std::transform(line.begin(), line.end(), line.begin(), toupper);

    if (dataReached && !line.empty()) {
      if (hasTargets)
        writeNewClass(line, dataset.getTargets(), instanceNo);
      writeNewTrainingDataEntry(line, dataset.getData(), instanceNo);
      instanceNo++;
    }

    if (line.find("@DATA", 0) != line.npos) {
      dataReached = true;
    }
  }

  return dataset;
}

void ARFFTools::writeNewTrainingDataEntry(const std::string& arffLine,
                                          sgpp::base::DataMatrix& destination, size_t instanceNo) {
  size_t cur_pos = 0;
  size_t cur_find = 0;
  size_t dim = destination.getNcols();
  std::string cur_value;
  double dbl_cur_value;

  for (size_t i = 0; i < dim; i++) {
    cur_find = arffLine.find(",", cur_pos);
    cur_value = arffLine.substr(cur_pos, cur_find - cur_pos);
    dbl_cur_value = atof(cur_value.c_str());
    destination.set(instanceNo, i, dbl_cur_value);
    cur_pos = cur_find + 1;
  }
}

void ARFFTools::writeNewClass(const std::string& arffLine, sgpp::base::DataVector& destination,
                              size_t instanceNo) {
  size_t cur_pos = arffLine.find_last_of(",");
  std::string cur_value = arffLine.substr(cur_pos + 1);
  double dbl_cur_value = atof(cur_value.c_str());
  destination.set(instanceNo, dbl_cur_value);
}

// void ARFFTools::writeAlpha(std::string tfilename, sgpp::base::DataVector& source)
// {
//
// }

// void ARFFTools::readAlpha(std::string tfilename, sgpp::base::DataVector& destination)
// {
//
// }

}  // namespace datadriven
}  // namespace sgpp
