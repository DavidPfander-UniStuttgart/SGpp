// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org
#pragma once

#include <mpi.h>

#include <sgpp/datadriven/operation/hash/OperationDensityOCLMultiPlatform/OpFactory.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/MPIEnviroment.hpp>
#include <sgpp/datadriven/operation/hash/OperationMPI/MPIWorkerBase.hpp>

#include <algorithm>
#include <exception>
#include <sstream>
#include <string>

namespace sgpp {
namespace datadriven {
namespace clusteringmpi {

template <class T>
class MPIWorkerPackageBase : virtual public MPIWorkerBase {
 protected:
  bool opencl_node;
  int packagesize_multiplier;
  bool overseer_node;
  MPI_Comm &master_worker_comm;
  MPI_Comm &sub_worker_comm;
  bool prefetching;
  size_t packages_per_worker;
  bool force_balancing;
  std::shared_ptr<base::OCLOperationConfiguration> parameters;

  MPI_Datatype mpi_typ;
  long secondary_workpackage[2];
  size_t size;

  void divide_workpackages(long *package, T *erg) {
    // Divide into more work packages
    size_t packagesize = size;
    if (size == 0) {
      const size_t desired_packagecount = MPIEnviroment::get_sub_worker_count() * packages_per_worker;
      packagesize = package[1] / desired_packagecount;
    } else if (force_balancing) {
      size_t number_steps = package[1] / (size * MPIEnviroment::get_sub_worker_count());
      if (number_steps > 0) {
        std::cout << "number steps:" << number_steps << std::endl;
        const size_t last_step_size = package[1] % (size * MPIEnviroment::get_sub_worker_count());
        std::cout << "last step:" << last_step_size << std::endl;
        packagesize += (last_step_size / (MPIEnviroment::get_sub_worker_count() * number_steps));
        // The rest of the workitems will be handled per default by the last package
      }
    }
    SimpleQueue<T> workitem_queue(package[0], package[1], packagesize, sub_worker_comm,
                                  MPIEnviroment::get_sub_worker_count(), verbose, prefetching);
    int chunkid = package[0];
    size_t messagesize = 0;
    while (!workitem_queue.is_finished()) {
      // Store result
      messagesize = workitem_queue.receive_result( erg, packagesize_multiplier);
    }
  }

  virtual void receive_and_send_initial_data(void) = 0;
  virtual void begin_opencl_operation(long *workpackage) = 0;
  virtual void finalize_opencl_operation(T *result_buffer, long *workpackage) = 0;

 private:
  void augment_ocl_configuration(std::string chosen_device_name, size_t device_select) {
    if (opencl_node) {
      // Get possible device names for error message
      std::vector<std::string> possible_devicenames;
      for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
        json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
        for (std::string &deviceName : platformNode["DEVICES"].keys()) {
          possible_devicenames.push_back(deviceName);
        }
      }
      // Insert/Update select on node level using opencl_platform and opencl_device
      bool found_device = false;
      for (std::string &platformName : (*parameters)["PLATFORMS"].keys()) {
        json::Node &platformNode = (*parameters)["PLATFORMS"][platformName];
        for (std::string &deviceName : platformNode["DEVICES"].keys()) {
          json::Node &deviceNode = platformNode["DEVICES"][deviceName];
          if (chosen_device_name == std::string("Unknown")) {
            chosen_device_name = deviceName;
            found_device = true;
          }
          if (deviceName == chosen_device_name) {
            if (deviceNode.contains("COUNT")) {
              std::cerr << "Warning: Node " << MPIEnviroment::get_node_rank()
                        << " is going to delete COUNT entry for " << deviceName
                        << " and will insert its own SELECT entry with the"
                        << "value given in the MPI config!" << std::endl;
              deviceNode.removeAttribute("COUNT");
            }
            if (deviceNode.contains("SELECT")) {
              std::cerr << "Warning: Node " << MPIEnviroment::get_node_rank()
                        << " is going to overwrite SELECT entry for "
                        << deviceName << " with the value given in the MPI config" << std::endl;
              deviceNode.replaceIDAttr("SELECT", static_cast<int64_t>(device_select));
            } else {
              deviceNode.addIDAttr("SELECT", static_cast<int64_t>(device_select));
            }
            found_device = true;
            break;
          } else {
            if (deviceNode.contains("COUNT")) {
              std::cerr << "Warning: Node " << MPIEnviroment::get_node_rank()
                        << " is going to delete COUNT entry for " << deviceName
                        << " and will insert its own SELECT entry with the"
                        << "value given in the MPI config!" << std::endl;
              deviceNode.removeAttribute("COUNT");
            }
            if (deviceNode.contains("SELECT")) {
              std::cerr << "Warning: Node " << MPIEnviroment::get_node_rank()
                        << " is going to overwrite SELECT entry for "
                        << deviceName << " with the value given in the MPI config" << std::endl;
              deviceNode.replaceIDAttr("SELECT", static_cast<int64_t>(9999));
            } else {
              deviceNode.addIDAttr("SELECT", static_cast<int64_t>(9999));
            }
          }
        }
      }
      if (!found_device) {
        std::stringstream errorString;
        errorString << "Specified device \"" << chosen_device_name
                    << "\" could not be found on MPI rank "
                    << MPIEnviroment::get_node_rank() << std::endl;
        errorString << "Specify one device for each opencl worker in the"
                    << " MPI config file with OPENCL_DEVICE_NAME.\n"
                    << "To pick a specific device of a certain name use "
                    << "OPENCL_DEVICE_SELECT for an additional ID.\n";
        errorString << "Possible device names: " << std::endl;
        for (auto &name : possible_devicenames) {
          errorString << name << std::endl;
        }
        throw sgpp::base::operation_exception(errorString.str());
      }
    }
  }

 public:
  MPIWorkerPackageBase(std::string operationName, int multiplier)
      : MPIWorkerBase(operationName),
        opencl_node(false),
        packagesize_multiplier(multiplier),
        overseer_node(false),
        force_balancing(true),
        master_worker_comm(MPIEnviroment::get_input_communicator()),
        sub_worker_comm(MPIEnviroment::get_communicator()),
        prefetching(false),
        parameters(nullptr) {
    if (std::is_same<T, int>::value) {
      mpi_typ = MPI_INT;
    } else if (std::is_same<T, float>::value) {
      mpi_typ = MPI_FLOAT;
    } else if (std::is_same<T, double>::value) {
      mpi_typ = MPI_DOUBLE;
    } else if (std::is_same<T, long>::value) {
      mpi_typ = MPI_LONG;
    } else {
      std::stringstream errorString;
      errorString << "Unsupported datatyp in class MPIWorkerPackageBase." << std::endl
                  << "Template class needs to be int, long, float or double." << std::endl;
      throw std::logic_error(errorString.str());
    }
    if (MPIEnviroment::get_sub_worker_count() > 0) {
      overseer_node = true;
      opencl_node = false;
    } else {
      overseer_node = false;
      opencl_node = true;
    }
    std::string chosen_device_name("Unknown");
    size_t device_select = 0;
    if (MPIEnviroment::get_configuration().contains("PREFERED_PACKAGESIZE"))
      std::cerr << "Warning! Flag PREFERED_PACKAGESIZE is deprecated!"
                << " Use PACKAGESIZE instead or rely on the defaults." << std::endl;
    if (MPIEnviroment::get_configuration().contains("PACKAGESIZE"))
      size = MPIEnviroment::get_configuration()["PACKAGESIZE"].getUInt();
    else
      size = 0;
    if (MPIEnviroment::get_configuration().contains("REDISTRIBUTE"))
      std::cerr << "Warning! Flag REDISTRIBUTE is deprecated and will be ignored!"
                << std::endl;
    if (MPIEnviroment::get_configuration().contains("PREFETCHING"))
      prefetching = MPIEnviroment::get_configuration()["PREFETCHING"].getBool();
    if (MPIEnviroment::get_configuration().contains("OPENCL_DEVICE_NAME"))
      chosen_device_name = MPIEnviroment::get_configuration()["OPENCL_DEVICE_NAME"].get();
    if (MPIEnviroment::get_configuration().contains("OPENCL_DEVICE_SELECT"))
      device_select = MPIEnviroment::get_configuration()["OPENCL_DEVICE_SELECT"].getUInt();
    if (MPIEnviroment::get_configuration().contains("PACKAGAGES_PER_WORKER"))
      packages_per_worker = MPIEnviroment::get_configuration()["PACKAGES_PER_WORKER"].getUInt();
    else
      packages_per_worker = 2;
    if (MPIEnviroment::get_configuration().contains("FORCE_BALANCING"))
      force_balancing = MPIEnviroment::get_configuration()["FORCE_BALANCING"].getBool();

    if (MPIEnviroment::get_configuration().contains("PACKAGAGES_PER_WORKER") &&
        MPIEnviroment::get_configuration().contains("PACKAGESIZE")) {
      std::stringstream errorString;
      errorString << "Both PACKAGESIZE and PACKAGES_PER_WORKER are specified for node"
                  << MPIEnviroment::get_node_rank() << std::endl
                  << "These are mutually exclusive! Use only one of them" << std::endl;
      throw std::logic_error(errorString.str());
    }
    if (MPIEnviroment::get_configuration().contains("FORCE_BALANCING") &&
        !MPIEnviroment::get_configuration().contains("PACKAGESIZE") &&
        MPIEnviroment::get_configuration()["FORCE_BALANCING"].getBool()) {
      std::stringstream errorString;
      errorString << "FORCE_BALANCING is used without PACKAGESIZE on node "
                  << MPIEnviroment::get_node_rank() << std::endl;
      errorString << "FORCE_BALANCING only works with specified PACKAGESIZE. Otherwise the"
                  << "number of packages is known anyway and there is no need for rebalancing"
                  << std::endl;
      throw std::logic_error(errorString.str());
    }

    // receive opencl configuration
    MPI_Status stat;
    int messagesize = 0;
    MPI_Probe(0, 1, master_worker_comm, &stat);
    MPI_Get_count(&stat, MPI_CHAR, &messagesize);
    char *serialized_conf = new char[messagesize];
    MPI_Recv(serialized_conf, messagesize, MPI_CHAR, stat.MPI_SOURCE, stat.MPI_TAG,
             master_worker_comm, &stat);
    // Send OCL configuration
    for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
      MPI_Send(serialized_conf, messagesize, MPI_CHAR, dest, 1, sub_worker_comm);
    parameters = std::make_shared<base::OCLOperationConfiguration>();
    parameters->deserialize(serialized_conf);
    delete[] serialized_conf;

    // Augment opencl config if necessary
    augment_ocl_configuration(chosen_device_name, device_select);

  }
  MPIWorkerPackageBase(std::string operationName, int multiplier, std::string ocl_conf_filename)
      : MPIWorkerBase(operationName),
        opencl_node(false),
        packagesize_multiplier(multiplier),
        overseer_node(false),
        force_balancing(true),
        master_worker_comm(MPIEnviroment::get_input_communicator()),
        sub_worker_comm(MPIEnviroment::get_communicator()),
        prefetching(false),
        parameters(nullptr) {
    if (std::is_same<T, int>::value) {
      mpi_typ = MPI_INT;
    } else if (std::is_same<T, float>::value) {
      mpi_typ = MPI_FLOAT;
    } else if (std::is_same<T, double>::value) {
      mpi_typ = MPI_DOUBLE;
    } else if (std::is_same<T, long>::value) {
      mpi_typ = MPI_LONG;
    } else {
      std::stringstream errorString;
      errorString << "Unsupported datatyp in class MPIWorkerPackageBase." << std::endl
                  << "Template class needs to be int, long, float or double." << std::endl;
      throw std::logic_error(errorString.str());
    }
    if (MPIEnviroment::get_sub_worker_count() > 0) {
      overseer_node = true;
      opencl_node = false;
    } else {
      overseer_node = false;
      opencl_node = true;
    }
    std::string chosen_device_name("Unknown");
    size_t device_select = 0;
    if (MPIEnviroment::get_configuration().contains("PREFERED_PACKAGESIZE"))
      std::cerr << "Warning! Flag PREFERED_PACKAGESIZE is deprecated!"
                << " Use PACKAGESIZE instead or rely on the defaults." << std::endl;
    if (MPIEnviroment::get_configuration().contains("PACKAGESIZE"))
      size = MPIEnviroment::get_configuration()["PACKAGESIZE"].getUInt();
    else
      size = 0;
    if (MPIEnviroment::get_configuration().contains("REDISTRIBUTE"))
      std::cerr << "Warning! Flag REDISTRIBUTE is deprecated and will be ignored!"
                << std::endl;
    if (MPIEnviroment::get_configuration().contains("PREFETCHING"))
      prefetching = MPIEnviroment::get_configuration()["PREFETCHING"].getBool();
    if (MPIEnviroment::get_configuration().contains("OPENCL_DEVICE_NAME"))
      chosen_device_name = MPIEnviroment::get_configuration()["OPENCL_DEVICE_NAME"].get();
    if (MPIEnviroment::get_configuration().contains("OPENCL_DEVICE_SELECT"))
      device_select = MPIEnviroment::get_configuration()["OPENCL_DEVICE_SELECT"].getUInt();
    if (MPIEnviroment::get_configuration().contains("PACKAGAGES_PER_WORKER"))
      packages_per_worker = MPIEnviroment::get_configuration()["PACKAGES_PER_WORKER"].getUInt();
    else
      packages_per_worker = 2;
    if (MPIEnviroment::get_configuration().contains("FORCE_BALANCING"))
      force_balancing = MPIEnviroment::get_configuration()["FORCE_BALANCING"].getBool();

    if (MPIEnviroment::get_configuration().contains("PACKAGAGES_PER_WORKER") &&
        MPIEnviroment::get_configuration().contains("PACKAGESIZE")) {
      std::stringstream errorString;
      errorString << "Both PACKAGESIZE and PACKAGES_PER_WORKER are specified for node"
                  << MPIEnviroment::get_node_rank() << std::endl
                  << "These are mutually exclusive! Use only one of them" << std::endl;
      throw std::logic_error(errorString.str());
    }
    if (MPIEnviroment::get_configuration().contains("FORCE_BALANCING") &&
        !MPIEnviroment::get_configuration().contains("PACKAGESIZE") &&
        MPIEnviroment::get_configuration()["FORCE_BALANCING"].getBool()) {
      std::stringstream errorString;
      errorString << "FORCE_BALANCING is used without PACKAGESIZE on node "
                  << MPIEnviroment::get_node_rank() << std::endl;
      errorString << "FORCE_BALANCING only works with specified PACKAGESIZE. Otherwise the"
                  << "number of packages is known anyway and there is no need for rebalancing"
                  << std::endl;
      throw std::logic_error(errorString.str());
    }

    parameters = std::make_shared<base::OCLOperationConfiguration>(ocl_conf_filename);
    std::ostringstream sstream;
    parameters->serialize(sstream, 0);
    std::string serialized_conf = sstream.str();
    char *conf_message = new char[serialized_conf.length() + 1];
    std::copy(serialized_conf.begin(), serialized_conf.end(), conf_message);
    conf_message[serialized_conf.size()] = '\0';
    // Send OCL configuration
    for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
      MPI_Send(conf_message, static_cast<int>(serialized_conf.size() + 1), MPI_CHAR, dest, 1,
               sub_worker_comm);
    delete[] conf_message;

    // Augment opencl config if necessary
    augment_ocl_configuration(chosen_device_name, device_select);
  }
  virtual ~MPIWorkerPackageBase() {}

  void start_worker_main(void) {
    MPI_Status stat;
    MPI_Request request[2];
    receive_and_send_initial_data();
    // Work Loop
    long datainfo[2];
    T *partial_results[2];
    partial_results[0] = NULL;
    partial_results[1] = NULL;
    long buffersizes[2];
    buffersizes[0] = 0;
    buffersizes[1] = 0;
    bool first_package = true;
    unsigned int currentbuffer = 0;
    do {
      if (!prefetching || first_package) {
        // Receive Workpackage
        MPI_Probe(0, 1, master_worker_comm, &stat);
        MPI_Recv(datainfo, 2, MPI_LONG, 0, stat.MPI_TAG, master_worker_comm, &stat);
        if (verbose) {
          std::cout << "Received workpackage [" << datainfo[0] << "," << datainfo[1] << "] on "
                    << MPIEnviroment::get_node_rank() << std::endl;
        }
        first_package = false;
      } else {
        datainfo[0] = secondary_workpackage[0];
        datainfo[1] = secondary_workpackage[1];
      }
      // Check for exit
      if (datainfo[0] == -2 && datainfo[1] == -2) {
        if (verbose) {
          std::cerr << "Node" << MPIEnviroment::get_node_rank() << " received exit signal"
                    << std::endl;
        }
        for (int dest = 1; dest < MPIEnviroment::get_sub_worker_count() + 1; dest++)
          MPI_Send(datainfo, 2, MPI_LONG, dest, 1, sub_worker_comm);
        break;
      } else if (datainfo[0] == -1 && datainfo[1] == -1) {
        // Receive exitpackage
        MPI_Probe(0, 1, master_worker_comm, &stat);
        MPI_Recv(secondary_workpackage, 2, MPI_LONG, 0, stat.MPI_TAG, master_worker_comm, &stat);
        continue;
      } else {
        if (datainfo[1] * packagesize_multiplier != buffersizes[currentbuffer] ||
            partial_results[currentbuffer] == NULL) {
          if (partial_results[currentbuffer] != NULL) delete[] partial_results[currentbuffer];
          partial_results[currentbuffer] = new T[datainfo[1] * packagesize_multiplier];
          buffersizes[currentbuffer] = datainfo[1] * packagesize_multiplier;
        }
        if (opencl_node) {
          // Run partial opencl operation
          begin_opencl_operation(datainfo);
          if (prefetching) {
            // Prefetch secondary workpackage
            MPI_Probe(0, 1, master_worker_comm, &stat);
            MPI_Recv(secondary_workpackage, 2, MPI_LONG, 0, stat.MPI_TAG, master_worker_comm, &stat);
            if (verbose) {
              std::cout << "Received secondary workpackage [" << secondary_workpackage[0] << ","
                        << secondary_workpackage[1] << "] on " << MPIEnviroment::get_node_rank()
                        << std::endl;
            }
          }
          // Finish opencl operation
          finalize_opencl_operation(partial_results[currentbuffer], datainfo);
        } else {
          if (prefetching) {
            // Prefetch secondary workpackage
            MPI_Probe(0, 1, master_worker_comm, &stat);
            MPI_Recv(secondary_workpackage, 2, MPI_LONG, 0, stat.MPI_TAG, master_worker_comm, &stat);
            if (verbose) {
              std::cout << "Received workpackage [" << secondary_workpackage[0] << ","
                        << secondary_workpackage[1] << "] on " << MPIEnviroment::get_node_rank()
                        << std::endl;
            }
          }
          divide_workpackages(datainfo, partial_results[currentbuffer]);
        }
        // Send results back
        MPI_Isend(partial_results[currentbuffer], datainfo[1] * packagesize_multiplier, mpi_typ, 0,
                  1, master_worker_comm, request + currentbuffer);
        currentbuffer = (currentbuffer + 1) % 2;
        // Wait for the old message beofre reusing the buffer
        if (buffersizes[currentbuffer] != 0) MPI_Wait(request + currentbuffer, &stat);
      }
    } while (true);
    if (partial_results[0] != NULL) delete[] partial_results[0];
    if (partial_results[1] != NULL) delete[] partial_results[1];
  }
};

}  // namespace clusteringmpi
}  // namespace datadriven
}  // namespace sgpp
