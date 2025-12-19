#ifndef OSRM_NB_PARTITIONERCONFIG_H
#define OSRM_NB_PARTITIONERCONFIG_H

#include "osrm/partitioner_config.hpp"

#include <nanobind/nanobind.h>

using osrm::partitioner::PartitionerConfig;

void init_PartitionerConfig(nanobind::module_& m);

#endif //OSRM_NB_PARTITIONERCONFIG_H
