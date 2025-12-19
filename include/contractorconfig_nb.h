#ifndef OSRM_NB_CONTRACTORCONFIG_H
#define OSRM_NB_CONTRACTORCONFIG_H

#include "osrm/contractor_config.hpp"

#include <nanobind/nanobind.h>

using osrm::contractor::ContractorConfig;

void init_ContractorConfig(nanobind::module_& m);

#endif //OSRM_NB_CONTRACTORCONFIG_H
