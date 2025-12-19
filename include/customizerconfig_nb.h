#ifndef OSRM_NB_CUSTOMIZERCONFIG_H
#define OSRM_NB_CUSTOMIZERCONFIG_H

#include "osrm/customizer_config.hpp"

#include <nanobind/nanobind.h>

using osrm::customizer::CustomizationConfig;

void init_CustomizerConfig(nanobind::module_& m);

#endif //OSRM_NB_CUSTOMIZERCONFIG_H
