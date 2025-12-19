#ifndef OSRM_NB_EXTRACTORCONFIG_H
#define OSRM_NB_EXTRACTORCONFIG_H

#include "osrm/extractor_config.hpp"

#include <nanobind/nanobind.h>

using osrm::extractor::ExtractorConfig;

void init_ExtractorConfig(nanobind::module_& m);

#endif //OSRM_NB_EXTRACTORCONFIG_H
