#include "contractorconfig_nb.h"

#include "osrm/contractor_config.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>

namespace nb = nanobind;

void init_ContractorConfig(nb::module_& m) {
    using osrm::contractor::ContractorConfig;

    nb::class_<ContractorConfig>(m, "ContractorConfig", nb::is_final())
        .def(nb::init<>())
        .def("UseDefaultOutputNames", &ContractorConfig::UseDefaultOutputNames,
             nb::arg("base"),
             "Set default output names based on base path")
        .def("IsValid", &ContractorConfig::IsValid,
             "Check if config is valid")
        .def_rw("requested_num_threads", &ContractorConfig::requested_num_threads,
                "Number of threads to use (0 = auto-detect)");
}
