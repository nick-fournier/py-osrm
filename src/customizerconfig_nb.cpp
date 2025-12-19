#include "customizerconfig_nb.h"

#include "osrm/customizer_config.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>

namespace nb = nanobind;

void init_CustomizerConfig(nb::module_& m) {
    using osrm::customizer::CustomizationConfig;

    nb::class_<CustomizationConfig>(m, "CustomizationConfig", nb::is_final())
        .def(nb::init<>())
        .def("UseDefaultOutputNames", &CustomizationConfig::UseDefaultOutputNames,
             nb::arg("base"),
             "Set default output names based on base path")
        .def_rw("requested_num_threads", &CustomizationConfig::requested_num_threads,
                "Number of threads to use (0 = auto-detect)");
}
