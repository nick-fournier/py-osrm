#include "customizerconfig_nb.h"

#include "osrm/customizer_config.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void init_CustomizerConfig(nb::module_& m) {
    using osrm::customizer::CustomizationConfig;

    nb::class_<CustomizationConfig>(m, "CustomizationConfig", nb::is_final())
        .def(nb::init<>())
        .def("UseDefaultOutputNames", &CustomizationConfig::UseDefaultOutputNames,
             nb::arg("base"),
             "Set default output names based on base path")
        .def_rw("requested_num_threads", &CustomizationConfig::requested_num_threads,
                "Number of threads to use (0 = auto-detect)")
        .def_prop_rw("segment_speed_lookup_paths",
            [](const CustomizationConfig &c) {
                return c.updater_config.segment_speed_lookup_paths;
            },
            [](CustomizationConfig &c, std::vector<std::string> paths) {
                c.updater_config.segment_speed_lookup_paths = std::move(paths);
            },
            "List of CSV file paths for segment speed updates")
        .def_prop_rw("turn_penalty_lookup_paths",
            [](const CustomizationConfig &c) {
                return c.updater_config.turn_penalty_lookup_paths;
            },
            [](CustomizationConfig &c, std::vector<std::string> paths) {
                c.updater_config.turn_penalty_lookup_paths = std::move(paths);
            },
            "List of CSV file paths for turn penalty updates");
}
