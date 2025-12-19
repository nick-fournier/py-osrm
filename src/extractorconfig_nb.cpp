#include "extractorconfig_nb.h"

#include "osrm/extractor_config.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/filesystem.h>

namespace nb = nanobind;

void init_ExtractorConfig(nb::module_& m) {
    using osrm::extractor::ExtractorConfig;

    nb::class_<ExtractorConfig>(m, "ExtractorConfig", nb::is_final())
        .def(nb::init<>())
        .def("UseDefaultOutputNames", &ExtractorConfig::UseDefaultOutputNames,
             nb::arg("base"),
             "Set default output names based on base path")
        .def_rw("input_path", &ExtractorConfig::input_path,
                "Path to input .osm, .osm.bz2, or .osm.pbf file")
        .def_rw("profile_path", &ExtractorConfig::profile_path,
                "Path to Lua routing profile")
        .def_rw("location_dependent_data_paths", &ExtractorConfig::location_dependent_data_paths,
                "Paths to location-dependent data files")
        .def_rw("data_version", &ExtractorConfig::data_version,
                "Version string for the data")
        .def_rw("requested_num_threads", &ExtractorConfig::requested_num_threads,
                "Number of threads to use (0 = auto-detect)")
        .def_rw("small_component_size", &ExtractorConfig::small_component_size,
                "Size threshold for small components")
        .def_rw("use_metadata", &ExtractorConfig::use_metadata,
                "Whether to include metadata in output")
        .def_rw("parse_conditionals", &ExtractorConfig::parse_conditionals,
                "Whether to parse conditional restrictions")
        .def_rw("use_locations_cache", &ExtractorConfig::use_locations_cache,
                "Whether to use locations cache")
        .def_rw("dump_nbg_graph", &ExtractorConfig::dump_nbg_graph,
                "Whether to dump the NBG graph for debugging");
}
