#include "partitionerconfig_nb.h"

#include "osrm/partitioner_config.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

void init_PartitionerConfig(nb::module_& m) {
    using osrm::partitioner::PartitionerConfig;

    nb::class_<PartitionerConfig>(m, "PartitionerConfig", nb::is_final())
        .def(nb::init<>())
        .def("UseDefaultOutputNames", &PartitionerConfig::UseDefaultOutputNames,
             nb::arg("base"),
             "Set default output names based on base path")
        .def_rw("requested_num_threads", &PartitionerConfig::requested_num_threads,
                "Number of threads to use (0 = auto-detect)")
        .def_rw("balance", &PartitionerConfig::balance,
                "Balance parameter for partitioning")
        .def_rw("boundary_factor", &PartitionerConfig::boundary_factor,
                "Boundary factor for partitioning")
        .def_rw("num_optimizing_cuts", &PartitionerConfig::num_optimizing_cuts,
                "Number of optimizing cuts")
        .def_rw("small_component_size", &PartitionerConfig::small_component_size,
                "Size threshold for small components")
        .def_rw("max_cell_sizes", &PartitionerConfig::max_cell_sizes,
                "Maximum cell sizes for each level");
}
