#include "types/approach_nb.h"

#include "engine/approach.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

void init_Approach(nb::module_& m) {
    using osrm::engine::Approach;

    nb::enum_<Approach>(m, "Approach", "How to approach a waypoint")
        .value("CURB", Approach::CURB, "Approach waypoint from curb side")
        .value("UNRESTRICTED", Approach::UNRESTRICTED, "Approach waypoint from any direction");
}
