#include "osrm/osrm.hpp"
#include "osrm/engine_config.hpp"
#include "osrm/status.hpp"
#include "osrm/match_parameters.hpp"
#include "osrm/nearest_parameters.hpp"
#include "osrm/route_parameters.hpp"
#include "osrm/table_parameters.hpp"
#include "osrm/tile_parameters.hpp"
#include "osrm/trip_parameters.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <stdexcept>
#include <cstdlib>
#include <vector>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "engineconfig_nb.h"
#include "preprocessing_nb.h"
#include "utility/osrm_utility.h"
#include "types/approach_nb.h"
#include "types/bearing_nb.h"
#include "types/coordinate_nb.h"
#include "types/jsoncontainer_nb.h"
#include "types/optional_nb.h"
#include "parameters/baseparameter_nb.h"
#include "parameters/matchparameter_nb.h"
#include "parameters/nearestparameter_nb.h"
#include "parameters/routeparameter_nb.h"
#include "parameters/tableparameter_nb.h"
#include "parameters/tileparameter_nb.h"
#include "parameters/tripparameter_nb.h"

namespace nb = nanobind;

// Global cleanup handler to prevent TBB thread pool from hanging on exit
static void cleanup_osrm_tbb() {
    // OSRM uses Intel TBB which creates threads that don't cleanly shut down
    // during normal Python exit. Using quick_exit bypasses static destructors
    // and TBB cleanup, preventing the hang.
    std::fflush(stdout);
    std::fflush(stderr);
    std::quick_exit(0);
}

NB_MODULE(osrm_ext, m) {
    namespace api = osrm::engine::api;
    namespace json = osrm::util::json;

    using osrm::OSRM;
    using osrm::engine::EngineConfig;
    using osrm::engine::api::MatchParameters;
    using osrm::engine::api::NearestParameters;
    using osrm::engine::api::RouteParameters;
    using osrm::engine::api::TableParameters;
    using osrm::engine::api::TileParameters;
    using osrm::engine::api::TripParameters;

    init_EngineConfig(m);
    init_Preprocessing(m);

    init_Approach(m);
    init_Bearing(m);
    init_Coordinate(m);
    init_JSONContainer(m);
    init_Optional(m);

    init_BaseParameters(m);
    init_NearestParameters(m);
    init_TableParameters(m);
    init_RouteParameters(m);
    init_MatchParameters(m);
    init_TripParameters(m);
    init_TileParameters(m);

    nb::class_<OSRM>(m, "OSRM", nb::is_final())
        .def(nb::init<EngineConfig&>(), "Instantiates an instance of OSRM.\n\n"
            "Examples:\n\
                >>> import osrm\n\
                >>> py_osrm = osrm.OSRM('.tests/test_data/ch/monaco.osrm')\n\
                >>> py_osrm = osrm.OSRM(\n\
                        algorithm = 'CH',\n\
                        storage_config = '.tests/test_data/ch/monaco.osrm',\n\
                        max_locations_trip = 3,\n\
                        max_locations_viaroute = 3,\n\
                        max_locations_distance_table = 3,\n\
                        max_locations_map_matching = 3,\n\
                        max_results_nearest = 1,\n\
                        max_alternatives = 1,\n\
                        default_radius = 'unlimited'\n\
                    )\n\n"
            "Args:\n\
                storage_config (string): File path string to storage config.\n\
                EngineConfig (osrm.osrm_ext.EngineConfig): Keyword arguments from the EngineConfig class.\n\n"
            "Returns:\n\
                __init__ (osrm.OSRM): A OSRM object.\n\n"
            "Raises:\n\
                RuntimeError: On invalid OSRM EngineConfig parameters."
            )
        .def("__init__", [](OSRM* t, const std::string& storage_path) { 
            EngineConfig config;
            config.storage_config = osrm::storage::StorageConfig(storage_path);

            if(!config.IsValid()) {
                throw std::runtime_error("Required files are missing");
            }

            new (t) OSRM(config);
        })
        .def("__init__", [](OSRM* t, const nb::kwargs& kwargs) {
            EngineConfig config;
            osrm_nb_util::populate_cfg_from_kwargs(kwargs, config);

            if(!config.IsValid()) {
                throw std::runtime_error("Config Parameters are Invalid");
            }

            new (t) OSRM(config);
        })
        .def("Match", [](OSRM* t, const MatchParameters& params) {
            if(!params.IsValid()) {
                throw std::runtime_error("Invalid Match Parameters");
            }

            json::Object result;
            osrm::engine::Status status;
            {
                nb::gil_scoped_release release;
                status = t->Match(params, result);
            }
            osrm_nb_util::check_status(status, result);

            return result;
        }, "Matches/snaps given GPS points to the road network in the most plausible way.\n\n"
            "Examples:\n\
                >>> res = py_osrm.Match(match_params)\n\n"
            "Args:\n\
                match_params (osrm.MatchParameters): MatchParameters Object.\n\n"
            "Returns:\n\
                (json): [A Match JSON Response](https://project-osrm.org/docs/v5.24.0/api/#match-service).\n\n"
            "Raises:\n\
                RuntimeError: On invalid MatchParameters."
            )
        .def("Nearest", [](OSRM* t, const NearestParameters& params) {
            if(!params.IsValid()) {
                throw std::runtime_error("Invalid Nearest Parameters");
            }

            json::Object result;
            osrm::engine::Status status;
            {
                nb::gil_scoped_release release;
                status = t->Nearest(params, result);
            }
            osrm_nb_util::check_status(status, result);

            return result;
        }, "Snaps a coordinate to the street network and returns the nearest matches.\n\n"
            "Examples:\n\
                >>> res = py_osrm.Nearest(nearest_params)\n\n"
            "Args:\n\
                nearest_params (osrm.NearestParameters): NearestParameters Object.\n\n"
            "Returns:\n\
                (json): [A Nearest JSON Response](https://project-osrm.org/docs/v5.24.0/api/#nearest-service).\n\n"
            "Raises:\n\
                RuntimeError: On invalid NearestParameters."
            )
        .def("Route", [](OSRM* t, const RouteParameters& params) {
            if(!params.IsValid()) {
                throw std::runtime_error("Invalid Route Parameters");
            }

            json::Object result;
            osrm::engine::Status status;
            {
                nb::gil_scoped_release release;
                status = t->Route(params, result);
            }
            osrm_nb_util::check_status(status, result);

            return result;
        }, "Finds the fastest route between coordinates in the supplied order.\n\n"
            "Examples:\n\
                >>> res = py_osrm.Route(route_params)\n\n"
            "Args:\n\
                route_params (osrm.RouteParameters): RouteParameters Object.\n\n"
            "Returns:\n\
                (json): [A Route JSON Response](https://project-osrm.org/docs/v5.24.0/api/#route-service).\n\n"
            "Raises:\n\
                RuntimeError: On invalid RouteParameters."
            )
        .def("BatchRoute", [](OSRM* t, const std::vector<RouteParameters>& params_list) {
            // Validate all params with GIL held for error reporting
            for (size_t i = 0; i < params_list.size(); ++i) {
                if (!params_list[i].IsValid()) {
                    throw std::runtime_error(
                        "Invalid Route Parameters at index " + std::to_string(i));
                }
            }

            std::vector<json::Object> results(params_list.size());
            std::vector<osrm::engine::Status> statuses(params_list.size());

            {
                nb::gil_scoped_release release;
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, params_list.size()),
                    [&](const tbb::blocked_range<size_t>& range) {
                        for (size_t i = range.begin(); i != range.end(); ++i) {
                            statuses[i] = t->Route(params_list[i], results[i]);
                        }
                    }
                );
            }

            nb::list py_results;
            for (size_t i = 0; i < results.size(); ++i) {
                if (statuses[i] == osrm::engine::Status::Ok) {
                    py_results.append(nb::cast(std::move(results[i])));
                } else {
                    py_results.append(nb::none());
                }
            }
            return py_results;
        }, "Route multiple OD pairs in parallel using native C++ threading (TBB).\n\n"
            "Args:\n\
                params_list (list[osrm.RouteParameters]): List of RouteParameters objects.\n\n"
            "Returns:\n\
                list: Route result dicts (None for failed routes).\n"
            )
        .def("Table", [](OSRM* t, const TableParameters& params) {
            if(!params.IsValid()) {
                throw std::runtime_error("Invalid Table Parameters");
            }

            json::Object result;
            osrm::engine::Status status;
            {
                nb::gil_scoped_release release;
                status = t->Table(params, result);
            }
            osrm_nb_util::check_status(status, result);

            return result;
        }, "Computes the duration of the fastest route between all pairs of supplied coordinates.\n\n"
            "Examples:\n\
                >>> res = py_osrm.Table(table_params)\n\n"
            "Args:\n\
                table_params (osrm.TableParameters): TableParameters Object.\n\n"
            "Returns:\n\
                (json): [A Table JSON Response](https://project-osrm.org/docs/v5.24.0/api/#table-service).\n\n"
            "Raises:\n\
                RuntimeError: On invalid TableParameters."
            )
        .def("Tile", [](OSRM* t, const TileParameters& params) {
            if(!params.IsValid()) {
                throw std::runtime_error("Invalid Tile Parameters");
            }

            std::string result;
            osrm::engine::Status status = t->Tile(params, result);
            nb::object obj = nb::bytes(result.c_str(), result.size());

            return obj;
        }, "Computes the duration of the fastest route between all pairs of supplied coordinates.\n\n"
            "Examples:\n\
                >>> res = py_osrm.Tile(tile_params)\n\n"
            "Args:\n\
                tile_params (osrm.TileParameters): TileParameters Object.\n\n"
            "Returns:\n\
                (json): [A Tile JSON Response](https://project-osrm.org/docs/v5.24.0/api/#tile-service).\n\n"
            "Raises:\n\
                RuntimeError: On invalid TileParameters."
            )
        .def("Trip", [](OSRM* t, const TripParameters& params) {
            if(!params.IsValid()) {
                throw std::runtime_error("Invalid Trip Parameters");
            }

            json::Object result;
            osrm::engine::Status status;
            {
                nb::gil_scoped_release release;
                status = t->Trip(params, result);
            }
            osrm_nb_util::check_status(status, result);

            return result;
        }, "Solves the Traveling Salesman Problem using a greedy heuristic (farthest-insertion algorithm).\n\n"
            "Examples:\n\
                >>> res = py_osrm.Trip(trip_params)\n\n"
            "Args:\n\
                trip_params (osrm.TripParameters): TripParameters Object.\n\n"
            "Returns:\n\
                (json): [A Trip JSON Response](https://project-osrm.org/docs/v5.24.0/api/#trip-service).\n\n"
            "Raises:\n\
                RuntimeError: On invalid TripParameters."
            );

    // Register cleanup handler to prevent TBB thread pool from hanging on exit
    // This must be done after all bindings are created
    try {
        nb::module_ atexit = nb::module_::import_("atexit");
        atexit.attr("register")(nb::cpp_function(cleanup_osrm_tbb));
    } catch (...) {
        // If atexit registration fails, continue anyway
        // The worst case is the hang continues to occur
    }
}
