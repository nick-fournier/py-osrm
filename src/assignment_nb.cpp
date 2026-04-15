/*
 * C++ accumulation for traffic assignment.
 *
 * Combines OSRM routing + volume accumulation into a single call.
 * Accepts numpy coordinate/volume arrays directly, builds
 * RouteParameters in C++, routes via OSRM, and accumulates — no
 * Python per-trip overhead at all.
 */

#include "assignment_nb.h"

#include "osrm/osrm.hpp"
#include "osrm/coordinate.hpp"
#include "osrm/route_parameters.hpp"
#include "osrm/status.hpp"
#include "util/json_container.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;
namespace json = osrm::util::json;

using osrm::OSRM;
using osrm::engine::api::RouteParameters;

struct PairHash {
    std::size_t operator()(std::pair<uint64_t,uint64_t> const& p) const noexcept {
        auto h1 = std::hash<uint64_t>{}(p.first);
        auto h2 = std::hash<uint64_t>{}(p.second);
        h1 ^= h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2);
        return h1;
    }
};

using EdgeMap = std::unordered_map<std::pair<uint64_t,uint64_t>, int, PairHash>;

static inline double as_number(const json::Value& v) {
    return std::get<json::Number>(v).value;
}
static inline const std::string& as_string(const json::Value& v) {
    return std::get<json::String>(v).value;
}
static inline const json::Array& as_array(const json::Value& v) {
    return std::get<json::Array>(v);
}
static inline const json::Object& as_object(const json::Value& v) {
    return std::get<json::Object>(v);
}

struct NewEdge {
    uint64_t from_id;
    uint64_t to_id;
    double   length_m;
    double   speed_kmh;
};

void init_Assignment(nb::module_& m) {

    m.def("batch_route_accumulate",
        [](OSRM* engine,
           nb::ndarray<double, nb::ndim<2>, nb::c_contig, nb::device::cpu> coords,
           nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> volumes,
           nb::ndarray<uint64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> edge_ids,
           int    n_threads,
           bool   return_routes,
           int    departure_period,
           double period_duration,
           nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu> departure_offsets,
           int    n_periods)
    {

        // Optionally cap TBB parallelism
        std::unique_ptr<tbb::global_control> tbb_ctl;
        if (n_threads > 0) {
            tbb_ctl = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism,
                static_cast<size_t>(n_threads));
        }

        const size_t n_trips  = coords.shape(0);
        const size_t n_edges0 = edge_ids.shape(0);
        const double* c_ptr   = coords.data();
        const double* vol_ptr = volumes.data();
        const bool has_period = (departure_period >= 0 && period_duration > 0);
        const double* offset_ptr = has_period ? departure_offsets.data() : nullptr;

        // Period-attributed 2D accumulation when n_periods > 0
        const bool use_2d = (has_period && n_periods > 0);

        // ── 1. build RouteParameters in C++ ──────────────────────
        std::vector<RouteParameters> params(n_trips);
        {
            using osrm::util::FloatLongitude;
            using osrm::util::FloatLatitude;
            using osrm::util::Coordinate;

            const auto ann_type =
                RouteParameters::AnnotationsType::Nodes |
                RouteParameters::AnnotationsType::Distance |
                RouteParameters::AnnotationsType::Speed;

            for (size_t i = 0; i < n_trips; ++i) {
                auto& rp = params[i];
                rp.annotations = true;
                rp.annotations_type = ann_type;
                if (return_routes) {
                    rp.overview = RouteParameters::OverviewType::Full;
                    rp.geometries = RouteParameters::GeometriesType::Polyline6;
                }
                rp.coordinates.push_back(
                    Coordinate{FloatLongitude{c_ptr[i*4]}, FloatLatitude{c_ptr[i*4+1]}});
                rp.coordinates.push_back(
                    Coordinate{FloatLongitude{c_ptr[i*4+2]}, FloatLatitude{c_ptr[i*4+3]}});

                // Multi-period: set departure context per trip
                if (has_period) {
                    rp.departure_period = static_cast<std::size_t>(departure_period);
                    rp.period_duration = period_duration;
                    if (offset_ptr)
                        rp.departure_time_offset = offset_ptr[i];
                }
            }
        }

        // ── 2. build edge lookup from numpy arrays ───────────────
        EdgeMap emap;
        emap.reserve(n_edges0 * 2);
        const uint64_t* eid_ptr = edge_ids.data();
        for (size_t i = 0; i < n_edges0; ++i) {
            emap[{eid_ptr[i * 2], eid_ptr[i * 2 + 1]}] = static_cast<int>(i);
        }

        // ── 3. BatchRoute (TBB parallel, GIL released) ──────────
        std::vector<json::Object>          results(n_trips);
        std::vector<osrm::engine::Status>  statuses(n_trips);
        auto t_route_start = std::chrono::steady_clock::now();
        {
            nb::gil_scoped_release release;
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n_trips),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        statuses[i] = engine->Route(params[i], results[i]);
                    }
                }
            );
        }
        auto t_route_end = std::chrono::steady_clock::now();
        double route_ms = std::chrono::duration<double, std::milli>(
            t_route_end - t_route_start).count();

        // ── 4. accumulate volume (sequential, in C++) ───────────
        auto t_accum_start = std::chrono::steady_clock::now();

        // Sparse accumulation: collect (period, edge_idx, volume) triples
        // instead of building a dense (n_periods × n_edges) array.
        struct FlowEntry { int period; int edge_idx; double vol; };
        std::vector<FlowEntry> sparse_flow;

        // 1D mode: volume[edge] (legacy, no period attribution)
        std::vector<double> volume;
        if (!use_2d) {
            volume.resize(n_edges0, 0.0);
        }

        double tstt = 0.0;
        std::vector<NewEdge> new_edges;
        std::vector<std::string> route_geoms;
        std::vector<double> trip_durations(n_trips, 0.0);
        if (return_routes) route_geoms.resize(n_trips);

        for (size_t ti = 0; ti < n_trips; ++ti) {
            if (statuses[ti] != osrm::engine::Status::Ok) continue;

            const auto& root = results[ti];
            auto routes_it = root.values.find("routes");
            if (routes_it == root.values.end()) continue;
            const auto& routes_arr = as_array(routes_it->second);
            if (routes_arr.values.empty()) continue;

            const auto& route = as_object(routes_arr.values[0]);
            double route_dur = as_number(route.values.at("duration"));
            double trip_vol  = vol_ptr[ti];
            tstt += trip_vol * route_dur;
            trip_durations[ti] = route_dur;

            if (return_routes) {
                auto geom_it = route.values.find("geometry");
                if (geom_it != route.values.end()) {
                    route_geoms[ti] = as_string(geom_it->second);
                }
            }

            // Per-trip departure offset for period computation
            double trip_dep_offset = (has_period && offset_ptr)
                ? offset_ptr[ti] : 0.0;
            double cumulative_time_s = 0.0;

            const auto& legs_arr = as_array(route.values.at("legs"));

            for (const auto& leg_val : legs_arr.values) {
                const auto& leg = as_object(leg_val);
                const auto& ann = as_object(leg.values.at("annotation"));
                const auto& nodes_arr = as_array(ann.values.at("nodes"));
                const auto& dist_arr  = as_array(ann.values.at("distance"));
                const auto& speed_arr = as_array(ann.values.at("speed"));

                size_t n_seg = nodes_arr.values.size() - 1;
                for (size_t si = 0; si < n_seg; ++si) {
                    uint64_t from_id = static_cast<uint64_t>(
                        as_number(nodes_arr.values[si]));
                    uint64_t to_id   = static_cast<uint64_t>(
                        as_number(nodes_arr.values[si + 1]));

                    // Segment travel time from distance/speed annotations
                    double seg_dist_m = (si < dist_arr.values.size())
                        ? as_number(dist_arr.values[si]) : 0.0;
                    double seg_speed_mps = (si < speed_arr.values.size())
                        ? as_number(speed_arr.values[si]) : 0.01;
                    if (seg_speed_mps < 0.01) seg_speed_mps = 0.01;
                    double seg_time_s = seg_dist_m / seg_speed_mps;

                    auto it = emap.find({from_id, to_id});
                    int idx;

                    if (it != emap.end()) {
                        idx = it->second;
                    } else {
                        // New edge: grow volume vector in lockstep
                        if (!use_2d) {
                            idx = static_cast<int>(volume.size());
                            volume.push_back(0.0);
                        } else {
                            // For sparse mode, just assign next index
                            idx = static_cast<int>(n_edges0 + new_edges.size());
                        }
                        emap[{from_id, to_id}] = idx;

                        double spd_kmh = seg_speed_mps * 3.6;
                        if (spd_kmh < 1.0) spd_kmh = 1.0;
                        new_edges.push_back({from_id, to_id, seg_dist_m, spd_kmh});
                    }

                    // Attribute volume to the correct period
                    if (use_2d) {
                        double mid_time = cumulative_time_s + seg_time_s * 0.5;
                        double total_time = trip_dep_offset + mid_time;
                        int seg_period = departure_period
                            + static_cast<int>(total_time / period_duration);
                        if (seg_period < 0) seg_period = 0;
                        if (seg_period >= n_periods) seg_period = n_periods - 1;
                        sparse_flow.push_back({seg_period, idx, trip_vol});
                    } else {
                        volume[idx] += trip_vol;
                    }

                    cumulative_time_s += seg_time_s;
                }
            }
        }

        // ── 5. pack results into numpy / Python objects ─────────
        auto t_accum_end = std::chrono::steady_clock::now();
        double accum_ms = std::chrono::duration<double, std::milli>(
            t_accum_end - t_accum_start).count();
        nb::list py_new_edges;
        for (const auto& ne : new_edges) {
            py_new_edges.append(nb::make_tuple(
                ne.from_id, ne.to_id, ne.length_m, ne.speed_kmh));
        }

        nb::object py_geoms_obj = nb::none();
        if (return_routes) {
            nb::list py_geoms;
            for (const auto& g : route_geoms) {
                py_geoms.append(g);
            }
            py_geoms_obj = py_geoms;
        }

        // Per-trip durations array
        double* dur_buf = new double[n_trips];
        std::memcpy(dur_buf, trip_durations.data(), n_trips * sizeof(double));
        nb::capsule dur_owner(dur_buf, [](void* p) noexcept {
            delete[] static_cast<double*>(p);
        });
        size_t dur_shape[1] = {n_trips};
        auto py_durations = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
            dur_buf, 1, dur_shape, dur_owner);

        if (use_2d) {
            // Return sparse COO: (periods, edge_indices, volumes) arrays
            size_t n_entries = sparse_flow.size();
            int32_t* p_buf = new int32_t[n_entries];
            int32_t* e_buf = new int32_t[n_entries];
            double*  f_buf = new double[n_entries];
            for (size_t i = 0; i < n_entries; ++i) {
                p_buf[i] = sparse_flow[i].period;
                e_buf[i] = sparse_flow[i].edge_idx;
                f_buf[i] = sparse_flow[i].vol;
            }

            nb::capsule p_owner(p_buf, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
            nb::capsule e_owner(e_buf, [](void* p) noexcept { delete[] static_cast<int32_t*>(p); });
            nb::capsule f_owner(f_buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
            size_t shape1[1] = {n_entries};

            auto py_periods = nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(p_buf, 1, shape1, p_owner);
            auto py_edges   = nb::ndarray<nb::numpy, int32_t, nb::ndim<1>>(e_buf, 1, shape1, e_owner);
            auto py_flows   = nb::ndarray<nb::numpy, double,  nb::ndim<1>>(f_buf, 1, shape1, f_owner);

            auto py_sparse = nb::make_tuple(py_periods, py_edges, py_flows);
            return nb::make_tuple(py_sparse, tstt, py_new_edges, py_geoms_obj, py_durations, route_ms, accum_ms);
        } else {
            // Return 1D volume: (n_total_edges,)
            size_t n_total = volume.size();
            double* v_buf = new double[n_total];
            std::memcpy(v_buf, volume.data(), n_total * sizeof(double));

            nb::capsule owner(v_buf, [](void* p) noexcept {
                delete[] static_cast<double*>(p);
            });
            size_t shape[1] = {n_total};
            auto py_vol = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
                v_buf, 1, shape, owner);

            return nb::make_tuple(py_vol, tstt, py_new_edges, py_geoms_obj, py_durations, route_ms, accum_ms);
        }
    },
    nb::arg("engine"),
    nb::arg("coords"),
    nb::arg("volumes"),
    nb::arg("edge_ids"),
    nb::arg("n_threads") = 0,
    nb::arg("return_routes") = false,
    nb::arg("departure_period") = -1,
    nb::arg("period_duration") = 0.0,
    nb::arg("departure_offsets") = nb::ndarray<double, nb::ndim<1>, nb::c_contig, nb::device::cpu>(),
    nb::arg("n_periods") = 0,
    "Route OD pairs and accumulate link volume in C++.\n\n"
    "Accepts (n,4) coordinate array [o_lon, o_lat, d_lon, d_lat] and\n"
    "builds RouteParameters internally — no Python param construction.\n"
    "Uses JSON route results (correct uint64 OSM node IDs at any scale).\n\n"
    "n_threads: 0 = use all cores, >0 = cap TBB parallelism.\n"
    "return_routes: if True, also returns per-trip encoded polyline6 strings.\n"
    "departure_period: period index for multi-period routing (-1 = disabled).\n"
    "period_duration: seconds per period (e.g. 900 for 15-min).\n"
    "departure_offsets: per-trip seconds into departure period.\n"
    "n_periods: total number of periods for 2D attribution (0 = 1D legacy).\n\n"
    "Returns (volume, tstt, new_edges, route_geometries, trip_durations).\n"
    "volume is 2D (n_periods, n_edges) when n_periods > 0, else 1D (n_edges,).\n"
    "route_geometries is a list of polyline6 strings when return_routes=True,\n"
    "otherwise None.\n"
    "trip_durations is a 1D array of per-trip travel times in seconds."
    );
}
