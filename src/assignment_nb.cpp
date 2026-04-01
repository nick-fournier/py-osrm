/*
 * C++ accumulation for traffic assignment.
 *
 * Combines OSRM BatchRoute + density/volume accumulation into a single
 * call.  Routes into flatbuffers (not JSON) so annotation data is
 * accessed via typed array pointers — no string-keyed hash maps, no
 * std::variant extraction, just direct memory reads.
 *
 * The function mirrors the logic of
 *   assignment_loop.py :: _route_and_accumulate_with_paths
 * but executes entirely in C++.
 */

#include "assignment_nb.h"

#include "osrm/osrm.hpp"
#include "osrm/route_parameters.hpp"
#include "osrm/status.hpp"
#include "engine/api/base_result.hpp"
#include "engine/api/flatbuffers/fbresult_generated.h"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/tuple.h>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;
namespace fbresult = osrm::engine::api::fbresult;

using osrm::OSRM;
using osrm::engine::api::RouteParameters;
using osrm::engine::api::ResultT;

// ── hash for (uint64, uint64) edge keys ──────────────────────────────
struct PairHash {
    std::size_t operator()(std::pair<uint64_t,uint64_t> const& p) const noexcept {
        auto h1 = std::hash<uint64_t>{}(p.first);
        auto h2 = std::hash<uint64_t>{}(p.second);
        h1 ^= h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2);
        return h1;
    }
};

using EdgeMap = std::unordered_map<std::pair<uint64_t,uint64_t>, int, PairHash>;

struct NewEdge {
    uint64_t from_id;
    uint64_t to_id;
    double   length_m;
    double   speed_kmh;
};

struct TripPath {
    int      trip_index;
    double   duration_s;
    std::vector<int>    edge_indices;
    std::vector<double> density_contributions;
};

void init_Assignment(nb::module_& m) {

    m.def("batch_route_accumulate",
        [](OSRM* engine,
           const std::vector<RouteParameters>& params,
           nb::ndarray<uint64_t, nb::ndim<2>, nb::c_contig, nb::device::cpu> edge_ids,
           nb::ndarray<double,   nb::ndim<1>, nb::c_contig, nb::device::cpu> freeflow_kmh,
           nb::ndarray<double,   nb::ndim<1>, nb::c_contig, nb::device::cpu> volumes,
           double bin_width_hr,
           double min_speed_kmh,
           double default_jam_density,
           int    default_n_lanes)
    {
        const size_t n_trips  = params.size();
        const size_t n_edges0 = edge_ids.shape(0);

        // ── 1. build edge lookup from numpy arrays ───────────────
        EdgeMap emap;
        emap.reserve(n_edges0 * 2);
        const uint64_t* eid_ptr = edge_ids.data();
        for (size_t i = 0; i < n_edges0; ++i) {
            emap[{eid_ptr[i * 2], eid_ptr[i * 2 + 1]}] = static_cast<int>(i);
        }

        // ── 2. Route into flatbuffers (TBB parallel, GIL released) ──
        //    Each thread gets its own FlatBufferBuilder.  After routing
        //    we keep the finished buffer bytes for sequential walking.
        struct FBRoute {
            std::vector<uint8_t> buf;
            osrm::engine::Status status;
        };
        std::vector<FBRoute> fb_routes(n_trips);

        {
            nb::gil_scoped_release release;
            tbb::parallel_for(
                tbb::blocked_range<size_t>(0, n_trips),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        flatbuffers::FlatBufferBuilder fbb(1024);
                        ResultT result = std::move(fbb);
                        fb_routes[i].status = engine->Route(params[i], result);
                        if (fb_routes[i].status == osrm::engine::Status::Ok) {
                            auto& builder = std::get<flatbuffers::FlatBufferBuilder>(result);
                            auto* ptr = builder.GetBufferPointer();
                            auto  sz  = builder.GetSize();
                            fb_routes[i].buf.assign(ptr, ptr + sz);
                        }
                    }
                }
            );
        }

        // ── 3. accumulate (sequential, in C++) ──────────────────
        std::vector<double> density(n_edges0, 0.0);
        std::vector<double> volume(n_edges0, 0.0);
        double tstt = 0.0;

        std::vector<NewEdge> new_edges;
        std::vector<TripPath> paths;
        paths.reserve(n_trips);

        const double* ff_ptr  = freeflow_kmh.data();
        const double* vol_ptr = volumes.data();

        for (size_t ti = 0; ti < n_trips; ++ti) {
            if (fb_routes[ti].status != osrm::engine::Status::Ok) continue;
            if (fb_routes[ti].buf.empty()) continue;

            auto* fb = fbresult::GetFBResult(fb_routes[ti].buf.data());
            auto* routes = fb->routes();
            if (!routes || routes->size() == 0) continue;

            auto* route = routes->Get(0);
            double route_dur = static_cast<double>(route->duration());
            double trip_vol  = vol_ptr[ti];
            tstt += trip_vol * route_dur;

            TripPath tp;
            tp.trip_index = static_cast<int>(ti);
            tp.duration_s = route_dur;

            auto* legs = route->legs();
            if (!legs) { paths.push_back(std::move(tp)); continue; }

            for (size_t li = 0; li < legs->size(); ++li) {
                auto* leg = legs->Get(li);
                auto* ann = leg->annotations();
                if (!ann) continue;

                auto* fb_nodes = ann->nodes();
                auto* fb_speed = ann->speed();
                auto* fb_dist  = ann->distance();
                if (!fb_nodes || fb_nodes->size() < 2) continue;

                size_t n_seg = fb_nodes->size() - 1;
                for (size_t si = 0; si < n_seg; ++si) {
                    uint64_t from_id = fb_nodes->Get(si);
                    uint64_t to_id   = fb_nodes->Get(si + 1);

                    auto it = emap.find({from_id, to_id});
                    int idx;

                    if (it != emap.end()) {
                        idx = it->second;
                    } else {
                        idx = static_cast<int>(density.size());
                        emap[{from_id, to_id}] = idx;
                        density.push_back(0.0);
                        volume.push_back(0.0);

                        double dist = (fb_dist && si < fb_dist->size())
                            ? static_cast<double>(fb_dist->Get(si)) : 0.0;
                        double spd  = (fb_speed && si < fb_speed->size())
                            ? static_cast<double>(fb_speed->Get(si)) * 3.6 : 1.0;
                        new_edges.push_back({from_id, to_id, dist, spd});
                    }

                    volume[idx] += trip_vol;

                    double seg_speed_kmh = (fb_speed && si < fb_speed->size())
                        ? static_cast<double>(fb_speed->Get(si)) * 3.6 : 0.0;
                    if (seg_speed_kmh < min_speed_kmh) {
                        if (static_cast<size_t>(idx) < n_edges0) {
                            seg_speed_kmh = ff_ptr[idx];
                        } else {
                            // newly discovered — annotation IS freeflow
                            seg_speed_kmh = new_edges[idx - n_edges0].speed_kmh;
                        }
                    }

                    double dd = trip_vol / (seg_speed_kmh * bin_width_hr);
                    density[idx] += dd;
                    tp.edge_indices.push_back(idx);
                    tp.density_contributions.push_back(dd);
                }
            }

            paths.push_back(std::move(tp));
        }

        // ── 4. pack results into numpy / Python objects ─────────
        size_t n_total = density.size();

        double* d_buf = new double[n_total];
        double* v_buf = new double[n_total];
        std::memcpy(d_buf, density.data(), n_total * sizeof(double));
        std::memcpy(v_buf, volume.data(),  n_total * sizeof(double));

        nb::capsule d_owner(d_buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });
        nb::capsule v_owner(v_buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });

        size_t shape[1] = {n_total};
        auto py_density = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
            d_buf, 1, shape, d_owner);
        auto py_volume  = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
            v_buf, 1, shape, v_owner);

        nb::list py_new_edges;
        for (const auto& ne : new_edges) {
            py_new_edges.append(nb::make_tuple(
                ne.from_id, ne.to_id, ne.length_m, ne.speed_kmh));
        }

        nb::list py_paths;
        for (const auto& tp : paths) {
            size_t pn = tp.edge_indices.size();
            int*    ei_buf = new int[pn];
            double* dc_buf = new double[pn];
            std::memcpy(ei_buf, tp.edge_indices.data(), pn * sizeof(int));
            std::memcpy(dc_buf, tp.density_contributions.data(), pn * sizeof(double));

            nb::capsule ei_own(ei_buf, [](void* p) noexcept { delete[] static_cast<int*>(p); });
            nb::capsule dc_own(dc_buf, [](void* p) noexcept { delete[] static_cast<double*>(p); });

            size_t ps[1] = {pn};
            auto py_ei = nb::ndarray<nb::numpy, int, nb::ndim<1>>(
                ei_buf, 1, ps, ei_own);
            auto py_dc = nb::ndarray<nb::numpy, double, nb::ndim<1>>(
                dc_buf, 1, ps, dc_own);

            py_paths.append(nb::make_tuple(
                tp.trip_index, tp.duration_s, py_ei, py_dc));
        }

        return nb::make_tuple(py_density, py_volume, tstt, py_paths, py_new_edges);
    },
    nb::arg("engine"),
    nb::arg("params"),
    nb::arg("edge_ids"),
    nb::arg("freeflow_kmh"),
    nb::arg("volumes"),
    nb::arg("bin_width_hr"),
    nb::arg("min_speed_kmh"),
    nb::arg("default_jam_density"),
    nb::arg("default_n_lanes"),
    "Route OD pairs and accumulate link density/volume in C++.\n\n"
    "Routes into flatbuffers for zero-overhead annotation access.\n\n"
    "Returns (density, volume, tstt, paths, new_edges)."
    );
}
