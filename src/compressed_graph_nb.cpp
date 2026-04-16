/*
 * Read OSRM's compressed node-based graph and build a lookup table
 * mapping annotation node pairs → compressed edge indices.
 *
 * This solves the micro-segment problem: OSRM annotations unpack
 * compressed edges back to full OSM-node granularity, creating 1-11m
 * edges with ~1 km/h freeflow. By reading the compressed graph, we
 * accumulate flow per compressed edge (correct freeflow/distance).
 */

#include "compressed_graph_nb.h"

#include "extractor/files.hpp"
#include "extractor/compressed_node_based_graph_edge.hpp"
#include "extractor/segment_data_container.hpp"
#include "extractor/packed_osm_ids.hpp"
#include "extractor/node_data_container.hpp"
#include "extractor/travel_mode.hpp"
#include "util/coordinate.hpp"
#include "util/coordinate_calculation.hpp"
#include "util/typedefs.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>

#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;
namespace fs = std::filesystem;

// Helper: extract uint64_t OSM ID from PackedOSMIDs at given internal NodeID
static inline uint64_t get_osm_id(const osrm::extractor::PackedOSMIDs& ids, NodeID nid) {
    OSMNodeID osm_id = ids[nid];
    return static_cast<uint64_t>(osm_id);
}

void init_CompressedGraph(nb::module_& m) {

    nb::class_<CompressedGraphLookup>(m, "CompressedGraphLookup")
        .def_prop_ro("n_edges", &CompressedGraphLookup::n_edges)
        .def_prop_ro("n_pairs", &CompressedGraphLookup::n_pairs)
        .def_ro("skipped_non_driving", &CompressedGraphLookup::skipped_non_driving)
        .def("find", &CompressedGraphLookup::find,
             nb::arg("from_osm"), nb::arg("to_osm"),
             "Look up node pair. Returns edge index or -1 if not found.")
        .def("distance", &CompressedGraphLookup::distance, nb::arg("edge_idx"))
        .def("freeflow", &CompressedGraphLookup::freeflow, nb::arg("edge_idx"));

    m.def("load_compressed_graph",
        [](const std::string& osrm_base_path) {
            // File paths
            const auto nbg_nodes_path = fs::path(osrm_base_path + ".nbg_nodes");
            const auto geometry_path  = fs::path(osrm_base_path + ".geometry");
            const auto ebg_nodes_path = fs::path(osrm_base_path + ".ebg_nodes");

            // ── 1. Read node coordinates and OSM IDs ──────────────
            std::vector<osrm::util::Coordinate> coordinates;
            osrm::extractor::PackedOSMIDs osm_node_ids;
            osrm::extractor::files::readNodes(nbg_nodes_path, coordinates, osm_node_ids);

            const size_t n_nodes = coordinates.size();

            // ── 2. Read segment data (compressed edge geometries) ─
            osrm::extractor::SegmentDataContainer segment_data;
            osrm::extractor::files::readSegmentData(geometry_path, segment_data);

            const size_t n_geom = segment_data.GetNumberOfGeometries();

            // ── 2b. Read edge-based node data for travel mode ─────
            // Build geometry_id → travel_mode map so we only index
            // driving-mode geometries (skip footways, cycleways, etc.)
            osrm::extractor::EdgeBasedNodeDataContainer node_data;
            osrm::extractor::files::readNodeData(ebg_nodes_path, node_data);

            // Map: geometry PackedGeometryID → travel mode
            // Multiple edge-based nodes may reference the same geometry;
            // we just need one mode per geometry (they should agree).
            std::vector<uint8_t> geom_mode(n_geom, osrm::extractor::TRAVEL_MODE_INACCESSIBLE);
            for (NodeID nid = 0; nid < node_data.NumberOfNodes(); ++nid) {
                auto geom = node_data.GetGeometryID(nid);
                if (geom.id < n_geom) {
                    geom_mode[geom.id] = node_data.GetTravelMode(nid);
                }
            }

            // ── 3. Build per-compressed-edge data + node pair lookup ─
            //
            // For each geometry entry (= one compressed edge direction):
            //   - Get the forward node sequence
            //   - Convert internal NodeIDs to OSM IDs
            //   - Compute total distance (haversine between consecutive nodes)
            //   - Compute total duration (sum of forward durations, stored as deciseconds)
            //   - Map each consecutive OSM node pair to this compressed edge index
            //
            // We use geometry_id as the compressed edge index (uint32).
            // Forward and reverse directions share the same geometry_id but have
            // different node orderings. We store both directions.

            // Edge metadata arrays (indexed by compressed edge index)
            std::vector<uint64_t> edge_from_osm;  // first OSM node
            std::vector<uint64_t> edge_to_osm;     // last OSM node
            std::vector<double>   edge_distance_m;  // total distance
            std::vector<double>   edge_freeflow_kmh; // freeflow speed

            edge_from_osm.reserve(n_geom * 2);
            edge_to_osm.reserve(n_geom * 2);
            edge_distance_m.reserve(n_geom * 2);
            edge_freeflow_kmh.reserve(n_geom * 2);

            // Node pair → compressed edge index
            std::unordered_map<std::pair<uint64_t,uint64_t>, uint32_t, PairHashCG> node_pair_to_edge;
            node_pair_to_edge.reserve(segment_data.GetNumberOfSegments() * 2);

            uint32_t edge_idx = 0;
            size_t skipped_non_driving = 0;
            const uint32_t INVALID_DUR = (1u << SEGMENT_DURATION_BITS) - 1;

            for (size_t g = 0; g < n_geom; ++g) {
                // Skip non-driving geometries (footways, cycleways, etc.)
                if (geom_mode[g] != osrm::extractor::TRAVEL_MODE_DRIVING) {
                    ++skipped_non_driving;
                    continue;
                }

                auto geom_id = static_cast<osrm::extractor::SegmentDataContainer::DirectionalGeometryID>(g);

                // Get forward node sequence
                auto fwd_nodes = segment_data.GetForwardGeometry(geom_id);

                // Need at least 2 nodes for an edge
                size_t n_nodes_in_geom = 0;
                for ([[maybe_unused]] auto _ : fwd_nodes) ++n_nodes_in_geom;
                if (n_nodes_in_geom < 2) continue;

                // Collect nodes into vector for random access
                std::vector<NodeID> node_seq;
                node_seq.reserve(n_nodes_in_geom);
                for (auto nid : fwd_nodes) {
                    node_seq.push_back(nid);
                }

                // Compute total distance via haversine between consecutive nodes
                double total_dist_m = 0.0;
                for (size_t i = 0; i + 1 < node_seq.size(); ++i) {
                    if (node_seq[i] < n_nodes && node_seq[i+1] < n_nodes) {
                        total_dist_m += osrm::util::coordinate_calculation::greatCircleDistance(
                            coordinates[node_seq[i]], coordinates[node_seq[i+1]]);
                    }
                }

                // Convert endpoint NodeIDs to OSM IDs
                uint64_t first_osm = get_osm_id(osm_node_ids, node_seq.front());
                uint64_t last_osm  = get_osm_id(osm_node_ids, node_seq.back());

                // Helper: sum durations, skipping INVALID entries
                auto sum_durations = [&](auto durations) -> std::pair<double, bool> {
                    double total = 0.0;
                    bool valid = false;
                    for (auto dur : durations) {
                        SegmentDuration sd = dur;
                        auto raw = static_cast<SegmentDuration::value_type>(sd);
                        if (raw != INVALID_DUR) {
                            total += static_cast<double>(raw) / 10.0;
                            valid = true;
                        }
                    }
                    return {total, valid};
                };

                // ── Forward direction ──
                auto [fwd_dur_s, fwd_valid] = sum_durations(
                    segment_data.GetForwardDurations(geom_id));

                if (fwd_valid) {
                    double freeflow_kmh = 1.0;
                    if (fwd_dur_s > 0.001) {
                        freeflow_kmh = (total_dist_m / fwd_dur_s) * 3.6;
                        if (freeflow_kmh < 1.0) freeflow_kmh = 1.0;
                    }

                    edge_from_osm.push_back(first_osm);
                    edge_to_osm.push_back(last_osm);
                    edge_distance_m.push_back(total_dist_m);
                    edge_freeflow_kmh.push_back(freeflow_kmh);

                    for (size_t i = 0; i + 1 < node_seq.size(); ++i) {
                        uint64_t osm_a = get_osm_id(osm_node_ids, node_seq[i]);
                        uint64_t osm_b = get_osm_id(osm_node_ids, node_seq[i+1]);
                        node_pair_to_edge[{osm_a, osm_b}] = edge_idx;
                    }
                    ++edge_idx;
                }

                // ── Reverse direction ──
                auto [rev_dur_s, rev_valid] = sum_durations(
                    segment_data.GetReverseDurations(geom_id));

                if (rev_valid) {
                    double rev_freeflow_kmh = 1.0;
                    if (rev_dur_s > 0.001) {
                        rev_freeflow_kmh = (total_dist_m / rev_dur_s) * 3.6;
                        if (rev_freeflow_kmh < 1.0) rev_freeflow_kmh = 1.0;
                    }

                    edge_from_osm.push_back(last_osm);
                    edge_to_osm.push_back(first_osm);
                    edge_distance_m.push_back(total_dist_m);
                    edge_freeflow_kmh.push_back(rev_freeflow_kmh);

                    for (size_t i = node_seq.size() - 1; i > 0; --i) {
                        uint64_t osm_a = get_osm_id(osm_node_ids, node_seq[i]);
                        uint64_t osm_b = get_osm_id(osm_node_ids, node_seq[i-1]);
                        node_pair_to_edge[{osm_a, osm_b}] = edge_idx;
                    }
                    ++edge_idx;
                }
            }

            // ── 4. Build CompressedGraphLookup object ──────────────
            auto lookup = new CompressedGraphLookup();
            lookup->pair_to_edge = std::move(node_pair_to_edge);
            lookup->edge_distance_m = std::move(edge_distance_m);
            lookup->edge_freeflow_kmh = std::move(edge_freeflow_kmh);
            lookup->skipped_non_driving = skipped_non_driving;

            return lookup;
        },
        nb::arg("osrm_base_path"),
        nb::rv_policy::take_ownership,
        "Load OSRM compressed graph and build node-pair → edge lookup.\n\n"
        "Returns a CompressedGraphLookup object that can be passed to\n"
        "batch_route_accumulate for correct freeflow speed resolution.\n"
        "Built once, reused across all batch calls (no per-batch rebuild).\n"
    );
}
