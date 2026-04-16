#pragma once

#include <nanobind/nanobind.h>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace nb = nanobind;

struct PairHashCG {
    std::size_t operator()(std::pair<uint64_t,uint64_t> const& p) const noexcept {
        auto h1 = std::hash<uint64_t>{}(p.first);
        auto h2 = std::hash<uint64_t>{}(p.second);
        h1 ^= h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2);
        return h1;
    }
};

// Holds compressed graph data: node-pair → compressed edge index lookup,
// plus per-edge freeflow speed and distance. Built once, reused across batches.
struct CompressedGraphLookup {
    std::unordered_map<std::pair<uint64_t,uint64_t>, uint32_t, PairHashCG> pair_to_edge;
    std::vector<double> edge_distance_m;
    std::vector<double> edge_freeflow_kmh;
    size_t skipped_non_driving = 0;

    // Look up a node pair. Returns compressed edge index, or -1 if not found.
    int find(uint64_t from_osm, uint64_t to_osm) const {
        auto it = pair_to_edge.find({from_osm, to_osm});
        return (it != pair_to_edge.end()) ? static_cast<int>(it->second) : -1;
    }

    double distance(int edge_idx) const { return edge_distance_m[edge_idx]; }
    double freeflow(int edge_idx) const { return edge_freeflow_kmh[edge_idx]; }
    size_t n_edges() const { return edge_distance_m.size(); }
    size_t n_pairs() const { return pair_to_edge.size(); }
};

void init_CompressedGraph(nb::module_& m);
