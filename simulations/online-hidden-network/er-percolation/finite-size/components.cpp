// Incrementally maintain component sizes while vertices are occupied.
#include <cstdint>
#include <iterator>
#include <set>
#include <vector>

extern "C" void component_scan(int n, const int32_t* offsets, const int32_t* neighbors,
                               const int32_t* order, int bins, const int32_t* occupied,
                               int32_t* result) {
    std::vector<int> parent(n, -1), size(n, 1);
    std::multiset<int> sizes;
    auto root = [&](int v) {
        while (v != parent[v]) {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        return v;
    };
    int count = 0;
    for (int j = 0; j < bins; ++j) {
        while (count < occupied[j]) {
            int v = order[count++], a = v;
            parent[v] = v;
            int merged_size = 1;
            for (int e = offsets[v]; e < offsets[v + 1]; ++e) {
                int w = neighbors[e];
                if (parent[w] < 0) continue;
                int b = root(w);
                if (a == b) continue;
                sizes.erase(sizes.find(size[b]));
                if (merged_size < size[b]) {
                    parent[a] = b;
                    a = b;
                } else {
                    parent[b] = a;
                }
                merged_size += size[b];
                size[a] = merged_size;
            }
            sizes.insert(merged_size);
        }
        auto it = sizes.rbegin();
        result[2*j] = it == sizes.rend() ? 0 : *it++;
        result[2*j+1] = it == sizes.rend() ? 0 : *it;
    }
}
