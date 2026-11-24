// Ascending occupancy scan for all-or-nothing hyperedges.
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <set>
#include <vector>

extern "C" void scan(int n, int h, const int32_t* edges, const int32_t* vertex_order,
                     const int32_t* edge_order, int bins, const int32_t* vertices_at,
                     const int32_t* edges_at, int32_t* out) {
    std::vector<int> parent(n), size(n, 1), frequency(n+1, 0);
    std::set<int> sizes;
    auto root = [&](int v) {
        while (v != parent[v]) {
            parent[v] = parent[parent[v]];
            v = parent[v];
        }
        return v;
    };
    auto add = [&](int s) { if (frequency[s]++ == 0) sizes.insert(s); };
    auto remove = [&](int s) { if (--frequency[s] == 0) sizes.erase(s); };
    int vc = 0, ec = 0, components = 0;
    for (int j = 0; j < bins; ++j) {
        while (vc < vertices_at[j]) {
            int v = vertex_order[vc++];
            parent[v] = v;
            add(1);
            ++components;
        }
        while (ec < edges_at[j]) {
            const int32_t* members = edges + h * edge_order[ec++];
            int a = root(members[0]);
            for (int i = 1; i < h; ++i) {
                int b = root(members[i]);
                if (a == b) continue;
                if (size[a] < size[b]) std::swap(a,b);
                remove(size[a]);
                remove(size[b]);
                parent[b] = a;
                size[a] += size[b];
                add(size[a]);
                --components;
            }
        }
        auto it = sizes.rbegin();
        int first = it == sizes.rend() ? 0 : *it;
        int second = first;
        if (first && frequency[first] == 1) {
            ++it;
            second = it == sizes.rend() ? 0 : *it;
        }
        out[5*j] = first;
        out[5*j+1] = second;
        out[5*j+2] = ec;
        out[5*j+3] = vc;
        out[5*j+4] = components;
    }
}
