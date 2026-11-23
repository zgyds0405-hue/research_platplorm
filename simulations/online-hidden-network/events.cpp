#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>

// Integrate holding times over a window. The caller samples its initial state
// from the exact finite-time law for an initially all-online population.
extern "C" double observe(int N, int n, double on, double off, double duration,
                          std::uint64_t seed, double *occupancy) {
    std::mt19937_64 rng(seed);
    auto uniform = [&]() { return ((rng() >> 12) + 0.5) * 0x1.0p-52; };
    double t = 0, integral = 0;
    while (t < duration) {
        double birth = (N - n) * on;
        double total = birth + n * off;
        double dt = std::min(-std::log(uniform()) / total, duration - t);
        integral += dt * n;
        if (occupancy) occupancy[n] += dt;
        t += dt;
        if (t < duration) n += uniform() * total < birth ? 1 : -1;
    }
    return integral / duration;
}
