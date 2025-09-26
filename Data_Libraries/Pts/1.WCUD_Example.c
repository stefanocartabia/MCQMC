// FAIL; DO NOT USE THIS 


#include <stdint.h>
#include <stdio.h>

// ---- Step 1: construct mat[] from given (m,taps) polynomial ----

// One LFSR step returning a single bit
static inline uint32_t lfsr_step_bit(uint32_t *reg, int m, uint32_t taps) {
    uint32_t msb = (*reg >> (m-1)) & 1u;
    *reg = ((*reg << 1) & ((1u<<m)-1u));
    if (msb) *reg ^= taps;
    return msb;
}

// Build mat[i]: the 32-bit word output starting from basis state e_i
static void build_mat(int m, uint32_t taps, uint32_t mat[]) {
    for (int i = 0; i < m; i++) {
        uint32_t reg = (1u << i);
        uint32_t word = 0;
        for (int k = 0; k < 32; k++) {
            word = (word << 1) | lfsr_step_bit(&reg, m, taps);
        }
        mat[i] = word;
    }
}

// ---- Step 2: Harase-style state transition ----

static inline uint32_t taus_step(uint32_t *state, int m, const uint32_t mat[]) {
    uint32_t s = *state;
    uint32_t x = 0;
    uint32_t t = 1u << (m-1);
    for (int i = 0; i < m; i++, t >>= 1) {
        if (s & t) x ^= mat[i];
    }
    *state = x;
    return x;
}

// ---- Step 3: main program ----

int main(void) {
    int m = 6;
    uint32_t taps = 0x2Du;   // x^6+x^5+x^3+x^2+1

    uint32_t mat[6];
    build_mat(m, taps, mat);

    uint32_t state = 1u; // nonzero seed
    unsigned N = (1u << m) - 1; // 63

    for (unsigned i = 0; i < N; i++) {
        uint32_t word = taus_step(&state, m, mat);
        double u = word * (1.0 / 4294967296.0);
        printf("%.10f\n", u);
    }
    return 0;
}
