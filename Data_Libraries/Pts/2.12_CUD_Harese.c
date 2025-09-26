#include <stdio.h>
#include <stdlib.h>

// Initial seed state
static unsigned int state12 = 0xd6945853U;

/* Tausworthe/LFSR generator with period 2^12 - 1 */
unsigned int rng12(void)
{
    unsigned int x = 0U;
    unsigned int t = 0x80000000U;
    unsigned int mat[12] = {
    0x510e0fc6, 0x79890825, 0xedca8bd4, 0x27eb4a2c, 0x42fbaad0,
    0xa17dd568, 0xd0beeab4, 0xb9517a9c, 0xdca8bd4e, 0x6e545ea7,
    0xe6242095, 0xa21c1f8c
    };
    for (int i = 0; i < 12; i++) {
        if (state12 & t) {
            x ^= mat[i];
        }
        t >>= 1;
    }
    state12 = x;
    return state12;
}

int main(void)
{
    unsigned int m = 12;
    unsigned int N = (1U << m) - 1;   // N = 2^m - 1 = 4095

    double *values = malloc(N * sizeof(double));
    if (values == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // generate the full sequence (no explicit seed output)
    for (unsigned int i = 0; i < N; i++) {
        values[i] = rng12() * (1.0 / 4294967296.0);
    }

    FILE *fp = fopen("C:/Users/mussi/Documents/Manhattan/Leuven/MCQMC/Data_Libraries/Pts/2.12rng.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Cannot open output file\n");
        free(values);
        return 1;
    }

    for (unsigned int i = 0; i < N; i++) {
        fprintf(fp, "%10.8f\n", values[i]);
    }

    fclose(fp);
    free(values);

    return 0;
}

