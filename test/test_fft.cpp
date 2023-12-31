#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <cstdint>
#include <cryptography_cuda/src/lib.h>
#include <gmp.h>
#include "../c/alt_bn128.hpp"
#include <time.h>
#include "fft.hpp"
#include <random>

#define assertm(exp, msg) assert(((void)msg, exp))

using namespace AltBn128;

__uint128_t g_lehmer64_state = 0xAAAAAAAAAAAAAAAALL;

void print_char_array(uint8_t *p, uint32_t size)
{
    for (int i = 0; i < size; i++)
    {

        if (i % 32 == 0)
        {
            printf("\n");
        }
        printf("%02x", p[i]);
    }
}

int main(int argc, char **argv)
{
    printf("start fft \n");
    int lg_n_size = 3;
    int N = 1 << lg_n_size;

    std::string raw_data_str_array[8] = {
        "30644e72e131a029b85045b68181585d2833e84879b9709045d13bda42a5dc53",
        "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000",
        "0",
        "0",
        "0",
        "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000",
        "0",
        "0"};

    // for (int i = 0; i < N * 32; i++)
    // {
    //     uint8_t random_int = random_byte();
    //     *(raw_data + i) = i % 32 == 31 ? 0 : random_int; // TODO: this is to make the input less than MOD; otherwise, the test will fail
    // }
    // // print_char_array(raw_data, N*32);

    AltBn128::FrElement *cpu_data_in = new AltBn128::FrElement[N];
    for (int i = 0; i < N; i++)
    {
        Fr.fromString(cpu_data_in[i], raw_data_str_array[i], 16);
    }

    FFT<typename Engine::Fr> fft(N);
    double start, end;
    double cpu_time_used;
    start = omp_get_wtime();
    fft.ifft(cpu_data_in, N);
    end = omp_get_wtime();
    cpu_time_used = ((double)(end - start));
    printf("\n Time used fft (us): %.3lf\n", cpu_time_used * 1000000); // lf stands for long float

    for (int i = 0; i < N; i++)
    {
        std::string result = Fr.toString(cpu_data_in[i],  16);
        std::cout << result << std::endl;
    }

    // fr_t *gpu_data_in = (fr_t *)malloc(N * sizeof(fr_t));
    // for (int i = 0; i < N; i++)
    // {
    //     gpu_data_in[i] = *(fr_t *)(raw_data + 32 * i);
    // }
    // delete[] raw_data;

    // size_t device_id = 0;
    // compute_ntt(device_id, gpu_data_in, lg_n_size, Ntt_Types::InputOutputOrder::NN, Ntt_Types::Direction::forward, Ntt_Types::Type::standard);

    // for (int i = 0; i < N; i++)
    // {
    //     uint8_t cpu_result[32], gpu_result[32];
    //     Fr.toRprLE(cpu_data_in[i], cpu_result, 32);
    //     memcpy(gpu_result, (uint8_t *)(gpu_data_in + i), 32);

    //     for (int i = 0; i < 32; i++)
    //     {

    //         ASSERT_EQ(cpu_result[i], (gpu_result[i]));
    //     }
    // }
    // delete[] cpu_data_in;
    // delete[] gpu_data_in;
}
