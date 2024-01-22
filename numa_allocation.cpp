//
// Created by buettner on 22.01.24.
//
#include <sycl/sycl.hpp>
#include <iostream>

/*
 * This benchmark tests memory bandwidth for different ways of buffer initialization.
 * This is mostly important on AMD CPUs, where you have multiple NUMA domains per socket.
 *
 * The first benchmark is a standard parallel initialization, so data should be placed in the correct NUMA domain.
 * In the second benchmark, data is initialized by the main thread, so the data might be in the wrong NUMA domain.
 * In the third test, data is copied in from a vector with queue::copy. Depending on the implementation, data may
 * or may not be in the correct NUMA domain for each thread.
 * In the fourth test, the buffer is first zeroed by a parallel_for loop and then initialized by queue::copy. This should
 * yield approximately the same bandwidth as in the first benchmark.
 * The last test uses handler::fill to zero the buffer and then copies the data with queue::copy.
 */

void benchmark_copy(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions,
        sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& b);
void parallel_init(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions);
void sequential_init(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions);
void copy_init(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions);
void first_touch_copy_init(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions);
void fill_copy_init(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions);
void host_ptr_init(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions);

int main(int argc, char* argv[])
{
    sycl::queue queue({ sycl::property::queue::enable_profiling() });

    std::cout << "Device name: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    bool has_profiling = queue.get_device().has(sycl::aspect::queue_profiling);
    if (!has_profiling) {
        std::cout << "Device does not support profiling with events!\n";
    }

    size_t len = 64*8'000'000;
    if (argc>1) {
        size_t n = std::atol(argv[1]);
        len = 64*n;
    }
    std::cout << "Total data volume: " << len*sizeof(float)*2 << " bytes.\n";
    constexpr size_t repetitions = 10;

    parallel_init(queue, has_profiling, len, repetitions);
    sequential_init(queue, has_profiling, len, repetitions);
    copy_init(queue, has_profiling, len, repetitions);
    first_touch_copy_init(queue, has_profiling, len, repetitions);
    fill_copy_init(queue, has_profiling, len, repetitions);
    host_ptr_init(queue, has_profiling, len, repetitions);
}

void parallel_init(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions)
{
    sycl::buffer<float, 1> a{ sycl::range<1>{ len }};
    sycl::buffer<float, 1> b{ sycl::range<1>{ len }};

    queue.submit([&](sycl::handler& cgh) {
        auto aAcc = a.get_access<sycl::access_mode::discard_write>(cgh);
        cgh.parallel_for<class Init>(len, [=](sycl::id<1> id) {
            aAcc[id] = 1.0f;
        });
    }).wait();

    std::cout << "Vector copy with length " << len << ", parallel initialization, " << repetitions << " repetitions.\n";
    benchmark_copy(queue, has_profiling, len, repetitions, a, b);
}

void sequential_init(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions)
{
    sycl::buffer<float, 1> a{ sycl::range<1>{ len }};
    sycl::buffer<float, 1> b{ sycl::range<1>{ len }};

    {
        auto aAcc = a.get_host_access();
        for (size_t i = 0; i<len; i++) {
            aAcc[i] = 1.0f;
        }
    }

    std::cout << "Vector copy with length " << len << ", sequential initialization, " << repetitions
              << " repetitions.\n";
    benchmark_copy(queue, has_profiling, len, repetitions, a, b);
}

void copy_init(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions)
{
    std::vector<float> a_data(len, 1.0f);

    sycl::buffer<float, 1> a{ sycl::range<1>{ len }};
    sycl::buffer<float, 1> b{ sycl::range<1>{ len }};

    {
        auto aAcc = a.get_host_access();
        queue.copy(a_data.data(), aAcc.get_pointer(), len).wait();
    }

    std::cout << "Vector copy with length " << len << ", copy initialization, " << repetitions
              << " repetitions.\n";
    benchmark_copy(queue, has_profiling, len, repetitions, a, b);
}

void first_touch_copy_init(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions)
{
    std::vector<float> a_data(len, 1.0f);

    sycl::buffer<float, 1> a{ sycl::range<1>{ len }};
    sycl::buffer<float, 1> b{ sycl::range<1>{ len }};

    queue.submit([&](sycl::handler& cgh) {
        auto aAcc = a.get_access<sycl::access_mode::discard_write>(cgh);
        cgh.parallel_for(len, [=](sycl::id<1> id) {
            aAcc[id] = 0.0f;
        });
    }).wait();

    {
        auto aAcc = a.get_host_access();
        queue.copy(a_data.data(), aAcc.get_pointer(), len).wait();
    }

    std::cout << "Vector copy with length " << len << ", parallel first touch + copy initialization, " << repetitions
              << " repetitions.\n";
    benchmark_copy(queue, has_profiling, len, repetitions, a, b);
}

void fill_copy_init(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions)
{
    std::vector<float> a_data(len, 1.0f);

    sycl::buffer<float, 1> a{ sycl::range<1>{ len }};
    sycl::buffer<float, 1> b{ sycl::range<1>{ len }};

    queue.submit([&](sycl::handler& cgh) {
        auto aAcc = a.get_access<sycl::access_mode::discard_write>(cgh);
        float init = 0.0f;
        cgh.fill(aAcc, init);
    }).wait();

    {
        auto aAcc = a.get_host_access();
        queue.copy(a_data.data(), aAcc.get_pointer(), len).wait();
    }

    std::cout << "Vector copy with length " << len << ", fill + copy initialization, " << repetitions
              << " repetitions.\n";
    benchmark_copy(queue, has_profiling, len, repetitions, a, b);
}

void host_ptr_init(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions)
{
    std::vector<float> a_data(len, 1.0f);

    sycl::buffer<float, 1> a{ a_data.data(), sycl::range<1>{ len }};
    sycl::buffer<float, 1> b{ sycl::range<1>{ len }};

    std::cout << "Vector copy with length " << len << ", host pointer initialization, " << repetitions
              << " repetitions.\n";
    benchmark_copy(queue, has_profiling, len, repetitions, a, b);
}

void benchmark_copy(sycl::queue& queue, bool has_profiling, size_t len, const size_t repetitions,
        sycl::buffer<float, 1>& a, sycl::buffer<float, 1>& b)
{
    for (size_t i = 0; i<repetitions; i++) {
        auto ev = queue.submit([&](sycl::handler& cgh) {
            auto aAcc = a.get_access<sycl::access_mode::read>(cgh);
            auto bAcc = b.get_access<sycl::access_mode::discard_write>(cgh);
            cgh.parallel_for<class GPUTriad>(len, [=](sycl::id<1> id) {
                bAcc[id] = aAcc[id];
            });
        });

        if (has_profiling) {
            unsigned long duration_ns = ev.get_profiling_info<sycl::info::event_profiling::command_end>()
                    -ev.get_profiling_info<sycl::info::event_profiling::command_start>();
            size_t memory_size = 2*len*sizeof(float);
            std::cout << "Estimated bandwidth: " << static_cast<double>(memory_size)/static_cast<double>(duration_ns)
                      << " GByte/s (" << duration_ns << " ns)\n";
        }
        else {
            ev.wait();
        }
    }
}

