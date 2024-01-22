NUMA Allocation benchmark
----
The purpose of this test is to test memory bandwidth with SYCL on NUMA systems.
As there are differnt ways to initialize the data of `sycl::buffer`s, these tests aim to provide some
quantitative bandwidth results.

In particular, it tests:

- Sequential initialization from the main thread (with `get_host_pointer`)
- Parallel initialization from the worker threads (with `parallel_for`)
- Initialization by copying data from a different memory region (`queue.copy`)
- Doing parallel first touch (with `parallel_for`) and then copying with `queue.copy`
- Using `handler.fill` for initialization and then `queue.copy`
- Specifying a host pointer in the `buffer` constructor

Feel free to point out any issues with these tests.

Measured results:
----

Stated values are approximate and only to give an idea on what to expect. Please measure your system.


2x AMD EPYC 7543 (each 32 cores, codename "Milan").

|             | sequential init | parallel init | copy    | parallel first touch + copy | fill + copy | host pointer |   
|-------------|-----------------|---------------|---------|-----------------------------|-------------|--------------|  
| AdaptiveCpp | 54 GB/s         | 200 GB/s      | 54 GB/s | 190 GB/s                    | 190 GB/s    | 54 GB/s      |
| oneAPI      | 75 GB/s         | 75 GB/s       | 90 GB/s | 90 GB/s                     | 57 GB/s     | 57 GB/s      |

<details>
    <summary>NUMA topology (from likwid-topology)</summary>

```
********************************************************************************
NUMA Topology
********************************************************************************
NUMA domains:           8
--------------------------------------------------------------------------------
Domain:                 0
Processors:             ( 0 1 2 3 4 5 6 7 )
Distances:              10 12 12 12 32 32 32 32
Free memory:            31161.5 MB
Total memory:           31813.9 MB
--------------------------------------------------------------------------------
Domain:                 1
Processors:             ( 8 9 10 11 12 13 14 15 )
Distances:              12 10 12 12 32 32 32 32
Free memory:            31964 MB
Total memory:           32252.9 MB
--------------------------------------------------------------------------------
Domain:                 2
Processors:             ( 16 17 18 19 20 21 22 23 )
Distances:              12 12 10 12 32 32 32 32
Free memory:            30847.1 MB
Total memory:           32252.9 MB
--------------------------------------------------------------------------------
Domain:                 3
Processors:             ( 24 25 26 27 28 29 30 31 )
Distances:              12 12 12 10 32 32 32 32
Free memory:            31976.5 MB
Total memory:           32240.9 MB
--------------------------------------------------------------------------------
Domain:                 4
Processors:             ( 32 33 34 35 36 37 38 39 )
Distances:              32 32 32 32 10 12 12 12
Free memory:            32064.7 MB
Total memory:           32252.9 MB
--------------------------------------------------------------------------------
Domain:                 5
Processors:             ( 40 41 42 43 44 45 46 47 )
Distances:              32 32 32 32 12 10 12 12
Free memory:            32001.7 MB
Total memory:           32252.9 MB
--------------------------------------------------------------------------------
Domain:                 6
Processors:             ( 48 49 50 51 52 53 54 55 )
Distances:              32 32 32 32 12 12 10 12
Free memory:            32012.1 MB
Total memory:           32205.5 MB
--------------------------------------------------------------------------------
Domain:                 7
Processors:             ( 56 57 58 59 60 61 62 63 )
Distances:              32 32 32 32 12 12 12 10
Free memory:            32036.2 MB
Total memory:           32249.3 MB
--------------------------------------------------------------------------------
```
</details>

2x AMD EPYC 7352 (each 24 cores, codename "Rome").

|             | sequential init | parallel init | copy      | parallel first touch + copy | fill + copy | host pointer |   
|-------------|-----------------|---------------|-----------|-----------------------------|-------------|--------------|  
| AdaptiveCpp | 35 GB/s         | 90 GB/s       | 26.5 GB/s | 90 GB/s                     | 90 GB/s     | 26.5 GB/s    |
| oneAPI      | 35 GB/s         | 70 GB/s       | 50 GB/s   | 70 GB/s                     | 27 GB/s     | 27 GB/s      |

<details>
    <summary>NUMA topology (from likwid-topology)</summary>

```
********************************************************************************
NUMA Topology
********************************************************************************
NUMA domains:           8
--------------------------------------------------------------------------------
Domain:                 0
Processors:             ( 0 48 1 49 2 50 3 51 4 52 5 53 )
Distances:              10 12 12 12 32 32 32 32
Free memory:            31073.6 MB
Total memory:           31798.8 MB
--------------------------------------------------------------------------------
Domain:                 1
Processors:             ( 6 54 7 55 8 56 9 57 10 58 11 59 )
Distances:              12 10 12 12 32 32 32 32
Free memory:            31939.6 MB
Total memory:           32205.1 MB
--------------------------------------------------------------------------------
Domain:                 2
Processors:             ( 12 60 13 61 14 62 15 63 16 64 17 65 )
Distances:              12 12 10 12 32 32 32 32
Free memory:            30946.1 MB
Total memory:           32252.5 MB
--------------------------------------------------------------------------------
Domain:                 3
Processors:             ( 18 66 19 67 20 68 21 69 22 70 23 71 )
Distances:              12 12 12 10 32 32 32 32
Free memory:            31937.1 MB
Total memory:           32240.5 MB
--------------------------------------------------------------------------------
Domain:                 4
Processors:             ( 24 72 25 73 26 74 27 75 28 76 29 77 )
Distances:              32 32 32 32 10 12 12 12
Free memory:            31429.6 MB
Total memory:           32252.5 MB
--------------------------------------------------------------------------------
Domain:                 5
Processors:             ( 30 78 31 79 32 80 33 81 34 82 35 83 )
Distances:              32 32 32 32 12 10 12 12
Free memory:            31944.2 MB
Total memory:           32252.5 MB
--------------------------------------------------------------------------------
Domain:                 6
Processors:             ( 36 84 37 85 38 86 39 87 40 88 41 89 )
Distances:              32 32 32 32 12 12 10 12
Free memory:            31929.2 MB
Total memory:           32252.5 MB
--------------------------------------------------------------------------------
Domain:                 7
Processors:             ( 42 90 43 91 44 92 45 93 46 94 47 95 )
Distances:              32 32 32 32 12 12 12 10
Free memory:            31425.5 MB
Total memory:           32246.4 MB
--------------------------------------------------------------------------------
```
</details>


Values could be influenced by thread pinning.
