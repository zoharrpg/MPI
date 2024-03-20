#include <algorithm>
#include <iostream>
#include <unistd.h>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <string>
#include <vector>
#include <random>

#include <mpi.h>

#include "wireroute.h"

void print_stats(const std::vector <std::vector<int>> &occupancy) {
    int max_occupancy = 0;
    long long total_cost = 0;

    for (const auto &row: occupancy) {
        for (const int count: row) {
            max_occupancy = std::max(max_occupancy, count);
            total_cost += count * count;
        }
    }

    std::cout << "Max occupancy: " << max_occupancy << '\n';
    std::cout << "Total cost: " << total_cost << '\n';
}

void write_output(const std::vector <Wire> &wires, const int num_wires, const std::vector <std::vector<int>> &occupancy,
                  const int dim_x, const int dim_y, const int nproc, std::string input_filename) {
    if (std::size(input_filename) >= 4 && input_filename.substr(std::size(input_filename) - 4) == ".txt") {
        input_filename.resize(std::size(input_filename) - 4);
    }

    const std::string occupancy_filename = input_filename + "_occupancy_" + std::to_string(nproc) + ".txt";
    const std::string wires_filename = input_filename + "_wires_" + std::to_string(nproc) + ".txt";

    std::ofstream out_occupancy(occupancy_filename, std::fstream::out);
    if (!out_occupancy) {
        std::cerr << "Unable to open file: " << occupancy_filename << '\n';
        exit(EXIT_FAILURE);
    }

    out_occupancy << dim_x << ' ' << dim_y << '\n';
    for (const auto &row: occupancy) {
        for (const int count: row) {
            out_occupancy << count << ' ';
        }
        out_occupancy << '\n';
    }

    out_occupancy.close();

    std::ofstream out_wires(wires_filename, std::fstream::out);
    if (!out_wires) {
        std::cerr << "Unable to open file: " << wires_filename << '\n';
        exit(EXIT_FAILURE);
    }

    out_wires << dim_x << ' ' << dim_y << '\n' << num_wires << '\n';

    for (const auto &[start_x, start_y, end_x, end_y, bend1_x, bend1_y]: wires) {
        out_wires << start_x << ' ' << start_y << ' ' << bend1_x << ' ' << bend1_y << ' ';

        if (start_y == bend1_y) {
            // first bend was horizontal

            if (end_x != bend1_x) {
                // two bends

                out_wires << bend1_x << ' ' << end_y << ' ';
            }
        } else if (start_x == bend1_x) {
            // first bend was vertical

            if (end_y != bend1_y) {
                // two bends

                out_wires << end_x << ' ' << bend1_y << ' ';
            }
        }
        out_wires << end_x << ' ' << end_y << '\n';
    }

    out_wires.close();
}


double fixed_probability(const double prob = -1) {
    static double fixed_prob = -1;
    if (prob > 0) {
        fixed_prob = prob;
    }
    return fixed_prob;
}

bool random_happen() {
    static thread_local std::mt19937
    generator(std::random_device{}());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    static const double probability = fixed_probability();
    return distribution(generator) < probability;
}

int num_bends(const Wire &wire) {
    /* Returns the number of bends in the wire */
    return (wire.start_x != wire.bend1_x) + (wire.bend1_x != wire.end_x) + (wire.start_y != wire.bend1_y) +
           (wire.bend1_y != wire.end_y) - 1;
}

template<bool CalculateDeltaCost, bool UpdateOccupancy>
cost_t update_point(const int x, const int y, std::vector <std::vector<int>> &occupancy, const int delta) {
    /* Update the occupancy count of a point (x, y) */
    cost_t delta_cost = 0;
    if constexpr(CalculateDeltaCost)
    {
        delta_cost = (occupancy[y][x] + delta) * (occupancy[y][x] + delta) - occupancy[y][x] * occupancy[y][x];
    }
    if constexpr(UpdateOccupancy)
    {
        occupancy[y][x] += delta;
    }
    return delta_cost;
}

template<bool CalculateDeltaCost, bool UpdateOccupancy>
cost_t update_wire_no_bend(const Wire &wire, std::vector <std::vector<int>> &occupancy, const int delta) {
    cost_t delta_cost = 0;
    if (wire.start_x == wire.end_x) {
        // Vertical wire
        for (int y = std::min(wire.start_y, wire.end_y); y <= std::max(wire.start_y, wire.end_y); y++) {
            delta_cost += update_point<CalculateDeltaCost, UpdateOccupancy>(wire.start_x, y, occupancy, delta);
        }
    } else {
        // Horizontal wire
        for (int x = std::min(wire.start_x, wire.end_x); x <= std::max(wire.start_x, wire.end_x); x++) {
            delta_cost += update_point<CalculateDeltaCost, UpdateOccupancy>(x, wire.start_y, occupancy, delta);
        }
    }
    return delta_cost;
}

template<bool CalculateDeltaCost, bool UpdateOccupancy>
cost_t update_wire_one_bend(const Wire &wire, std::vector <std::vector<int>> &occupancy, const int delta) {
    cost_t delta_cost = 0;
    if (wire.start_x == wire.bend1_x) {
        // Vertical first bend
        int direction = wire.start_y < wire.bend1_y ? 1 : -1;
        for (int y = wire.start_y; y != wire.bend1_y; y += direction) {
            delta_cost += update_point<CalculateDeltaCost, UpdateOccupancy>(wire.start_x, y, occupancy, delta);
        }

        // Horizontal second bend
        for (int x = std::min(wire.bend1_x, wire.end_x); x <= std::max(wire.bend1_x, wire.end_x); x++) {
            delta_cost += update_point<CalculateDeltaCost, UpdateOccupancy>(x, wire.bend1_y, occupancy, delta);
        }
    } else {
        // Horizontal first bend
        int direction = wire.start_x < wire.bend1_x ? 1 : -1;
        for (int x = wire.start_x; x != wire.bend1_x; x += direction) {
            delta_cost += update_point<CalculateDeltaCost, UpdateOccupancy>(x, wire.start_y, occupancy, delta);
        }

        // Vertical second bend
        for (int y = std::min(wire.bend1_y, wire.end_y); y <= std::max(wire.bend1_y, wire.end_y); y++) {
            delta_cost += update_point<CalculateDeltaCost, UpdateOccupancy>(wire.bend1_x, y, occupancy, delta);
        }
    }

    return delta_cost;
}

template<bool CalculateDeltaCost, bool UpdateOccupancy>
cost_t update_wire_two_bends(const Wire &wire, std::vector <std::vector<int>> &occupancy, const int delta) {
    cost_t delta_cost = 0;
    if (wire.start_x == wire.bend1_x) {
        // Vertical first bend
        int direction = wire.start_y < wire.bend1_y ? 1 : -1;
        for (int y = wire.start_y; y != wire.bend1_y; y += direction) {
            delta_cost += update_point<CalculateDeltaCost, UpdateOccupancy>(wire.start_x, y, occupancy, delta);
        }

        // Horizontal second bend
        direction = wire.bend1_x < wire.end_x ? 1 : -1;
        for (int x = wire.bend1_x; x != wire.end_x; x += direction) {
            delta_cost += update_point<CalculateDeltaCost, UpdateOccupancy>(x, wire.bend1_y, occupancy, delta);
        }

        // Vertical third
        for (int y = std::min(wire.bend1_y, wire.end_y); y <= std::max(wire.bend1_y, wire.end_y); y++) {
            delta_cost += update_point<CalculateDeltaCost, UpdateOccupancy>(wire.end_x, y, occupancy, delta);
        }
    } else {
        // Horizontal first bend
        int direction = wire.start_x < wire.bend1_x ? 1 : -1;
        for (int x = wire.start_x; x != wire.bend1_x; x += direction) {
            delta_cost += update_point<CalculateDeltaCost, UpdateOccupancy>(x, wire.start_y, occupancy, delta);
        }

        // Vertical second bend
        direction = wire.bend1_y < wire.end_y ? 1 : -1;
        for (int y = wire.bend1_y; y != wire.end_y; y += direction) {
            delta_cost += update_point<CalculateDeltaCost, UpdateOccupancy>(wire.bend1_x, y, occupancy, delta);
        }

        // Horizontal second
        for (int x = std::min(wire.bend1_x, wire.end_x); x <= std::max(wire.bend1_x, wire.end_x); x++) {
            delta_cost += update_point<CalculateDeltaCost, UpdateOccupancy>(x, wire.end_y, occupancy, delta);
        }
    }

    return delta_cost;
}

template<bool CalculateDeltaCost, bool UpdateOccupancy>
cost_t update_wire(const Wire &wire, std::vector <std::vector<int>> &occupancy, const int delta) {
    cost_t delta_cost = 0;
    switch (int num_bend = num_bends(wire)) {
        case 0:
            // No bends
            delta_cost = update_wire_no_bend<CalculateDeltaCost, UpdateOccupancy>(wire, occupancy, delta);
            break;
        case 1:
            // One bend
            delta_cost = update_wire_one_bend<CalculateDeltaCost, UpdateOccupancy>(wire, occupancy, delta);
            break;
        case 2:
            // Two bends
            delta_cost = update_wire_two_bends<CalculateDeltaCost, UpdateOccupancy>(wire, occupancy, delta);
            break;
        default:
            std::cerr << "Invalid number of bends: " << num_bend << '\n';
            exit(EXIT_FAILURE);
    }
    return delta_cost;
}

cost_t initialize(const std::vector <Wire> &wires, std::vector <std::vector<int>> &occupancy) {
    /* Initialize occupancy matrix */
    for (unsigned int i = 0; i < std::size(wires); i++) {
        update_wire<false, true>(wires[i], occupancy, 1);
    }
    return 0;
}

template<bool CalculateDeltaCost>
cost_t set_bend(int index, std::vector <std::vector<int>> *occupancy, Wire &wire) {
    int delta_x = std::abs(wire.start_x - wire.end_x);
    cost_t delta_cost = 0;
    if (index <= delta_x) {
        wire.bend1_x = wire.start_x + (wire.start_x < wire.end_x ? index : -index);
        wire.bend1_y = wire.start_y;
    } else {
        index -= delta_x;
        wire.bend1_x = wire.start_x;
        wire.bend1_y = wire.start_y + (wire.start_y < wire.end_y ? index : -index);
    }
    if constexpr(CalculateDeltaCost)
    {
        delta_cost = update_wire<true, false>(wire, *occupancy, 1);
    }

    return delta_cost;
}

void random_bend(Wire &wire) {
    int delta_x = std::abs(wire.end_x - wire.start_x);
    int delta_y = std::abs(wire.end_y - wire.start_y);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis_x(1, delta_x + delta_y);

    set_bend<false>(dis_x(gen), nullptr, wire);
}

void across_wires(std::vector <Wire> &wires, std::vector <std::vector<int>> &occupancy, const int iterations,
                  const int batch_size, int nproc, int pid) {
    auto num_wires = static_cast<int>(std::size(wires));
    MPI_Status status;
    for (int iter = 0; iter < iterations; iter++) {
        for (int start = 0; start < num_wires; start += batch_size) {
            int end = std::min(start + batch_size, num_wires);
            // Remove all wires in the batch from the occupancy matrix
            for (int i = start; i < end; i++) {
                update_wire<false, true>(wires[i], occupancy, -1);
            }
            int span = (end - start + nproc - 1) / nproc;
            int span_start = start + pid * span;
            int span_end = std::min(span_start + span, end);
            for (int wire_index = span_start; wire_index < span_end; wire_index++) {
                if (random_happen()) {
                    random_bend(wires[wire_index]);
                    update_wire<true, true>(wires[wire_index], occupancy, 1);
                    continue;
                }
                cost_t delta_cost = std::numeric_limits<cost_t>::max();
                Wire wire = wires[wire_index];
                // If the wire is horizontal or vertical, skip
                if (num_bends(wire) == 0) {
                    continue;
                }

                int delta_x = std::abs(wire.start_x - wire.end_x);
                int delta_y = std::abs(wire.start_y - wire.end_y);
                for (int i = 1; i <= delta_x + delta_y; i++) {
                    cost_t _delta_cost = set_bend<true>(i, &occupancy, wire);
                    if (_delta_cost < delta_cost) {
                        delta_cost = _delta_cost;
                        wires[wire_index].bend1_x = wire.bend1_x;
                        wires[wire_index].bend1_y = wire.bend1_y;
                    }
                }
            }
            if (pid != 0) {
                MPI_Send(&wires[span_start], (span_end - span_start) * sizeof(Wire), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            } else {
                for (int source = 1; source < nproc; source++) {
                    int source_start = start + source * span;
                    int source_end = std::min(source_start + span, end);
                    MPI_Recv(&wires[source_start], (source_end - source_start) * sizeof(Wire), MPI_BYTE, source, 0, MPI_COMM_WORLD,
                             &status);
                }
            }
            MPI_Bcast(&wires[start], (end - start) * sizeof(Wire), MPI_BYTE, 0, MPI_COMM_WORLD);

            for (int wire_index = start; wire_index < end; wire_index++) {
                update_wire<false, true>(wires[wire_index], occupancy, 1);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    const auto init_start = std::chrono::steady_clock::now();
    int pid;
    int nproc;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    // Get process rank
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    // Get total number of processes specificed at start of run
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    std::string input_filename;
    double SA_prob = 0.1;
    int SA_iters = 5;
    char parallel_mode = '\0';
    int batch_size = 1;

    // Read command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "f:p:i:m:b:")) != -1) {
        switch (opt) {
            case 'f':
                input_filename = optarg;
                break;
            case 'p':
                SA_prob = atof(optarg);
                break;
            case 'i':
                SA_iters = atoi(optarg);
                break;
            case 'm':
                parallel_mode = *optarg;
                break;
            case 'b':
                batch_size = atoi(optarg);
                break;
            default:
                if (pid == 0) {
                    std::cerr << "Usage: " << argv[0]
                              << " -f input_filename [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
                }

                MPI_Finalize();
                exit(EXIT_FAILURE);
        }
    }

    // Check if required options are provided
    if (empty(input_filename) || SA_iters <= 0 || (parallel_mode != 'A' && parallel_mode != 'W') || batch_size <= 0) {
        if (pid == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " -f input_filename [-p SA_prob] [-i SA_iters] -m parallel_mode -b batch_size\n";
        }

        MPI_Finalize();
        exit(EXIT_FAILURE);
    }

    if (pid == 0) {
        std::cout << "Number of processes: " << nproc << '\n';
        std::cout << "Simulated annealing probability parameter: " << SA_prob << '\n';
        std::cout << "Simulated annealing iterations: " << SA_iters << '\n';
        std::cout << "Input file: " << input_filename << '\n';
        std::cout << "Parallel mode: " << parallel_mode << '\n';
        std::cout << "Batch size: " << batch_size << '\n';
    }

    int dim_x, dim_y, num_wires;
    std::vector <Wire> wires;
    std::vector <std::vector<int>> occupancy;


    if (pid == 0) {
        std::ifstream fin(input_filename);

        if (!fin) {
            std::cerr << "Unable to open file: " << input_filename << ".\n";
            exit(EXIT_FAILURE);
        }

        /* Read the grid dimension and wire information from file */
        fin >> dim_x >> dim_y >> num_wires;

        wires.resize(num_wires);
        for (auto &wire: wires) {
            fin >> wire.start_x >> wire.start_y >> wire.end_x >> wire.end_y;
            wire.bend1_x = wire.start_x;
            wire.bend1_y = wire.end_y;
        }
    }
    MPI_Bcast(&num_wires, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (pid != 0) {
        wires.resize(num_wires);
    }

    MPI_Bcast(&dim_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dim_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
    occupancy.resize(dim_y, std::vector<int>(dim_x));

    /* Initialize any additional data structures needed in the algorithm */
    if (pid == 0) {
        std::sort(wires.begin(), wires.end(), [](const Wire &a, const Wire &b) {
            int distance_a = std::abs(a.start_x - a.end_x) + std::abs(a.start_y - a.end_y);
            int distance_b = std::abs(b.start_x - b.end_x) + std::abs(b.start_y - b.end_y);
            return distance_a > distance_b; // Descending order
        });
    }

    MPI_Bcast(wires.data(), num_wires * sizeof(Wire), MPI_BYTE, 0, MPI_COMM_WORLD);
    initialize(wires, occupancy);

    fixed_probability(SA_prob);

    if (pid == 0) {
        const double init_time = std::chrono::duration_cast < std::chrono::duration <
                                 double >> (std::chrono::steady_clock::now() - init_start).count();
        std::cout << "Initialization time (sec): " << std::fixed << std::setprecision(10) << init_time << '\n';
    }

    const auto compute_start = std::chrono::steady_clock::now();
    switch (parallel_mode) {
        case 'W':
            throw std::runtime_error("Within wire mode not implemented");
            break;
        case 'A':
            across_wires(wires, occupancy, SA_iters, batch_size, nproc, pid);
            break;
    }


    if (pid == 0) {
        const double compute_time = std::chrono::duration_cast < std::chrono::duration <
                                    double >> (std::chrono::steady_clock::now() - compute_start).count();
        std::cout << "Computation time (sec): " << std::fixed << std::setprecision(10) << compute_time << '\n';
    }

    if (pid == 0) {
        /* Write wires and occupancy matrix to files */
        print_stats(occupancy);
        write_output(wires, num_wires, occupancy, dim_x, dim_y, nproc, input_filename);
    }

    // Cleanup
    MPI_Finalize();
}
