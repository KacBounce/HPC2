import matplotlib.pyplot as plt

num_processes = [2, 4, 8, 16, 24, 25]
results = {}

for p in num_processes:
    with open(f"result_{p}.txt", "r") as f:
        lines = f.readlines()[1:]
        results[p] = []
        for line in lines:
            matrix_size, num_procs, seq_time_gen, par_time_gen, seq_time_pso, par_time_pso = map(
                float, line.strip().split(","))
            results[p].append((int(matrix_size), int(num_procs),
                               seq_time_gen, par_time_gen, seq_time_pso, par_time_pso))

matrix_sizes = sorted(
    set(res[0] for res_list in results.values() for res in res_list))

speedups_genetic = {p: [] for p in num_processes}
efficiencies_genetic = {p: [] for p in num_processes}

speedups_pso = {p: [] for p in num_processes}
efficiencies_pso = {p: [] for p in num_processes}

for p in num_processes:
    for matrix_size in matrix_sizes:
        relevant_data = [entry for entry in results[p]
                         if entry[0] == matrix_size]

        if relevant_data:
            seq_time_gen = relevant_data[0][2]
            par_time_gen = relevant_data[0][3]
            speedups_genetic[p].append(seq_time_gen / par_time_gen)
            efficiencies_genetic[p].append(speedups_genetic[p][-1] / p)

            seq_time_pso = relevant_data[0][4]
            par_time_pso = relevant_data[0][5]
            speedups_pso[p].append(seq_time_pso / par_time_pso)
            efficiencies_pso[p].append(speedups_pso[p][-1] / p)

plt.figure(figsize=(10, 6))
for p in num_processes:
    plt.plot(matrix_sizes, speedups_genetic[p],
             label=f"Genetic - S(p) for {p} processes", marker='o')
plt.xlabel("Matrix Size (N)")
plt.ylabel("Speedup (S(p))")
plt.title("Speedup vs Matrix Size (Genetic Algorithm) - Logarithmic Scale")
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for p in num_processes:
    plt.plot(matrix_sizes, speedups_pso[p],
             label=f"PSO - S(p) for {p} processes", marker='x')
plt.xlabel("Matrix Size (N)")
plt.ylabel("Speedup (S(p))")
plt.title("Speedup vs Matrix Size (PSO Algorithm) - Logarithmic Scale")
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for p in num_processes:
    plt.plot(matrix_sizes, efficiencies_genetic[p],
             label=f"Genetic - E(p) for {p} processes", marker='o')
plt.xlabel("Matrix Size (N)")
plt.ylabel("Efficiency (E(p))")
plt.title("Efficiency vs Matrix Size (Genetic Algorithm) - Logarithmic Scale")
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for p in num_processes:
    plt.plot(matrix_sizes, efficiencies_pso[p],
             label=f"PSO - E(p) for {p} processes", marker='x')
plt.xlabel("Matrix Size (N)")
plt.ylabel("Efficiency (E(p))")
plt.title("Efficiency vs Matrix Size (PSO Algorithm) - Logarithmic Scale")
plt.yscale('log')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for matrix_size in matrix_sizes:
    matrix_speedups_genetic = [
        speedups_genetic[p][matrix_sizes.index(matrix_size)] for p in num_processes]
    plt.plot(num_processes, matrix_speedups_genetic,
             label=f"Genetic - S(p) for {matrix_size}x{matrix_size} matrix", marker='o')

plt.xlabel("Number of Processes (P)")
plt.ylabel("Speedup (S(p))")
plt.title("Speedup vs Number of Processes (Genetic Algorithm)")
plt.axvline(x=24, color='red', linestyle='--', label="logical processors: 24")
plt.axvline(x=16, color='blue', linestyle='--', label="cores: 16")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for matrix_size in matrix_sizes:
    matrix_speedups_pso = [
        speedups_pso[p][matrix_sizes.index(matrix_size)] for p in num_processes]
    plt.plot(num_processes, matrix_speedups_pso,
             label=f"PSO - S(p) for {matrix_size}x{matrix_size} matrix", marker='x')

plt.xlabel("Number of Processes (P)")
plt.ylabel("Speedup (S(p))")
plt.title("Speedup vs Number of Processes (PSO Algorithm)")
plt.axvline(x=24, color='red', linestyle='--', label="logical processors: 24")
plt.axvline(x=16, color='blue', linestyle='--', label="cores: 16")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for matrix_size in matrix_sizes:
    matrix_efficiencies_genetic = [
        efficiencies_genetic[p][matrix_sizes.index(matrix_size)] for p in num_processes]
    plt.plot(num_processes, matrix_efficiencies_genetic,
             label=f"Genetic - E(p) for {matrix_size}x{matrix_size} matrix", marker='o')

plt.xlabel("Number of Processes (P)")
plt.ylabel("Efficiency (E(p))")
plt.title("Efficiency vs Number of Processes (Genetic Algorithm)")
plt.axvline(x=24, color='red', linestyle='--', label="logical processors: 24")
plt.axvline(x=16, color='blue', linestyle='--', label="cores: 16")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for matrix_size in matrix_sizes:
    matrix_efficiencies_pso = [
        efficiencies_pso[p][matrix_sizes.index(matrix_size)] for p in num_processes]
    plt.plot(num_processes, matrix_efficiencies_pso,
             label=f"PSO - E(p) for {matrix_size}x{matrix_size} matrix", marker='x')

plt.xlabel("Number of Processes (P)")
plt.ylabel("Efficiency (E(p))")
plt.title("Efficiency vs Number of Processes (PSO Algorithm)")
plt.axvline(x=24, color='red', linestyle='--', label="logical processors: 24")
plt.axvline(x=16, color='blue', linestyle='--', label="cores: 16")
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.show()
