import csv
import matplotlib.pyplot as plt
from math import floor

# =========================
# PARAMETERS
# =========================
N = 6
TOTAL_VARS = N * N * 4

SOLUTION_FILE = "solution_labeled.csv"
COORD_FILE = "Coordinates.csv"


# =========================
# MAPPING FUNCTION
# =========================
def yToX(t,n):
    i = floor(t/n) % n + 1
    j = t % n + 1
    k = floor(t/(n**2)) + 1
    return (i,j,k)

def xToY(i,j,k,n):
    return (i-1)*n + (j-1) + (k-1)*(n**2)

# =========================
# LOAD SOLUTION VECTOR
# =========================
def load_sample_from_csv(filename, n_vars):
    sample = {}

    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            idx = int(row[0])
            val = int(float(row[1]))  # handles "1.0"
            sample[idx] = val
    x_vec = [sample.get(i, 0) for i in range(n_vars)]
    return x_vec


# =========================
# LOAD COORDINATES
# =========================
def load_coordinates(filename):
    coords = {}

    with open(filename, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            idx = int(row[0])
            x = float(row[1])
            y = float(row[2])
            coords[idx] = (x, y)
    coords_subset = dict(list(coords.items())[:N])
    # print(coords_subset)
    return coords_subset


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    # --- load data ---
    x_vec = load_sample_from_csv(SOLUTION_FILE, TOTAL_VARS)
    coords = load_coordinates(COORD_FILE)

    print("Vector size:", len(x_vec))
    sum=0
    for i in range (len(x_vec)):
        if x_vec[i]!=0:
            sum+=(yToX(i,N)[2])
            print(f"Cable from {yToX(i,N)[0]} to {yToX(i,N)[1]} with width {yToX(i,N)[2]}")
    print("Sum of x:", sum)
    print("Turbines loaded:", len(coords))


    # =========================
    # EXTRACT ACTIVE EDGES
    # =========================
    edges = []

    for t in range(TOTAL_VARS):
        if x_vec[t] == 1:
            i, j, k = yToX(t, N)
            if i != j:
                edges.append((i, j, k))
                print(i,j,k)


    print("Active edges:", len(edges))


    # =========================
    # COLOR MAP FOR CABLE TYPES
    # =========================
    color_map = {
        1: "blue",
        2: "green",
        3: "orange",
        4: "red"
    }


    # =========================
    # PLOT
    # =========================
    plt.figure(figsize=(10, 8))

    # --- plot turbines ---
    xs = [coords[i][0] for i in coords]
    ys = [coords[i][1] for i in coords]

    plt.scatter(xs, ys)

    # label turbines (optional)
    for i in coords:
        plt.text(coords[i][0], coords[i][1], str(i), fontsize=8)


    # --- plot directed edges ---
    for (i, j, k) in edges:
        if i in coords and j in coords:
            x1, y1 = coords[i]
            x2, y2 = coords[j]

            dx = x2 - x1
            dy = y2 - y1

            plt.arrow(
                x1, y1,
                dx, dy,
                length_includes_head=True,
                head_width=5,      # adjust if needed
                alpha=0.7,
                color=color_map.get(k, "black")
            )


    # =========================
    # FINALIZE
    # =========================
    plt.title("Wind Farm Cable Network (Directed i → j)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid()

    # legend (manual)
    for k, color in color_map.items():
        plt.plot([], [], color=color, label=f"k={k}")
    plt.legend()

    # plt.show()