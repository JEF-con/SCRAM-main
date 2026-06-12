
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for academic papers
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6)
})

# Data 1: Varying Model Size (Clients=100)
# Size | Data Gen | Arith Share | SC Gen/Ver | Delta | Delta_ij | Key Gen | Sec Cos | Clustering
data_size = [
    [300000, 0.4574, 0.5771, 2.8604, 0.3255, 26.92, 0.0033, 2.6918, 0.0069],
    [600000, 0.6757, 0.9048, 4.5824, 0.636, 72.1714, 0.0222, 2.5671, 0.0057],
    [900000, 0.9101, 1.2341, 6.7532, 0.8262, 105.4289, 0.0527, 2.7837, 0.0057],
    [1200000, 1.2414, 1.7227, 8.962, 0.8001, 144.3034, 0.0083, 3.0374, 0.0057],
    [1500000, 1.3608, 2.0319, 10.5424, 1.0035, 175.8945, 0.0031, 2.4113, 0.0056],
    [1800000, 1.5518, 2.4147, 12.4944, 1.227, 222.9809, 0.0051, 2.503, 0.0047]
]

# Data 2: Varying Client Count (Size=1500k)
# Clients | Data Gen | Arith Share | SC Gen/Ver | Delta | Delta_ij | Key Gen | Sec Cos | Clustering
data_clients = [
    [10, 0.4138, 0.2219, 1.2006, 0.1544, 1.6516, 0.011, 0.0298, 0.0018],
    [20, 0.3961, 0.4265, 2.3778, 0.3246, 7.5422, 0.0053, 0.1135, 0.003],
    [30, 0.5439, 0.5414, 3.2447, 0.4579, 14.8584, 0.0189, 0.3555, 0.0064],
    [40, 0.6973, 0.8343, 4.3221, 0.5527, 29.7078, 0.022, 0.3897, 0.0036],
    [50, 0.7601, 0.9648, 5.2859, 0.646, 46.7436, 0.0037, 0.6043, 0.0034],
    [60, 0.8548, 1.1227, 6.3287, 0.7426, 65.9288, 0.0118, 0.9996, 0.004],
    [70, 1.0103, 1.4577, 10.8, 0.7658, 91.9761, 0.0071, 1.204, 0.0045],
    [80, 1.1377, 1.6643, 8.6113, 0.9333, 119.4993, 0.0144, 1.6107, 0.0048],
    [90, 1.3407, 1.8492, 12.3973, 1.124, 150.3783, 0.0076, 2.135, 0.0062],
    [100, 1.4362, 1.901, 10.6445, 1.0229, 181.0454, 0.0202, 2.425, 0.0057],
    [150, 2.1902, 2.8737, 24.9855, 1.8459, 405.4025, 0.0374, 5.9032, 0.0086]
]

labels = ["Data Gen", "Arith Share", "SC Gen/Ver", "Delta", "Delta_ij", "Key Gen", "Sec Cos", "Clustering"]
# Corresponding to columns 1 to 8 (index 0 is x-axis value)

def plot_data(data, x_label, title, filename):
    data = np.array(data)
    x = data[:, 0]
    y_components = data[:, 1:]
    
    # Calculate Total Time
    y_total = np.sum(y_components, axis=1)
    
    plt.figure()
    
    # Plot Total Time
    plt.plot(x, y_total, label='Total Time', color='black', linewidth=2.5, linestyle='--')
    
    # Plot Components
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
    for i in range(y_components.shape[1]):
        # Filter out components that are negligible if needed, or plot all
        # Here we plot all, but Delta_ij is dominant
        label = labels[i]
        plt.plot(x, y_components[:, i], label=label, marker=markers[i % len(markers)], linewidth=1.5)

    plt.xlabel(x_label)
    plt.ylabel('Time (seconds)')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    
    plt.savefig(filename, dpi=300)
    print(f"Generated {filename}")

# Plot 1: Varying Model Size
repo_root = Path(__file__).resolve().parents[2]
out_dir = repo_root / "results" / "benchmark" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
plot_data(data_size, "Model Parameter Size", "Computation Time vs Model Size (Clients=100)", out_dir / "benchmark_size_scaling.png")

# Plot 2: Varying Client Count
plot_data(data_clients, "Number of Clients", "Computation Time vs Client Count (Size=1500k)", out_dir / "benchmark_clients_scaling.png")
