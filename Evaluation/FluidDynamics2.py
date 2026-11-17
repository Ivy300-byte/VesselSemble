import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

# ==========================================
# 1. KAGGLE INPUT / OUTPUT PATHS ONLY
# ==========================================
CSV_PATH = "/kaggle/input/datasets/ifeeumeh/csv-reports/nnUNet-on-3Dircadb.csv"
OUTPUT_DIR = "/kaggle/working/tda_self_calibrated_cfd"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# 2. DERIVE BIOPHYSICAL PARAMETERS FROM TDA
# ==========================================
def fahraeus_lindqvist_viscosity(
    diameter_mm: float, hematocrit: float = 0.45
) -> float:
  """Computes exact apparent blood viscosity (Pa*s) from vessel diameter (mm)

  using the Pries et al. (1992) in-vitro/in-vivo Fåhræus-Lindqvist law.
  """
  D_um = max(diameter_mm * 1000.0, 4.0)  # Convert mm to micrometers
  mu_plasma = 0.0012  # Plasma viscosity at 37C (Pa*s)

  # Relative viscosity at H_d = 0.45 (Pries empirical formulation)
  mu_45 = (
      220.0 * np.exp(-1.3 * D_um)
      + 3.2
      - 2.44 * np.exp(-0.06 * (D_um**0.645))
  )

  # Adjust for hematocrit H_d
  C = (0.8 + np.exp(-0.075 * D_um)) * (
      -1.0 + 1.0 / (1.0 + 1e-11 * (D_um**12))
  ) + 1.0 / (1.0 + 1e-11 * (D_um**12))
  mu_rel = 1.0 + (mu_45 - 1.0) * (
      ((1.0 - hematocrit) ** C - 1.0) / ((1.0 - 0.45) ** C - 1.0)
  )
  return float(mu_plasma * mu_rel)


def derive_biophysical_parameters(row: pd.Series, hematocrit: float = 0.45):
  """Infers physical inlet diameter, physiological inlet Reynolds number,

  fluid density, and root viscosity directly from TDA skeleton metrics.
  """
  bifurcations = max(int(row["bifurcation_count"]), 1)
  avg_branch_mm = float(row["average_branch_length_mm"])
  tortuosity = max(float(row["mean_tortuosity"]), 1.0)

  # 1. Generational depth N from TDA bifurcation count: N = log2(B + 1)
  gen_depth = np.log2(bifurcations + 1.0)

  # 2. Infer mean network diameter from anatomical segment aspect ratio (L / D ~ 12)
  chord_len_mm = avg_branch_mm / tortuosity
  aspect_ratio = 12.0  # Conserved vascular segment length-to-diameter ratio
  mean_network_diam_mm = chord_len_mm / aspect_ratio

  # 3. Back-calculate Root Inlet Diameter D_0 via Murray's Law (D_mean = D_0 * 2^(-N / 6))
  inlet_diam_mm = mean_network_diam_mm * (2.0 ** (gen_depth / 6.0))

  # 4. Derive physiological inlet Reynolds number from allometric arterial flow scaling
  physiological_inlet_re = 65.0 * (inlet_diam_mm**1.55)

  # 5. Couple blood density and root viscosity to hematocrit & inferred D_0
  rho_plasma, rho_rbc = 1025.0, 1100.0
  rho = rho_plasma * (1.0 - hematocrit) + rho_rbc * hematocrit
  mu_inlet = fahraeus_lindqvist_viscosity(inlet_diam_mm, hematocrit)

  return {
      "inlet_diam_mm": inlet_diam_mm,
      "physiological_inlet_re": physiological_inlet_re,
      "rho": rho,
      "mu_inlet": mu_inlet,
      "gen_depth": gen_depth,
  }


# ==========================================
# 3. BUILD & SOLVE TDA-CALIBRATED GRAPH
# ==========================================
def solve_self_calibrated_tda_graph(
    row: pd.Series, save_dir: str, plot_graph: bool = True
):
  case_id = str(row["case_id"]).strip()
  n_bifurcations = int(row["bifurcation_count"])
  n_loops = int(row["false_loops_h1"])
  total_len_mm = float(row["total_vessel_length_mm"])
  avg_branch_m = float(row["average_branch_length_mm"]) * 1e-3
  tortuosity = max(float(row["mean_tortuosity"]), 1.0001)

  # Derive all physical parameters strictly from this row's TDA metrics
  phys = derive_biophysical_parameters(row)
  d0_mm = phys["inlet_diam_mm"]
  target_re0 = phys["physiological_inlet_re"]
  rho = phys["rho"]
  mu0 = phys["mu_inlet"]

  n_tree_edges = 2 * n_bifurcations + 1
  scaled_branch_len_mm = total_len_mm / n_tree_edges
  L_m = scaled_branch_len_mm * 1e-3

  G = nx.Graph()
  G.add_node(0, depth=0, pos=(0.0, 0.0))
  G.add_node(1, depth=1, pos=(0.0, -scaled_branch_len_mm))

  r0_m = (d0_mm * 0.5) * 1e-3
  g0 = (np.pi * (r0_m**4)) / (8.0 * mu0 * L_m)
  G.add_edge(0, 1, gen=0, diam_m=2 * r0_m, mu=mu0, g=g0)

  # Fixed: Initialize active_tips with node 1 and start bifurcations_created at 0
  active_tips = [1]
  next_node_id = 2
  bifurcations_created = 0

  while bifurcations_created < n_bifurcations and active_tips:
    parent = active_tips.pop(0)
    gen = G.nodes[parent]["depth"]
    px, py = G.nodes[parent]["pos"]

    # Murray's Law diameter + diameter-specific Fåhræus-Lindqvist viscosity
    d_gen_mm = d0_mm * (2.0 ** (-gen / 3.0))
    mu_gen = fahraeus_lindqvist_viscosity(d_gen_mm)
    r_m = (d_gen_mm * 0.5) * 1e-3
    g_edge = (np.pi * (r_m**4)) / (8.0 * mu_gen * L_m)

    spread_x = (total_len_mm * 0.15) / (1.5**gen)
    for side in [-1, 1]:
      child = next_node_id
      next_node_id += 1
      G.add_node(
          child,
          depth=gen + 1,
          pos=(px + side * spread_x, py - scaled_branch_len_mm),
      )
      G.add_edge(parent, child, gen=gen, diam_m=2 * r_m, mu=mu_gen, g=g_edge)
      active_tips.append(child)

    bifurcations_created += 1

  # Add H1 collateral loops
  nodes_by_depth = {}
  for n, d in G.nodes(data=True):
    if n > 1:
      nodes_by_depth.setdefault(d["depth"], []).append(n)

  loops_added = 0
  for depth in sorted(nodes_by_depth.keys(), reverse=True):
    lvl = nodes_by_depth[depth]
    for i in range(len(lvl) - 1):
      if loops_added >= n_loops:
        break
      u, v = lvl[i], lvl[i + 1]
      if not G.has_edge(u, v):
        d_loop_mm = d0_mm * (2.0 ** (-depth / 3.0))
        mu_loop = fahraeus_lindqvist_viscosity(d_loop_mm)
        r_m = (d_loop_mm * 0.5) * 1e-3
        g_loop = (np.pi * (r_m**4)) / (8.0 * mu_loop * L_m)
        G.add_edge(u, v, gen=depth, diam_m=2 * r_m, mu=mu_loop, g=g_loop)
        loops_added += 1
    if loops_added >= n_loops:
      break

  # Solve Kirchhoff Matrix
  inlet_node = 0
  outlet_nodes = [n for n, deg in G.degree() if deg == 1 and n != inlet_node]
  nodes = list(G.nodes())
  node_idx = {n: i for i, n in enumerate(nodes)}

  L_mat = np.zeros((len(nodes), len(nodes)))
  rhs = np.zeros(len(nodes))

  for u, v, data in G.edges(data=True):
    i, j = node_idx[u], node_idx[v]
    g = data["g"]
    L_mat[i, i] += g
    L_mat[j, j] += g
    L_mat[i, j] -= g
    L_mat[j, i] -= g

  for b_node in [inlet_node] + outlet_nodes:
    idx = node_idx[b_node]
    L_mat[idx, :] = 0.0
    L_mat[idx, idx] = 1.0
    rhs[idx] = 1.0 if b_node == inlet_node else 0.0

  unit_p = np.linalg.solve(L_mat, rhs)

  inlet_d_m = d0_mm * 1e-3
  target_q = ((target_re0 * mu0) / (rho * inlet_d_m)) * (
      np.pi * (inlet_d_m / 2.0) ** 2
  )
  unit_q = sum(
      d["g"] * (unit_p[node_idx[u]] - unit_p[node_idx[v]])
      for u, v, d in G.edges(inlet_node, data=True)
  )
  pressures = unit_p * (target_q / unit_q)

  rc_m = avg_branch_m / (2.0 * np.sqrt(max(1e-6, tortuosity - 1.0)))
  re_list, dean_list = [], []

  for u, v, data in G.edges(data=True):
    dp = abs(pressures[node_idx[u]] - pressures[node_idx[v]])
    q = data["g"] * dp
    d_m = data["diam_m"]
    mu_e = data["mu"]
    vel = q / (np.pi * (d_m / 2.0) ** 2)
    re_val = (rho * vel * d_m) / mu_e
    dean_val = re_val * np.sqrt(d_m / (2.0 * rc_m))
    data["reynolds"] = re_val
    re_list.append(re_val)
    dean_list.append(dean_val)

  if plot_graph:
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.get_node_attributes(G, "pos")
    edges = list(G.edges(data=True))
    edge_re = [d["reynolds"] for _, _, d in edges]

    nx.draw_networkx_nodes(G, pos, node_size=18, node_color="#222222", ax=ax)
    nx.draw_networkx_nodes(
        G, pos, nodelist=[inlet_node], node_size=90, node_color="cyan", ax=ax
    )
    ec = nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for u, v, _ in edges],
        edge_color=edge_re,
        edge_cmap=plt.cm.plasma,
        width=2.0,
        ax=ax,
    )
    plt.colorbar(ec, ax=ax, label="Local Reynolds Number (Re)")
    ax.set_title(
        f"{case_id} — TDA-Calibrated Hydraulic Graph\n"
        f"Inferred D0: {d0_mm:.2f} mm | Inferred Re0: {target_re0:.1f}"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"{case_id}_calibrated_reynolds.png"), dpi=200
    )
    plt.close(fig)

  return {
      "case_id": case_id,
      "Inferred_Inlet_Diam_mm": round(d0_mm, 3),
      "Inferred_Inlet_Re": round(target_re0, 2),
      "Inferred_Viscosity_mPas": round(mu0 * 1000.0, 3),
      "Mean_Branch_Re": round(float(np.mean(re_list)), 2),
      "Terminal_Outlet_Re": round(float(np.min(re_list)), 2),
      "Max_Dean_Number": round(float(np.max(dean_list)), 2),
      "Required_Inlet_Pressure_Pa": round(
          float(pressures[node_idx[inlet_node]]), 2
      ),
  }


# ==========================================
# 4. RUN ACROSS YOUR CSV
# ==========================================
df = pd.read_csv(CSV_PATH)
results = [
    solve_self_calibrated_tda_graph(row, OUTPUT_DIR)
    for _, row in df.iterrows()
    if str(row["status"]).strip() == "Success"
]

out_df = pd.merge(df, pd.DataFrame(results), on="case_id", how="left")
out_df.to_csv(
    os.path.join(OUTPUT_DIR, "tda_self_calibrated_reynolds.csv"), index=False
)

print(
    out_df[[
        "case_id",
        "Inferred_Inlet_Diam_mm",
        "Inferred_Inlet_Re",
        "Mean_Branch_Re",
        "Max_Dean_Number",
        "Required_Inlet_Pressure_Pa",
    ]].to_string(index=False)
)
