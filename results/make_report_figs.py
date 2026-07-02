import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "report_figs")
os.makedirs(OUT, exist_ok=True)

C_RND, C_SCA, C_HRL = "#E24B4A", "#BA7517", "#1D9E75"

# ---- Fig 1: QoS violation vs CPU capacity (resource adaptivity) ----
plt.figure(figsize=(6, 4))
plt.plot([6, 7, 7.5, 8], [94.4, 85.8, 36.8, 0.0], "o-", color=C_RND, label="Random")
plt.plot([7, 7.5], [88.0, 61.7], "s--", color=C_SCA, label="SCA (classical)")
plt.plot([7, 7.5], [38.6, 0.03], "D-", color=C_HRL, linewidth=2.5, label="HRL (ours)")
plt.xlabel("CPU capacity (GHz)")
plt.ylabel("URLLC QoS violation (%)")
plt.title("QoS violation vs CPU capacity")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig1_qos_vs_cpu.png"), dpi=130)
plt.close()

# ---- Fig 2: three-way comparison at cpu=7.5 ----
methods = ["Random", "SCA", "HRL"]
colors = [C_RND, C_SCA, C_HRL]
qos = [36.8, 61.7, 0.03]
delay_h = [482, 766, 188]
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
b0 = ax[0].bar(methods, qos, color=colors)
ax[0].set_yscale("log")
ax[0].set_ylabel("URLLC QoS violation (%, log)")
ax[0].set_title("QoS violation @ cpu=7.5")
for r, v in zip(b0, qos):
    ax[0].text(r.get_x() + r.get_width() / 2, v, f"{v}", ha="center", va="bottom")
b1 = ax[1].bar(methods, delay_h, color=colors)
ax[1].set_ylabel("URLLC end-to-end delay (ms)")
ax[1].set_title("URLLC delay @ cpu=7.5")
for r, v in zip(b1, delay_h):
    ax[1].text(r.get_x() + r.get_width() / 2, v, f"{v}", ha="center", va="bottom")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig2_threeway_cpu75.png"), dpi=130)
plt.close()

# ---- Fig 3: sharp capacity cliff via load_scale (cpu=7.5) ----
load = [1.0, 1.02, 1.05, 1.08, 1.1, 1.2, 1.3, 1.45]
rqos = [36.8, 65.2, 85.3, 88.5, 91.2, 92.4, 95.2, 96.5]
plt.figure(figsize=(6, 4))
plt.plot(load, rqos, "o-", color=C_RND, label="Random")
plt.axvspan(1.0, 1.03, color=C_HRL, alpha=0.15, label="narrow learnable band")
plt.xlabel("load_scale (offered-load multiplier)")
plt.ylabel("Random QoS violation (%)")
plt.title("Sharp capacity cliff (cpu=7.5)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "fig3_load_cliff.png"), dpi=130)
plt.close()

print("figures saved to", OUT)
