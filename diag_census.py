import numpy as np
import glob

d = 'results/dilute_spectrum_shell/imc_32g_standard'
a = 1.459e-2

def load_snap(t_str):
    f = '{}/snapshot_t_{}ns.npz'.format(d, t_str)
    return np.load(f)

snap93 = load_snap('0.93000')
snap97 = load_snap('0.97000')

r = snap93['r_centers']
edges = snap93['r_edges']
vols = (4.0/3.0)*np.pi*(edges[1:]**3 - edges[:-1]**3)

Er93 = snap93['E_rad']
Er97 = snap97['E_rad']
Tr_si93 = snap93['T_rad']
Tr_si97 = snap97['T_rad']
Tr_cen93 = (Er93 / a)**0.25
Tr_cen97 = (Er97 / a)**0.25

hdr = "{:>7}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}".format(
    "r", "Tr_si93", "Tr_cen93", "Tr_si97", "Tr_cen97", "Ecen_rat")
print(hdr)
print("-" * len(hdr))
for i in range(20):
    e93 = Er93[i] * vols[i]
    e97 = Er97[i] * vols[i]
    rat = e97 / e93 if e93 > 0 else float('nan')
    print("{:7.3f}  {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}  {:10.4f}".format(
        r[i], Tr_si93[i], Tr_cen93[i], Tr_si97[i], Tr_cen97[i], rat))

print()
print("Total E_cen_cell0-5 at t=0.93: {:.4e}".format(np.sum(Er93[:6]*vols[:6])))
print("Total E_cen_cell0-5 at t=0.97: {:.4e}".format(np.sum(Er97[:6]*vols[:6])))
print("Total E_cen r>10 at t=0.93: {:.4e}".format(np.sum(Er93[r>10]*vols[r>10])))
print("Total E_cen r>10 at t=0.97: {:.4e}".format(np.sum(Er97[r>10]*vols[r>10])))
