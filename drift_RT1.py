from matplotlib import rc
from scipy import integrate
import numpy as np
import rt1mag
import matplotlib.pyplot as plt
import scipy.optimize

class DriftRT1:
    rc('text', usetex=True)
    def __init__(self):
        self.r = np.linspace(0.3, 1.0, 700)

    def load_nez0_profile(self, filename):
        #nez0 = np.load("rs_nez0_20171111.npz")
        #nez0 = np.load("rs_nez0_t11_20171111.npz")
        nez0 = np.load(filename)
        rs = nez0["rs"]
        ne_profile_z0 = nez0["ne_profile_z0"]

        return rs, ne_profile_z0

    def load_ne2Dz0_profile(self):
        nez0 = np.load("ne2D_rs_nez0_20180223_52_r2_v2.npz")
        rs = nez0["rs"]
        ne_profile_z0 = nez0["ne_profile_z0"]

        return rs, ne_profile_z0

    def savetxt_nez0(self):
        rs, ne_profile_z0 = self.load_ne2Dz0_profile()
        plt.plot(rs, ne_profile_z0)
        plt.show()
        np.savetxt("rs_nez0_20180223_52_r2_v2.txt", np.c_[rs, ne_profile_z0], delimiter=',')


    def load_TVHeII(self):
        TVHeII = np.load("r_THeII_VHeII_20171223_68to80.npz")
        r_buf = TVHeII["arr_sightline"]
        Ti_buf = TVHeII["T_rarr"]
        Tierr_buf = TVHeII["Terr_rarr"]
        Vi = TVHeII["V_rarr"]
        Vierr = TVHeII["Verr_rarr"]

        Ti = Ti_buf[~np.isnan(Ti_buf)]
        Tierr = Tierr_buf[~np.isnan(Ti_buf)]
        r = r_buf[~np.isnan(Ti_buf)]
        Ti = np.append(Ti, 0)
        r = np.append(r, 1.0)

        return r, Ti, Tierr, r_buf, Vi, Vierr

    def func_T(self, r, T0, a1, psix):
        #psix = psi_z0 = np.array([])
        psi_z0 = np.array([])
        psi0 = rt1mag.psi(1.0, 0.0, separatrix=True) # psi at the vacuum chamber
        for i, r_val in enumerate(r):
            #buf_psix  = rt1mag.psi(r_val, 0.0, separatrix=True) # psi上のBが最小となる直線上の密度最大値
            buf_psi_z0 = rt1mag.psi(r_val, 0.0, separatrix=True)
            #psix = np.append(psix, buf_psix)
            psi_z0 = np.append(psi_z0, buf_psi_z0)

        return T0*np.exp(-a1*((psi_z0-psix)/psi0)**2)

    def fit_T(self, rs):
        r, Ti, Tierr, r_V, Vi, Vierr = self.load_TVHeII()
        Ti[3] = (Ti[2] + Ti[4])/2
        Ti[5] = (Ti[4] + Ti[6])/2
        Ti[10] = -10
        pinit = np.array([2.0, 1, 0.008])
        popt, pcov = scipy.optimize.curve_fit(self.func_T, r, Ti, p0=pinit, bounds=(0, [5., 2., 0.01]))
        fit_Ti = self.func_T(rs, *popt)
        #plt.plot(r, Ti, "o")
        #plt.plot(rs, fit_Ti)
        #plt.show()

        return fit_Ti, r_V, Vi, Vierr

    def cal_dlnTe_over_dlnNe(self):
        plt.figure(figsize=(5,2), dpi=150)
        rs, ne_profile_z0_t15 = self.load_nez0_profile("rs_nez0_20171111.npz")
        _, ne_profile_z0_t11 = self.load_nez0_profile("rs_nez0_t11_20171111.npz")
        Ti, r_Vi, Vi, Vierr = self.fit_T(rs)
        dlnTe = np.gradient(Ti, rs[0]-rs[1])/Ti
        dlnNe_t15 = np.gradient(ne_profile_z0_t15, rs[0]-rs[1])/ne_profile_z0_t15
        dlnNe_t11 = np.gradient(ne_profile_z0_t11, rs[0]-rs[1])/ne_profile_z0_t11
        Te_t15 = -integrate.cumtrapz(5*np.gradient(Ti, rs[0]-rs[1]), rs)
        Te_t11 = -integrate.cumtrapz(3*np.gradient(Ti, rs[0]-rs[1]), rs)
        dlnTe_t15 = np.gradient(Te_t15, rs[0]-rs[1])/Te_t15
        dlnTe_t11 = np.gradient(Te_t11, rs[0]-rs[1])/Te_t11
        #plt.plot(rs, dlnNe_t11)
        #plt.plot(rs, dlnTe)
        #plt.plot(rs, 2*dlnTe/dlnNe_t11, label='t=1.1sec')
        #plt.plot(rs, 3*dlnTe/dlnNe_t15, label='t=1.5sec')
        #plt.plot(rs, dlnTe/dlnNe_t11, label='t=1.1sec')
        #plt.plot(rs, dlnTe/dlnNe_t15, label='t=1.5sec')
        #plt.plot(rs[:-1], dlnTe_t11/dlnNe_t11[:-1], label='t=1.1sec', color='blue')
        #plt.plot(rs[:-1], dlnTe_t15/dlnNe_t15[:-1], label='t=1.4sec', color='red')
        plt.plot(rs, Ti*ne_profile_z0_t11/np.max(Ti*ne_profile_z0_t11), label='t=1.1sec', color='blue')
        plt.plot(rs, Ti*ne_profile_z0_t15/np.max(Ti*ne_profile_z0_t15), label='t=1.4sec', color='red')
        #plt.plot(rs, ne_profile_z0_t11/np.max(ne_profile_z0_t11), label='t=1.1sec', color='blue')
        #plt.plot(rs, ne_profile_z0_t15/np.max(ne_profile_z0_t15), label='t=1.4sec', color='red')
        #plt.plot(rs[:-1], Te_t11, label='t=1.1sec', color='blue')
        #plt.plot(rs[:-1], Te_t15, label='t=1.4sec', color='red')
        #plt.plot(rs, 2*Ti, label='t=1.1sec')
        #plt.plot(rs, 3*Ti, label='t=1.4sec')
        #plt.plot(rs[:-1], dlnTe_t15+0.1, label='t=1.4sec')
        #plt.plot(rs[:-1], dlnTe_t11, label='t=1.1sec')
        plt.ylim(0, 1)
        plt.xlim(0.4, 1)
        #plt.hlines(2/3, xmin=0.4, xmax=1, linestyles='dotted')
        plt.legend(loc='upper right')
        #plt.ylabel(r'$\eta = d$ ln $T_e/d$ ln $n_e$')
        #plt.ylabel(r'$n_e [10^{17}m^{-3}]$')
        #plt.ylabel(r'$T_e$ [eV]')
        plt.ylabel(r'$P_e/P_{e(MAX)}$')
        plt.xlabel('R [m]')
        plt.tight_layout()
        plt.show()

    def cal_drift(self):
        z = 0.0
        Zeff = 2.0
        #Ti = 5.0
        br = bz = np.array([])
        rs, ne_profile_z0 = self.load_nez0_profile()
        ne_profile_z0 *= 0.8
        ne_profile_z0 += 0.1
        Ti, r_Vi, Vi, Vierr = self.fit_T(rs)
        pi = Ti*ne_profile_z0
        for i, r in enumerate(rs):
            buf_br, buf_bz = rt1mag.bvec(r, z, separatrix=True)
            bz = np.append(bz, buf_bz)
            br = np.append(br, buf_br)
        gradB = np.gradient(bz, rs[1]-rs[0])
        grad_pi = np.gradient(pi, rs[1]-rs[0])
        V_curv = 2*Ti/(Zeff*bz*rs)
        V_gradB = -Ti*gradB/(Zeff*bz*bz)
        V_dia = -Ti*grad_pi/(Zeff*bz*pi)
        V_curv_gradB_dia = V_curv + V_gradB + V_dia
        V_ExB = np.zeros(Vi.__len__())
        for i in range(Vi.__len__()):
            idx = np.abs(np.asarray(rs - r_Vi[i])).argmin()
            V_ExB[i] = Vi[i] - V_curv_gradB_dia[idx]



        freq_drift = np.abs(V_ExB/(2*np.pi*r_Vi))

        #plt.xkcd()
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.plot(rs, ne_profile_z0, label="$n_i$")
        plt.plot(rs, Ti, label="$T_i$")
        plt.plot(rs, pi, label="$P_i$")
        #plt.plot(rs, grad_pi)
        plt.xlim(0.35, 1.0)
        plt.ylim(0, 30)
        plt.xlabel("r [m]")
        plt.ylabel("ni, Ti, Pi")
        plt.legend()
        plt.show()

        #plt.plot(rs, -V_curv, label="$V_{curv}$")
        #plt.plot(rs, -V_gradB, label="$V_{gradB}$")
        #plt.plot(rs, -V_dia, label="$V_{dia}$")
        plt.plot(rs, -(V_curv+V_gradB+V_dia), label="$V_{curv} + V_{gradB} + V_{dia}$", linewidth=3)
        plt.plot(r_Vi, -Vi, "o", label="$V_{spectroscopy}$", color='black')
        plt.plot(r_Vi, -V_ExB, "^", label="$V_{ExB}$", color="red", markersize=10)
        #plt.plot(self.r, bz)
        #plt.plot(self.r, gradB)
        plt.xlim(0.35, 0.8)
        plt.ylim(-1000, 4000)
        plt.legend(loc='upper right', fontsize=14)
        plt.ylabel("Velocity [m/s]", fontsize=14)
        plt.xlabel("r [m]", fontsize=14)
        plt.hlines([0], 0.35, 0.8, linestyles='dashed')
        plt.tight_layout()
        plt.show()

        plt.plot(r_Vi, freq_drift, "^", color="red")
        plt.xlabel("r [m]", fontsize=14)
        plt.ylabel("Drift \nFrequency \n[Hz]", fontsize=14)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    dr = DriftRT1()
    #dr.cal_drift()
    #dr.savetxt_nez0()
    dr.cal_dlnTe_over_dlnNe()
