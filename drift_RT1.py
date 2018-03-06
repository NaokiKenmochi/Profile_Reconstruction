import numpy as np
import rt1mag
import matplotlib.pyplot as plt
import scipy.optimize

class DriftRT1:
    def __init__(self):
        self.r = np.linspace(0.3, 1.0, 700)

    def load_nez0_profile(self):
        nez0 = np.load("rs_nez0_20171111.npz")
        rs = nez0["rs"]
        ne_profile_z0 = nez0["ne_profile_z0"]

        return rs, ne_profile_z0

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
        pinit = np.array([2.0, 1, 0.008])
        popt, pcov = scipy.optimize.curve_fit(self.func_T, r, Ti, p0=pinit, bounds=(0, [5., 2., 0.01]))
        fit_Ti = self.func_T(rs, *popt)
        #plt.plot(r, Ti, "o")
        #plt.plot(rs, fit_Ti)
        #plt.show()

        return fit_Ti, r_V, Vi, Vierr

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


        #plt.plot(rs, ne_profile_z0)
        #plt.plot(rs, pi)
        #plt.xlim(0.35, 1.0)
        #plt.show()

        #plt.plot(rs, pi)
        #plt.plot(rs, grad_pi)
        #plt.xlim(0.35, 1.0)
        #plt.show()

        plt.plot(rs, V_curv, label="V_curv")
        plt.plot(rs, V_gradB, label="V_gradB")
        plt.plot(rs, V_dia, label="V_dia")
        plt.plot(rs, V_curv+V_gradB+V_dia, label="V_curv + V_gradB + V_dia")
        plt.plot(r_Vi, Vi, "o", label="V_spectroscopy")
        plt.plot(r_Vi, V_ExB, "^", label="V_ExB", color="red")
        #plt.plot(self.r, bz)
        #plt.plot(self.r, gradB)
        plt.xlim(0.35, 0.8)
        plt.ylim(-5000, 1000)
        plt.legend()
        plt.ylabel("Velocity [m/s]")
        plt.xlabel("r [m]")
        plt.show()

if __name__ == "__main__":
    dr = DriftRT1()
    dr.cal_drift()
