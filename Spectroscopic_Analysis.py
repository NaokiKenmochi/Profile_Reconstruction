"""分光器で計測したスペクトルを元にイオン温度，フロー速度を求めるモジュールです
"""
from scipy.optimize import curve_fit
from scipy import integrate
from Abel_ne import Abel_ne

__author__  = "Naoki Kenmochi <kenmochi@edu.k.u-tokyo.ac.jp>"
__version__ = "0.1.0"
__date__    = "6 Feb 2018"

import numpy as np
import matplotlib.pyplot as plt
import abel


class SpectAnal(Abel_ne):

    def __init__(self):
        self.file_path = "/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/Spectroscopy/d20161206sp/spectr_161206_18to27.txt"
        self.instwid = 0.025129#0.024485
        self.sightline_spect = 1e-3*np.array([385, 422, 475, 526, 576, 623, 667, 709, 785])


    def load_spec_data(self):
        data_org = np.loadtxt(self.file_path, delimiter='\t', skiprows=1)
        data = np.zeros((1024, 9))
        d_order = np.array([0, 4, 1, 5, 2, 7, 3, 8, 9])
        for i in range(9):
            data[:, i] = data_org[:, d_order[i]]
        wavelength = np.linspace(462.268, 474.769, 1024)
        return data, wavelength


    def gauss(self, x, k0, k1, k2, k3):
        """
        gauss function (see Igor Pro)
        :param x:
        :param k0:
        :param k1:
        :param k2:
        :param k3:
        :return:
        """
        return k0 + k1*np.exp(-((x-k2)/k3)**2)

    def accurategauss(self, x, k0, k1, k2, k3, k4):

        return k0 + k1*np.exp(-((x-k2-k3)/k4)**2)

    def HefitverS_const_wl(self, x, k0, k1, k3, k4):
        const_values = np.array([10,40,468.538,0,0.01,468.541,468.557,468.57,468.576,468.58,468.583,468.591])

        return k0+k1*np.exp(-((x-const_values[2]-k3)/k4)**2)+k1*0.522681*np.exp(-((x-const_values[5]-k3)/k4)**2)+ \
               k1*0.261347*np.exp(-((x-const_values[6]-k3)/k4)**2)+k1*5.09218*np.exp(-((x-const_values[7]-k3)/k4)**2)+ \
               k1*0.205956*np.exp(-((x-const_values[8]-k3)/k4)**2)+k1*4.70328*np.exp(-((x-const_values[9]-k3)/k4)**2)+ \
               k1*0.235159*np.exp(-((x-const_values[10]-k3)/k4)**2)+k1*0.104284*np.exp(-((x-const_values[11]-k3)/k4)**2)


    def HefitverS(self, x, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11):

        return k0+k1*np.exp(-((x-k2-k3)/k4)**2)+k1*0.522681*np.exp(-((x-k5-k3)/k4)**2)+ \
               k1*0.261347*np.exp(-((x-k6-k3)/k4)**2)+k1*5.09218*np.exp(-((x-k7-k3)/k4)**2)+ \
               k1*0.205956*np.exp(-((x-k8-k3)/k4)**2)+k1*4.70328*np.exp(-((x-k9-k3)/k4)**2)+ \
               k1*0.235159*np.exp(-((x-k10-k3)/k4)**2)+k1*0.104284*np.exp(-((x-k11-k3)/k4)**2)

    def exp(self, x, k0, k1, k2):
        return k0 + k1*np.exp(-k2*x)

    def gauss_fitting_CIII(self, isAbel=False, spline=False, convolve=False):
        wavelength, sightline_spect, data, spect_local = super().abelic_spectroscopy(spline=spline, convolve=convolve)

        CT_arr = np.array([])
        CTerr_arr = np.array([])
        CV_arr = np.array([])
        CVerr_arr = np.array([])
        Cint_arr = np.array([])
        bounds_st = 464.6
        bounds_ed = 464.83
        init_values = np.array([10, 500, 464.742, 0.01])

        if(isAbel==True):
            data = spect_local
            label = 'Local'
        else:
            label = 'Line-Integrated'

        plt.figure(figsize=(12, 8))
        plt.subplot(221)
        for (num_pos, x) in enumerate(sightline_spect):
            popt_0, pcov_0 = curve_fit(self.gauss, wavelength, data[:, num_pos], p0=init_values)

            plt.plot(wavelength, data[:, num_pos], label='r=%5.3f' % sightline_spect[num_pos])
            #In IgorPro #ToDo Igorとの違いを確認
            #$("W_" + chstr)	 ={$wr[pcsrA], wavemax($wr, xcsrA, xcsrB) - $wr[pcsrA], 464.742, W_coef[2] - 464.742, W_coef[3]}
            init_values_2 = np.array([popt_0[0], popt_0[1], popt_0[2]-464.742, popt_0[3]])
            popt, pcov = curve_fit(lambda wavelength, k0, k1, k3, k4: self.accurategauss(wavelength, k0, k1, 464.742, k3, k4),
                                   wavelength, data[:, num_pos], p0=init_values_2)
            popt = np.insert(popt, 2, init_values[2])

            sigma = np.sqrt(np.diag(pcov))
            #TODO Errorの値が大きく出ている？　要確認
            CT = 469000000*12*(np.sqrt(popt[4]**2 - self.instwid**2)/464.742)**2
            CTerr = 469000000*12*(np.sqrt((np.abs(popt[4]) + np.abs(sigma[3]))**2 - self.instwid**2)/464.742)**2 - CT
            Cdx = popt[3]
            CV = 299800000*Cdx/464.742
            CVerr = 299800000*sigma[2]/464.742
            Cint = integrate.quad(self.accurategauss, bounds_st, bounds_ed,
                                 args=(popt[0], popt[1], popt[2], popt[3], popt[4]))[0] - popt[0]*(bounds_ed-bounds_st)

            CT_arr = np.append(CT_arr, CT)
            CTerr_arr = np.append(CTerr_arr, CTerr)
            CV_arr = np.append(CV_arr, CV)
            CVerr_arr = np.append(CVerr_arr, CVerr)
            Cint_arr = np.append(Cint_arr, Cint)

            print('========= r = %5.3f m =========' % sightline_spect[num_pos])
            print('T_C = %5.3f ± %5.3f eV' % (CT, CTerr))
            print('V_C = %5.3f ± %5.3f m/s' % (CV, CVerr))
            print('C_int = %5.3f' % Cint)

            plt.plot(wavelength, self.accurategauss(wavelength, popt_0[0], popt_0[1], 464.742, popt_0[2]-464.742, popt_0[3]), '-o',
                     label='accurategauss fitting(r=%5.3fm' % sightline_spect[num_pos])
            #plt.plot(wavelength, self.gauss(wavelength, *popt_0), '-o', label='gauss fitting(r=%5.3fm' % sightline_spect[num_pos])
            #plt.ylim(0, 1e5)

        plt.legend()
        plt.xlim(464.4, 465.2)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Intensity [a.u.]')

        plt.subplot(222)
        plt.plot(sightline_spect, Cint_arr, '-o', color='green', label=label)
        plt.legend()
        #plt.title(label + ', spectr_161206_18to27', loc='right', fontsize=20)
        plt.title(label + ', 161111#36-44', loc='right', fontsize=20)
        plt.xlabel('r [m]')
        plt.ylabel('Intensity of CIII [a.u.]')

        plt.subplot(223)
        plt.errorbar(sightline_spect, CT_arr, yerr=CTerr_arr, fmt='ro')
        plt.xlabel('r [m]')
        plt.ylabel('$T_{CIII}$ [eV]')
        plt.ylim(0, 50)

        plt.subplot(224)
        plt.errorbar(sightline_spect, CV_arr, yerr=CVerr_arr, fmt='bo')
        plt.xlabel('r [m]')
        plt.ylabel('$V_{CIII}$ [m/s]')
        plt.tight_layout()
        plt.show()

    def gauss_fitting_HeI(self, isAbel=False, spline=False, convolve=False):
        wavelength, sightline_spect, data, spect_local = super().abelic_spectroscopy(spline=spline, convolve=convolve)
        init_values = np.array([10, 500, 471.31457, 0.01])
        HeIT_arr = np.array([])
        HeITerr_arr = np.array([])
        HeIint_arr = np.array([])
        bounds_st = 471.2
        bounds_ed = 471.5

        if(isAbel==True):
            data = spect_local
            label = 'Local'
        else:
            label = 'Line-Integrated'

        plt.figure(figsize=(12, 8))
        plt.subplot(221)

        for (num_pos, x) in enumerate(sightline_spect):
            popt, pcov = curve_fit(self.gauss, wavelength, data[:, num_pos], p0=init_values)

            plt.plot(wavelength, data[:, num_pos], label='r=%5.3f' % sightline_spect[num_pos])
            sigma = np.sqrt(np.diag(pcov))
            #TODO Errorの値が大きく出ている？　要確認
            HeIT = 469000000*4*(np.sqrt(popt[3]**2 - self.instwid**2)/471.31457)**2
            HeITerr = 469000000*4*(np.sqrt((np.abs(popt[3]) + np.abs(sigma[3]))**2 - self.instwid**2)/471.31457)**2 - HeIT
            HeIdx = popt[2]+sigma[2]-471.31457
            HeIint = integrate.quad(self.gauss, bounds_st, bounds_ed,
                                  args=(popt[0], popt[1], popt[2], popt[3]))[0] - popt[0]*(bounds_ed-bounds_st)

            HeIT_arr = np.append(HeIT_arr, HeIT)
            HeITerr_arr = np.append(HeITerr_arr, HeITerr)
            HeIint_arr = np.append(HeIint_arr, HeIint)

            print('========= r = %5.3f m =========' % sightline_spect[num_pos])
            print('T_HeI = %5.3f ± %5.3f eV' % (HeIT, HeITerr))
            print('HeI_int = %5.3f' % HeIint)

            plt.plot(wavelength, self.gauss(wavelength, *popt), '-o',
                     label='gauss fitting(r=%5.3fm' % sightline_spect[num_pos])
            #plt.plot(wavelength, self.gauss(wavelength, *popt_0), '-o', label='gauss fitting(r=%5.3fm' % sightline_spect[num_pos])
            #plt.ylim(0, 1e5)

        plt.legend()
        plt.xlim(471.1342, 471.49935)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Intensity [a.u.]')

        plt.subplot(222)
        plt.plot(sightline_spect, HeIint_arr, '-o', color='green', label=label)
        plt.legend()
        #plt.title(label + ', spectr_161206_18to27', loc='right', fontsize=20)
        plt.title(label + ', 161111#36-44', loc='right', fontsize=20)
        plt.xlabel('r [m]')
        plt.ylabel('Intensity of HeI [a.u.]')

        plt.subplot(223)
        plt.errorbar(sightline_spect, HeIT_arr, yerr=HeITerr_arr, fmt='ro')
        plt.xlabel('r [m]')
        plt.ylabel('$T_{HeII}$ [eV]')
        plt.ylim(0, 5)
        plt.tight_layout()
        plt.show()

    def gauss_fitting_HeII(self, isAbel=False, spline=False, convolve=False):
        #data, wavelength = self.load_spec_data()    #TODO アーベル変換したスペクトルに対して解析する
        wavelength, sightline_spect, data, spect_local = super().abelic_spectroscopy(spline=spline, convolve=convolve)

        #popt, pcov = curve_fit(self.gauss, wavelength, data[:, 3], bounds=([0, 0, 471.20, 0], [np.inf, np.inf, 471.40, 1.0]))
        #popt, pcov = curve_fit(self.gauss, wavelength, data[:, 3], bounds=([0, 0, 468.45, 0], [np.inf, np.inf, 468.65, 1.0]))
        bounds_st = np.zeros(12)
        bounds_ed = np.ones(12)
        bounds_ed *= np.inf
        bounds_st[3] = 468.45
        bounds_ed[3] = 468.65
        bounds_ed[4] = 1.0
        init_values = np.array([10, 40, 0, 0.01])
        HeT_arr = np.array([])
        HeTerr_arr = np.array([])
        HeV_arr = np.array([])
        HeVerr_arr = np.array([])
        Heint_arr = np.array([])
        Hedx_arr = np.array([])

        if(isAbel==True):
            data = spect_local
            label = 'Local'
        else:
            label = 'Line-Integrated'

        plt.figure(figsize=(12, 8))
        plt.subplot(221)

        for (num_pos, x) in enumerate(sightline_spect):
            popt, pcov = curve_fit(self.HefitverS_const_wl, wavelength, data[:, num_pos], p0=init_values)
            sigma = np.sqrt(np.diag(pcov))
            HeT = 469000000*4*(np.sqrt(popt[3]**2 - self.instwid**2)/468.565)**2
            HeTerr = 469000000*4*(np.sqrt((np.abs(popt[3]) + np.abs(sigma[3]))**2 - self.instwid**2)/468.565)**2 - HeT
            Hedx = popt[2]
            HeV = 299800000*Hedx/468.565
            HeVerr = 299800000*sigma[2]/468.565
            Heint = integrate.quad(self.HefitverS_const_wl, bounds_st[3], bounds_ed[3],
                                   args=(popt[0], popt[1], popt[2], popt[3]))[0] - popt[0]*(bounds_ed[3]-bounds_st[3])

            #HeV = self.modify_center_for_V(Hedx, -0.007, "He")
            HeT_arr = np.append(HeT_arr, HeT)
            HeTerr_arr = np.append(HeTerr_arr, HeTerr)
            HeV_arr = np.append(HeV_arr, HeV)
            HeVerr_arr = np.append(HeVerr_arr, HeVerr)
            Heint_arr = np.append(Heint_arr, Heint)
            Hedx_arr = np.append(Hedx_arr, Hedx)

            print('========= r = %5.3f m =========' % sightline_spect[num_pos])
            print('T_He = %5.3f ± %5.3f eV' % (HeT, HeTerr))
            print('V_He = %5.3f ± %5.3f m/s' % (HeV, HeVerr))
            print('He_int = %5.3f' % Heint)
            print('Hdx = %5.3f' % Hedx)

            plt.plot(wavelength, data[:, num_pos], label='r=%5.3f' % sightline_spect[num_pos])
            #plt.plot(wavelength[710:770], data[710:770, 3], label='Line-integrated')
            plt.plot(wavelength, self.HefitverS_const_wl(wavelength, *popt), '-o', label='gauss fitting(r=%5.3fm' % sightline_spect[num_pos])
        print('Center of Hedx: %5.3f' % np.average(Hedx_arr))
        for (num_pos, x) in enumerate(sightline_spect):
            HeV_arr[num_pos] = self.modify_center_for_V(Hedx_arr[num_pos], np.average(Hedx_arr), CorHe="C") #TODO   dxcenterの与え方　要注意
            print('========= r = %5.3f m =========' % sightline_spect[num_pos])
            print('V_He = %5.3f ± %5.3f m/s' % (HeV_arr[num_pos], HeVerr_arr[num_pos]))
        plt.legend()
        #plt.title('$k0+k1*exp\{-((x-k2)/k3)^2\}$\nk0=%5.3f±%5.3f, \nk1=%5.3f±%5.3f, \nk2=%5.3f±%5.3f, \nk3=%5.3f±%5.3f'
        #plt.xlim(471.2, 471.4)
        plt.xlim(468.2, 469.0)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Intensity [a.u.]')

        plt.subplot(222)
        plt.plot(sightline_spect, Heint_arr, '-o', color='green', label=label)
        plt.legend()
        #plt.title(label + ', spectr_161206_18to27', loc='right', fontsize=20)
        plt.title(label + ', 161111#36-44', loc='right', fontsize=20)
        plt.xlabel('r [m]')
        plt.ylabel('Intensity of HeII [a.u.]')

        plt.subplot(223)
        plt.errorbar(sightline_spect, HeT_arr, yerr=HeTerr_arr, fmt='ro')
        plt.xlabel('r [m]')
        plt.ylabel('$T_{HeII}$ [eV]')
        plt.ylim(0, 20)

        plt.subplot(224)
        plt.errorbar(sightline_spect, HeV_arr, yerr=HeVerr_arr, fmt='bo')
        plt.xlabel('r [m]')
        plt.ylabel('$V_{HeII}$ [m/s]')
        plt.tight_layout()
        plt.show()

    def gauss_fitting(self, Species, isAbel=False, spline=False, convolve=False):   #TODO   現状ではHeIIの処理のみ．　HeIとC（Ar）の処理も追加する
        wavelength, sightline_spect, data, spect_local = super().abelic_spectroscopy(spline=spline, convolve=convolve)

        bounds_st = np.array([471.2, 468.45, 464.6])    #[HeI, HeII, CIII]
        bounds_ed = np.array([471.5, 468.65, 464.83])    #[HeI, HeII, CIII]
        T_arr = np.array([])
        Terr_arr = np.array([])
        V_arr = np.array([])
        Verr_arr = np.array([])
        int_arr = np.array([])
        dx_arr = np.array([])

        if(isAbel==True):
            data = spect_local
            label = 'Local'
        else:
            label = 'Line-Integrated'

        plt.figure(figsize=(12, 8))
        plt.subplot(221)

        for (num_pos, x) in enumerate(sightline_spect):
            if(Species=="HeI"):
                init_values = np.array([10, 500, 471.31457, 0.01])
                popt, pcov = curve_fit(self.gauss, wavelength, data[:, num_pos], p0=init_values)
                plt.plot(wavelength, data[:, num_pos], label='r=%5.3f' % sightline_spect[num_pos])
                sigma = np.sqrt(np.diag(pcov))
                #TODO Errorの値が大きく出ている？　要確認
                T = 469000000*4*(np.sqrt(popt[3]**2 - self.instwid**2)/471.31457)**2
                Terr = 469000000*4*(np.sqrt((np.abs(popt[3]) + np.abs(sigma[3]))**2 - self.instwid**2)/471.31457)**2 - T
                dx = popt[2]+sigma[2]-471.31457
                int = integrate.quad(self.gauss, bounds_st[0], bounds_ed[0],
                                     args=(popt[0], popt[1], popt[2], popt[3]))[0] - popt[0]*(bounds_ed[0]-bounds_st[0])
                V = 0.0
                Verr = 0.0

            elif(Species=="HeII"):
                init_values = np.array([10, 40, 0, 0.01])
                popt, pcov = curve_fit(self.HefitverS_const_wl, wavelength, data[:, num_pos], p0=init_values)
                sigma = np.sqrt(np.diag(pcov))
                T = 469000000*4*(np.sqrt(popt[3]**2 - self.instwid**2)/468.565)**2
                Terr = 469000000*4*(np.sqrt((np.abs(popt[3]) + np.abs(sigma[3]))**2 - self.instwid**2)/468.565)**2 - T
                dx = popt[2]
                V = 299800000*dx/468.565
                Verr = 299800000*sigma[2]/468.565
                int = integrate.quad(self.HefitverS_const_wl, bounds_st[1], bounds_ed[1],
                                       args=(popt[0], popt[1], popt[2], popt[3]))[0] - popt[0]*(bounds_ed[1]-bounds_st[1])

            elif(Species=="CIII"):
                init_values = np.array([10, 500, 464.742, 0.01])
                popt_0, pcov_0 = curve_fit(self.gauss, wavelength, data[:, num_pos], p0=init_values)

                plt.plot(wavelength, data[:, num_pos], label='r=%5.3f' % sightline_spect[num_pos])
                #In IgorPro #ToDo Igorとの違いを確認
                #$("W_" + chstr)	 ={$wr[pcsrA], wavemax($wr, xcsrA, xcsrB) - $wr[pcsrA], 464.742, W_coef[2] - 464.742, W_coef[3]}
                init_values_2 = np.array([popt_0[0], popt_0[1], popt_0[2]-464.742, popt_0[3]])
                popt, pcov = curve_fit(lambda wavelength, k0, k1, k3, k4: self.accurategauss(wavelength, k0, k1, 464.742, k3, k4),
                                       wavelength, data[:, num_pos], p0=init_values_2)
                popt = np.insert(popt, 2, init_values[2])
                sigma = np.sqrt(np.diag(pcov))
                #TODO Errorの値が大きく出ている？　要確認
                T = 469000000*12*(np.sqrt(popt[4]**2 - self.instwid**2)/464.742)**2
                Terr = 469000000*12*(np.sqrt((np.abs(popt[4]) + np.abs(sigma[3]))**2 - self.instwid**2)/464.742)**2 - T
                dx = popt[3]
                V = 299800000*dx/464.742
                Verr = 299800000*sigma[2]/464.742
                int = integrate.quad(self.accurategauss, bounds_st[2], bounds_ed[2],
                                  args=(popt[0], popt[1], popt[2], popt[3], popt[4]))[0] - popt[0]*(bounds_ed[2]-bounds_st[2])
            else:
                print("Enter 'HeI', 'HeII', or 'CIII'")
                return


            T_arr = np.append(T_arr, T)
            Terr_arr = np.append(Terr_arr, Terr)
            V_arr = np.append(V_arr, V)
            Verr_arr = np.append(Verr_arr, Verr)
            int_arr = np.append(int_arr, int)
            dx_arr = np.append(dx_arr, dx)

            print('========= r = %5.3f m =========' % sightline_spect[num_pos])
            print('T_%s = %5.3f ± %5.3f eV' % (Species, T, Terr))
            print('V_%s = %5.3f ± %5.3f m/s' % (Species, V, Verr))
            print('%s_int = %5.3f' % (Species, int))
            print('%sdx = %5.3f' % (Species, dx))

            plt.plot(wavelength, data[:, num_pos], label='r=%5.3f' % sightline_spect[num_pos])
            if(Species=="HeI"):
                plt.plot(wavelength, self.gauss(wavelength, *popt), '-o',
                         label='gauss fitting(r=%5.3fm' % sightline_spect[num_pos])
                plt.xlim(471.1342, 471.49935)
            elif(Species=="HeII"):
                plt.plot(wavelength, self.HefitverS_const_wl(wavelength, *popt), '-o', label='gauss fitting(r=%5.3fm' % sightline_spect[num_pos])
                plt.xlim(468.2, 469.0)
            elif(Species=="CIII"):
                plt.plot(wavelength, self.accurategauss(wavelength, popt_0[0], popt_0[1], 464.742, popt_0[2]-464.742, popt_0[3]), '-o',
                         label='accurategauss fitting(r=%5.3fm' % sightline_spect[num_pos])
                plt.xlim(464.4, 465.2)

        print('Center of %sdx: %5.3f' % (Species, np.average(dx_arr)))
        for (num_pos, x) in enumerate(sightline_spect):
            V_arr[num_pos] = self.modify_center_for_V(dx_arr[num_pos], np.average(dx_arr), Species=Species) #TODO   dxcenterの与え方　要注意
            print('========= r = %5.3f m =========' % sightline_spect[num_pos])
            print('V_%s = %5.3f ± %5.3f m/s' % (Species, V_arr[num_pos], Verr_arr[num_pos]))

        plt.legend()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Intensity [a.u.]')

        plt.subplot(222)
        plt.plot(sightline_spect, int_arr, '-o', color='green', label=label)
        plt.legend()
        #plt.title(label + ', spectr_161206_18to27', loc='right', fontsize=20)
        plt.title(label + ', 161111#36-44', loc='right', fontsize=20)
        plt.xlabel('r [m]')
        plt.ylabel('Intensity of HeII [a.u.]')

        plt.subplot(223)
        plt.errorbar(sightline_spect, T_arr, yerr=Terr_arr, fmt='ro')
        plt.xlabel('r [m]')
        plt.ylabel('$T_{%s}$ [eV]' % Species)
        plt.ylim(0, 20)

        plt.subplot(224)
        plt.errorbar(sightline_spect, V_arr, yerr=Verr_arr, fmt='bo')
        plt.xlabel('r [m]')
        plt.ylabel('$V_{%s}$ [m/s]' % Species)
        plt.tight_layout()
        plt.show()


    def modify_center_for_V(self, dx, dxcent, Species):
        """
        modifiy spectral center and recalculate velocity
        :param dx:
        :param dxcent: int
        :param Species: char
        :return: Velocity of CorHe
        """
        if(Species=="HeI"):
            V = 0.0
        elif(Species=="HeII"):
            V = 299800000*(dx - dxcent)/468.565
        elif(Species=="CIII"):
            V = 299800000*(dx - dxcent)/464.742

        return V



if __name__ == '__main__':
    span = SpectAnal()
    #span.gauss_fitting_HeII(isAbel=False, spline=False, convolve=False)
    #span.gauss_fitting_CIII(isAbel=True, spline=False, convolve=False)
    #span.gauss_fitting_HeI(isAbel=False, spline=False, convolve=False)
    span.gauss_fitting(Species="CIII", isAbel=False, spline=False, convolve=False)

