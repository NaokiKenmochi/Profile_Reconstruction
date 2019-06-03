"""分光器で計測したスペクトルを元にイオン温度，フロー速度を求めるモジュールです
"""
from scipy.optimize import curve_fit
from scipy import integrate
from Abel_ne import Abel_ne

__author__  = "Naoki Kenmochi <kenmochi@edu.k.u-tokyo.ac.jp>"
__version__ = "1.0.0"
__date__    = "3 Mar. 2018"

import numpy as np
import matplotlib.pyplot as plt
import platform
import subprocess
import os.path


class SpectAnal(Abel_ne):

    def __init__(self, date, arr_shotNo, LOCALorPPL, instwid, lm0, dlm, opp_ch):
        """

        :param date: 解析対象の実験日
        :param arr_shotNo: 解析対象のショット番号（array）．複数選択すると積算したデータを解析する．
        :param LOCALorPPL: "LOCAL": ローカルに保存してあるデータを解析．"PPL": PPLサーバー(PC: spcectra）に保存してあるデータを解析
        :param instwid: 分光器の較正データを使用
        :param lm0:  分光器の較正データを使用
        :param dlm:  分光器の較正データを使用
        :param opp_ch: 対向チャンネルを指定
        """
        self.date = date
        self.arr_shotnum = arr_shotNo
        self.LOCALorPPL = LOCALorPPL
        self.file_path = "/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/Spectroscopy/d20161206sp/spectr_161206_18to27.txt"
        #self.instwid = 0.025129
        #self.instwid = 0.024485 #20161206
        self.instwid = instwid
        self.lm0 = lm0
        self.dlm = dlm
        self.opp_ch = opp_ch
        self.sightline_spect = 1e-3*np.array([385, 422, 475, 526, 576, 623, 667, 709, 785])

    def load_spec_data(self):
        if(platform.system() == 'Darwin'):
            path_OS = os.path.expanduser("~/mount_point/spectra/")

        elif(platform.system() == 'Windows'):
            path_OS = "//spectra/"

        if self.LOCALorPPL == "PPL":
            data = np.zeros((1024, 10))
            for (i, x) in enumerate(self.arr_shotnum):
                file_path = path_OS + "C/rt1sp/d" + str(self.date) + "sp/d" + str(self.arr_shotnum[i]) + ".asc"
                data_org = np.loadtxt(file_path)
                data += data_org[::-1, 1:]
            data /= self.arr_shotnum.__len__()
            wavelength = np.linspace(self.lm0, self.lm0 + self.dlm*1024, 1024)
        elif self.LOCALorPPL == "LOCAL":
            data_org = np.loadtxt(self.file_path, delimiter='\t', skiprows=1)
            data = np.zeros((1024, 9))
            d_order = np.array([0, 4, 1, 5, 2, 7, 3, 8, 9])
            for i in range(9):
                data[:, i] = data_org[:, d_order[i]]
            wavelength = np.linspace(self.lm0, self.lm0 + self.dlm*1024, 1024)
            #wavelength = np.linspace(462.268, 474.769, 1024)
        else:
            print("Enter 'PPL' or 'LOCAL'.")
            return

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

    def make_spectrum(self, Ti, Vi):
        wavelength = np.linspace(468.2, 468.8, 6)
        dx = Vi*468.565/299800000
        k0 = 0.0
        k1 = 1.0
        k3 = dx
        k4 = np.sqrt((468.565*np.sqrt(Ti/(469000000*4)))**2+self.instwid**2)
        fig,axes = plt.subplots(nrows=5, ncols=2)
        cnt=0
        #for i in range(wavelength.__len__()):
        for i in range(5):
            for j in range(1):
                y = self.HefitverS_const_wl(wavelength[cnt], k0, k1, k3, k4)
                #plt.imshow(y, cmap='jet')
                axes[i,j].imshow(y, cmap='jet')
                #plt.show()
                #plt.clf()
                cnt+=1
        #plt.plot(wavelength, y)
        plt.show()


    def load_image(self, p_opt_best, p_opt_best_2ndPeak):
        images = np.load("dataset_SimCIS/SimCIS_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.npz" % \
                         (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                          p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]))
        image_local = images['image_local']
        image = images['image']
        image_1ref = images['image_1ref']


        Ti_local = 70*image_local[::-1,:]
        Ti_projection = (image + image_1ref).T
        Vi_2D = np.zeros(np.shape(Ti_projection))
        spectrum_2D_local = self.make_spectrum(Ti_local, Vi_2D)
        spectrum_2D_projection = self.make_spectrum(Ti_projection, Vi_2D)

        plt.figure(figsize=(16,12))
        plt.subplot(1,2,1)
        plt.imshow(image_local[::-1,:], cmap='jet')
        plt.title("1st: %s\n2nd: %s\nLocal" % (p_opt_best, p_opt_best_2ndPeak))
        plt.subplot(1,2,2)
        plt.imshow((image + image_1ref).T, cmap='jet')
        plt.title("Projection")
        plt.show()


    def gauss_fitting(self, Species, isAbel=False, spline=False, convolve=False, isPLOT=True, is1CH=False, CH=None):
        bounds_st = np.array([471.2, 468.45, 464.6])    #[HeI, HeII, CIII]
        bounds_ed = np.array([471.5, 468.65, 464.83])    #[HeI, HeII, CIII]
        T_arr = np.array([])
        Terr_arr = np.array([])
        V_arr = np.array([])
        Verr_arr = np.array([])
        int_arr = np.array([])
        dx_arr = np.array([])

        #DXCENT = -0.0165

        if(isAbel==True):
            wavelength, sightline_spect, data, spect_local = super().abelic_spectroscopy(spline=spline, convolve=convolve)
            data = spect_local
            label = 'Local'
        else:
            data, wavelength = self.load_spec_data()
            sightline_spect = np.arange(10)
            label = 'Line-Integrated'

        #dataからオフセットを差し引く
        offset_data = np.average(data[:50, :], axis=0)
        data -= offset_data

        #TODO   Igor(Procedure_common.ipf)内のint_ratioを適用する必要があるのか要確認
        if(isPLOT==True):
            plt.figure(figsize=(12, 8))
            plt.subplot(221)

        for (num_pos, x) in enumerate(sightline_spect):
            try:
                if(Species=="HeI"):
                    init_values = np.array([10, 500, 471.31457, 0.01])
                    popt, pcov = curve_fit(self.gauss, wavelength, data[:, num_pos], p0=init_values)
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

                    #In IgorPro #ToDo Igorとの違いを確認
                    #$("W_" + chstr)	 ={$wr[pcsrA], wavemax($wr, xcsrA, xcsrB) - $wr[pcsrA], 464.742, W_coef[2] - 464.742, W_coef[3]}
                    init_values_2 = np.array([popt_0[0], popt_0[1], popt_0[2]-464.742, popt_0[3]])
                    popt, pcov = curve_fit(lambda wavelength, k0, k1, k3, k4: self.accurategauss(wavelength, k0, k1, 464.742, k3, k4),
                                           wavelength, data[:, num_pos], p0=init_values_2)
                    popt = np.insert(popt, 2, init_values[2])
                    sigma = np.sqrt(np.diag(pcov))
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

            except Exception as e:
                print("!!!!!!!!!!!!!!!!! ERROR in ch.%d !!!!!!!!!!!!!!!!!!!!" % num_pos)
                print(e.args)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                T = Terr = V = Verr = int = np.nan

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

            if(isPLOT==True):
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
        DXCENT = (dx_arr[self.opp_ch[0]-1]+dx_arr[self.opp_ch[1]-1])/2.0
        print('Center of %sdx: %5.3f' % (Species, DXCENT))
        for (num_pos, x) in enumerate(sightline_spect):
            #V_arr[num_pos] = self.modify_center_for_V(dx_arr[num_pos], np.average(dx_arr), Species=Species) #TODO   dxcenterの与え方　要注意
            V_arr[num_pos] = self.modify_center_for_V(dx_arr[num_pos], dxcent=DXCENT, Species=Species) #TODO   dxcenterの与え方　要注意
            print('========= r = %5.3f m =========' % sightline_spect[num_pos])
            print('V_%s(Calib.) = %5.3f ± %5.3f m/s' % (Species, V_arr[num_pos], Verr_arr[num_pos]))

        if(isPLOT==True):
            plt.legend(fontsize=8, loc='right')
            plt.xlabel('Wavelength [nm]')
            plt.ylabel('Intensity [a.u.]')
            plt.ylim(0, 100)

            plt.subplot(222)
            plt.plot(sightline_spect, int_arr, '-o', color='green', label=label)
            plt.legend(fontsize=12, loc='best')
            plt.title('Date: %d, Shot No.: %d-%d, %s' % (self.date, self.arr_shotnum[0], self.arr_shotnum[-1], label), loc='right', fontsize=20)
            #plt.title(label + ', spectr_161206_18to27', loc='right', fontsize=20)
            plt.xlabel('r [m]')
            plt.ylabel('Intensity of %s [a.u.]' % Species)

            plt.subplot(223)
            plt.errorbar(sightline_spect, T_arr, yerr=Terr_arr, fmt='ro')
            plt.xlabel('r [m]')
            plt.ylabel('$T_{%s}$ [eV]' % Species)
            plt.ylim(0, 20)

            plt.subplot(224)
            plt.errorbar(sightline_spect, V_arr, yerr=Verr_arr, fmt='bo')
            plt.xlabel('r [m]')
            plt.ylabel('$V_{%s}$ [m/s]' % Species)
            plt.ylim(-1e4, 1e4)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0.2)
            plt.show()

        return T_arr, Terr_arr, V_arr, Verr_arr, int_arr, data, wavelength


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

def make_profile(date, ch, Species):
    label = 'Line-Integrated'
    #arr_shotnum = np.array([31, 32, 33, 34, 35])
    #arr_sightline = np.array([379, 484, 583, 689, 820])
    arr_shotnum = np.arange(68, 81)
    arr_sightline = 1e-3*np.array([422, 475, 526, 576, 623, 667, 709, 785, 785, 667, 576, 475, 385])
    #arr_shotnum = np.array([87, 54, 89, 91, 93, 95, 97, 99, 101])    #For 23Feb2018
    #arr_sightline_buf = 1e-3*np.array([422, 385, 475, 526, 576, 623, 667, 709, 785])
    #arr_sightline = np.zeros(2*arr_sightline_buf.__len__())
    #arr_sightline[1::2] = arr_sightline[::2] = arr_sightline_buf[:]
    #arr_shotnum = np.r_[np.array([58, 59]), np.arange(87, 103)]

    T_rarr = np.array([])
    Terr_rarr = np.array([[]])
    V_rarr = np.array([])
    Verr_rarr = np.array([])
    int_rarr = np.array([])
    data_r = np.array([])

    plt.figure(figsize=(12, 8))
    plt.subplot(221)

    _mount()

    for i, shotnum in enumerate(arr_shotnum):
        print("\n\n\nCalculate T%s and V%s for #%d in %d" % (Species, Species, shotnum, date))
        span = SpectAnal(date=date, arr_shotNo=[shotnum], LOCALorPPL="PPL",
                         #instwid=0.016831, lm0=462.195, dlm=0.0122182, opp_ch=[5, 6])   #7-11 Nov. 2017
                         instwid=0.017867, lm0=462.235, dlm=0.0122165, opp_ch=[5, 6])   #19-23 Dec. 2017
                         #instwid=0.020104, lm0=462.255, dlm=0.0122178, opp_ch=[5, 6])   #19-23 Feb. 2018
        T_arr, Terr_arr, V_arr, Verr_arr, int_arr, data, wavelength = span.gauss_fitting(Species=Species, isAbel=False,
                                                                                         spline=False, convolve=False, isPLOT=False)
        T_rarr = np.append(T_rarr, T_arr[ch])
        Terr_rarr = np.append(Terr_rarr, Terr_arr[ch])
        V_rarr = np.append(V_rarr, V_arr[ch])
        Verr_rarr = np.append(Verr_rarr, Verr_arr[ch])
        int_rarr = np.append(int_rarr, int_arr[ch])
        data_r = np.append(data_r, data[:, ch])
        if(Species=="HeI"):
            plt.xlim(471.1342, 471.49935)
        elif(Species=="HeII"):
            plt.xlim(468.2, 469.0)
        elif(Species=="CIII"):
            plt.xlim(464.4, 465.2)
        plt.plot(wavelength, data[:, ch], label='r=%5.3f' % arr_sightline[i])


    plt.legend(fontsize=8, loc='right')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity [a.u.]')
    plt.ylim(0, 100)

    plt.subplot(222)
    plt.plot(arr_sightline, int_rarr, '-o', color='green', label=label)
    plt.legend(fontsize=12, loc='best')
    plt.title('Date: %d, Shot No.: %d-%d, %s' % (date, arr_shotnum[0], arr_shotnum[-1], label), loc='right', fontsize=20)
    plt.xlabel('r [m]')
    plt.ylabel('Intensity of %s [a.u.]' % Species)

    plt.subplot(223)
    plt.errorbar(arr_sightline, T_rarr, yerr=Terr_rarr, fmt='ro')
    plt.xlabel('r [m]')
    plt.ylabel('$T_{%s}$ [eV]' % Species)
    plt.ylim(0, 20)

    plt.subplot(224)
    plt.errorbar(arr_sightline, V_rarr, yerr=Verr_rarr, fmt='bo')
    plt.xlabel('r [m]')
    plt.ylabel('$V_{%s}$ [m/s]' % Species)
    plt.ylim(-1e4, 1e4)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    plt.show()

    np.savez("r_THeII_VHeII_20171223_68to80.npz",
             arr_sightline=arr_sightline, T_rarr=T_rarr, Terr_rarr=Terr_rarr, V_rarr=V_rarr, Verr_rarr=Verr_rarr)

def _mount():
    if(platform.system() == 'Darwin'):
        try:
            cmd = 'mount_smbfs //rt-1:ringtrap@172.16.107.131/C ~/mount_point/spectra/C'
            subprocess.check_call(cmd, shell=True)

        except Exception as e:
            if(e.args[0] == 64):
                print("!!!spectra is already mouted !!!")
            elif(e.args[0] == 68):
                print("Error; mount_smbfs: server connection failed: No route to host")
            else:
                print("!!!!!!!!!!!!!!!!! ERROR !!!!!!!!!!!!!!!!!!!!")
                print(e.args)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")



if __name__ == '__main__':
    span = SpectAnal(date=20171223, arr_shotNo=[80], LOCALorPPL="LOCAL",
    #                 #instwid=0.020104, lm0=462.2546908755637, dlm=0.01221776718749085, opp_ch=[5, 6])
    #                 #instwid=0.016831, lm0=462.195, dlm=0.0122182, opp_ch=[5, 6])   #7-11 Nov. 2017
                     instwid=0.017867, lm0=462.235, dlm=0.0122165, opp_ch=[5, 6])   #19-23 Dec. 2017
    #span.make_spectrum(Ti=0, Vi=0)
    span.load_image(p_opt_best=[1, 11, 1.10, 0.54], p_opt_best_2ndPeak=[0.10, 1, 0.10, 0.70])
    #span.gauss_fitting(Species="HeII", isAbel=True, spline=False, convolve=False)
    #make_profile(date=20171223, ch=1, Species="HeII")

