"""線積分計測のアーベル変換を行い，局所値を求めるモジュールです
"""

from sightline_ne import sightline_ne
from scipy import interpolate, signal
from matplotlib import gridspec

__author__  = "Naoki Kenmochi <kenmochi@edu.k.u-tokyo.ac.jp>"
__version__ = "1.0.0"
__date__    = "31 Aug 2018"

import numpy as np
import matplotlib.pyplot as plt
import abel
import matplotlib.ticker

class Abel_ne(sightline_ne):
    """
    線積分値を局所値にアーベル変換するクラスです
    """

    def __init__(self):
        super().__init__()
        print("Load constructor of Abel_ne")

    def get_nel(self):
        nl_y = super().calc_ne()
        plt.plot(self.sight_line_para, nl_y)
        #plt.show()

        recon = abel.basex.basex_transform(nl_y, verbose=True, basis_dir=None,
                dr=0.03, direction='inverse')
        #recon = abel.Transform(nl_y, direction='inverse', method='linbasex').transform

        plt.plot(self.sight_line_para, recon)
        plt.show()

    def abelic_r0(self):
        """
        以下の条件の計測に対するアーベル変換を行います
        計測視線：等間隔
        計測開始：中心
        計測終了：最外殻

        :return:
        """
        nl_y = super().calc_ne()

        dr = self.sight_line_para[1] - self.sight_line_para[0]
        N = self.num_para
        A = np.eye(self.num_para)
        for k in range(1, N+1):
            A[k-1, k-1] = dr*(k**2*np.arccos((k-1)/k) - (k-1)*np.sqrt(2*k-1))
            for j in range(1, N-k+1):
                A[k-1, k+j-1] = dr*((k+j)**2*np.arccos((k-1)/(k+j)) - (k-1)*np.sqrt(2*k*(j+1)+j**2-1)  \
                + (k+j-1)**2*np.arccos(k/(k+j-1)) - k*np.sqrt(2*k*(j-1)+(j-1)**2) \
                - (k+j)**2*np.arccos(k/(k+j)) + k*np.sqrt(2*k*j+j**2) \
                - (k+j-1)**2*np.arccos((k-1)/(k+j-1)) + (k-1)*np.sqrt(2*k*j+j**2-2*j))

        ne = np.linalg.solve(A, nl_y)

    def make_integration_matrix_uneven_dr(self, sight_line, d):
        """
        以下の条件の計測に対するアーベル変換行列を計算します
        計測視線：不等間隔
        計測開始：中心より外側
        計測終了：最外殻

        カメラ計測のために立体角を考慮します

        :param sight_line:
        :return:
        """

        k_st = np.int(sight_line[0]/(sight_line[1]-sight_line[0]))
        N = sight_line.__len__()    #np.int(self.sight_line_para[-1]/dr)
        A = np.eye(N)
        S0 = np.ones(N)
        for k in range(k_st+1, k_st+N+1):
            S0[k-k_st-1] = (sight_line[-1] + d)**2 - sight_line[k-k_st-1]**2
            if(k==k_st+1):
                dr = sight_line[1] - sight_line[0]
            elif(k==k_st+N):
                dr = sight_line[-1] - sight_line[-2]
            else:    #TODO   要検証
                #dr = (sight_line[k-k_st] - sight_line[k-2-k_st])/2
                dr = np.sqrt((((sight_line[k-k_st]-sight_line[k-k_st-1])**2 + (sight_line[k-k_st-1]-sight_line[k-2-k_st])**2))/2)
            A[k-1-k_st, k-1-k_st] = dr*(k**2*np.arccos((k-1)/k) - (k-1)*np.sqrt(2*k-1))
            for j in range(1, k_st+N-k+1):
                if(j==k_st+N-k):
                    dr = sight_line[-1] - sight_line[-2]
                else:    #TODO   要検証
                    #dr = (sight_line[j+k-k_st] - sight_line[j+k-2-k_st])/2
                    dr = np.sqrt(((sight_line[j+k-k_st]-sight_line[j+k-k_st-1])**2 + (sight_line[j+k-k_st-1]-sight_line[j+k-2-k_st])**2)/2)
                A[k-1-k_st, k+j-1-k_st] = dr*((k+j)**2*np.arccos((k-1)/(k+j)) - (k-1)*np.sqrt(2*k*(j+1)+j**2-1) \
                                              + (k+j-1)**2*np.arccos(k/(k+j-1)) - k*np.sqrt(2*k*(j-1)+(j-1)**2) \
                                              - (k+j)**2*np.arccos(k/(k+j)) + k*np.sqrt(2*k*j+j**2) \
                                              - (k+j-1)**2*np.arccos((k-1)/(k+j-1)) + (k-1)*np.sqrt(2*k*j+j**2-2*j))

        S = np.diag(1/S0)
        return A, S

    def abelic_uneven_dr(self, nl, sight_line):
        """
        以下の条件の計測に対するアーベル変換を行います
        計測視線：不等間隔
        計測開始：中心より外側
        計測終了：最外殻

        nlが２次元の場合，２次元のnを返す
        :param nl:
        :param sight_line:
        :return:
        """
        A, _ = self.make_integration_matrix_uneven_dr(sight_line, d=0)

        if(nl.ndim == 1):
            n = np.linalg.solve(A, nl)
        else:
            #n = np.zeros(nl.shape, dtype=complex)
            n = np.zeros(nl.shape)
            for i in range(nl.__len__()):
                n[i, :] = np.linalg.solve(A, nl[i, :])  #for文を回さず同じ動作をさせたい

        return n

    def abelic_uneven_dr_wrt_solid_angle(self, nl, sight_line, d):
        """
        以下の条件の計測に対するアーベル変換を行います
        計測視線：不等間隔
        計測開始：中心より外側
        計測終了：最外殻

        カメラ計測のために立体角を考慮します

        nlが２次元の場合，２次元のnを返す
        :param nl:
        :param sight_line:
        :return:
        """

        A, S = self.make_integration_matrix_uneven_dr(sight_line, d)

        if(nl.ndim == 1):
            n = np.linalg.solve(np.dot(A, S), nl)
        else:
            #n = np.zeros(nl.shape, dtype=complex)
            n = np.zeros(nl.shape)
            for i in range(nl.__len__()):
                n[i, :] = np.linalg.solve(np.dot(A, S), nl[i, :])  #for文を回さず同じ動作をさせたい

        return n

    def make_nel_uneven_dr_wrt_solid_angle(self, ne, sight_line, d):
        """
        局所値から立体角を考慮した視線積分値に変換します
        計測視線：不等間隔
        計測開始：中心より外側
        計測終了：最外殻


        nlが２次元の場合，２次元のnを返す
        :param nl:
        :param sight_line:
        :return:
        """
        A, S = self.make_integration_matrix_uneven_dr(sight_line, d)

        if(ne.ndim == 1):
            nl = np.dot(np.dot(A, S), ne)
        else:
            #n = np.zeros(nl.shape, dtype=complex)
            nl = np.zeros(ne.shape)
            for i in range(ne.__len__()):
                nl[i, :] = np.dot(np.dot(A, S), ne[i, :])  #for文を回さず同じ動作をさせたい

        return nl

    def abelic(self, nl, sight_line):
        '''
        以下の条件の計測に対するアーベル変換を行います
        計測視線：等間隔
        計測開始：中心より外側
        計測終了：最外殻
        Abel Matrix Calculation
        :param nl: line-integrated value
        :param sight_line: position of sight line
        :return: local value
        '''

        dr = sight_line[1] - sight_line[0]
        k_st = np.int(sight_line[0]/dr)
        N = sight_line.__len__()    #np.int(self.sight_line[-1]/dr)
        A = np.eye(N)
        for k in range(k_st+1, k_st+N+1):
            A[k-1-k_st, k-1-k_st] = dr*(k**2*np.arccos((k-1)/k) - (k-1)*np.sqrt(2*k-1))
            #for j in range(k_st+1, k_st+N-k+1):
            for j in range(1, k_st+N-k+1):
                A[k-1-k_st, k+j-1-k_st] = dr*((k+j)**2*np.arccos((k-1)/(k+j)) - (k-1)*np.sqrt(2*k*(j+1)+j**2-1) \
                                              + (k+j-1)**2*np.arccos(k/(k+j-1)) - k*np.sqrt(2*k*(j-1)+(j-1)**2) \
                                              - (k+j)**2*np.arccos(k/(k+j)) + k*np.sqrt(2*k*j+j**2) \
                                              - (k+j-1)**2*np.arccos((k-1)/(k+j-1)) + (k-1)*np.sqrt(2*k*j+j**2-2*j))

        n = np.linalg.solve(A, nl)

        return n

    def abelic_spectroscopy(self, spline=False, convolve=False):
        """
        1m分光器で計測されたスペクトルに対するアーベル変換を行います
        :param spline:
        :return:
        """
        data_org = np.loadtxt("/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/Spectroscopy/d20161206sp/spectr_161206_18to27.txt", delimiter='\t', skiprows=1)
        d_order = np.array([0, 4, 1, 5, 2, 7, 3, 8, 9])
        wavelength = np.linspace(462.268, 474.769, 1024)
        sightline_spect = 1e-3*np.array([385, 422, 475, 526, 576, 623, 667, 709, 785])

        #data_org = np.loadtxt("/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/Spectroscopy/d20161111sp/spectr_161111_22to33.txt", delimiter='\t', skiprows=1)
        #d_order = np.array([0, 5, 1, 6, 2, 7, 3, 8, 4])

        #data_org_ch2 = np.loadtxt("/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/Spectroscopy/d20161111sp/spectr_161111_ch2_35to44.txt", delimiter='\t', skiprows=1)
        #data_org_ch5 = np.loadtxt("/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/Spectroscopy/d20161111sp/spectr_161111_ch5_36to44.txt", delimiter='\t', skiprows=1)
        #data_org = np.c_[data_org_ch2, data_org_ch5]
        #sightline_spect_ch2 = 1e-3*np.array([385, 422, 475, 526, 576, 623, 667, 709, 785])
        #sightline_spect_ch5 = 1e-3*np.array([379, 432, 484, 535, 583, 630, 689, 745, 820])
        #sightline_spect = sightline_spect_ch5#np.sort(np.r_[sightline_spect_ch2, sightline_spect_ch5])
        ##d_order = np.array([9, 0, 5, 14, 3, 10, 6, 15, 1, 11, 7, 16, 2, 12, 8, 17, 4, 13])
        ##d_order = np.array([0, 5, 3, 6, 1, 7, 2, 8, 4])
        #d_order = np.array([9, 14, 10, 15, 11, 16, 12, 17, 13])

        #data_org = np.loadtxt("/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/Spectroscopy/d20161111sp/spectr_161111_ch56_50.txt", delimiter='\t', skiprows=1)
        #data_org = np.loadtxt("/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/Spectroscopy/d20161206sp/spectr_ch56_161206_18to27.txt", delimiter='\t', skiprows=1)
        #sightline_spect = 1e-3*np.array([450, 451])
        #d_order = np.array([0, 1])

        #wavelength = np.linspace(462.261, 474.761, 1024)
        data = np.zeros((1024, sightline_spect.__len__()))
        for i in range(sightline_spect.__len__()):
            data[:, i] = data_org[:, d_order[i]]

        if(convolve==True):
            spect_convolved = np.zeros((data[:, 0].__len__(), sightline_spect.__len__()))
            for i in range(data[:, 0].__len__()):
                num_convolve = 5
                b = np.ones(num_convolve)/num_convolve
                spect_convolved[i, :] = np.convolve(data[i, :], b, mode='same')
            data = spect_convolved
        #for i in range(sightline_spect.__len__()):
        #    plt.plot(wavelength, data[:, i], label=('r=%.3fm' % sightline_spect[i]))
        plt.plot(sightline_spect, data[515, :], '-^', label='Line-integrated')
        plt.title('Line-integrated, 20161206, #18-27')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Intensity [a.u.]')
        plt.legend()
        plt.tight_layout()
        #plt.show()

        if(spline == True):
            dr = (sightline_spect[-1] - sightline_spect[0])/((sightline_spect.__len__()-1)*2)
            f = interpolate.interp1d(sightline_spect, data)#, kind='zero')
            sightline_spect = np.arange(sightline_spect[0],sightline_spect[-1], dr)
            data = f(sightline_spect)

        plt.plot(sightline_spect, data[515, :], '-^', label='Line-integrated')
        spect_local = self.abelic_uneven_dr(data, sightline_spect)
        spect_integrated_wrtSolidAngle = self.make_nel_uneven_dr_wrt_solid_angle(spect_local, sightline_spect, d=0.3)
        #plt.plot(wavelength, spect_local, label='local')
        plt.plot(sightline_spect[1:], spect_local[515, 1:], '-o', label='Local')
        plt.plot(sightline_spect[1:], spect_integrated_wrtSolidAngle[515, 1:], '-x', label='Line-integrated wrt Solid Angle')
        #for i in range(sightline_spect.__len__()):
        #    plt.plot(wavelength, spect_local[:, i], label=('r=%.3fm' % sightline_spect[i]))
        plt.xlabel('r [m]')
        plt.ylabel('Intensity [a.u.]')
        plt.title('468.6nm, 20161111, #22-33')
        #plt.title('Local, 20161206, #18-27')
        #plt.xlabel('Wavelength [nm]')
        #plt.ylabel('Intensity [a.u.]')
        plt.tight_layout()
        plt.legend()
        plt.show()

        plt.figure(figsize=(16,9))
        WAVELENGTH, R_POL = np.meshgrid(wavelength, sightline_spect)
        plt.contourf(WAVELENGTH, R_POL, spect_local.T, cmap='jet')
        plt.colorbar()
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('r [mm]')
        plt.show()

        return wavelength, sightline_spect, data, spect_local

    def abelic_SX(self, spline=False):
        """
        X線計測されたスペクトルに対するアーベル変換を行います
        :param spline:
        :return:
        """
        file_path = '/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/SX/171109'
        file_name = np.array(['73_78', '62_67', '53_58', '45_50'])
        for (i, x) in enumerate(file_name):
            data_buf = np.load(file_path + '/' + file_name[i] + '.npz')
            if(i==0):
                data = data_buf['count']
                energy = data_buf['energy']
            else:
                data = np.c_[data, data_buf['count']]
        data_org = data
        sightline_SX_org = np.array([0.40, 0.45, 0.50, 0.55])
        sightline_SX = np.array([0.40, 0.45, 0.50, 0.55])

        if(spline == True):
            dr = (sightline_SX_org[-1] - sightline_SX_org[0])/((sightline_SX_org.__len__()-1)*3.0)
            f = interpolate.interp1d(sightline_SX_org, data, kind='cubic')
            sightline_SX = np.arange(sightline_SX_org[0],sightline_SX_org[-1], dr)
            data = f(sightline_SX)

        SX_local = self.abelic_uneven_dr(data, sightline_SX)

        plt.figure(figsize=(12,8))
        plt.subplot(221)
        plt.plot(sightline_SX_org, data_org[20, :], '-^', label='Line-integrated')
        if(spline==True):
            plt.plot(sightline_SX, data[20, :], '-^', label='Line-integrated (interpolated)')
        plt.plot(sightline_SX, SX_local[20, :], '-o', label='Local')
        plt.xlabel('r [m]')
        plt.ylabel('count at %deV' % energy[20])
        plt.legend()
        #plt.show()

        plt.subplot(223)
        for i in range(sightline_SX_org.__len__()):
            plt.plot(energy, data[:, i], label=('r=%.3fm' % sightline_SX_org[i]))
        plt.title('Line-integrated, 20171109')
        plt.xlabel('Energy [eV]')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        #plt.show()

        plt.subplot(224)
        for i in range(sightline_SX.__len__()):
            plt.plot(energy, SX_local[:, i], label=('r=%.3fm' % sightline_SX[i]))
        plt.title('Local, 20171109')
        plt.xlabel('Energy [eV]')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.legend()
        #plt.show()

        plt.subplot(222)
        ENERGY, R_SX = np.meshgrid(energy, sightline_SX)
        plt.contourf(ENERGY, R_SX, SX_local.T, cmap='jet')
        plt.colorbar()
        plt.xlabel('Energy [nm]')
        plt.ylabel('r [mm]')
        plt.show()

        np.savez('SX_Local_171109.npz', energy=energy, count=SX_local)

    def abelic_pol(self, spline=False):
        """
        ポリクロメータの光量に対するアーベル変換を行います
        :param spline:
        :return:
        """
        file_path = "/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/Polychromator/20171223"
        file_name_wGP = "Pol_wGP_20171223_68to80.npz"
        file_name_woGP = "Pol_woGP_20171223_81to89.npz"
        pol_wGP = np.load(file_path + '/' + file_name_wGP)
        pol_woGP = np.load(file_path + '/' + file_name_woGP)

        pol = pol_wGP['Pol710nm']

        sightline_pol = 1e-3*np.array([379, 432, 484, 535, 583, 630, 689, 745, 820])

        pol_convolved = np.zeros((pol[:, 0].__len__(), sightline_pol.__len__()))
        for i in range(sightline_pol.__len__()):
            num_convolve = 100
            b = np.ones(num_convolve)/num_convolve
            pol_convolved[:, i] = np.convolve(pol[:, i], b, mode='same')
        #num_convolve = 100
        #b = np.ones((num_convolve, 9))/num_convolve/4.5
        #pol_convolved = signal.convolve2d(pol, b, boundary='symm', mode='same')    #TODO   convolve2dが使用できるか要検証
        plt.plot(pol[:, 1])
        plt.plot(pol_convolved[:, 1])
        plt.show()

        if(spline == True):
            dr = (sightline_pol[-1] - sightline_pol[0])/((sightline_pol.__len__()-1)*2)
            #f = interpolate.interp1d(sightline_pol, pol_av, kind='cubic')
            f = interpolate.interp1d(sightline_pol, pol_convolved, kind='cubic')
            sightline_pol = np.arange(sightline_pol[0], sightline_pol[-1]+dr, dr)
            #pol_av = f(sightline_pol)
            pol_convolved = f(sightline_pol)

        #pol_local = self.abelic_uneven_dr(pol_av, sightline_pol)
        #plt.plot(sightline_pol, pol_local, label='pol_local')
        plt.plot(sightline_pol, pol_convolved[12500, :], label='pol_convolved')
        pol_local_2 = self.abelic_uneven_dr(pol_convolved, sightline_pol)
        plt.plot(sightline_pol, pol_local_2[12500, :], label='pol_local_2')
        plt.legend()
        plt.show()
        plt.figure(figsize=(16,9))
        time = np.linspace(1, 2, 10000)
        TIME, R_POL = np.meshgrid(time, sightline_pol)
        plt.contourf(TIME, R_POL, pol_local_2[10000:20000, :].T, cmap='jet')
        plt.colorbar()
        plt.xlabel('Time [sec]')
        plt.ylabel('r [mm]')
        plt.title('delta ne_wGP - ne_woGP (Pol710nm)')
        plt.show()


    def abelic_ne(self, spline=False):
        """
        干渉計で計測した線積分電子密度に対するアーベル変換を行います
        :param spline:
        :return:
        """
        nl_y_ori, rs, ne_profile_z0 = super().calc_ne()

        if(spline == True):
            dr = (self.sight_line_para[-1] - self.sight_line_para[0])/((self.sight_line_para.__len__()-1)*3)
            rr = np.arange(self.sight_line_para[0], self.sight_line_para[-1], dr)
            f = interpolate.interp1d(self.sight_line_para, nl_y_ori, kind='cubic')
            nl_y = f(rr)
        else:
            rr = self.sight_line_para
            nl_y = nl_y_ori

        #ne = self.abelic(nl_y, rr)
        ne = self.abelic_uneven_dr(nl_y, rr)
        nel = self.make_nel_uneven_dr_wrt_solid_angle(ne, rr, d=0.3)
        ne_2 = self.abelic_uneven_dr_wrt_solid_angle(nel, rr, d=0.3)

        #ne_test = np.linspace(20, 0, 20)
        #nl_y_test = np.dot(A, ne_test)
        #ne_test_2 = np.linalg.solve(A, nl_y_test)
        #plt.plot(ne_test)
        #plt.plot(ne_test_2)
        #plt.plot(nl_y_test)
        #plt.show()
        return ne, rs, rr, ne_profile_z0, nl_y_ori, nel, ne_2

    def abelic_pol_stft(self, spline=False, abel=True):
        """
        390nm, 730nm, 710nm, 450nmの順で格納
        390nm, 450nmの比を用いて電子密度を計算
        730nm, 710nmの比を用いて電子温度を計算
        """
        vmin = 0.0
        vmax = 2e-9
        #vmax = 5e-10
        #vmax = 3e-6
        time_offset_stft = 0.0

        #STFTの時間・周波数平均をとるための設定値
        t_st = 1.25#1.25
        t_ed = 1.35#1.35
        f_st = 575
        f_ed = 625

        file_path = '/Users/kemmochi/PycharmProjects/RT1DataBrowser/'
        #file_name = "Pol_stft_20180223_87to101.npz"
        #file_name = "Pol_ratio_stft_20180223_87to101.npz"
        #file_name = "Pol_ratio_woffset_stft_complex_20180223_87to101.npz"
        file_name = "Pol_ratio_woffset_stft_20180223_87to101.npz"
        #file_name = "Pol_ratio_woffset_stft_20171223_80to75.npz"
        #file_name = "Pol_ratio_woffset_stft_20171223_97to75.npz"
        #file_name = "Pol_woffset_stft_20171223_80to75.npz"
        data_buf = np.load(file_path + file_name)

        r_pol = data_buf["r_pol"]
        f = data_buf["f"]
        t = data_buf["t"]
        Zxx_4D = data_buf["Zxx_4D"]

        if(spline == True):
            pass

        #Zxx_4D_local = np.zeros(np.shape(Zxx_4D), dtype=complex)
        Zxx_4D_local = np.zeros(np.shape(Zxx_4D))
        if(abel == True):
            for i_time in range(t.__len__()):
                for i_ch in range(np.shape(Zxx_4D)[2]):
                    #Zxx_4D_local[:, i_time, i_ch, :] = self.abelic_uneven_dr(np.abs(Zxx_4D[:, i_time, i_ch, :]), r_pol)
                    Zxx_4D_local[:, i_time, i_ch, :] = self.abelic_uneven_dr(Zxx_4D[:, i_time, i_ch, :], r_pol)

        else:
            Zxx_4D_local = Zxx_4D

        #Zxxの周波数，時間平均を格納する配列
        arr_avZxx = np.zeros((r_pol.__len__(), np.shape(Zxx_4D)[2]))

        #plt.plot(r_pol, np.abs(Zxx_4D_local[12, 60, 0, :]))
        #plt.plot(r_pol, np.abs(Zxx_4D_local[12, 60, 1, :]))
        #plt.show()

        #plt.figure(figsize=(20, 16))
        fig, axes = plt.subplots(nrows=r_pol.__len__(), ncols=np.shape(Zxx_4D)[2], figsize=(20, 16))
        gs = gridspec.GridSpec(9, np.shape(Zxx_4D)[2])
        gs.update(hspace=0.1, wspace=0.15)
        for i_r in range(r_pol.__len__()):
            for i_wl in range(np.shape(Zxx_4D)[2]):
                ax0 = plt.subplot(gs[i_r, i_wl])
                im = plt.pcolormesh(t + time_offset_stft, f, np.abs(Zxx_4D_local[:, :, i_wl, i_r]), vmin=vmin, vmax=vmax)
                arr_avZxx[i_r, i_wl] = self.cal_mean_value(t, f, t_st=t_st-time_offset_stft, t_ed=t_ed-time_offset_stft, f_st=f_st, f_ed=f_ed, Zxx=Zxx_4D_local[:,:,i_wl, i_r])
                #arr_avZxx[i_r, i_wl] = self.cal_mean_value(t, f, t_st=1.5, t_ed=1.6, f_st=500, f_ed=550, Zxx=Zxx_4D_local[:,:,i_wl, i_r])
                if(i_r == r_pol.__len__()-1):
                    ax0.set_xlabel("Time [sec]")
                elif(i_r != r_pol.__len__()-1):
                    ax0.tick_params(labelbottom=False)
                ax0.set_ylabel("r=%dmm\nFreq. [Hz]" % (r_pol[i_r]))
                ax0.set_xlim(0.5, 2.5)
                ax0.set_ylim([0, 2000])
                if(i_wl==np.shape(Zxx_4D)[2]-1 and i_r==0):
                    #plt.title("%s" % (file_name), loc='right', fontsize=20, fontname="Times New Roman")
                    plt.title("Abel Inversion, %s\n447/388 (ne sensitive)" % (file_name), loc='right', fontsize=20, fontname="Times New Roman")
                    #plt.title("%s\n447/388 (ne sensitive)" % (file_name), loc='right', fontsize=20, fontname="Times New Roman")
                if(i_wl==0 and i_r==0):
                    plt.title("728/706 (Te sensitive)", loc='right', fontsize=20, fontname="Times New Roman")
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

        rs, ne_profile_z0 = self.load_nez0_profile()
        plt.rcParams['font.family'] ='sans-serif'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        plt.rcParams['ytick.direction'] = 'in'
        fig, ax1 = plt.subplots()
        ax1.plot(rs, ne_profile_z0, label='Δ$n_e$', color='blue')
        ax1.hlines(y=0, xmin=0.35, xmax=0.85, colors='black', linestyles='dashed')
        #ax1.fill_between(rs, ne_profile_z0, where=ne_profile_z0>=0, facecolors='red', interpolate=True)
        #ax1.fill_between(rs, ne_profile_z0, where=ne_profile_z0<=0, facecolors='blue', interpolate=True)
        ax2 = ax1.twinx()
        #ax2.plot(r_pol*1e-3, arr_avZxx[:, 0])
        ax2.plot(r_pol*1e-3, arr_avZxx[:, 1], "-o", color='red', label='PSD')
        ax2.set_ylabel('PSD')
        plt.title("t: %.2f-%.2f sec, f: %d-%d Hz" % (t_st, t_ed, f_st, f_ed))
        plt.xlim(0.35, 0.85)
        ax1.set_ylabel('Δ$n_e [10^{16} m^{-3}]$')
        ax1.set_xlabel('r [m]')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.show()

    def plot_ne_nel(self, spline=False):
        ne, rs, rr, ne_profile_z0, nl_y_ori, nel, ne_2 = self.abelic_ne(spline=spline)
        plt.rcParams['font.family'] ='sans-serif'
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['xtick.top'] = 'True'
        plt.rcParams['ytick.right'] = 'True'
        plt.rcParams['ytick.direction'] = 'in'
        plt.figure(figsize=(6,4))
        #plt.plot(self.sight_line_para, ne)
        #plt.plot(rr, nl_y, "-^", label='nel')
        plt.plot(rs, ne_profile_z0, color='black', label='$n_e$(original)')
        plt.plot(self.sight_line_para, nl_y_ori, "-^", color='blue', label='$n_eL$')
        plt.plot(rr[:-1], ne[:-1], "-o", color='red', label='$n_e$(reconstructed)')
        plt.plot(rr[:-1], ne_2[:-1], "o", color='green', label='$n_e$(wrt Solid Angle)')
        plt.plot(rr[:-1], nel[:-1], "o", color='blue', label='$n_eL$(wrt Solid Angle)')
        plt.legend(fontsize=10, loc='upper left')
        plt.tick_params(labelsize=12)
        plt.xlim(0, 1.0)
        plt.ylim(0, 35)
        plt.xlabel('r [m]', fontsize=16)
        plt.ylabel(r'$\mathrm{[10^{16}\,m^{-3}]}$', fontsize=16)
        plt.tight_layout()
        plt.show()

    def cal_mean_value(self, t, f, t_st, t_ed, f_st, f_ed, Zxx):
        #平均を求める際の時間(t)，周波数(f)の範囲とそのindexを取得
        idx_tst = np.abs(np.asarray(t - t_st)).argmin()
        idx_ted = np.abs(np.asarray(t - t_ed)).argmin()
        idx_fst = np.abs(np.asarray(f - f_st)).argmin()
        idx_fed = np.abs(np.asarray(f - f_ed)).argmin()

        return np.mean(np.abs(Zxx[idx_fst:idx_fed, idx_tst:idx_ted]))

    def load_nez0_profile(self):
        nez0 = np.load("rs_nez0_35_t15mt11.npz")
        #nez0 = np.load("rs_nez0_20171111.npz")
        rs = nez0["rs"]
        ne_profile_z0 = nez0["ne_profile_z0"]

        return rs, ne_profile_z0

if __name__ == '__main__':
    abne = Abel_ne()
    #abne.plot_ne_nel(spline=True)
    #abne.abelic_pol(spline=True)
    #abne.abelic_pol_stft(spline=True, abel=True)
    abne.abelic_spectroscopy(spline=True, convolve=False)
    #abne.abelic_SX(spline=True)
