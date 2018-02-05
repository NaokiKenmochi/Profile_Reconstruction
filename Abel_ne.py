"""線積分計測のアーベル変換を行い，局所値を求めるモジュールです
"""

from sightline_ne import sightline_ne
from scipy import interpolate

__author__  = "Naoki Kenmochi <kenmochi@edu.k.u-tokyo.ac.jp>"
__version__ = "0.0.0"
__date__    = "5 Feb 2018"

import numpy as np
import matplotlib.pyplot as plt
import abel

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

    def abelic_uneven_dr(self, nl, sight_line):
        """
        以下の条件の計測に対するアーベル変換を行います
        計測視線：不等間隔
        計測開始：中心より外側
        計測終了：最外殻

        :param nl:
        :param sight_line:
        :return:
        """

        k_st = np.int(sight_line[0]/(sight_line[1]-sight_line[0]))
        N = sight_line.__len__()    #np.int(self.sight_line_para[-1]/dr)
        A = np.eye(N)
        for k in range(k_st+1, k_st+N+1):
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

        n = np.linalg.solve(A, nl)

        return n

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

        pol = pol_wGP['Pol730nm']

        pol_av = np.average(pol[11000:12000, :], axis = 0)
        #sightline_pol = np.linspace(0.38, 0.82, 9)
        sightline_pol = 1e-3*np.array([379, 432, 484, 535, 583, 630, 689, 745, 820])

        if(spline == True):
            dr = (sightline_pol[-1] - sightline_pol[0])/((sightline_pol.__len__()-1)*2)
            f = interpolate.interp1d(sightline_pol, pol_av, kind='cubic')
            sightline_pol = np.arange(sightline_pol[0], sightline_pol[-1]+dr, dr)
            pol_av = f(sightline_pol)

        pol_local = self.abelic_uneven_dr(pol_av, sightline_pol)
        plt.plot(sightline_pol, pol_local, label='pol_local')
        plt.plot(sightline_pol, pol_av, label='pol_av')
        plt.legend()
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

        #ne_test = np.linspace(20, 0, 20)
        #nl_y_test = np.dot(A, ne_test)
        #ne_test_2 = np.linalg.solve(A, nl_y_test)
        #plt.plot(ne_test)
        #plt.plot(ne_test_2)
        #plt.plot(nl_y_test)
        #plt.show()
        return ne, rs, rr, ne_profile_z0, nl_y_ori

    def plot_ne_nel(self, spline=False):
        ne, rs, rr, ne_profile_z0, nl_y_ori = self.abelic_ne(spline=spline)
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
        plt.plot(rr[:-1], ne[:-1], "o", color='red', label='$n_e$(reconstructed)')
        plt.legend(fontsize=10, loc='upper left')
        plt.tick_params(labelsize=12)
        plt.xlim(0, 1.0)
        plt.ylim(0, 35)
        plt.xlabel('r [m]', fontsize=16)
        plt.ylabel(r'$\mathrm{[10^{16}\,m^{-3}]}$', fontsize=16)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    abne = Abel_ne()
    #abne.plot_ne_nel(spline=True)
    abne.abelic_pol(spline=True)
