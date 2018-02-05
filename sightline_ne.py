import rt1mag as rt1
import numpy as np
from   scipy.optimize import minimize, differential_evolution
import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib as mplt
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
np.set_printoptions(linewidth=500, formatter={'float': '{: 0.3f}'.format})
import time

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

class sightline_ne(object):
    def __init__(self):
        print("Load constructor of sightline_ne")
        ###計測視線###
        self.sight_line_perp = np.array([0.42, 0.45, 0.48, 0.51, 0.54, 0.57, 0.60, 0.62, 0.70])
        #self.sight_line_para = np.arange(0.45, 0.85, 0.05)  #np.array([0.42, 0.45, 0.48, 0.51, 0.54, 0.57, 0.60, 0.63, 0.66, 0.69])
        self.sight_line_para = np.array([0.45, 0.50, 0.53, 0.60, 0.67, 0.70, 0.75, 0.78, 0.80])     #TODO test for uneven dr
        self.num_perp = self.sight_line_perp.__len__()
        self.num_para = self.sight_line_para.__len__()
        ###各計測視線でのプラズマ領域###
        self.nz = self.nx = 1000  # number of grids along the line of sight
        self.nu = self.nr = 100
        self.separatrix = True  # if pure dipole configuration, change this to 'False'
        self.gaussian = 'single'  # 'single' or 'double'
        self.z_perp_st_ed = np.array([[-0.50, 0.50],[-0.50, 0.50],[-0.50, 0.50],[-0.50, 0.50],[-0.50, 0.50],[-0.50, 0.50],[-0.50, 0.50],[-0.50, 0.50],[-0.50, 0.50]])#, [-0.27, 0.42], [-0.24, 0.39]])
        x_para_st_ed = np.zeros((self.num_para, 2))    #np.array([[0.0, 0.78], [0.0, 0.78], [0.0, 0.78], [0.0, 0.78], [0.0, 0.78], [0.0, 0.78], [0.0, 0.78], [0.0, 0.78], [0.0, 0.78], [0.0, 0.78]])
        x_para_st_ed[:,1] = 1.00    #0.78
        self.x_para_st_ed = x_para_st_ed
        self.rms = np.linspace(0.4, 0.6, 100)   # z=0における密度最大値候補 (single gaussian)
        self.w = np.array(self.rms.shape)
        self.error_at_rms = np.zeros(self.w)
        self.nl_z_mes_ori = np.array([0.0e17,0.0e17,0.0e17,0.0e17,0.0e17,0.0e17,0.0e17,0.0e17, 0.61e17, 0.42e17])
        self.nl_z_mes = self.nl_z_mes_ori*1.0e-16# normalize by 1e-16
        self.nl_y_mes_ori = np.ones(self.num_para) #np.array([2.37e17])
        self.nl_y_mes = self.nl_y_mes_ori*1.0e-16# normalize by 1e-16

        self.psi0  = rt1.psi(1.0, 0.0, self.separatrix) # psi at the vacuum chamber
        self.params = {'backend': 'pdf',
                  'axes.labelsize': 20,
                  'text.fontsize': 20,
                  'legend.fontsize': 25,
                  'xtick.labelsize': 20,
                  'ytick.labelsize': 20,
                  'text.usetex': False,
                  'font.family': 'sans-serif',
                  'axes.unicode_minus': True}

#----------------------------------------------------#
#             set IF measured values                 #
#----------------------------------------------------#

# ####計測視線###
# r620_perp = 0.62  # [m]
# r700_perp = 0.70
# r450_para = 0.45
#
#
#
# z620 = np.linspace(-0.27, 0.42, nz)
# z700 = np.linspace(-0.24, 0.39, nz)
# x450 = np.linspace( 0.0, 0.780, nx)
#
#
# nl_y450_mes_ori = 2.37e17#2.34e17 #12.62e17 #2.34e17 #2.64e17 #1.55e17 #2.77e17  #1.92e17  #* 2 * 0.780  # Xray mod
# nl_z620_mes_ori = 0.61e17#0.675e17 #3.837e17 #0.675e17 #0.94e17 #2.96e16 #0.67e17 #4.45e16  #* (0.28 + 0.44)# Xray mod
# nl_z700_mes_ori = 0.42e17#0.285e17 #3.32e17 #0.285e17 #0.425e17 #2.12e16 #0.39e17 #3.22e16  #* (0.23 + 0.39)# Xray mod
#
# nl_y450_mes = nl_y450_mes_ori*1e-16 # normalize by 1e-16
# nl_z620_mes = nl_z620_mes_ori*1e-16 # normalize by 1e-16
# nl_z700_mes = nl_z700_mes_ori*1e-16 # normalize by 1e-16

#----------------------------------------------------#
#                 setting for plot                   #
#----------------------------------------------------#
    def fmt(self, x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)


#----------------------------------------------------#
#                   def ne(r, z)                     #
#----------------------------------------------------#
    def ne_single_gaussian(self, r, z, psix, *p):
        n1, a1, b1, rm = p

        br, bz = rt1.bvec(r, z, self.separatrix)
        bb = np.sqrt(br**2 + bz**2)

        if r == 0.0:
            return n1 * np.exp(-a1 * abs((rt1.psi(r, 0.0, self.separatrix)-psix)/self.psi0)**2)

        if rt1.check_coilcase(r, z):
            return 0.0

        else:
            return n1 * np.exp(-a1*abs((rt1.psi(r, z, self.separatrix) - psix)/self.psi0)**2) * (bb/rt1.b0(r, z, self.separatrix))**(-b1)


#----------------------------------------------------#
#                   def ne(r, z)                     #
#----------------------------------------------------#
    def ne_test(self, r, z, psix, *p):      #解析的な分布でテスト
        n1, a1, b1, rm = p

        br, bz = rt1.bvec(r, z, self.separatrix)
        bb = np.sqrt(br**2 + bz**2)

        if rt1.check_coilcase(r, z):
            return 0.0

        elif(rt1.psi(r, z, self.separatrix)>=self.psi0+0.001):
            return n1*(0.8-r)
        else:
            return 0.0
#----------------------------------------------------#
#              def psi_term, B_term                  #
#----------------------------------------------------#
    def psi_term(self, r, z, psix, *p):
        n1, a1, b1, rm = p

        if rt1.check_coilcase(r, z):
            return 0.0
        else:
            return np.exp(-a1*abs((rt1.psi(r, z, self.separatrix) - psix)/self.psi0)**2)

    def B_term(self, r, z, *p):
        n1, a1, b1, rm = p

        br, bz = rt1.bvec(r, z, self.separatrix)
        bb = np.sqrt(br**2 + bz**2)

        if rt1.check_coilcase(r, z):
            return 0.0
        else:
            return (bb/rt1.b0(r, z, self.separatrix))**(-b1)

#--------------------------------------------------------------------------------------#
#             determine error between the measurement and optimization                 #
#--------------------------------------------------------------------------------------#

    def err_single_gaussian(self, p, disp_flg):
        n1, a1, b1, rm = p
        psix = rt1.psi(rm, 0.0, self.separatrix)
        #   line integral along horizontal y=45 chord

        nl_y450 = 0.0
        for i, x in enumerate(x450):
            nl_y450 = nl_y450 + np.exp(-a1*abs((psi_x450[i] - psix)/self.psi0)**2)*n1*dx450
        nl_y450 = 2.0*nl_y450
        error_y450 = (nl_y450 - nl_y450_mes)**2/(nl_y450_mes)**2


        #   line integral along vertical y=60, 70 chord
        nl_z = 0.0
        z_perp = np.linspace(z_perp_st_ed[0,0], z_perp_st_ed[0,1], nz)
        dz = z_perp[1] - z_perp[0]
        for j, z in enumerate(z_perp):
            nl_z = nl_z + n1*np.exp(-a1*abs((psi_z[j] - psix)/self.psi0)**2)*(bb[j]/b0[j])**(-b1)*dz
        error_z = (nl_z - nl_z_mes[0])**2/(nl_z_mes[0])**2

        nl_z700 = 0.0
        for j, z in enumerate(z700):
            nl_z700 = nl_z700 + n1*np.exp(-a1*abs((psi_z700[j] - psix)/self.psi0)**2)*(bb700[j]/b0700[j])**(-b1)*dz700
        error_z700 = (nl_z700 - nl_z700_mes)**2/(nl_z700_mes)**2

        error = [error_y450, error_z, error_z700]


        #    print (  'n1, a1, b1 =' , p)
        #    print (  'y400: ', nl_y400, '/', nl_y400_mes)
        #    print (  'y450: ', nl_y450, '/', nl_y450_mes)
        #    print (  'y500: ', nl_y500, '/', nl_y500_mes)
        #    print (  'y550: ', nl_y550, '/', nl_y550_mes)
        #    print (  'y620: ', nl_y620, '/', nl_y620_mes)
        #    print (  'z620: ', nl_z620, '/', nl_z620_mes)
        #    print (  'z700: ', nl_z700, '/', nl_z700_mes)
        #    print (  'z840: ', nl_z840, '/', nl_z840_mes)
        #    print ('  err = ', sum(error[4:7]))

        return sum(error[0:3])

    def elapsed_time(self, t1):
        t2 = time.time()
        print("Elapsed time is ", "{:7.3}".format(t2 - t1), " sec")

    def calc_nel(self, p_opt, psi_x, psi_z, bb, b0):
        psix = rt1.psi(p_opt[3], 0.0, self.separatrix) # psi上のBが最小となる直線上の密度最大値

        rs = np.linspace( 0.1, 1.001, 200)
        zs = np.linspace(-0.4, 0.401, 200)
        r_mesh, z_mesh = np.meshgrid(rs, zs)

        ne_profile = np.array([list(map(lambda r, z : self.ne_single_gaussian(r, z, psix, *p_opt), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
        #ne_profile = np.array([list(map(lambda r, z : self.ne_test(r, z, psix, *p_opt), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))    #解析的な分布でテスト
        # profile_zでのdensity profileの表示
        profile_z = 0.0  # プロファイルが見たい任意のz [m]
        profile_z_index = np.searchsorted(zs, profile_z)
        ne_profile_z0 = ne_profile[:][profile_z_index]

        n1, a1, b1, rm = p_opt

        nl_y = np.zeros(self.num_para)
        nl_z = np.zeros(self.num_perp)
        error_y = np.zeros(self.num_para)
        error_z = np.zeros(self.num_perp)

        for k in range(self.num_para):
            x_para = np.linspace(self.x_para_st_ed[k,0], self.x_para_st_ed[k,1], self.nx)
            dx = x_para[1] - x_para[0]
            for i, x in enumerate(x_para):
                nl_y[k] += np.exp(-a1*abs((psi_x[k, i] - psix)/self.psi0)**2)*n1*dx
                #if(psi_x[k, i]>=self.psi0+0.001):#検算のためのテスト
                #    rx = np.sqrt(x**2 + self.sight_line_para[k]**2)
                #    nl_y[k] += n1*(0.8-rx)*dx #n1*dx
                #else:
                #    nl_y[k] += 0.0
            nl_y[k] = 2.0*nl_y[k]
            error_y[k] = (nl_y[k] - self.nl_y_mes[k])**2/(self.nl_y_mes[k])**2

        #   line integral along vertical y=60, 70 chord
        for i in range(self.num_perp):
            z_perp = np.linspace(self.z_perp_st_ed[i,0], self.z_perp_st_ed[i,1], self.nz)
            dz = z_perp[1] - z_perp[0]
            for j, z in enumerate(z_perp):
                nl_z[i] += n1*np.exp(-a1*abs((psi_z[i, j] - psix)/self.psi0)**2)*(bb[i, j]/b0[i, j])**(-b1)*dz
            error_z[i] = (nl_z[i] - self.nl_z_mes[i])**2/(self.nl_z_mes[i])**2


        for i in range(self.num_para):
            print('y%2f: %5f/%5f' % (self.sight_line_para[i], nl_y[i], self.nl_y_mes[i]))
        for j in range(self.num_perp):
            print('z%2f: %5f/%5f' % (self.sight_line_perp[j], nl_z[j], self.nl_z_mes[j]))

        for i in range(self.num_para):
            print('error_y%2f: %5f' % (self.sight_line_para[i], error_y[i]))
        for j in range(self.num_perp):
            print('error_z%2f: %5f' % (self.sight_line_perp[j], error_z[j]))

        #plt.plot(self.sight_line_perp, nl_z, label='ne_perp')
        #plt.plot(self.sight_line_para, nl_y, label='ne_para')
        #plt.legend()
        #plt.show()

        return nl_y, rs, ne_profile_z0


    #def view_profile(rm, p_opt):
    def view_profile(self, p_opt, psi_x, psi_z, bb, b0):
        psix = rt1.psi(p_opt[3], 0.0, self.separatrix) # psi上のBが最小となる直線上の密度最大値
        #     Export figure
        rs = np.linspace( 0.1, 1.001, 200)
        zs = np.linspace(-0.4, 0.401, 200)
        r_mesh, z_mesh = np.meshgrid(rs, zs)

        ne_profile = np.array([list(map(lambda r, z : self.ne_single_gaussian(r, z, psix, *p_opt), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
        #ne_profile = np.array([list(map(lambda r, z : self.ne_test(r, z, psix, *p_opt), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))    #解析的な分布でテスト
        psi_term_profile = np.array([list(map(lambda r, z : self.psi_term(r, z, psix, *p_opt), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
        B_term_profile = np.array([list(map(lambda r, z : self.B_term(r, z, *p_opt), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
        psi = np.array([list(map(lambda r, z : rt1.psi(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
        coilcase_truth_table = np.array([list(map(lambda r, z : rt1.check_coilcase(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
        psi[coilcase_truth_table == True] = 0
        #np.save('ne2D_29_fled_r2', ne_profile)

        #ne_profile = np.load("ne2D_35_t10_r1.npy")
        #ne_profile_t10 = np.load("ne2D_35_t10_r1.npy")
        #ne_profile_t11 = np.load("ne2D_35_t11_r1.npy")
        #ne_profile_t15 = np.load("ne2D_35_t15_r1.npy")
        #ne_profile = ne_profile_t11 - ne_profile_t10

        # density profileの表示
        levels = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014]
        plt.figure(figsize=(8, 5))
        plt.subplot(111)
        img = plt.imshow(ne_profile, origin='lower', cmap='jet',
                         #img = plt.imshow(ne_profile, origin='lower', cmap=plt.cm.seismic,
                         #                 norm = MidpointNormalize(midpoint=0),
                         #                                  #vmin=-np.amax(ne_profile), vmax=np.amax(ne_profile),
                         vmax=37,
                         extent=(rs.min(), rs.max(), zs.min(), zs.max()))
        plt.contour(r_mesh, z_mesh, ne_profile, colors=['k'])
        plt.contour(r_mesh, z_mesh, psi, colors=['white'], levels=levels)
        plt.title(r'$n_\mathrm{e}$')
        plt.xlabel(r'$r\mathrm{\ [m]}$')
        plt.ylabel(r'$z\mathrm{\ [m]}$')
        # plt.gca():現在のAxesオブジェクトを返す
        divider = make_axes_locatable(plt.gca())
        # カラーバーの位置
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(img, cax=cax)
        #cb.set_clim(0,6.4)
        cb.set_label(r'$\mathrm{[10^{16}\,m^{-3}]}$')
        plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
        plt.show()

        # psi term profileの表示
        plt.figure(figsize=(8, 5))
        plt.subplot(111)
        img = plt.imshow(psi_term_profile, origin='lower', cmap='plasma',
                         extent=(rs.min(), rs.max(), zs.min(), zs.max()))
        plt.contour(r_mesh, z_mesh, psi_term_profile, colors=['k'])
        plt.contour(r_mesh, z_mesh, psi, colors=['white'], levels=levels)
        plt.title(r'$\psi term$')
        plt.xlabel(r'$r\mathrm{\ [m]}$')
        plt.ylabel(r'$z\mathrm{\ [m]}$')
        # plt.gca():現在のAxesオブジェクトを返す
        divider = make_axes_locatable(plt.gca())
        # カラーバーの位置
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(img, cax=cax)
        cb.set_clim(0,6.4)
        cb.set_label(r'$\mathrm{[10^{16}\,m^{-3}]}$')
        plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
        plt.show()

        # B term profileの表示
        plt.figure(figsize=(8, 5))
        plt.subplot(111)
        img = plt.imshow(B_term_profile, origin='lower', cmap='plasma',
                         extent=(rs.min(), rs.max(), zs.min(), zs.max()))

        plt.contour(r_mesh, z_mesh, B_term_profile, colors=['k'])
        plt.contour(r_mesh, z_mesh, psi, colors=['white'], levels=levels)
        plt.title(r'$B term$')
        plt.xlabel(r'$r\mathrm{\ [m]}$')
        plt.ylabel(r'$z\mathrm{\ [m]}$')
        # plt.gca():現在のAxesオブジェクトを返す
        divider = make_axes_locatable(plt.gca())
        # カラーバーの位置
        cax = divider.append_axes("right", "5%", pad="3%")
        cb = plt.colorbar(img, cax=cax)
        cb.set_clim(0,6.4)
        cb.set_label(r'$\mathrm{[10^{16}\,m^{-3}]}$')
        plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
        plt.show()

        # profile_zでのdensity profileの表示
        profile_z = 0.0  # プロファイルが見たい任意のz [m]
        profile_z_index = np.searchsorted(zs, profile_z)
        ne_profile_z0 = ne_profile[:][profile_z_index]

        fig, ax = plt.subplots(1)
        ax.plot(rs, ne_profile_z0)
        plt.draw()
        plt.show()


    def calc_ne(self):
        mpl.rcParams.update(self.params)

        print('Running with view mode')
        #rm_best = 0.5424
        #p_opt_best = [31.182,  4.464,  0.782,  0.516]
        #p_opt_best = [35.389,  6.629,  1.800,  0.549]
        #p_opt_best = [29.241,  4.295,  1.698,  0.550]
        p_opt_best = [28.227,  5.897,  1.903,  0.550]
        #p_opt_best = [1.0, 0.0, 0.0, 0.0]   #検算のための値


        t_start = time.time()
        print('')
        print(time.ctime())
        print('')


        #     calculate psi, bb, b0 along the line of sights beforehand
        psi_x = np.zeros((self.num_para, self.nx))
        psi_z = np.zeros((self.num_perp, self.nz))
        bb = np.zeros((self.num_perp, self.nz))
        b0 = np.zeros((self.num_perp, self.nz))

        for k in range(self.num_para):
            x_para = np.linspace(self.x_para_st_ed[k,0], self.x_para_st_ed[k,1], self.nx)
            for i, x in enumerate(x_para):
                rx = np.sqrt(x**2 + self.sight_line_para[k]**2)
                psi_x[k, i] = rt1.psi(rx, 0.0, self.separatrix)

        for i in range(self.num_perp):
            z_perp = np.linspace(self.z_perp_st_ed[i,0], self.z_perp_st_ed[i,1], self.nz)
            for j, z in enumerate(z_perp):
                psi_z[i, j] = rt1.psi(self.sight_line_perp[i], z, self.separatrix)
                br, bz = rt1.bvec(self.sight_line_perp[i], z, self.separatrix)
                bb[i, j] = np.sqrt(br**2 + bz**2)
                b0[i, j] = rt1.b0(self.sight_line_perp[i], z, self.separatrix)

        print('                                      ')
        print('      start 1st optimization....      ')
        print('                                      ')
        err_max = 1e10

        print('')
        self.elapsed_time(t_start)
        print('')
        #psix  = rt1.psi(rm_best, 0.0, separatrix)
        psix  = rt1.psi(p_opt_best[3], 0.0, self.separatrix)

        #view_profile(rm_best, p_opt_best)
        #self.view_profile(p_opt_best, psi_x, psi_z, bb, b0)
        nl_y = self.calc_nel(p_opt_best, psi_x, psi_z, bb, b0)
        return nl_y



if __name__ == '__main__':
    sl_ne = sightline_ne()
    sl_ne.calc_ne()
