"""
Created on Fri Nov  17 21:03:10 2017

@author: nishiura
"""

import rt1mag as rt1
import numpy as np
import physical_constant as pc
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

gaussian = 'single'  # 'single' or 'double'

#----------------------------------------------------#
#             set IF measured values                 #
#----------------------------------------------------#

###計測視線###
r620_perp = 0.62  # [m]
r700_perp = 0.70
r450_para = 0.45


###各計測視線でのプラズマ領域###
nz = nx = 1000  # number of grids along the line of sight
nu = nr = 100
separatrix = True  # if pure dipole configuration, change this to 'False'

z620 = np.linspace(-0.27, 0.42, nz)
z700 = np.linspace(-0.24, 0.39, nz)
x450 = np.linspace( 0.0, 0.780, nx)

rms = np.linspace(0.4, 0.6, 100)   # z=0における密度最大値候補 (single gaussian)
w = np.array(rms.shape)
error_at_rms = np.zeros(w)

nl_y450_mes_ori = 2.832228e17#2.828068e17#2.85e17#2.80e17#2.54e17#3.02e17#2.37e17#2.34e17 #12.62e17 #2.34e17 #2.64e17 #1.55e17 #2.77e17  #1.92e17  #* 2 * 0.780  # Xray mod
nl_z620_mes_ori = 0.821236e17#0.836155e17#0.87e17#0.78e17#0.77e17#0.97e17#1.00e17#0.61e17#0.675e17 #3.837e17 #0.675e17 #0.94e17 #2.96e16 #0.67e17 #4.45e16  #* (0.28 + 0.44)# Xray mod
nl_z700_mes_ori = 0.468873e17#0.479757e17#0.48e17#0.50e17#0.51e17#0.52e17#0.53e17#0.42e17#0.285e17 #3.32e17 #0.285e17 #0.425e17 #2.12e16 #0.39e17 #3.22e16  #* (0.23 + 0.39)# Xray mod

nl_y450_mes = nl_y450_mes_ori*1e-16 # normalize by 1e-16
nl_z620_mes = nl_z620_mes_ori*1e-16 # normalize by 1e-16
nl_z700_mes = nl_z700_mes_ori*1e-16 # normalize by 1e-16

#----------------------------------------------------#
#                 setting for plot                   #
#----------------------------------------------------#
def fmt(x, pos):
	a, b = '{:.2e}'.format(x).split('e')
	b = int(b)
	return r'${} \times 10^{{{}}}$'.format(a, b)

params = {'backend': 'pdf',
          'axes.labelsize': 20,
          'text.fontsize': 20,
          'legend.fontsize': 25,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'text.usetex': False,
			'font.family': 'sans-serif',
          'axes.unicode_minus': True}
mpl.rcParams.update(params)

#----------------------------------------------------#
#                   def ne(r, z)                     #
#----------------------------------------------------#
def ne_single_gaussian(r, z, *p):
	n1, a1, b1, rm = p

	br, bz = rt1.bvec(r, z, separatrix)
	bb = np.sqrt(br**2 + bz**2)

	if r == 0.0:
		return n1 * np.exp(-a1 * abs((rt1.psi(r, 0.0, separatrix)-psix)/psi0)**2)

	if rt1.check_coilcase(r, z):
		return 0.0
	else:
		return n1 * np.exp(-a1*abs((rt1.psi(r, z, separatrix) - psix)/psi0)**2) * (bb/rt1.b0(r, z, separatrix))**(-b1)


#----------------------------------------------------#
#              def psi_term, B_term                  #
#----------------------------------------------------#    
def psi_term(r, z, *p):
	n1, a1, b1, rm = p

	if rt1.check_coilcase(r, z):
		return 0.0
	else:
		return np.exp(-a1*abs((rt1.psi(r, z, separatrix) - psix)/psi0)**2)

def B_term(r, z, *p):
	n1, a1, b1, rm = p

	br, bz = rt1.bvec(r, z, separatrix)
	bb = np.sqrt(br**2 + bz**2)

	if rt1.check_coilcase(r, z):
		return 0.0
	else:
		return (bb/rt1.b0(r, z, separatrix))**(-b1)

#--------------------------------------------------------------------------------------#
#             determine error between the measurement and optimization                 #
#--------------------------------------------------------------------------------------#

def err_single_gaussian(p, disp_flg):
    n1, a1, b1, rm = p
    psix  = rt1.psi(rm, 0.0, separatrix)
    #   line integral along horizontal y=45 chord

    nl_y450 = 0.0
    for i, x in enumerate(x450):
        nl_y450 = nl_y450 + np.exp(-a1*abs((psi_x450[i] - psix)/psi0)**2)*n1*dx450
    nl_y450 = 2.0*nl_y450
    error_y450 = (nl_y450 - nl_y450_mes)**2/(nl_y450_mes)**2
    

    #   line integral along vertical y=60, 70 chord
    nl_z620 = 0.0
    for j, z in enumerate(z620):
        nl_z620 = nl_z620 + n1*np.exp(-a1*abs((psi_z620[j] - psix)/psi0)**2)*(bb620[j]/b0620[j])**(-b1)*dz620
    error_z620 = (nl_z620 - nl_z620_mes)**2/(nl_z620_mes)**2

    nl_z700 = 0.0
    for j, z in enumerate(z700):
        nl_z700 = nl_z700 + n1*np.exp(-a1*abs((psi_z700[j] - psix)/psi0)**2)*(bb700[j]/b0700[j])**(-b1)*dz700
    error_z700 = (nl_z700 - nl_z700_mes)**2/(nl_z700_mes)**2

    error = [error_y450, error_z620, error_z700]
    

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

def elapsed_time(t1):
    t2 = time.time()
    print("Elapsed time is ", "{:7.3}".format(t2 - t1), " sec")
    
#def view_profile(rm, p_opt):
def view_profile(p_opt):
    mpl.rcParams['font.family'] = 'Arial'
    psix  = rt1.psi(p_opt[3], 0.0, separatrix) # psi上のBが最小となる直線上の密度最大値
    n1, a1, b1, rm = p_opt

    nl_y450 = 0.0
    nl_z620 = 0.0
    nl_z700 = 0.0
    
    nl_y450 = 0.0
    for i, x in enumerate(x450):
        nl_y450 = nl_y450 + np.exp(-a1*abs((psi_x450[i] - psix)/psi0)**2)*n1*dx450
    nl_y450 = 2.0*nl_y450
    error_y450 = (nl_y450 - nl_y450_mes)**2/(nl_y450_mes)**2
   
    #   line integral along vertical y=60, 70 chord
    nl_z620 = 0.0
    for j, z in enumerate(z620):
        nl_z620 = nl_z620 + n1*np.exp(-a1*abs((psi_z620[j] - psix)/psi0)**2)*(bb620[j]/b0620[j])**(-b1)*dz620
    error_z620 = (nl_z620 - nl_z620_mes)**2/(nl_z620_mes)**2

    nl_z700 = 0.0
    for j, z in enumerate(z700):
        nl_z700 = nl_z700 + n1*np.exp(-a1*abs((psi_z700[j] - psix)/psi0)**2)*(bb700[j]/b0700[j])**(-b1)*dz700
    error_z700 = (nl_z700 - nl_z700_mes)**2/(nl_z700_mes)**2

    print (  'y450: ', nl_y450, '/', nl_y450_mes)
    print (  'z620: ', nl_z620, '/', nl_z620_mes)
    print (  'z700: ', nl_z700, '/', nl_z700_mes)

    print (  'error_y450: ', error_y450)
    print (  'error_z620: ', error_z620)
    print (  'error_z700: ', error_z700)

    #     Export figure
    rs = np.linspace( 0.1, 1.001, 200)
    zs = np.linspace(-0.4, 0.401, 200)
    r_mesh, z_mesh = np.meshgrid(rs, zs)

    ne_profile = np.array([list(map(lambda r, z : ne_single_gaussian(r, z, *p_opt_best), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
    psi_term_profile = np.array([list(map(lambda r, z : psi_term(r, z, *p_opt_best), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
    B_term_profile = np.array([list(map(lambda r, z : B_term(r, z, *p_opt_best), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
    psi = np.array([list(map(lambda r, z : rt1.psi(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
    coilcase_truth_table = np.array([list(map(lambda r, z : rt1.check_coilcase(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
    psi[coilcase_truth_table == True] = 0
    #np.save('ne2D_20180223_52_r2', ne_profile)

    ne2D = np.load("ne2D_rs_nez0_20180223_52_r2_v2.npz")
    ne_profile = ne2D["ne_profile"]
    #ne_profile = np.load("ne2D_20180223_52_r2.npy")
    #ne_profile_t10 = np.load("ne2D_35_t10_r1.npy")
    #ne_profile_t11 = np.load("ne2D_35_t11_r1.npy")
    #ne_profile_t15 = np.load("ne2D_35_t15_r1.npy")
    #ne_profile_t16 = np.load("ne2D_20180223_64_t164_r2.npy")
    #ne_profile_t10 = np.load("ne2D_20180223_64_t10_r2.npy")
    #ne_profile = ne_profile_t11


    # density profileの表示
    levels = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014]
    plt.figure(figsize=(8, 5))
    plt.subplot(111)
    img = plt.imshow(ne_profile, origin='lower', cmap='jet',
    #img = plt.imshow(ne_profile, origin='lower', cmap=plt.cm.seismic,
                     #norm = MidpointNormalize(midpoint=0),
                                      #vmin=-np.amax(ne_profile), vmax=np.amax(ne_profile),
                     vmax=36,
                     extent=(rs.min(), rs.max(), zs.min(), zs.max()))
    plt.contour(r_mesh, z_mesh, ne_profile, colors=['k'], linewidth=0.5)
    plt.contour(r_mesh, z_mesh, psi, colors=['white'], linewidths=0.5, levels=levels)

    #< O - cutoffとR - cutoffに関するコード >
    mag_strength = np.array(
        [list(map(lambda r, z: np.sqrt(rt1.bvec(r, z, separatrix)[0] ** 2 + rt1.bvec(r, z, separatrix)[1] ** 2),
                  r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
    ne_IF_profile = ne_profile * 1e16
    omega_ce = pc.q * mag_strength / pc.me
    omega_pe = 4 * np.pi * ne_IF_profile * pc.q ** 2 / pc.me
    omega_R = omega_ce / 2 + ((omega_ce / 2) ** 2 + omega_pe ** 2) ** (1 / 2)
    f_R = omega_R / (2 * np.pi)
    f_O = np.sqrt(pc.q ** 2 * ne_IF_profile / (pc.epsilon_0 * pc.me)) / (2 * np.pi)
    levels_cutoff = [2.45]
    #plt.contour(r_mesh, z_mesh, f_R / 1e9, colors='lime', linewidths=2, levels=levels_cutoff)
    #plt.contour(r_mesh, z_mesh, f_O / 1e9, colors='lime', linewidths=2, linestyles='dashed', levels=levels_cutoff)
    plt.contour(r_mesh, z_mesh, f_O / 1e9, colors='lime', linewidths=2, levels=levels_cutoff)

    #< 電子サイクロトロン共鳴面に関するコード >
    levels_resonance = [875e-4]
    levels_resonance_2nd = [875e-4 / 2]
    plt.contour(r_mesh, z_mesh, mag_strength, colors='red', linewidths=3, levels=levels_resonance)
    ##plt.contour(r_mesh, z_mesh, mag_strength, colors='red', linewidths=3, linestyles='dashed',
    #            levels=levels_resonance_2nd)

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
    plt.savefig("ne2D_rs_nez0_20180223_52_r2_v2.eps", format='eps', dpi=600)

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
    #np.savez_compressed('ne2D_rs_nez0_20180223_52_r2_v3', ne_profile=ne_profile, rs=rs, ne_profile_z0=ne_profile_z0)

    fig, ax = plt.subplots(1)
    ax.plot(rs, ne_profile_z0)
    ax.plot(0.45*np.ones(3), np.array([1.06, 1.28, 1.50]))
    plt.draw()
    plt.show()
    
if __name__ == '__main__':
    view_mode = True

    if view_mode is True:
        print('Running with view mode')
        #rm_best = 0.5424
        #p_opt_best = [35.389,  6.629,  1.800,  0.549]
        #p_opt_best = [29.241,  4.295,  1.698,  0.550]
        #p_opt_best = [28.227,  4.897,  1.903,  0.542]
        #p_opt_best = [31.182,  4.464,  0.782,  0.516]
        p_opt_best = [35.210, 6.341, 1.501, 0.544]


    t_start = time.time()
    print('')
    print(time.ctime())
    print('')

    dx450 = x450[1] - x450[0]
    dz620 = z620[1] - z620[0]
    dz700 = z700[1] - z700[0]

    #     calculate psi, bb, b0 along the line of sights beforehand
    psi0  = rt1.psi(1.0, 0.0, separatrix) # psi at the vacuum chamber
    psi_x450 = np.zeros(len(x450))
    psi_z620 = np.zeros(len(z620))
    psi_z700 = np.zeros(len(z700))
    bb620  = np.zeros(len(z620))
    bb700  = np.zeros(len(z700))
    b0620  = np.zeros(len(z620))
    b0700  = np.zeros(len(z700))

    for i, x in enumerate(x450):
        rx = np.sqrt(x**2 + r450_para**2)
        psi_x450[i] = rt1.psi(rx, 0.0, separatrix)

    for j, z in enumerate(z620):
    	psi_z620[j] = rt1.psi(r620_perp, z, separatrix)
    	br, bz   = rt1.bvec(r620_perp, z, separatrix)
    	bb620[j]  = np.sqrt(br**2 + bz**2)
    	b0620[j]  = rt1.b0(r620_perp, z, separatrix)

    for j, z in enumerate(z700):
    	psi_z700[j] = rt1.psi(r700_perp, z, separatrix)
    	br, bz   = rt1.bvec(r700_perp, z, separatrix)
    	bb700[j]  = np.sqrt(br**2 + bz**2)
    	b0700[j]  = rt1.b0(r700_perp, z, separatrix)

    print ('                                      ')
    print ('      start 1st optimization....      ')
    print ('                                      ')
    err_max = 1e10

    if not(view_mode is True):
        if gaussian == 'single':

            initial_guess = [24.0, 10.0, 2.0, 0.5]  # n1, a1, b1
            bounds = [(0.1, 500.0), (1.0, 100.0), (0.01, 2.0), (0.4, 0.6)]
        #    initial_guess = [24.0, 100, 8.0]  # n1, a1, b1
        #    bounds = [(0.1, 100.0), (1.0, 1000.0), (0.1, 40.0)]

#            for i, rm in enumerate(rms):
#                print('rm = ', rm)
                # psixはグローバル変数。メソッドerrの中で用いる
#               psix  = rt1.psi(rm, 0.0, separatrix) # psi上のBが最小となる直線上の密度最大値
                #     optimization by least square method
                # メソッドerrの返り値errが最小となるようなn1, a1, b1の値を求める
        #        result = minimize(err_single_gaussian, # minimize wrt the noisy data
        #                          initial_guess,
        #                          bounds=bounds,
        #                          tol=1e-7,
        #                          args=tuple([False]),
        #                          #method="Nelder-Mead",
        #                          # method="Powell",
        #                          # method="CG",        # this method supports eps
        #                          # method="BFGS",      # this method supports eps
        #                          # method="Newton-CG", # this method supports eps
        #                          # method="Anneal",
        #                          method="L-BFGS-B",  # this method supports bounds
        #                          # method="TNC",       # this method supports bounds
        #                          # method="COBYLA",
        #                          # method="SLSQP",     # this method supports bounds and eps(single)
        #                          # method="dogleg",
        #                          # method="trust-ncg",
        #                          # options={'eps':eps, 'disp':True}
        #                          # options={'maxiter':10000,'disp':True}
        #                          )
                #result = differential_evolution(err_single_gaussian, bounds, args=tuple([False]))
            result = differential_evolution(err_single_gaussian, bounds, args=tuple([False]))

            p_opt = result.x
            err_ = err_single_gaussian(p_opt, True)
            print('error =', err_)
            if err_ < err_max:
                err_max = err_
                p_opt_best = p_opt


            print (' -------------------------------------')
            print ('|     1st optimization finished      |')
            print (' -------------------------------------')
            print ('initial guess[n1, a1, b1, rm] = ', initial_guess)
            print ('bounds [n1, a1, b1, rm]       = ', bounds)
            print ('fitted       [n1, a1, b1, rm] = ', p_opt_best)
            print ('error', err_max)
            #print ('rm', rm_best)
            print ('rm', p_opt_best[3])

    print('')
    elapsed_time(t_start)
    print('')
    #psix  = rt1.psi(rm_best, 0.0, separatrix)
    psix  = rt1.psi(p_opt_best[3], 0.0, separatrix)

    #view_profile(rm_best, p_opt_best)
    view_profile(p_opt_best)

    