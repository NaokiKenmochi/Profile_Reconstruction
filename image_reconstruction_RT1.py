import numpy as np
import matplotlib.pyplot as plt
import ne_profile_r2
import rt1mag as rt1
import time
import pandas
import codecs
import cv2
import os
import matplotlib.colors
import csv
#import numba
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import optimize
from scipy.interpolate import RectBivariateSpline, interp1d
from bokeh.plotting import figure, output_file, show, reset_output
from multiprocessing import Pool
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 5

def unwrap_self_plot_projection_image_spline_wrt_1reflection_MP(arg, **kwarg):
    return ImageReconstruction.plot_projection_image_spline_wrt_1reflection_MP(*arg, **kwarg)

class ImageReconstruction:
    def __init__(self, n, k):
        #self.p_opt_best = [35.210, 6.341, 1.501, 0.544]
        self.n = n  #Refractive index
        self.k = k  #Extinction coefficient
        self.p_opt_best = [30, 18, 1.0, 0.5]
        self.R_cam = 1.34439317 #[m]
        self.r_num = 100
        self.z_num = 100
        self.theta_max = np.deg2rad(44)#50.23)#np.arcsin(1/self.R_cam)#np.pi/3
        self.phi_max = self.theta_max/2#np.pi/6
        print("Initialized\nRefractive index: %.4f, Extinction coefficient: %.4f\nPosition of detector: R=%4f[m]\
                \nDivision number, r: %d, z:%d\nTheta max: %.2f [rad], Phi max: %.2f" \
                % (self.n, self.k, self.R_cam, self.r_num, self.z_num, self.theta_max, self.phi_max))

    def projection_poroidally(self, theta, phi):
        dist_from_cam = np.linspace(0, 3.0, 200)
        r = np.sqrt(self.R_cam**2 + (dist_from_cam*np.cos(phi))**2 - 2*self.R_cam*dist_from_cam*np.cos(phi)*np.cos(theta))
        z = dist_from_cam*np.sin(phi)
        i = 0
        while rt1.check_coilcase(r[i], z[i]) == False and i<len(r)-1:
            i += 1
        else:
            r[i:] = self.R_cam
            z[i:] = 0.0


        psix  = rt1.psi(self.p_opt_best[3], 0.0, separatrix=True) # psi上のBが最小となる直線上の密度最大値
        psi0 = rt1.psi(1.0, 0.0, separatrix=True) # psi at the vacuum chamber
        ne = np.array([list(map(lambda r, z: ne_profile_r2.ne_single_gaussian(r, z, *self.p_opt_best, psix=psix, psi0=psi0), r, z))])

        return r, z, np.sum(ne)

    def projection_poroidally_wrt_1reflection(self, theta, phi):
        dist_from_cam = np.linspace(0, 4.0, 200)
        r = np.sqrt(self.R_cam**2 + (dist_from_cam*np.cos(phi))**2 - 2*self.R_cam*dist_from_cam*np.cos(phi)*np.cos(theta))
        z = dist_from_cam*np.sin(phi)
        i = 0
        j = 0
        while rt1.check_coilcase(r[i], z[i]) == False and rt1.check_vacuum_vessel(r[i], z[i]) == False and i<len(r)-1:
            i += 1
        else:
            i += 1

        while rt1.check_coilcase(r[i], z[i]) == False and rt1.check_vacuum_vessel(r[i], z[i]) == True and i<len(r)-1:
            i += 1
        else:
            j = i
            r[i] = r[i-1]

        while rt1.check_coilcase(r[i], z[i]) == False and rt1.check_vacuum_vessel(r[i], z[i]) == True and i<len(r)-1:
            r[i+1] = r[j-(i-j)-1]
            i += 1
        else:
            r[i:] = self.R_cam
            z[i:] = 0.0


        psix  = rt1.psi(self.p_opt_best[3], 0.0, separatrix=True) # psi上のBが最小となる直線上の密度最大値
        psi0 = rt1.psi(1.0, 0.0, separatrix=True) # psi at the vacuum chamber
        ne = np.array([list(map(lambda r, z: ne_profile_r2.ne_single_gaussian(r, z, *self.p_opt_best, psix=psix, psi0=psi0), r, z))])

        return r, z, np.sum(ne)

    def bokeh_local_image(self):
        output_file("local_image.html")
        levels = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014]
        #r = np.linspace(0.1, 1, self.r_num)
        #z = np.linspace(-0.4, 0.4, self.z_num)
        r = np.linspace(0.0, 1, self.r_num)
        z = np.linspace(-0.5, 0.5, self.z_num)
        r_mesh, z_mesh = np.meshgrid(r, z)
        psi = np.array([list(map(lambda r, z : rt1.psi(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))
        coilcase_truth_table = np.array([list(map(lambda r, z : rt1.check_coilcase(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))
        psi[coilcase_truth_table == True] = 0
        ne = self.make_ne_image(r, z)
        f = figure(x_range=(0, 1), y_range=(-0.5, 0.5))
        f.image(image=[ne], x=0, y=-0.5, dw=1, dh=1, palette='Spectral11')
        show(f)

    def plot_local_image(self, p_opt_best, p_opt_best_2ndPeak):
        #levels = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014]
        levels = [0.005, 0.006, 0.0063767, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014]
        #r = np.linspace(0.1, 1, self.r_num)
        #z = np.linspace(-0.4, 0.4, self.z_num)
        r = np.linspace(0.0, 1, self.r_num)
        z = np.linspace(-0.5, 0.5, self.z_num)
        r_mesh, z_mesh = np.meshgrid(r, z)
        psi = np.array([list(map(lambda r, z : rt1.psi(r, z, separatrix=True), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))
        coilcase_truth_table = np.array([list(map(lambda r, z : rt1.check_coilcase(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))
        psi[coilcase_truth_table == True] = 0
        ne = self.make_ne_image(r, z, p_opt_best, p_opt_best_2ndPeak)
        #plt.imshow(ne, cmap='jet', origin='lower', extent=(r.min(), r.max(), z.min(), z.max()), vmax=35)
        plt.imshow(ne, cmap='jet', origin='lower', extent=(r.min(), r.max(), z.min(), z.max()))
        plt.colorbar()
        plt.contour(r_mesh, z_mesh, psi, colors=['white'], linewidths=0.5, levels=levels)
        plt.title(r'$n_\mathrm{e}$')
        plt.xlabel(r'$r\mathrm{\ [m]}$')
        plt.ylabel(r'$z\mathrm{\ [m]}$')
        plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
        plt.show()

    def make_ne_image(self, r, z, p_opt_best, p_opt_best_2ndPeak):
        r_mesh, z_mesh = np.meshgrid(r, z)
        psi = np.array([list(map(lambda r, z : rt1.psi(r, z, separatrix=True), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))
        #coilcase_truth_table = np.array([list(map(lambda r, z : rt1.check_coilcase(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))
        #psi[coilcase_truth_table == True] = 0
        psix  = rt1.psi(p_opt_best[3], 0.0, separatrix=True) # psi上のBが最小となる直線上の密度最大値
        psi0  = rt1.psi(1.0, 0.0, separatrix=True) # psi at the vacuum chamber
        #psi0 = 0.002
        ne = np.array([list(map(lambda r, z : ne_profile_r2.ne_single_gaussian(r, z, *p_opt_best, psix=psix, psi0=psi0, bb_limit=True), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))

        # For double peak profile
        psix  = rt1.psi(p_opt_best_2ndPeak[3], 0.0, separatrix=True) # psi上のBが最小となる直線上の密度最大値
        ne += np.array([list(map(lambda r, z : ne_profile_r2.ne_single_gaussian(r, z, *p_opt_best_2ndPeak, psix=psix, psi0=psi0, bb_limit=True), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))

        # For third peak profile
        #p_opt_best_3rdPeak=[0.7, 7, 1.00, 0.55]
        #psix  = rt1.psi(p_opt_best_3rdPeak[3], 0.0, separatrix=True) # psi上のBが最小となる直線上の密度最大値
        #ne += np.array([list(map(lambda r, z : ne_profile_r2.ne_single_gaussian(r, z, *p_opt_best_3rdPeak, psix=psix, psi0=psi0, bb_limit=True), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))
        #ne = np.where(psi > 0.006375, ne, 0.0)
        #ne[psi < 0.006390] = 0.0
        #ne[psi < 0.006600] = 0.0
        #ne[psi < 0.0063767] = 0.0
        #ne[ne > 1.15] = 0.0

        return ne

    def spline_image(self):
        r = np.linspace(0.1, 1, self.r_num)
        z = np.linspace(-0.4, 0.4, self.z_num)
        image = self.make_ne_image(r, z)
        interp_image = RectBivariateSpline(z, r, image)
        r_2 = np.linspace(0.1, 1, 10*self.r_num)
        z_2 = np.linspace(-0.4, 0.4, 10*self.z_num)
        image_2 = interp_image(z_2, r_2)
        plt.imshow(image_2, cmap='jet', origin='lower', extent=(r_2.min(), r_2.max(), z_2.min(), z_2.max()))
        plt.colorbar()
        plt.title(r'$n_\mathrm{e}$')
        plt.xlabel(r'$r\mathrm{\ [m]}$')
        plt.ylabel(r'$z\mathrm{\ [m]}$')
        plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
        plt.show()

    def plot_projection(self):
        for j in range(self.r_num):
            for i in range(self.z_num):
                #r,z,_ = self.projection_poroidally(self.theta_max*j/self.r_num, self.phi_max - 2*self.phi_max*i/self.z_num)
                r,z,_ = self.projection_poroidally_wrt_1reflection(self.theta_max*j/self.r_num, self.phi_max - 2*self.phi_max*i/self.z_num)
                plt.plot(r, z, '.', markersize=3)
                #plt.plot(r, z)

        levels = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014]
        r = np.linspace(0.1, 1, 20*self.r_num)
        z = np.linspace(-0.4, 0.4, 20*self.z_num)
        r_mesh, z_mesh = np.meshgrid(r, z)
        psi = np.array([list(map(lambda r, z : rt1.psi(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))
        coilcase_truth_table = np.array([list(map(lambda r, z : rt1.check_coilcase(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))
        psi[coilcase_truth_table == True] = 0
        plt.contour(r_mesh, z_mesh, psi, colors=['black'], linewidths=0.5, levels=levels)
        plt.xlim(0, 1.2)
        plt.ylim(-0.4, 0.4)
        plt.xlabel(r'$r\mathrm{\ [m]}$')
        plt.ylabel(r'$z\mathrm{\ [m]}$')
        plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)

        plt.show()

    def plot_projection_image(self):
        image=np.zeros((self.r_num, self.z_num))
        for j in range(self.r_num):
            for i in range(self.z_num):
                #_,_,image[i, j] = self.projection_poroidally(self.theta_max*j/self.r_num, self.phi_max - 2*self.phi_max*i/self.z_num)
                _,_,image[i, j] = self.projection_poroidally_wrt_1reflection(self.theta_max*j/self.r_num, self.phi_max - 2*self.phi_max*i/self.z_num)

        plt.imshow(image, cmap='jet')
        plt.show()

    def plot_projection_image_spline_wrt_1reflection_v3(self, reflection_factor=1.0):
        #TODO   for文の中では視線の2次元行列をつくり，interp_imageはその2次元配列に対して一度にかける．それをsumなどを用いて積分する
        dist_from_cam = np.linspace(0, 4.0, 400)

        r_image = np.linspace(0, 1, self.r_num)
        z_image = np.linspace(-0.5, 0.5, self.z_num)
        image = self.make_ne_image(r_image, z_image)
        interp_image = RectBivariateSpline(z_image, r_image, image)
        image = np.zeros((self.r_num, self.z_num))

        theta = self.theta_max*np.arange(self.r_num)/self.r_num
        phi = self.phi_max - 2*self.phi_max*np.arange(self.z_num)/self.z_num
        z_sightline = np.ones((self.r_num, self.z_num, dist_from_cam.__len__()))
        r_sightline = np.ones((self.r_num, self.z_num, dist_from_cam.__len__()))
        z_sightline = z_sightline*np.sin(phi)[np.newaxis, :, np.newaxis]*dist_from_cam[np.newaxis, np.newaxis, :]
        r_sightline = np.sqrt(r_sightline*self.R_cam**2 + r_sightline*(np.cos(phi)[:, np.newaxis, np.newaxis]*dist_from_cam[np.newaxis, np.newaxis, :])**2 - 2*r_sightline*self.R_cam*np.cos(phi)[:, np.newaxis, np.newaxis]*np.cos(theta)[:, np.newaxis, np.newaxis]*dist_from_cam[np.newaxis, np.newaxis, :])
        for i in range(self.r_num):
            for j in range(self.z_num):
                i_sightline = 0
                buf = 0.0
                while rt1.check_coilcase(r_sightline[i, j, i_sightline], z_sightline[i, j, i_sightline]) == False and rt1.check_vacuum_vessel(r_sightline[i, j, i_sightline], z_sightline[i, j, i_sightline]) == False and i_sightline<dist_from_cam.__len__()-1:
                    i_sightline += 1
                else:
                    i_sightline += 1

                while rt1.check_coilcase(r_sightline[i, j, i_sightline], z_sightline[i, j, i_sightline]) == False and rt1.check_vacuum_vessel(r_sightline[i, j, i_sightline], z_sightline[i, j, i_sightline]) == True and i_sightline<dist_from_cam.__len__()-1:
                    i_sightline += 1
                    buf += interp_image(z_sightline[i, j, i_sightline], r_sightline[i, j, i_sightline])
                else:
                    j_sightline = i_sightline
                    r_sightline[i, j, i_sightline] = r_sightline[i, j, i_sightline-1]

                while rt1.check_coilcase(r_sightline[i, j, i_sightline], z_sightline[i, j, i_sightline]) == False and rt1.check_vacuum_vessel(r_sightline[i, j, i_sightline], z_sightline[i, j, i_sightline]) == True and i_sightline<dist_from_cam.__len__()-1:
                    r_sightline[i, j, i_sightline+1] = r_sightline[i, j, j_sightline-(i_sightline-j_sightline)-1]
                    i_sightline += 1
                    injection_angle = self.cal_injection_angle(dist_from_cam[i_sightline], z_sightline[i, j, i_sightline])
                    if injection_angle < np.pi:
                        R_s, R_p = self.cal_refractive_indices_metal(injection_angle, 2.6580, 2.8125)
                        reflection_factor = (R_s + R_p)/2
                    else:
                        reflection_factor = 0.0

                    test = interp_image(z_sightline[i, j, i_sightline], r_sightline[i, j, j_sightline-(i_sightline-j_sightline)-1]) * reflection_factor
                    buf += interp_image(z_sightline[i, j, i_sightline], r_sightline[i, j, j_sightline-(i_sightline-j_sightline)-1]) * reflection_factor
                else:
                    r_sightline[i, j, i_sightline:] = self.R_cam
                    z_sightline[i, j, i_sightline:] = 0.0
                image[j, i] = buf

        plt.title('Non-polarized, Reflection factor = %.1f' % reflection_factor)
        plt.imshow(image, cmap='jet')
        plt.savefig("integrated_image_test_v3.png")

    def plot_projection_image_spline_wrt_1reflection_v2(self, reflection_factor=1.0):
        dist_from_cam = np.linspace(0, 4.0, 400)

        r_image = np.linspace(0, 1, self.r_num)
        z_image = np.linspace(-0.5, 0.5, self.z_num)
        image = self.make_ne_image(r_image, z_image)
        #plt.imshow(image, cmap='jet')
        #plt.show()
        interp_image = RectBivariateSpline(z_image, r_image, image)
        image = np.zeros((self.r_num, self.z_num))
        for j in range(self.r_num):
            for i in range(self.z_num):
                theta = self.theta_max*j/self.r_num
                phi = self.phi_max - 2*self.phi_max*i/self.z_num
                r_sightline = np.sqrt(self.R_cam**2 + (dist_from_cam*np.cos(phi))**2 - 2*self.R_cam*dist_from_cam*np.cos(phi)*np.cos(theta))
                z_sightline = dist_from_cam*np.sin(phi)
                i_sightline = 0
                flg_reflection = 0
                buf = 0.0
                for _ in range(dist_from_cam.__len__()):
                    if rt1.check_coilcase(r_sightline[i_sightline], z_sightline[i_sightline]) == False and rt1.check_vacuum_vessel(r_sightline[i_sightline], z_sightline[i_sightline]) == False and i_sightline<len(r_sightline)-1:
                        i_sightline += 1
                    #else:
                    i_sightline += 1

                    if rt1.check_coilcase(r_sightline[i_sightline], z_sightline[i_sightline]) == False and rt1.check_vacuum_vessel(r_sightline[i_sightline], z_sightline[i_sightline]) == True and i_sightline<len(r_sightline)-1:
                        i_sightline += 1
                        buf += interp_image(z_sightline[i_sightline], r_sightline[i_sightline])
                    j_sightline = i_sightline
                    r_sightline[i_sightline] = r_sightline[i_sightline-1]

                    if rt1.check_coilcase(r_sightline[i_sightline], z_sightline[i_sightline]) == False and rt1.check_vacuum_vessel(r_sightline[i_sightline], z_sightline[i_sightline]) == True and i_sightline<len(r_sightline)-1:
                        if flg_reflection == 0:
                            flg_reflection == 1
                        r_sightline[i_sightline+1] = r_sightline[j_sightline-(i_sightline-j_sightline)-1]
                        i_sightline += 1
                        injection_angle = self.cal_injection_angle(dist_from_cam[i_sightline], z_sightline[i_sightline])
                        if injection_angle < np.pi:
                            R_s, R_p = self.cal_refractive_indices_metal(injection_angle, 2.6580, 2.8125)
                            reflection_factor = (R_s + R_p)/2
                        else:
                            reflection_factor = 0.0

                        buf += interp_image(z_sightline[i_sightline], r_sightline[j_sightline-(i_sightline-j_sightline)-1]) * reflection_factor
                    else:
                        r_sightline[i_sightline:] = self.R_cam
                        z_sightline[i_sightline:] = 0.0
                        break
                image[i, j] = buf

        plt.title('Non-polarized, Reflection factor = %.1f' % reflection_factor)
        plt.imshow(image, cmap='jet')
        plt.savefig("integrated_image_test_v2.png")

    def run(self):

        pool = Pool(processes=4)
        pool.apply_async(unwrap_self_plot_projection_image_spline_wrt_1reflection_MP, [])

    def plot_projection_image_spline_wrt_1reflection_MP(self, prep, pend, thread):
        dist_from_cam = np.linspace(0, 4.0, 400)

        r_image = np.linspace(0, 1, self.r_num)
        z_image = np.linspace(-0.5, 0.5, self.z_num)
        image = self.make_ne_image(r_image, z_image)
        #plt.imshow(image, cmap='jet')
        #plt.show()
        interp_image = RectBivariateSpline(z_image, r_image, image)
        image = np.zeros((self.r_num, self.z_num))
        for j in range(prep, pend):
            for i in range(self.z_num):
                theta = self.theta_max*j/self.r_num
                phi = self.phi_max - 2*self.phi_max*i/self.z_num
                r_sightline = np.sqrt(self.R_cam**2 + (dist_from_cam*np.cos(phi))**2 - 2*self.R_cam*dist_from_cam*np.cos(phi)*np.cos(theta))
                z_sightline = dist_from_cam*np.sin(phi)
                i_sightline = 0
                buf = 0.0
                while rt1.check_coilcase(r_sightline[i_sightline], z_sightline[i_sightline]) == False and rt1.check_vacuum_vessel(r_sightline[i_sightline], z_sightline[i_sightline]) == False and i_sightline<len(r_sightline)-1:
                    i_sightline += 1
                else:
                    i_sightline += 1

                while rt1.check_coilcase(r_sightline[i_sightline], z_sightline[i_sightline]) == False and rt1.check_vacuum_vessel(r_sightline[i_sightline], z_sightline[i_sightline]) == True and i_sightline<len(r_sightline)-1:
                    i_sightline += 1
                    buf += interp_image(z_sightline[i_sightline], r_sightline[i_sightline])
                else:
                    j_sightline = i_sightline
                    r_sightline[i_sightline] = r_sightline[i_sightline-1]

                while rt1.check_coilcase(r_sightline[i_sightline], z_sightline[i_sightline]) == False and rt1.check_vacuum_vessel(r_sightline[i_sightline], z_sightline[i_sightline]) == True and i_sightline<len(r_sightline)-1:
                    r_sightline[i_sightline+1] = r_sightline[j_sightline-(i_sightline-j_sightline)-1]
                    i_sightline += 1
                    injection_angle = self.cal_injection_angle(dist_from_cam[i_sightline], z_sightline[i_sightline])
                    if injection_angle < np.pi:
                        R_s, R_p = self.cal_refractive_indices_metal(injection_angle, 2.6580, 2.8125)
                        reflection_factor = (R_s + R_p)/2
                    else:
                        reflection_factor = 0.0

                    buf += interp_image(z_sightline[i_sightline], r_sightline[j_sightline-(i_sightline-j_sightline)-1]) * reflection_factor
                else:
                    r_sightline[i_sightline:] = self.R_cam
                    z_sightline[i_sightline:] = 0.0
                image[i, j] = buf

        plt.title('Non-polarized, Reflection factor = %.1f' % reflection_factor)
        plt.imshow(image, cmap='jet')
        plt.savefig("integrated_image_test.png")
        print("test multiprocessing")
        return image
    #@numba.autojit
    def plot_projection_image_spline_wrt_1reflection(self, reflection_factor=1.0):
        dist_from_cam = np.linspace(0, 4.0, 400)

        r_image = np.linspace(0, 1, self.r_num)
        z_image = np.linspace(-0.5, 0.5, self.z_num)
        image = self.make_ne_image(r_image, z_image)
        #plt.imshow(image, cmap='jet')
        #plt.show()
        interp_image = RectBivariateSpline(z_image, r_image, image)
        image = np.zeros((self.r_num, self.z_num))
        for j in range(self.r_num):
            for i in range(self.z_num):
                theta = self.theta_max*j/self.r_num
                phi = self.phi_max - 2*self.phi_max*i/self.z_num
                r_sightline = np.sqrt(self.R_cam**2 + (dist_from_cam*np.cos(phi))**2 - 2*self.R_cam*dist_from_cam*np.cos(phi)*np.cos(theta))
                z_sightline = dist_from_cam*np.sin(phi)
                i_sightline = 0
                buf = 0.0
                while rt1.check_coilcase(r_sightline[i_sightline], z_sightline[i_sightline]) == False and rt1.check_vacuum_vessel(r_sightline[i_sightline], z_sightline[i_sightline]) == False and i_sightline<len(r_sightline)-1:
                    i_sightline += 1
                else:
                    i_sightline += 1

                while rt1.check_coilcase(r_sightline[i_sightline], z_sightline[i_sightline]) == False and rt1.check_vacuum_vessel(r_sightline[i_sightline], z_sightline[i_sightline]) == True and i_sightline<len(r_sightline)-1:
                    i_sightline += 1
                    buf += interp_image(z_sightline[i_sightline], r_sightline[i_sightline])
                else:
                    j_sightline = i_sightline
                    r_sightline[i_sightline] = r_sightline[i_sightline-1]

                while rt1.check_coilcase(r_sightline[i_sightline], z_sightline[i_sightline]) == False and rt1.check_vacuum_vessel(r_sightline[i_sightline], z_sightline[i_sightline]) == True and i_sightline<len(r_sightline)-1:
                    r_sightline[i_sightline+1] = r_sightline[j_sightline-(i_sightline-j_sightline)-1]
                    i_sightline += 1
                    injection_angle = self.cal_injection_angle(dist_from_cam[i_sightline], z_sightline[i_sightline])
                    if injection_angle < np.pi:
                        R_s, R_p = self.cal_refractive_indices_metal(injection_angle, 2.6580, 2.8125)
                        reflection_factor = (R_s + R_p)/2
                    else:
                        reflection_factor = 0.0

                    buf += interp_image(z_sightline[i_sightline], r_sightline[j_sightline-(i_sightline-j_sightline)-1]) * reflection_factor
                else:
                    r_sightline[i_sightline:] = self.R_cam
                    z_sightline[i_sightline:] = 0.0
                image[i, j] = buf

        plt.title('Non-polarized, Reflection factor = %.1f' % reflection_factor)
        plt.imshow(image, cmap='jet')
        plt.savefig("integrated_image_test.png")
        #plt.title("No reflection")
        #plt.show()

    def plot_projection_image_spline(self):
        dist_from_cam = np.linspace(0, 2.0, 200)

        r_image = np.linspace(0, 1, self.r_num)
        #z_image = np.linspace(-1, 1, self.z_num)
        z_image = np.linspace(-0.5, 0.5, self.z_num)
        image = self.make_ne_image(r_image, z_image)
        interp_image = RectBivariateSpline(z_image, r_image, image)
        image = np.zeros((self.r_num, self.z_num))
        for j in range(self.r_num):
            for i in range(self.z_num):
                theta = self.theta_max*j/self.r_num
                phi = self.phi_max - 2*self.phi_max*i/self.z_num
                r_sightline = np.sqrt(self.R_cam**2 + (dist_from_cam*np.cos(phi))**2 - 2*self.R_cam*dist_from_cam*np.cos(phi)*np.cos(theta))
                z_sightline = dist_from_cam*np.sin(phi)
                i_sightline = 0
                buf = 0.0
                while rt1.check_coilcase(r_sightline[i_sightline], z_sightline[i_sightline]) == False and i_sightline<len(r_sightline)-1:
                    i_sightline += 1
                    buf += interp_image(z_sightline[i_sightline], r_sightline[i_sightline])
                else:
                    r_sightline[i_sightline:] = self.R_cam
                    z_sightline[i_sightline:] = 0.0

                image[i, j] = buf

        plt.imshow(image, cmap='jet')
        plt.show()

    def cal_refractive_indices(self, theta_i, n1, n2):
        R_s = np.abs((-n2*np.sqrt(1-(n1*np.sin(theta_i)/n2)**2) + n1*np.cos(theta_i))/(n2*np.sqrt(1-(n1*np.sin(theta_i)/n2)**2) + n1*np.cos(theta_i)))**2
        R_p = np.abs((n1*np.sqrt(1-(n1*np.sin(theta_i)/n2)**2) - n2*np.cos(theta_i))/(n1*np.sqrt(1-(n1*np.sin(theta_i)/n2)**2) + n2*np.cos(theta_i)))**2

        return R_s, R_p

    def cal_refractive_indices_metal(self, theta_i, n_R, n_I):
        '''
        金属の反射率を計算する．
        :param theta_i: 入射角 [rad]
        :param n_R: 屈折率の実数部
        :param n_I: 屈折率の虚数部（消光係数）
        :return: s偏光の反射率（絶対値），p偏光の反射率（絶対値）
        '''
        r_TE = (np.cos(theta_i) - np.sqrt((n_R**2 - n_I**2 - np.sin(theta_i)**2) + 2j*n_I*n_R))\
              /(np.cos(theta_i) + np.sqrt((n_R**2 - n_I**2 - np.sin(theta_i)**2) + 2j*n_I*n_R))
        #r_TE = 0.0 * theta_i
        r_TM = (-(n_R**2 - n_I**2 + 2j*n_R*n_I)*np.cos(theta_i) + np.sqrt((n_R**2 - n_I**2 - np.sin(theta_i)**2) + 2j*n_I*n_R))\
              /((n_R**2 - n_I**2 + 2j*n_R*n_I)*np.cos(theta_i) + np.sqrt((n_R**2 - n_I**2 - np.sin(theta_i)**2) + 2j*n_I*n_R))
        #r_TM = (-(n_R**2 - n_I**2 + 2j*n_R*n_I)*np.cos(theta_i) + np.sqrt((n_R**2 - n_I**2 - np.sin(theta_i)**2) + 2j*n_I*n_R))
        #r_TM = (-(n_R**2 - n_I**2 + 2*n_R*n_I)*np.cos(theta_i) + np.sqrt((n_R**2 - n_I**2 - np.sin(theta_i)**2) + 2*n_I*n_R))
        #r_TM = (-(n_R**2 - n_I**2 + 2*n_R*n_I)*np.cos(theta_i))

        return np.abs(r_TE)**2, np.abs(r_TM)**2

    def plot_refractive_indices(self, n1, n2):
        theta_i = np.linspace(0, np.pi/2, 100)
        R_s, R_p = self.cal_refractive_indices_metal(theta_i, n1, n2)
        plt.plot(theta_i, R_s, label='S-polarized')
        plt.plot(theta_i, R_p, label='P-polarized')
        plt.plot(theta_i, (R_s+R_p)/2, label='non-polarized')
        plt.ylim(0, 1)
        plt.xlabel('Angle of incidence [rad]')
        plt.ylabel('Reflectance')
        plt.xlim(0, np.pi/2)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

    def load_reflactive_index(self, input_wavelength_um):
        '''
        Feの反射率波長依存性データを読み込み
        REFERENCES: "P. B. Johnson and R. W. Christy. Optical constants of transition metals: Ti, V, Cr, Mn, Fe, Co, Ni, and Pd, <a href=\"https://doi.org/10.1103/PhysRevB.9.5056\"><i>Phys. Rev. B</i> <b>9</b>, 5056-5070 (1974)</a>"
        COMMENTS: "Room temperature"
        :return:
        指定した波長のときのFeの反射率を返す
        '''
        filepath = 'RefractiveIndexINFO_Fe.csv'
        with codecs.open(filepath, 'r', 'Shift-JIS', 'ignore') as f:
            data = pandas.read_csv(filepath)
        print("Load %s" % filepath)
        arr_data = np.array(data)
        n = arr_data[:, 1]
        k = arr_data[:, 2]
        #injection_angle = np.deg2rad(arr_data[:, 0])
        wavelength_um = arr_data[:, 0]
        interp_relative_index_n = interp1d(wavelength_um, n, kind='quadratic')
        interp_relative_index_k = interp1d(wavelength_um, k, kind='quadratic')
        #wavelength_um_100 = np.linspace(0.19, 1.9, 100)
        #plt.plot(wavelength_um, n)
        #plt.plot(wavelength_um, k)
        #plt.plot(wavelength_um_100, interp_relative_index_n(wavelength_um_100))
        #plt.plot(wavelength_um_100, interp_relative_index_k(wavelength_um_100))
        #plt.show()

        print("Wavelength: %.3f um" % input_wavelength_um)

        return interp_relative_index_n(input_wavelength_um), interp_relative_index_k(input_wavelength_um)


    def plot_refractive_indices_388_447_706_728(self):
        theta_i = np.linspace(0, np.pi/2, 100)
        wavelength = np.array([388., 447., 468., 706., 728.])
        wavelength /= 1000
        for i_wavelength,_ in enumerate(wavelength):
            n, k = self.load_reflactive_index(input_wavelength_um=wavelength[i_wavelength])
            R_s, R_p = self.cal_refractive_indices_metal(theta_i, n, k)
            #plt.plot(theta_i, R_s, label='S-polarized' + str(wavelength[i_wavelength]))
            #plt.plot(theta_i, R_p, label='P-polarized' + str(wavelength[i_wavelength]))
            plt.plot(theta_i, (R_s+R_p)/2, label='non-polarized (' + str(wavelength[i_wavelength]) + ' um)')
        plt.ylim(0, 1)
        plt.xlabel('Angle of incidence [rad]')
        plt.ylabel('Reflectance')
        plt.xlim(0, np.pi/2)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.show()

    def cal_injection_angle(self, dist_from_cam, z):
        cos_theta_i = (dist_from_cam**2 + (z**2 + 1) - self.R_cam**2)/(2*dist_from_cam*np.sqrt(z**2 + 1))

        return np.arccos(cos_theta_i)

    def ray_trace_3D_broadcast(self, dist_from_cam,THETA, PHI, I_ARRAY):
        d_dist_from_cam = dist_from_cam[1] - dist_from_cam[0]
        vec_i = np.array([-np.cos(THETA), np.sin(THETA), np.sin(PHI)])
        norm_vec_i = np.linalg.norm(vec_i, axis=0, ord=2)

        reflection_factor = 0.0

        x = self.R_cam + d_dist_from_cam*vec_i[0, :, :, :]*I_ARRAY/norm_vec_i
        y = d_dist_from_cam*vec_i[1, :, :, :]*I_ARRAY/norm_vec_i
        z = d_dist_from_cam*vec_i[2, :, :, :]*I_ARRAY/norm_vec_i
        x_1ref = np.zeros(x.shape)
        y_1ref = np.zeros(y.shape)
        z_1ref = np.zeros(z.shape)
        #===============================================================================================================
        #   真空容器との接触を判定
        #===============================================================================================================
        is_inside_vacuum_vessel_0 = np.where((np.abs(z)<0.35) & (np.sqrt(x**2 + y**2)<1.0))
        is_inside_vacuum_vessel_1 = np.where((np.abs(z)<0.35) & (np.sqrt(x**2 + y**2)>1.0))
        is_inside_vacuum_vessel_2 = np.where((np.abs(z)>0.35) & (np.abs(z)<0.53) & (np.sqrt(x**2 + y**2)>(0.8 + np.sqrt(0.04 - (np.abs(z) - 0.35)**2))))
        is_inside_vacuum_vessel_3 = np.where((np.abs(z)>0.53) & (np.sqrt(x**2 + y**2)>(0.8888889 - 2.7838*(np.abs(z)-0.53))))
        try:
            buf_arr_for_refcnt = np.ones(x.shape)
            x_reflection_point = np.ones(x.shape)
            y_reflection_point = np.ones(y.shape)
            z_reflection_point = np.ones(z.shape)
            vec_i_x = vec_i[0]
            vec_i_y = vec_i[1]
            vec_i_z = vec_i[2]
            #if is_inside_vacuum_vessel_1[0][-1] > is_inside_vacuum_vessel_0[0][0]:
            #if np.sum(is_inside_vacuum_vessel_1)>0:
            #    #    index_is_inside_vacuum_vessel = np.min(np.where(is_inside_vacuum_vessel_1 > is_inside_vacuum_vessel_0[0][0], is_inside_vacuum_vessel_1, np.inf)) - 1
            #    #    reflectionpoint = np.array([x[index_is_inside_vacuum_vessel], y[index_is_inside_vacuum_vessel], z[index_is_inside_vacuum_vessel]])
            #    is_inside_vacuum_vessel = is_inside_vacuum_vessel_1
            #    sign_vec_ref = np.array([1, 1, 1])
            if np.sum(is_inside_vacuum_vessel_2)>0:
                #elif np.sum(is_inside_vacuum_vessel_2)>0:
                #reflectionpoint = np.array([x[is_inside_vacuum_vessel_2], y[is_inside_vacuum_vessel_2], z[is_inside_vacuum_vessel_2]])
                is_inside_vacuum_vessel = is_inside_vacuum_vessel_2
                sign_vec_ref = np.array([1, 1, -1])
            #if np.sum(is_inside_vacuum_vessel_3)>0:
            #    #    index_is_inside_vacuum_vessel = np.min(np.argwhere((np.abs(z)>0.53) & (np.sqrt(x**2 + y**2)>(0.8888889 - 2.7838*(np.abs(z)-0.53))))) - 1
            #    #    reflectionpoint = np.array([x[index_is_inside_vacuum_vessel], y[index_is_inside_vacuum_vessel], z[index_is_inside_vacuum_vessel]])
            #    #    sign_vec_ref = np.array([1, -1, 1])
            #    is_inside_vacuum_vessel = is_inside_vacuum_vessel_3
            #    buf_arr_for_refcnt[is_inside_vacuum_vessel_3] = 0.0
            #    if(reflectionpoint[0]>0):
            #        vec_n = np.array([reflectionpoint[0], reflectionpoint[1], -np.sign(reflectionpoint[2])*(np.sqrt(reflectionpoint[0]**2 + reflectionpoint[1]**2)*2.7838)])
            #    else:
            #        vec_n = np.array([reflectionpoint[0], reflectionpoint[1], np.sign(reflectionpoint[2])*(np.sqrt(reflectionpoint[0]**2 + reflectionpoint[1]**2)*2.7838)])
            buf_arr_for_refcnt[is_inside_vacuum_vessel] = 0.0
            arr_for_refcnt = np.sum(buf_arr_for_refcnt, axis=2)
            arr_for_refcnt = arr_for_refcnt[:, :, np.newaxis]
            J_ARRAY = I_ARRAY - arr_for_refcnt*np.ones(x.shape)
            x_reflection_point[is_inside_vacuum_vessel] = x[is_inside_vacuum_vessel] - (d_dist_from_cam*vec_i_x[is_inside_vacuum_vessel]*(J_ARRAY[is_inside_vacuum_vessel]-1)/norm_vec_i[is_inside_vacuum_vessel])
            y_reflection_point[is_inside_vacuum_vessel] = y[is_inside_vacuum_vessel] - (d_dist_from_cam*vec_i_y[is_inside_vacuum_vessel]*(J_ARRAY[is_inside_vacuum_vessel]-1)/norm_vec_i[is_inside_vacuum_vessel])
            z_reflection_point[is_inside_vacuum_vessel] = z[is_inside_vacuum_vessel] - (d_dist_from_cam*vec_i_z[is_inside_vacuum_vessel]*(J_ARRAY[is_inside_vacuum_vessel]-1)/norm_vec_i[is_inside_vacuum_vessel])
            #if np.sum(is_inside_vacuum_vessel_1)>0:
            #    vec_n = np.array([-x_reflection_point, -y_reflection_point, 0])
            #    sign_vec_ref = np.array([1, 1, 1])
            if np.sum(is_inside_vacuum_vessel_2)>0:
                vec_n = np.array([-0.8*x_reflection_point, -0.8*y_reflection_point, 0.35*np.sign(z_reflection_point)])
                sign_vec_ref = np.array([1, 1, -1])
            #if np.sum(is_inside_vacuum_vessel_3)>0:
            #    vec_n = np.array([x_reflection_point, y_reflection_point, -np.sign(z_reflection_point)*(np.sqrt(z_reflection_point**2 + y_reflection_point**2)*2.7838)])

            vec_ref = self.cal_reflection_vector_broadcast(vec_i, vec_n)
            vec_ref_x = vec_ref[0]
            vec_ref_y = vec_ref[1]
            vec_ref_z = vec_ref[2]
            norm_vec_ref = np.linalg.norm(vec_ref, axis=0)
            x[is_inside_vacuum_vessel] = x_reflection_point[is_inside_vacuum_vessel] + d_dist_from_cam*vec_ref_x[is_inside_vacuum_vessel]*J_ARRAY[is_inside_vacuum_vessel]/norm_vec_ref[is_inside_vacuum_vessel]
            y[is_inside_vacuum_vessel] = y_reflection_point[is_inside_vacuum_vessel] + d_dist_from_cam*vec_ref_y[is_inside_vacuum_vessel]*J_ARRAY[is_inside_vacuum_vessel]/norm_vec_ref[is_inside_vacuum_vessel]
            z[is_inside_vacuum_vessel] = z_reflection_point[is_inside_vacuum_vessel] + d_dist_from_cam*vec_ref_z[is_inside_vacuum_vessel]*J_ARRAY[is_inside_vacuum_vessel]/norm_vec_ref[is_inside_vacuum_vessel]
            #x[is_inside_vacuum_vessel_2] = np.nan
            #y[is_inside_vacuum_vessel_2] = np.nan
            #z[is_inside_vacuum_vessel_2] = np.nan
            x_1ref = np.where(x_1ref==0, np.nan, x_1ref)
            #injection_angle = self.cal_injection_angle_for2vector(vec_i, vec_n)
            #R_s, R_p = self.cal_refractive_indices_metal(injection_angle, 2.6580, 2.8125)
            #reflection_factor = (R_s + R_p)/2
        except:
            import traceback
            traceback.print_exc()

        return x, y, z, x_1ref, y_1ref, z_1ref, reflection_factor

    def ray_trace_3D(self, dist_from_cam, theta, phi):
        #n, k = self.load_reflactive_index(input_wavelength_um=0.468)
        #n, k = 2.0509, 0.52747 #FB450-10 実測値
        n, k = 0.70897, 0.42432 #FL730-10 実測値

        d_dist_from_cam = dist_from_cam[1] - dist_from_cam[0]
        vec_i = np.array([-np.cos(theta), np.sin(theta), np.sin(phi)])
        norm_vec_i = np.linalg.norm(vec_i)
        x_1ref = np.zeros(dist_from_cam.__len__())
        y_1ref = np.zeros(dist_from_cam.__len__())
        z_1ref = np.zeros(dist_from_cam.__len__())

        reflection_factor = 0.0

        i_array = np.arange(dist_from_cam.__len__())
        x = self.R_cam + d_dist_from_cam*vec_i[0]*i_array/norm_vec_i
        y = d_dist_from_cam*vec_i[1]*i_array/norm_vec_i
        z = d_dist_from_cam*vec_i[2]*i_array/norm_vec_i
        #===============================================================================================================
        #   真空容器との接触を判定
        #===============================================================================================================
        is_inside_vacuum_vessel_0 = np.where((np.abs(z)<0.35) & (np.sqrt(x**2 + y**2)<1.0))
        is_inside_vacuum_vessel_1 = np.where((np.abs(z)<0.35) & (np.sqrt(x**2 + y**2)>1.0))
        is_inside_vacuum_vessel_2 = np.where((np.abs(z)>0.35) & (np.abs(z)<0.53) & (np.sqrt(x**2 + y**2)>(0.8 + np.sqrt(0.04 - (np.abs(z) - 0.35)**2))))
        is_inside_vacuum_vessel_3 = np.where((np.abs(z)>0.53) & (np.sqrt(x**2 + y**2)>(0.8888889 - 2.7838*(np.abs(z)-0.53))))
        try:
            if is_inside_vacuum_vessel_1[0][-1] > is_inside_vacuum_vessel_0[0][0]:
                index_is_inside_vacuum_vessel = np.min(np.where(is_inside_vacuum_vessel_1 > is_inside_vacuum_vessel_0[0][0], is_inside_vacuum_vessel_1, np.inf)) - 1
                reflectionpoint = np.array([x[index_is_inside_vacuum_vessel], y[index_is_inside_vacuum_vessel], z[index_is_inside_vacuum_vessel]])
                vec_n = np.array([-reflectionpoint[0], -reflectionpoint[1], 0])
                sign_vec_ref = np.array([1, 1, 1])
            elif np.sum(is_inside_vacuum_vessel_2)>0:
                index_is_inside_vacuum_vessel = np.min(np.argwhere((np.abs(z)>0.35) & (np.abs(z)<0.53) & (np.sqrt(x**2 + y**2)>(0.8 + np.sqrt(0.04 - (np.abs(z) - 0.35)**2))))) - 1
                reflectionpoint = np.array([x[index_is_inside_vacuum_vessel], y[index_is_inside_vacuum_vessel], z[index_is_inside_vacuum_vessel]])
                vec_n = np.array([-0.8*reflectionpoint[0], -0.8*reflectionpoint[1], 0.35*np.sign(reflectionpoint[2])])
                sign_vec_ref = np.array([1, 1, -1])
            elif np.sum(is_inside_vacuum_vessel_3)>0:
                index_is_inside_vacuum_vessel = np.min(np.argwhere((np.abs(z)>0.53) & (np.sqrt(x**2 + y**2)>(0.8888889 - 2.7838*(np.abs(z)-0.53))))) - 1
                reflectionpoint = np.array([x[index_is_inside_vacuum_vessel], y[index_is_inside_vacuum_vessel], z[index_is_inside_vacuum_vessel]])
                sign_vec_ref = np.array([1, -1, 1])
                if(reflectionpoint[0]>0):
                    vec_n = np.array([reflectionpoint[0], reflectionpoint[1], -np.sign(reflectionpoint[2])*(np.sqrt(reflectionpoint[0]**2 + reflectionpoint[1]**2)*2.7838)])
                else:
                    vec_n = np.array([reflectionpoint[0], reflectionpoint[1], np.sign(reflectionpoint[2])*(np.sqrt(reflectionpoint[0]**2 + reflectionpoint[1]**2)*2.7838)])
        except:
            pass
        try:
            vec_ref = self.cal_reflection_vector(vec_i, vec_n)
            norm_vec_ref = np.linalg.norm(vec_ref)
            j_array = np.arange(dist_from_cam.__len__()-index_is_inside_vacuum_vessel)
            x[index_is_inside_vacuum_vessel+1:] = np.nan
            y[index_is_inside_vacuum_vessel+1:] = np.nan
            z[index_is_inside_vacuum_vessel+1:] = np.nan
            x_1ref[:index_is_inside_vacuum_vessel] = np.nan
            y_1ref[:index_is_inside_vacuum_vessel] = np.nan
            z_1ref[:index_is_inside_vacuum_vessel] = np.nan
            x_1ref[index_is_inside_vacuum_vessel:] = reflectionpoint[0] + d_dist_from_cam*sign_vec_ref[0]*vec_ref[0]*j_array/norm_vec_ref
            y_1ref[index_is_inside_vacuum_vessel:] = reflectionpoint[1] + d_dist_from_cam*sign_vec_ref[1]*vec_ref[1]*j_array/norm_vec_ref
            z_1ref[index_is_inside_vacuum_vessel:] = reflectionpoint[2] + d_dist_from_cam*sign_vec_ref[2]*vec_ref[2]*j_array/norm_vec_ref
            injection_angle = self.cal_injection_angle_for2vector(vec_i, vec_n)
            R_s, R_p = self.cal_refractive_indices_metal(injection_angle, self.n, self.k)
            #R_s, R_p = self.cal_refractive_indices_metal(injection_angle, 2.6580, 2.8125)  #468nm
            #R_s, R_p = self.cal_refractive_indices_metal(injection_angle, 2.865, 3.235)   #730nm
            reflection_factor = (R_s + R_p)/2
        except:
            pass

        #===============================================================================================================
        #   コイルとの接触を判定
        #===============================================================================================================
        r_1ref = np.sqrt(x_1ref**2 + y_1ref**2)
        is_inside_fcoil_1ref_0 = np.argwhere((r_1ref >= 0.185) & (r_1ref <= 0.1850 + 0.1930) & (z_1ref >= -.0200) & (z_1ref <= .0200))
        is_inside_fcoil_1ref_1 = np.argwhere((r_1ref >= 0.185 + 0.0550) & (r_1ref <= 0.1850 + 0.1930 - 0.0550) & (z_1ref >= -.0750) & (z_1ref <= .0750))
        is_inside_fcoil_1ref_2 = np.argwhere(((r_1ref - (0.185  + 0.055))**2 + (np.abs(z_1ref) - (0.020))**2 <= 0.055**2))
        is_inside_fcoil_1ref_3 = np.argwhere(((r_1ref - (0.185 + 0.193 - 0.055))**2 + (np.abs(z_1ref) - (0.020))**2 <= 0.055**2))
        buf_index_1ref = []
        if np.sum(is_inside_fcoil_1ref_0)>0:
            buf_index_1ref.append(np.min(is_inside_fcoil_1ref_0))
        if np.sum(is_inside_fcoil_1ref_1)>0:
            buf_index_1ref.append(np.min(is_inside_fcoil_1ref_1))
        if np.sum(is_inside_fcoil_1ref_2)>0:
            buf_index_1ref.append(np.min(is_inside_fcoil_1ref_2))
        if np.sum(is_inside_fcoil_1ref_3)>0:
            buf_index_1ref.append(np.min(is_inside_fcoil_1ref_3))
        try:
            index_is_inside_fcoil_1ref = np.min(buf_index_1ref)
            x_1ref[index_is_inside_fcoil_1ref+1:] = np.nan
            y_1ref[index_is_inside_fcoil_1ref+1:] = np.nan
            z_1ref[index_is_inside_fcoil_1ref+1:] = np.nan
        except:
            pass
        r = np.sqrt(x**2 + y**2)
        is_inside_fcoil_0 = np.argwhere((r >= 0.185) & (r <= 0.1850 + 0.1930) & (z >= -.0200) & (z <= .0200))
        is_inside_fcoil_1 = np.argwhere((r >= 0.185 + 0.0550) & (r <= 0.1850 + 0.1930 - 0.0550) & (z >= -.0750) & (z <= .0750))
        is_inside_fcoil_2 = np.argwhere(((r - (0.185  + 0.055))**2 + (np.abs(z) - (0.020))**2 <= 0.055**2))
        is_inside_fcoil_3 = np.argwhere(((r - (0.185 + 0.193 - 0.055))**2 + (np.abs(z) - (0.020))**2 <= 0.055**2))
        #buf_index = dist_from_cam.__len__()
        buf_index = []
        buf_where_in_fcoil = []
        if np.sum(is_inside_fcoil_0)>0:
            buf_index.append(np.min(is_inside_fcoil_0))
            buf_where_in_fcoil.append(0)
        if np.sum(is_inside_fcoil_1)>0:
            buf_index.append(np.min(is_inside_fcoil_1))
            buf_where_in_fcoil.append(1)
        if np.sum(is_inside_fcoil_2)>0:
            buf_index.append(np.min(is_inside_fcoil_2))
            buf_where_in_fcoil.append(2)
        if np.sum(is_inside_fcoil_3)>0:
            buf_index.append(np.min(is_inside_fcoil_3))
            buf_where_in_fcoil.append(3)
        try:
            index_is_inside_fcoil = np.min(buf_index)
            where_in_fcoil = buf_where_in_fcoil[np.argmin(buf_index)]
            if where_in_fcoil == 0:
                reflectionpoint = np.array([x[index_is_inside_fcoil], y[index_is_inside_fcoil], z[index_is_inside_fcoil]])
                vec_n = np.array([reflectionpoint[0], reflectionpoint[1], 0])
            elif where_in_fcoil == 1:
                reflectionpoint = np.array([x[index_is_inside_fcoil], y[index_is_inside_fcoil], z[index_is_inside_fcoil]])
                vec_n = np.array([-reflectionpoint[0], -reflectionpoint[1], 0])
            elif where_in_fcoil == 2:
                reflectionpoint = np.array([x[index_is_inside_fcoil], y[index_is_inside_fcoil], z[index_is_inside_fcoil]])
                vec_n = np.array([-0.24*reflectionpoint[0], -0.24*reflectionpoint[1], 0.02*np.sign(reflectionpoint[2])])
            elif where_in_fcoil == 3:
                reflectionpoint = np.array([x[index_is_inside_fcoil], y[index_is_inside_fcoil], z[index_is_inside_fcoil]])
                vec_n = np.array([0.323*reflectionpoint[0], 0.323*reflectionpoint[1], 0.02*np.sign(reflectionpoint[2])])

            vec_ref = self.cal_reflection_vector(vec_i, vec_n)
            norm_vec_ref = np.linalg.norm(vec_ref)
            j_array = np.arange(dist_from_cam.__len__()-index_is_inside_fcoil)
            x[index_is_inside_fcoil+1:] = np.nan
            y[index_is_inside_fcoil+1:] = np.nan
            z[index_is_inside_fcoil+1:] = np.nan
            x_1ref[:index_is_inside_fcoil] = np.nan
            y_1ref[:index_is_inside_fcoil] = np.nan
            z_1ref[:index_is_inside_fcoil] = np.nan
            x_1ref[index_is_inside_fcoil:] = reflectionpoint[0] + d_dist_from_cam*vec_ref[0]*j_array/norm_vec_ref
            y_1ref[index_is_inside_fcoil:] = reflectionpoint[1] + d_dist_from_cam*vec_ref[1]*j_array/norm_vec_ref
            z_1ref[index_is_inside_fcoil:] = reflectionpoint[2] + d_dist_from_cam*vec_ref[2]*j_array/norm_vec_ref
            injection_angle = self.cal_injection_angle_for2vector(vec_i, vec_n)
            R_s, R_p = self.cal_refractive_indices_metal(injection_angle, self.n, self.k)
            reflection_factor = (R_s + R_p)/2
        except:
            pass
        #===============================================================================================================
        #   センタースタックとの接触を判定
        #===============================================================================================================
        try:
            r = np.sqrt(x**2 + y**2)
            index_is_inside_vacuum_vessel = np.min(np.argwhere((r <= 0.0826) & (z >= -.6600) & (z <= .6600)))-1
            reflectionpoint = np.array([x[index_is_inside_vacuum_vessel], y[index_is_inside_vacuum_vessel], z[index_is_inside_vacuum_vessel]])
            vec_n = np.array([-reflectionpoint[0], -reflectionpoint[1], 0])
            vec_ref = self.cal_reflection_vector(vec_i, vec_n)
            norm_vec_ref = np.linalg.norm(vec_ref)
            j_array = np.arange(dist_from_cam.__len__()-index_is_inside_vacuum_vessel)
            x[index_is_inside_vacuum_vessel+1:] = np.nan
            y[index_is_inside_vacuum_vessel+1:] = np.nan
            z[index_is_inside_vacuum_vessel+1:] = np.nan
            x_1ref[:index_is_inside_vacuum_vessel+1] = np.nan
            y_1ref[:index_is_inside_vacuum_vessel+1] = np.nan
            z_1ref[:index_is_inside_vacuum_vessel+1] = np.nan
            x_1ref[index_is_inside_vacuum_vessel:] = reflectionpoint[0] + d_dist_from_cam*vec_ref[0]*j_array/norm_vec_ref
            y_1ref[index_is_inside_vacuum_vessel:] = reflectionpoint[1] + d_dist_from_cam*vec_ref[1]*j_array/norm_vec_ref
            z_1ref[index_is_inside_vacuum_vessel:] = reflectionpoint[2] + d_dist_from_cam*vec_ref[2]*j_array/norm_vec_ref
            injection_angle = self.cal_injection_angle_for2vector(vec_i, vec_n)
            R_s, R_p = self.cal_refractive_indices_metal(injection_angle, self.n, self.k)
            reflection_factor = (R_s + R_p)/2
        except:
            pass
        try:
            r_1ref = np.sqrt(x_1ref**2 + y_1ref**2)
            index_is_inside_vacuum_vessel = np.min(np.argwhere((r_1ref <= 0.0826) & (z_1ref >= -.6600) & (z_1ref <= .6600)))-1
            x_1ref[index_is_inside_vacuum_vessel+1:] = np.nan
            y_1ref[index_is_inside_vacuum_vessel+1:] = np.nan
            z_1ref[index_is_inside_vacuum_vessel+1:] = np.nan
        except:
            pass
        #===============================================================================================================
        #   釣り上げコイルとの接触を判定
        #===============================================================================================================
        try:
            r = np.sqrt(x**2 + y**2)
            index_is_inside_vacuum_vessel = np.min(np.argwhere((r <= 0.485) & (z >= .4800)))-1
            x[index_is_inside_vacuum_vessel+1:] = np.nan
            y[index_is_inside_vacuum_vessel+1:] = np.nan
            z[index_is_inside_vacuum_vessel+1:] = np.nan
            x_1ref[index_is_inside_vacuum_vessel+1:] = np.nan
            y_1ref[index_is_inside_vacuum_vessel+1:] = np.nan
            z_1ref[index_is_inside_vacuum_vessel+1:] = np.nan
        except:
            pass
        try:
            r_1ref = np.sqrt(x_1ref**2 + y_1ref**2)
            index_is_inside_vacuum_vessel = np.min(np.argwhere((r_1ref <= 0.485) & (z_1ref >= .4800)))-1
            x_1ref[index_is_inside_vacuum_vessel+1:] = np.nan
            y_1ref[index_is_inside_vacuum_vessel+1:] = np.nan
            z_1ref[index_is_inside_vacuum_vessel+1:] = np.nan
        except:
            pass
        #is_inside_vacuum_vessel_0 = np.where((np.abs(z_1ref)<0.35) & (np.sqrt(x_1ref**2 + y_1ref**2)<1.0))
        #is_inside_vacuum_vessel_1 = np.where((np.abs(z_1ref)<0.35) & (np.sqrt(x_1ref**2 + y_1ref**2)>1.0))
        #try:
        #    if is_inside_vacuum_vessel_1[0][-1] > is_inside_vacuum_vessel_0[0][0]:
        #        index_is_inside_vacuum_vessel = np.min(np.where(is_inside_vacuum_vessel_1 > is_inside_vacuum_vessel_0[0][0], is_inside_vacuum_vessel_1, np.inf)) - 1
        #        reflectionpoint = np.array([x[index_is_inside_vacuum_vessel], y[index_is_inside_vacuum_vessel], z[index_is_inside_vacuum_vessel]])
        #        vec_n = np.array([-reflectionpoint[0], -reflectionpoint[1], 0])
        #        vec_ref = self.cal_reflection_vector(vec_i, vec_n)
        #        j_array = np.arange(dist_from_cam.__len__()-index_is_inside_vacuum_vessel)
        #        norm_vec_ref = np.linalg.norm(vec_ref)
        #        x_1ref[index_is_inside_vacuum_vessel:] = reflectionpoint[0] + d_dist_from_cam*vec_ref[0]*j_array/norm_vec_ref
        #        y_1ref[index_is_inside_vacuum_vessel:] = reflectionpoint[1] + d_dist_from_cam*vec_ref[1]*j_array/norm_vec_ref
        #        z_1ref[index_is_inside_vacuum_vessel:] = reflectionpoint[2] + d_dist_from_cam*vec_ref[2]*j_array/norm_vec_ref
        #except:
        #    pass

        return x, y, z, x_1ref, y_1ref, z_1ref, reflection_factor

    def cal_injection_angle_for2vector(self, vec_i, vec_n):
        inner = np.inner(vec_i, -vec_n)
        if inner < 0:
            inner = np.inner(vec_i, vec_n)
        norm = np.linalg.norm(vec_i) * np.linalg.norm(vec_n)
        c = inner/norm
        #injection_angle = np.arccos(np.clip(c, 0, 1.0))
        injection_angle = np.arccos(c)

        return injection_angle

    def cal_reflection_vector_broadcast(self, vec_i, vec_n):
        """
        入射ベクトルと法線ベクトルを入力として，反射ベクトルを返します
        :param vec_i:入射ベクトル
        :param vec_n: 反射ベクトル
        :return:
        """
        norm_vec_n = vec_n/np.linalg.norm(vec_n, axis=0)
        return vec_i - 2*(vec_i[0]*vec_n[0]+vec_i[1]*vec_n[1]+vec_i[2]*vec_n[2])*norm_vec_n

    def cal_reflection_vector(self, vec_i, vec_n):
        """
        入射ベクトルと法線ベクトルを入力として，反射ベクトルを返します
        :param vec_i:入射ベクトル
        :param vec_n: 反射ベクトル
        :return:
        """
        norm_vec_n = vec_n/np.linalg.norm(vec_n)
        return vec_i - 2*np.dot(vec_i, norm_vec_n)*norm_vec_n

    def plot_3Dto2D_broadcast(self):
        dist_from_cam = np.linspace(0, 2.5, 125)
        theta = self.theta_max*np.arange(self.r_num)/self.r_num
        phi = self.phi_max - 2*self.phi_max*np.arange(self.z_num)/self.z_num
        i_array = np.arange(dist_from_cam.__len__())
        THETA, PHI, I_ARRAY = np.meshgrid(theta, phi, i_array)
        r_image = np.linspace(0, 1, self.r_num)
        z_image = np.linspace(-0.5, 0.5, self.z_num)
        image = self.make_ne_image(r_image, z_image)
        interp_image = RectBivariateSpline(z_image, r_image, image)
        #image_buf = np.zeros((self.r_num, self.z_num, dist_from_cam.__len__()))
        #image_1ref_buf = np.zeros((self.r_num, self.z_num, dist_from_cam.__len__()))
        x, y, z, x_1ref, y_1ref, z_1ref, reflection_factor = self.ray_trace_3D_broadcast(dist_from_cam, THETA, PHI, I_ARRAY)
        r = np.sqrt(x**2 + y**2)
        r_1ref = np.sqrt(x_1ref**2 + y_1ref**2)
        image_buf = interp_image(z, r, grid=False)
        image_1ref_buf = reflection_factor*interp_image(z_1ref, r_1ref, grid=False)
        image_buf[np.isnan(image_buf)] = 0
        image_1ref_buf[np.isnan(image_1ref_buf)] = 0
        image = np.sum(image_buf, axis=2)
        image_1ref = np.sum(image_1ref_buf, axis=0)
        #plt.imshow((image + image_1ref), cmap='jet', interpolation='none')
        plt.imshow(image, cmap='jet', interpolation='none')

        plt.show()
        #plt.savefig("integrated_image_3Dto2D.png")

    def plot_3Dto2D(self, p_opt_best, p_opt_best_2ndPeak):
        dist_from_cam = np.linspace(0, 5, 250)
        theta = self.theta_max*np.arange(self.r_num)/self.r_num
        phi = self.phi_max - 2*self.phi_max*np.arange(self.z_num)/self.z_num
        r_image = np.linspace(0, 1, self.r_num)
        z_image = np.linspace(-0.5, 0.5, self.z_num)
        image_local = self.make_ne_image(r_image, z_image, p_opt_best, p_opt_best_2ndPeak)
        #image_local = image2ResizedGray(self.r_num, self.z_num)
        plt.figure(figsize=(16,12))
        plt.subplot(1,2,1)
        plt.imshow(image_local[::-1,:], cmap='jet', vmax=np.max(image_local[:np.int(0.8*self.r_num), :]), interpolation='none')
        plt.title("1st: [%d, %d, %.2f, %.2f]\n2nd: [%.2f, %d, %.2f, %.2f]\nLocal" % \
                  (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                   p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]))
        interp_image = RectBivariateSpline(z_image, r_image, image_local)
        image_buf = np.zeros((self.r_num, self.z_num, dist_from_cam.__len__()))
        image_1ref_buf = np.zeros((self.r_num, self.z_num, dist_from_cam.__len__()))
        #injection_angle, relative_illumination = self.load_relative_illumination()
        #interp_relative_illumination = interp1d(injection_angle, relative_illumination, kind='quadratic')
        for i in range(self.r_num):
            for j in range(self.z_num):
                x, y, z, x_1ref, y_1ref, z_1ref, reflection_factor = self.ray_trace_3D(dist_from_cam, theta[i], phi[j])
                #vec_i = np.array([-np.cos(theta[i]), np.sin(theta[i]), np.sin(phi[j])])
                #injection_angle = self.cal_injection_angle_for2vector(vec_i, np.array([-np.cos(self.theta_max/2), np.sin(self.theta_max/2), 0]))
                #try:
                #    relative_illumination = interp_relative_illumination(injection_angle)
                #except:
                #    relative_illumination = 1.0
                r = np.sqrt(x**2 + y**2)
                r_1ref = np.sqrt(x_1ref**2 + y_1ref**2)
                #plt.plot(np.sqrt(x**2 + y**2), z)
                #plt.plot(np.sqrt(x_1ref**2 + y_1ref**2), z_1ref)
                image_buf[i, j, :] = interp_image(z, r, grid=False)
                image_1ref_buf[i, j, :] = reflection_factor*interp_image(z_1ref, r_1ref, grid=False)
                #image_buf[i, j, :] = relative_illumination*interp_image(z, r, grid=False)
                #image_1ref_buf[i, j, :] = relative_illumination*reflection_factor*interp_image(z_1ref, r_1ref, grid=False)
                #image_1ref_buf[i, j, :] = interp_image(z_1ref, r_1ref, grid=False)
                #image_1ref_buf[i, j, :] = reflection_factor
                #image_1ref_buf[i, j, :] = relative_illumination/dist_from_cam.__len__()

        image_buf[np.isnan(image_buf)] = 0
        image_1ref_buf[np.isnan(image_1ref_buf)] = 0
        image = np.sum(image_buf, axis=2)
        image_1ref = np.sum(image_1ref_buf, axis=2)
        plt.subplot(1,2,2)
        #plt.imshow(image_1ref.T, cmap='jet')
        plt.imshow((image + image_1ref).T, cmap='jet', interpolation='none')
        plt.title("Projection")

        plt.tight_layout()
        #plt.show()
        #r = np.linspace(0, 1.0, self.r_num)
        #plt.plot(r, image[:, np.int(self.z_num/2)], label='w/o reflection')
        #plt.plot(r, image[:, np.int(self.z_num/2)] + image_1ref[:, np.int(self.z_num/2)], label='w/ reflection')
        #plt.legend()
        #plt.show()
        plt.savefig("SimCIS_n%.4f_k%.4f_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.png" % \
                    (self.n, self.k, p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3],\
                     p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]))
        np.savez("SimCIS_n%.4f_k%.4f_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.npz" % \
                 (self.n, self.k, p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3],\
                  p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]),\
                  image_local=image_local, image=image, image_1ref=image_1ref, p_opt_best=p_opt_best, p_opt_best_2ndPeak=p_opt_best_2ndPeak, dist_from_cam=dist_from_cam)

    def load_image(self, p_opt_best, p_opt_best_2ndPeak):
        images = np.load("dataset_SimCIS/SimCIS_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.npz" % \
                 (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                  p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]))
        image_local = images['image_local']
        image = images['image']
        image_1ref = images['image_1ref']

        plt.figure(figsize=(16,12))
        plt.subplot(1,2,1)
        plt.imshow(image_local[::-1,:], cmap='jet')
        plt.title("1st: %s\n2nd: %s\nLocal" % (p_opt_best, p_opt_best_2ndPeak))
        plt.subplot(1,2,2)
        plt.imshow((image + image_1ref).T, cmap='jet')
        plt.title("Projection")
        plt.show()


    def show_animation(self):
        fig = plt.figure(figsize=(10, 6))
        params = {
            'fig': fig,
            'func': self.plot_3D_ray,
            'interval': 10,
            'frames': 10,
            'repeat': False,
            'blit': True

        }
        anime = animation.FuncAnimation(**params)
        #anime.save('ray_3D.gif', writer='XXX')
        plt.show()

    def plot_3D_ray_broadcast(self, frame=None, showRay=True, showVV=False, showFC=False, showLC=False, showCS=False):
        # (x, y, z)
        dist_from_cam = np.linspace(0, 5, 125)
        theta = self.theta_max*np.arange(self.r_num)/self.r_num
        phi = self.phi_max - 2*self.phi_max*np.arange(self.z_num)/self.z_num
        i_array = np.arange(dist_from_cam.__len__())
        THETA, PHI, I_ARRAY = np.meshgrid(theta, phi, i_array)
        x, y, z, x_1ref, y_1ref, z_1ref, reflection_factor = self.ray_trace_3D_broadcast(dist_from_cam, THETA, PHI, I_ARRAY)
        fig = plt.figure()
        # 3Dでプロット
        ax = Axes3D(fig)
        if showRay:
            for i in range(self.r_num):
                for j in range(1):
                    j=0
                #for j in range(self.z_num):
                    ax.plot(x[i,j,:], y[i,j,:], z[i,0,:], "-", color="#00aa00", ms=1, mew=0.1)
                    ax.plot(x_1ref[i,j,:], y_1ref[i,j,:], z_1ref[i,j,:], "-", color="#aa0000", ms=10, mew=0.1)
        ax.set_aspect('equal')
        # 軸ラベル
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        set_axes_equal(ax)
        plt.show()

    def plot_3D_ray(self, frame=None, showRay=True, showVV=False, showFC=False, showLC=False, showCS=False):
        # (x, y, z)
        dist_from_cam = np.linspace(0, 3, 125)
        fig = plt.figure()
        # 3Dでプロット
        ax = Axes3D(fig)
        if showRay:
            theta = self.theta_max*np.arange(self.r_num)/self.r_num
            phi = self.phi_max - 2*self.phi_max*np.arange(self.z_num)/self.z_num
            for i in range(self.r_num):
                for j in range(self.z_num):
                    x, y, z, x_1ref, y_1ref, z_1ref,_ = self.ray_trace_3D(dist_from_cam, theta[i], phi[j])
                    #x, y, z, x_1ref, y_1ref, z_1ref,_ = self.ray_trace_3D(dist_from_cam, theta[i], np.pi/50)
                    #x, y, z, x_1ref, y_1ref, z_1ref,_ = self.ray_trace_3D(dist_from_cam, theta[i], 0)
                    #x, y, z, x_1ref, y_1ref, z_1ref, _ = self.ray_trace_3D(dist_from_cam, 0*np.pi/20, phi[j])
                    #x, y, z, x_1ref, y_1ref, z_1ref, _ = self.ray_trace_3D(dist_from_cam, frame*np.pi/20, phi[j])
                    ax.plot(x, y, z, "-", color="#00aa00", ms=1, mew=0.1)
                    ax.plot(x_1ref, y_1ref, z_1ref, "-", color="#aa0000", ms=1, mew=0.1)
        ax.set_aspect('equal')
        # 軸ラベル
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        us = np.linspace(0, 2.0 * np.pi, 32)
        #Center Stack
        if showCS:
            zs = np.linspace(-0.66, 0.66, 2)
            us, zs = np.meshgrid(us, zs)
            xs = 0.0826 * np.cos(us)
            ys = 0.0826 * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='black', alpha=0.2)

        #Levitation Coil
        if showLC:
            zs = np.linspace(0.48, 0.88, 2)
            us, zs = np.meshgrid(us, zs)
            xs = 0.485 * np.cos(us)
            ys = 0.485 * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='y', alpha=0.2)
            zs = np.array([0.48, 0.48])
            us, zs = np.meshgrid(us, zs)
            xs = np.cos(us)
            ys = np.sin(us)
            xs[0] *= 0.485
            ys[0] *= 0.485
            xs[1] *= 0.0826
            ys[1] *= 0.0826
            ax.plot_surface(xs, ys, zs, color='y', alpha=0.2)

        us = np.linspace(0, 1.0 * np.pi, 32)
        #Vacuum Vessel
        if showVV:
            zs = np.linspace(-0.35, 0.35, 2)
            us, zs = np.meshgrid(us, zs)
            xs = 1.0 * np.cos(us)
            ys = 1.0 * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='b', alpha=0.2)

            zs = np.linspace(0.35, 0.53, 10)
            us, zs = np.meshgrid(us, zs)
            xs = (0.8 + np.sqrt(0.04 - (zs - 0.35)**2)) * np.cos(us)
            ys = (0.8 + np.sqrt(0.04 - (zs - 0.35)**2)) * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='b', alpha=0.2)

            zs = np.linspace(-0.35, -0.53, 10)
            us, zs = np.meshgrid(us, zs)
            xs = (0.8 + np.sqrt(0.04 - (zs + 0.35)**2)) * np.cos(us)
            ys = (0.8 + np.sqrt(0.04 - (zs + 0.35)**2)) * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='b', alpha=0.2)

            zs = np.linspace(0.53, 0.66, 2)
            us, zs = np.meshgrid(us, zs)
            xs = (0.8888889 - 2.7838*(zs-0.53)) * np.cos(us)
            ys = (0.8888889 - 2.7838*(zs-0.53)) * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='b', alpha=0.2)

            zs = np.linspace(-0.53, -0.66, 2)
            us, zs = np.meshgrid(us, zs)
            xs = (0.8888889 + 2.7838*(zs+0.53)) * np.cos(us)
            ys = (0.8888889 + 2.7838*(zs+0.53)) * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='b', alpha=0.2)

        ##Coilcase
        us = np.linspace(0, 2.0 * np.pi, 32)
        if showFC:
            zs = np.linspace(-0.02, 0.02, 2)
            us, zs = np.meshgrid(us, zs)
            xs = 0.185 * np.cos(us)
            ys = 0.185 * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='r', alpha=1.0)
            xs = 0.378 * np.cos(us)
            ys = 0.378 * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='r', alpha=1.0)
            zs = np.linspace(0.02, 0.075, 10)
            us, zs = np.meshgrid(us, zs)
            xs = (0.24 - np.sqrt(0.055**2 - (zs - 0.02)**2)) * np.cos(us)
            ys = (0.24 - np.sqrt(0.055**2 - (zs - 0.02)**2)) * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='r', alpha=1.0)
            xs = (0.323+ np.sqrt(0.055**2 - (zs - 0.02)**2)) * np.cos(us)
            ys = (0.323+ np.sqrt(0.055**2 - (zs - 0.02)**2)) * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='r', alpha=1.0)
            zs = np.linspace(-0.02, -0.075, 10)
            us, zs = np.meshgrid(us, zs)
            xs = (0.24 - np.sqrt(0.055**2 - (zs + 0.02)**2)) * np.cos(us)
            ys = (0.24 - np.sqrt(0.055**2 - (zs + 0.02)**2)) * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='r', alpha=1.0)
            xs = (0.323+ np.sqrt(0.055**2 - (zs + 0.02)**2)) * np.cos(us)
            ys = (0.323+ np.sqrt(0.055**2 - (zs + 0.02)**2)) * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='r', alpha=1.0)

            zs = np.array([0.075, 0.075])
            us, zs = np.meshgrid(us, zs)
            xs = np.cos(us)
            ys = np.sin(us)
            xs[0] *= 0.24
            ys[0] *= 0.24
            xs[1] *= 0.323
            ys[1] *= 0.323
            ax.plot_surface(xs, ys, zs, color='r', alpha=1.0)
            zs = np.array([-0.075, -0.075])
            us, zs = np.meshgrid(us, zs)
            xs = np.cos(us)
            ys = np.sin(us)
            xs[0] *= 0.24
            ys[0] *= 0.24
            xs[1] *= 0.323
            ys[1] *= 0.323
            ax.plot_surface(xs, ys, zs, color='r', alpha=1.0)



        set_axes_equal(ax)
        plt.show()

    def load_relative_illumination(self):
        filepath = 'illuminatiouCIS.csv'
        with codecs.open(filepath, 'r', 'Shift-JIS', 'ignore') as f:
            data = pandas.read_csv(filepath)
        #with open(filepath) as f:
        #    data = list(csv.reader(f))
        arr_data = np.array(data)
        relative_illumination = arr_data[:, 1]
        injection_angle = np.deg2rad(arr_data[:, 0])

        #parameter0 = 0
        #popt, pcov = optimize.curve_fit(fit_func_relative_illumination, injection_angle, relative_illumination)
        #plt.plot(injection_angle, fit_func_relative_illumination(injection_angle, *popt))
        #plt.plot(injection_angle, relative_illumination)
        #plt.show()
        #print("LOADED %s", filepath)
        return injection_angle, relative_illumination

    def load_relative_illumination_2D(self):
        filepath = 'RelativeIllumination_100to2000d100mm.csv'
        with codecs.open(filepath, 'r', 'Shift-JIS', 'ignore') as f:
            data = pandas.read_csv(filepath)
        arr_data = np.array(data)
        relative_illumination = arr_data[:, 1]
        #injection_angle = np.deg2rad(arr_data[:, 0])
        injection_angle = arr_data[:, 0]

        distance = np.linspace(100, 2000, 20)
        for i in range(20):
            plt.plot(injection_angle, arr_data[:, 1+4*i], label=('%d mm' % distance[i]))
        plt.legend(title='Distance from \nL1 focal position', fontsize=12)
        plt.xlabel('Injection Angle [degree]')
        plt.ylabel('Relative Illumination Ratio')
        plt.ylim(0, 1.1)
        plt.show()
        #print("LOADED %s", filepath)

def fit_func_relative_illumination(x, a):
    return a*np.cos(x)**4


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)

def make_line_integrated_images(n, k, num_loop, frame=None, num_Process=None):
    imrec = ImageReconstruction(n, k)
    ratio_1st_per_2nd = [0.1, 0.25, 0.5, 1.0, 2.0, 10]
    num = 0
    if num_Process > 1:
        st_loop = np.int((num_loop+3)*frame/num_Process)
        ed_loop = np.int((num_loop+3)*(frame+1)/num_Process)
    else:
        st_loop = 0
        ed_loop = num_loop
        frame = 0
        num_Process = 1

    num_allstep = (num_loop+3)*(num_loop+1)**2*num_loop**4/2

    for j0 in range(st_loop, ed_loop):
        for i1 in range(num_loop):
            for i2 in range(num_loop):
                #for i3 in range(num_loop+1):
                for i3 in range(np.int((num_loop+1)/2)):
                    for j1 in range(num_loop):
                        for j2 in range(num_loop):
                            for j3 in range(num_loop+1):
                                p_opt_best = [1, 1 + 20*i1/num_loop, 0.1 + 2.0*i2/num_loop, 0.38 + 0.32*i3/num_loop]
                                #p_opt_best_2ndPeak = [10**(-1 + 2*j0/num_loop), 1 + 20*j1/num_loop, 0.1 + 2.0*j2/num_loop, 0.70 + 0.25*j3/num_loop]
                                p_opt_best_2ndPeak = [ratio_1st_per_2nd[j0], 1 + 20 * j1 / num_loop,
                                                      0.1 + 2.0 * j2 / num_loop, 0.70 + 0.25 * j3 / num_loop]
                                imrec.plot_3Dto2D(p_opt_best, p_opt_best_2ndPeak)
                                num+=1
                                print('Progress (%d/%d): %d/%d (%.2f percent)' % (frame+1, num_Process, num, num_allstep/num_Process, 100*num*num_Process/num_allstep))

def make_dataset_for_pix2pix(num_loop, frame=None, num_Process=None):
    ratio_1st_per_2nd = [0.1, 0.5, 1.0, 2.0, 10]
    if num_Process > 1:
        st_loop = np.int(num_loop*frame/num_Process)
        ed_loop = np.int(num_loop*(frame+1)/num_Process)
    else:
        st_loop = 0
        ed_loop = num_loop
        frame = 0
        num_Process = 1

    for j0 in range(num_loop+2):
    #for j0 in range(st_loop, ed_loop):
        for i3 in range(num_loop+1):
            for j3 in range(num_loop+1):
                for i1 in range(num_loop):
                    for i2 in range(num_loop):
                        for j1 in range(num_loop):
                            for j2 in range(num_loop):
                                try:
                                    p_opt_best = [1, 1 + 20 * i1 / num_loop, 0.1 + 2.0 * i2 / num_loop,
                                                  0.38 + 0.32 * i3 / num_loop]
                                    # p_opt_best_2ndPeak = [10**(-1 + 2*j0/num_loop), 1 + 20*j1/num_loop, 0.1 + 2.0*j2/num_loop, 0.70 + 0.25*j3/num_loop]
                                    p_opt_best_2ndPeak = [ratio_1st_per_2nd[j0], 1 + 20 * j1 / num_loop,
                                                          0.1 + 2.0 * j2 / num_loop, 0.70 + 0.25 * j3 / num_loop]
                                    #p_opt_best = [1, 1 + 20*i1/num_loop, 0.1 + 2.0*i2/num_loop, 0.38 + 0.32*i3/num_loop]
                                    #p_opt_best_2ndPeak = [10**(-1 + 2*j0/num_loop), 1 + 20*j1/num_loop, 0.1 + 2.0*j2/num_loop, 0.70 + 0.25*j3/num_loop]
                                    images = np.load("dataset_SimCIS_ltd_bb/SimCIS_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.npz" % \
                                                     (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                                                      p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]))
                                    image_local = images['image_local']
                                    image = images['image']
                                    image_1ref = images['image_1ref']

                                    plt.axis("off")
                                    plt.tick_params(bottom=False, left=False, right=False, top=False,labelbottom=False,labelleft=False,labelright=False,labeltop=False)
                                    plt.subplots_adjust(left=0., right=1., bottom=0., top=1.)
                                    plt.imshow(image_local[::-1,:], cmap='jet', interpolation=None)
                                    plt.savefig("dataset_SimCIS_ltd_bb/Local/SimCIS_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.png" % \
                                                (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                                                 p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]),\
                                                bbox_inches="tight", pad_inches=-0.04)
                                    plt.imshow((image + image_1ref).T, cmap='jet', interpolation=None)
                                    plt.savefig("dataset_SimCIS_ltd_bb/Projection/SimCIS_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.png" % \
                                                (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                                                 p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]), \
                                                bbox_inches="tight", pad_inches=-0.04)
                                except:
                                    #import traceback
                                    #traceback.print_exc()
                                    #print("No file: dataset_SimCIS/SimCIS_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.png" % \
                                    #      (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                                    #       p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]))
                                    pass

def png2video(num_loop):
    import cv2
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video = cv2.VideoWriter('video.mp4', fourcc, 20.0, (640, 480))

    for j0 in range(num_loop):
        for i1 in range(num_loop):
            for i2 in range(num_loop):
                for i3 in range(num_loop):
                    for j1 in range(num_loop):
                        for j2 in range(num_loop):
                            for j3 in range(num_loop):
                                try:
                                    p_opt_best = [1, 1 + 20*i1/num_loop, 0.1 + 2.0*i2/num_loop, 0.38 + 0.32*i3/num_loop]
                                    p_opt_best_2ndPeak = [10**(-1 + 2*j0/num_loop), 1 + 20*j1/num_loop, 0.1 + 2.0*j2/num_loop, 0.70 + 0.25*j3/num_loop]
                                    img = cv2.imread("dataset_SimCIS/SimCIS_1st%dmport _%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.png" % \
                                                (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                                                 p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]))
                                    img = cv2.resize(img, (640, 480))
                                    video.write(img)
                                    print(i1, i2, i3, j0, j1, j2, j3)
                                except:
                                    print("No file: dataset_SimCIS/SimCIS_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.png" % \
                                                     (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                                                      p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]))
                                    print(i1, i2, i3, j0, j1, j2, j3)
                                    pass

    video.release()

def mask_with_circle():
    arr_mask = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            if np.int((i-45)**2 + (j-50)**2) < 45**2:
                arr_mask[i, j] = 1
            if i<10 or i>80:
                arr_mask[i, j] = 0


    plt.imshow(arr_mask, cmap='jet')
    plt.savefig("mask_CIS.png")
    #plt.show()

def read_mask():
    img = cv2.imread("test_masked_CIS.png", cv2.IMREAD_COLOR)
    i_max, j_max, _ = np.shape(img)
    img_mask = np.zeros((i_max, j_max))
    for i in range(i_max):
        for j in range(j_max):
            if i>np.int(0.1*i_max) and i<np.int(0.8*i_max) and np.int((i - 0.45*i_max)**2 + (j - 0.5*j_max)**2) < (0.45*i_max)**2:
                img_mask[i, j] = 1

    img[img_mask==0] = [127, 0, 0]

    #img = cv2.imread("test_local.png", cv2.IMREAD_COLOR)
    #img_mask[:,:,0] = 127
    cv2.imwrite("maskedImg.png", img)
    #plt.imshow(img, cmap='jet')
    #plt.show()

def make_1maskedImg(p_opt_best, p_opt_best_2ndPeak):
    img = cv2.imread("test_masked_CIS.png", cv2.IMREAD_COLOR)
    i_max, j_max, _ = np.shape(img)
    img_mask = np.zeros((i_max, j_max))
    for i in range(i_max):
        for j in range(j_max):
            if i>np.int(0.1*i_max) and i<np.int(0.8*i_max) and np.int((i - 0.45*i_max)**2 + (j - 0.5*j_max)**2) < (0.45*i_max)**2:
                img_mask[i, j] = 1
            if i>np.int(0.41*i_max) and i<np.int(0.60*i_max) and j < np.int(0.31*j_max):
                img_mask[i, j] = 0
            if j > np.int(0.310*j_max) and np.int((i - 0.5*i_max)**2 + (j - 0.3*j_max)**2) < (0.09*i_max)**2:
                img_mask[i, j] = 0

    img[img_mask==0] = [127, 0, 0]
    plt.imshow(img)
    plt.show()

    images = np.load("SimCIS_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.npz" % \
                     (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                      p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]))
    image_local = images['image_local']
    image = images['image']
    image_1ref = images['image_1ref']

    plt.axis("off")
    plt.tick_params(bottom=False, left=False, right=False, top=False,labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    plt.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    plt.imshow(image_local[::-1,:], cmap='jet', interpolation=None)
    plt.savefig("dataset_SimCIS_ltd_bb/Local/SimCIS_3peaks_maskedFC_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.png" % \
                (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                 p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]), \
                bbox_inches="tight", pad_inches=-0.04)
    plt.imshow((image + image_1ref).T, cmap='jet', interpolation=None)
    plt.savefig("dataset_SimCIS_ltd_bb/Projection/SimCIS_3peaks_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.png" % \
                (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                 p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]), \
                bbox_inches="tight", pad_inches=-0.04)

    cv2.imwrite("maskedImg.png", img)
    img = cv2.imread("dataset_SimCIS_ltd_bb/Projection/SimCIS_3peaks_maskedFC_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.png" % \
                     (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                      p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]))
    img[img_mask==0] = [127, 0, 0]
    cv2.imwrite("dataset_SimCIS_ltd_bb_masked/SimCIS_3peaks_maskedFC_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.png" % \
                (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                 p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]), img)

def make_maskedImg(num_loop, frame=None, num_Process=None):
    ratio_1st_per_2nd = [0.1, 0.5, 1.0, 2.0, 10]
    img = cv2.imread("test_masked_CIS.png", cv2.IMREAD_COLOR)
    i_max, j_max, _ = np.shape(img)
    img_mask = np.zeros((i_max, j_max))
    for i in range(i_max):
        for j in range(j_max):
            if i>np.int(0.1*i_max) and i<np.int(0.8*i_max) and np.int((i - 0.45*i_max)**2 + (j - 0.5*j_max)**2) < (0.45*i_max)**2:
                img_mask[i, j] = 1

    img[img_mask==0] = [127, 0, 0]

    cv2.imwrite("maskedImg.png", img)
    #if num_Process > 1:
    #    st_loop = np.int(num_loop*frame/num_Process)
    #    ed_loop = np.int(num_loop*(frame+1)/num_Process)
    #else:
    #    st_loop = 0
    #    ed_loop = num_loop
    #    frame = 0
    #    num_Process = 1

    for j0 in range(num_loop+2):
    #for j0 in range(st_loop, ed_loop):
        for i3 in range(num_loop+1):
            for j3 in range(num_loop+1):
                for i1 in range(num_loop):
                    for i2 in range(num_loop):
                        for j1 in range(num_loop):
                            for j2 in range(num_loop):
                                try:
                                    p_opt_best = [1, 1 + 20 * i1 / num_loop, 0.1 + 2.0 * i2 / num_loop,
                                                  0.38 + 0.32 * i3 / num_loop]
                                    # p_opt_best_2ndPeak = [10**(-1 + 2*j0/num_loop), 1 + 20*j1/num_loop, 0.1 + 2.0*j2/num_loop, 0.70 + 0.25*j3/num_loop]
                                    p_opt_best_2ndPeak = [ratio_1st_per_2nd[j0], 1 + 20 * j1 / num_loop,
                                                          0.1 + 2.0 * j2 / num_loop, 0.70 + 0.25 * j3 / num_loop]
                                    #p_opt_best = [1, 1 + 20*i1/num_loop, 0.1 + 2.0*i2/num_loop, 0.38 + 0.32*i3/num_loop]
                                    #p_opt_best_2ndPeak = [10**(-1 + 2*j0/num_loop), 1 + 20*j1/num_loop, 0.1 + 2.0*j2/num_loop, 0.70 + 0.25*j3/num_loop]
                                    img = cv2.imread("dataset_SimCIS_ltd_bb/Projection/SimCIS_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.png" % \
                                                     (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                                                      p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]))
                                    img[img_mask==0] = [127, 0, 0]
                                    cv2.imwrite("dataset_SimCIS_ltd_bb_masked/SimCIS_1st%d_%d_%.1f_%.2f_2nd%.2f_%d_%.1f_%.2f.png" % \
                                                (p_opt_best[0], p_opt_best[1], p_opt_best[2], p_opt_best[3], \
                                                 p_opt_best_2ndPeak[0], p_opt_best_2ndPeak[1], p_opt_best_2ndPeak[2], p_opt_best_2ndPeak[3]), img)

                                except:
                                    #import traceback
                                    #traceback.print_exc()
                                    pass


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err

def rmse(imageA, imageB):
    """ Root Mean Squared Error """
    return np.sqrt(mse(imageA, imageB))


def nrmse(imageA, imageB):
    """ Normalized Root Mean Squared Error """
    return rmse(imageA, imageB) / (imageA.max() - imageA.min())

def compare_images(path_imageA, path_imageB):
    from skimage.measure import compare_ssim, compare_psnr
    # compute the mean squared error and structural similarity
    # index for the images
    imageA = cv2.imread(path_imageA, cv2.IMREAD_GRAYSCALE)
    imageB = cv2.imread(path_imageB, cv2.IMREAD_GRAYSCALE)
    m = mse(imageA, imageB)
    nrm = nrmse(imageA, imageB)
    s = compare_ssim(imageA, imageB)
    p = compare_psnr(imageA, imageB)

    #print(path_imageA, m, nrm, s, p)

    ## setup the figure
    #fig = plt.figure()
    #plt.suptitle("MSE: %.2f, \nSSIM: %.2f, \nPSNR: %.2f" % (m, s, p))

    ## show first image
    #ax = fig.add_subplot(1, 2, 1)
    ##plt.imshow(imageA, cmap=plt.cm.gray)
    #plt.imshow(imageA, cmap=plt.cm.jet)
    #plt.axis("off")

    ## show the second image
    #ax = fig.add_subplot(1, 2, 2)
    #plt.imshow(imageB, cmap=plt.cm.jet)
    #plt.axis("off")

    ## show the images
    #plt.show()

    return m, nrm, s, p

def compare_allimages_indir(path):
    m = []
    nrm = []
    s = []
    p = []
    s_max = 0.
    s_min = 1.
    for pathname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if filename.endswith('-outputs.png'):
                path_A = os.path.join(pathname, filename)
                path_B = path_A[:-11] + "targets.png"
                buf_m, buf_nrm, buf_s, buf_p = compare_images(path_A, path_B)
                m.append(buf_m.tolist())
                nrm.append(buf_nrm.tolist())
                s.append(buf_s.tolist())
                p.append(buf_p.tolist())
                if buf_s > 0.97:
                    print(filename)
                #if s_max < buf_s:
                #    s_max = buf_s
                #    path_A_smax = path_A
                #if s_min > buf_s:
                #    s_min = buf_s
                #    path_A_smin = path_A

    m_mean = np.mean(m)
    m_stdev = np.std(m)
    nrm_mean = np.mean(nrm)
    nrm_stdev = np.std(nrm)
    nrm_min = np.min(nrm)
    s_mean = np.mean(s)
    s_stdev = np.std(s)
    s_max = np.max(s)
    p_mean = np.mean(p)
    p_stdev = np.std(p)
    p_max = np.max(p)

    print(m_mean, m_stdev)
    print(nrm_mean, nrm_stdev, nrm_min)
    print(s_mean, s_stdev, s_max)
    print(p_mean, p_stdev, p_max)
    print("FINISH")

def load_intensityCIS():
    #data = np.load("/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/CoherenceImaging/intensity_CIS/CIS発光量/I0_2D_r_20180921d0055.txt.npy")
    data = np.load("/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/CoherenceImaging/intensity_CIS/発光量(ICH実験)/I0_2D_r_20180921d0069.txt.npy")
    #data = np.load("/Users/kemmochi/SkyDrive/Document/Study/Fusion/RT1/CoherenceImaging/intensity_CIS/発光量(ICH実験)/I0_2D_r_20171111d0037.txt.npy")
    data = data[::-1, :]
    orgHeight, orgWidth = data.shape[:2]
    offset = 100
    offset_up = 40
    offset_left = 30
    length_square = np.max((orgHeight+offset, orgWidth+offset))
    squareImg = np.zeros((length_square, length_square))
    start_up = np.int((length_square-offset-orgHeight)/2) + offset_up
    start_left = np.int((length_square-offset-orgWidth)/2) + offset_left
    squareImg[start_up:start_up+orgHeight, start_left:start_left + orgWidth] = data
    #plt.imshow(squareImg, cmap='jet')
    #plt.show()
    #size = (np.int(orgHeight/10), np.int(orgWidth/10))
    size = (256,256)
    #size = (100, 100)
    OpenCV_data = np.asarray(squareImg)
    resizedImg = cv2.resize(OpenCV_data, size, interpolation=cv2.INTER_CUBIC)
    resizedImg[:,:30] = 0.0
    resizedImg[:,-30:] = 0.0
    blur = cv2.blur(resizedImg, (5,3))
    gblur = cv2.GaussianBlur(resizedImg, (5, 5), 2)
    #mblur = cv2.medianBlur(resizedImg, ksize=5)
    #resizedImg = data.resize(size)
    blur0pad = np.pad(blur, [(40,60), (20,30)], "constant")
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.title('Original')
    plt.imshow(squareImg, cmap='jet')
    plt.xticks([]), plt.yticks([])
    plt.subplot(222)
    plt.title('Resized')
    plt.imshow(resizedImg, cmap='jet')
    plt.xticks([]), plt.yticks([])
    plt.subplot(223)
    plt.imshow(blur, cmap='jet')
    plt.title('Blurred')
    plt.xticks([]), plt.yticks([])
    plt.subplot(224)
    plt.imshow(gblur, cmap='jet')
    plt.title('GaussianBlurred')
    plt.xticks([]), plt.yticks([])
    plt.show()

    plt.axis("off")
    plt.tick_params(bottom=False, left=False, right=False, top=False,labelbottom=False,labelleft=False,labelright=False,labeltop=False)
    plt.subplots_adjust(left=0., right=1., bottom=0., top=1.)
    #plt.imshow(blur[22:-18, 7:-23], cmap='jet', interpolation=None, vmax=610)
    #plt.imshow(blur[26:-51, 13:-13], cmap='jet', interpolation=None, vmax=610)
    #plt.imshow(blur[29:-48, 13:-13], cmap='jet', interpolation=None, vmax=610)
    plt.imshow(blur, cmap='jet', interpolation=None, vmax=640)
    plt.savefig("intensity_CIS_woICH_sn69_256.png", bbox_inches="tight", pad_inches=-0.04)

def changecolormap(image, origin_cmap, target_cmap):
    r = np.linspace(0,1, 256)
    norm = matplotlib.colors.Normalize(0,1)
    mapvals = origin_cmap(norm(r))[:,:3]

    def get_value_from_cm(color):
        color=matplotlib.colors.to_rgb(color)
        #if color is already gray scale, dont change it
        if np.std(color) < 0.1:
            return color
        #otherwise return value from colormap
        distance = np.sum((mapvals - color)**2, axis=1)
        return target_cmap(r[np.argmin(distance)])[:3]

    newim = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            c = image[i,j,:3]
            newim[i,j, :3] =  get_value_from_cm(c)
    return newim

def image2ResizedGray(r_num, z_num):
    #buf_image_local = plt.imread('export_CIS_5_step518k.png')
    buf_image_local = plt.imread('export_intensity_CIS_wICH_sn70.png')
    #buf_image_local = plt.imread('export_intensity_CIS_woICH_sn69.png')
    #buf_image_local = plt.imread('export_test_CIS_square.png')
    #buf_image_local = cv2.resize(buf_image_local, (r_num, z_num))
    #image_local = changecolormap(buf_image_local, plt.cm.jet, plt.cm.viridis)
    image_local = changecolormap(buf_image_local, plt.cm.jet, plt.cm.gray)
    image_local = cv2.cvtColor(image_local, cv2.COLOR_BGR2GRAY)
    #image_local = cv2.resize(image_local, (r_num, z_num))
    i_max, j_max = np.shape(image_local)
    img_mask = np.zeros((i_max, j_max))
    for i in range(i_max):
        for j in range(j_max):
            if i>np.int(0.1*i_max) and i<np.int(0.8*i_max) and np.int((i - 0.45*i_max)**2 + (j - 0.5*j_max)**2) < (0.40*i_max)**2:
                img_mask[i, j] = 1

    image_local[(image_local>0.2) & (img_mask==0)] = 0.0
    #image_local = np.where((image_local>0.25) & (img_mask==0), 0.0, image_local)
    #image_local[img_mask==0] = 0.0
    image_local = image_local[::-1,:]
    #image_local[:15,:] = 0.0
    #image_local[-22:,:] = 0.0
    #image_local[:, -19:] = 0.0
    #image_local[:, 3:] = image_local[:, :-3]
    #image_local[3:, :] = image_local[:-3, :]
    #image_local[2:, :] = image_local[:-2, :]
    #image_local[:, 8:] = image_local[:, :-8]
    image_local[8:, :] = image_local[:-8, :]
    #image_local[5:, :] = image_local[:-5, :]

    #最大が１になるように規格化
    #image_local /= np.max(image_local)
    image_local /= 0.89


    fig = plt.figure(dpi=200)
    ax = plt.subplot(111)
    plt.tick_params(labelsize=15)

    separatrix = True  # if pure dipole configuration, change this to 'False'
    levels = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014]
    rs = np.linspace( 0.0, 1.001, 200)
    zs = np.linspace(-0.5, 0.501, 200)
    r_mesh, z_mesh = np.meshgrid(rs, zs)
    img = plt.imshow(image_local[::,:], origin='lower', cmap='jet', \
                     extent=(rs.min(), rs.max(), zs.min(), zs.max()), vmax=1.0) #vmax=0.9
    mag_strength = np.array(
        [list(map(lambda r, z: np.sqrt(rt1.bvec(r, z, separatrix)[0] ** 2 + rt1.bvec(r, z, separatrix)[1] ** 2),
                  r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
    levels_resonance_ICH_He2 = [2631e-4]    #He2+, 2MHzの共鳴面
    levels_resonance_ICH_He1 = [5263e-4]    #He+, 2MHzの共鳴面
    plt.contour(r_mesh, z_mesh, mag_strength, colors='red', linewidths=3, levels=levels_resonance_ICH_He2)
    plt.contour(r_mesh, z_mesh, mag_strength, colors='red', linewidths=3, linestyles='dashed',
                levels=levels_resonance_ICH_He1)
    psi = np.array([list(map(lambda r, z : rt1.psi(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
    coilcase_truth_table = np.array([list(map(lambda r, z : rt1.check_coilcase(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(rs), len(zs))
    psi[coilcase_truth_table == True] = 0
    plt.contour(r_mesh, z_mesh, psi, colors=['white'], linewidths=0.5, levels=levels)
    #plt.contour(r_mesh, z_mesh, mag_strength, colors='red', linewidths=3, levels=levels_resonance)
    ##plt.contour(r_mesh, z_mesh, mag_strength, colors='red', linewidths=3, linestyles='dashed',
    #            levels=levels_resonance_2nd)

    #plt.title(r'$n_\mathrm{e}$')
    plt.xlabel(r'$r\mathrm{\ [m]}$')
    plt.ylabel(r'$z\mathrm{\ [m]}$')
    plt.xlim(0.05, 0.95)
    plt.ylim(-0.3, 0.4)
    # plt.gca():現在のAxesオブジェクトを返す
    divider = make_axes_locatable(plt.gca())
    # カラーバーの位置
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = plt.colorbar(img, cax=cax)
    cb.set_clim(0,6.4)
    cb.ax.tick_params(labelsize=15)
    #cb.set_label('Intensity [a.u.]')
    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
    plt.tight_layout()
    plt.show()

    return image_local

if __name__ == '__main__':
    start = time.time()

    n, k = 1.7689, 0.60521 #FB450-10 実測値
    #n, k = 0.70249, 0.36890 #FL730-10 実測値
    make_line_integrated_images(n, k, num_loop=3, frame=0, num_Process=6)
    #png2video(3)
    #make_dataset_for_pix2pix(3)
    #mask_with_circle()
    #make_maskedImg(num_loop=3)
    #make_1maskedImg(p_opt_best=[1, 7, 0.77, 0.4], p_opt_best_2ndPeak=[0.5, 7, 1.43, 0.90])
    #path_A = "/Volumes/kemmochi/Documents/RT1/pix2pix-tensorflow/modified_docker/Projection2Local/masked/epochs100/images/SimCIS_1st1_7_0.8_0.38_2nd0.50_1_1.4_0.78-outputs.png"
    #path_B = "/Volumes/kemmochi/Documents/RT1/pix2pix-tensorflow/modified_docker/Projection2Local/masked/epochs100/images/SimCIS_1st1_7_0.8_0.38_2nd0.50_1_1.4_0.78-targets.png"
    #compare_images(path_A, path_B)
    #compare_allimages_indir("/Volumes/kemmochi/Documents/RT1/pix2pix-tensorflow/modified_docker/Projection2Local/masked/epochs100/images")
    #load_intensityCIS()
    #image2ResizedGray(r_num=100, z_num=100)

    #n, k = 0.70249, 0.36890 #FL730-10 実測値
    #n, k = 1.7689, 0.60521 #FB450-10 実測値
    #imrec = ImageReconstruction(n, k)
    #imrec.projection_poroidally(1.2, np.pi/4, np.pi/4)
    #imrec.projection_poroidally(0.9, 0, 0)
    #imrec.plot_projection()
    #imrec.plot_projection_image()
    #imrec.bokeh_local_image()
    #imrec.plot_local_image(p_opt_best=[30, 18, 1.0, 0.5], p_opt_best_2ndPeak=[20, 17, 0.1, 0.75])
    #imrec.plot_local_image(p_opt_best=[1, 7, 0.77, 0.4], p_opt_best_2ndPeak=[0.5, 7, 1.43, 0.90])
    #imrec.spline_image()
    #imrec.plot_projection_image_spline()
    #imrec.plot_projection_image_spline_wrt_1reflection(reflection_factor=0.5)
    #imrec.plot_projection_image_spline_wrt_1reflection_v3(reflection_factor=0.5)
    #imrec.run()
    #imrec.plot_refractive_indices(2.6580, 2.8125)
    #imrec.plot_refractive_indices(2.15, 0.8)
    #imrec.plot_refractive_indices(1.2, 1.1)
    #imrec.plot_refractive_indices(n, k)
    #imrec.plot_refractive_indices(0.70897, 0.42432)
    #imrec.plot_refractive_indices_388_447_706_728()
    #imrec.plot_3D_ray(showRay=True, showFC=True, showLC=True, showVV=True, showCS=True)
    #imrec.plot_3D_ray_broadcast(showRay=True, showFC=False, showLC=False, showVV=False, showCS=False)
    #imrec.load_relative_illumination_2D()
    #imrec.show_animation()
    #imrec.plot_3Dto2D_broadcast()
    #imrec.plot_3Dto2D(p_opt_best=[30, 18, 1.0, 0.5], p_opt_best_2ndPeak=[20, 17, 0.1, 0.75])
    #imrec.plot_3Dto2D(p_opt_best=[1, 7, 0.77, 0.59], p_opt_best_2ndPeak=[0.5, 7, 1.43, 0.87])
    #imrec.plot_3Dto2D(p_opt_best=[1, 7, 1.00, 0.49], p_opt_best_2ndPeak=[1.0, 13, 0.30, 0.75])
    #imrec.load_image(p_opt_best=[30, 18, 1.0, 0.5], p_opt_best_2ndPeak=[20, 17, 0.1, 0.75])
    #imrec.load_relative_illumination()

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

