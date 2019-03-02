import numpy as np
import matplotlib.pyplot as plt
import ne_profile_r2
import rt1mag as rt1
import time
#import numba
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RectBivariateSpline, interp1d
from bokeh.plotting import figure, output_file, show, reset_output
from multiprocessing import Pool
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

def unwrap_self_plot_projection_image_spline_wrt_1reflection_MP(arg, **kwarg):
    return ImageReconstruction.plot_projection_image_spline_wrt_1reflection_MP(*arg, **kwarg)

class ImageReconstruction:
    def __init__(self):
        #self.p_opt_best = [35.210, 6.341, 1.501, 0.544]
        self.p_opt_best = [30, 18, 1.0, 0.5]
        self.R_cam = 1.2 #[m]
        self.r_num = 100
        self.z_num = 100
        self.theta_max = np.arcsin(1/self.R_cam)#np.pi/3
        self.phi_max = self.theta_max/2#np.pi/6

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

    def plot_local_image(self):
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
        plt.imshow(ne, cmap='jet', origin='lower', extent=(r.min(), r.max(), z.min(), z.max()), vmax=35)
        plt.colorbar()
        plt.contour(r_mesh, z_mesh, psi, colors=['white'], linewidths=0.5, levels=levels)
        plt.title(r'$n_\mathrm{e}$')
        plt.xlabel(r'$r\mathrm{\ [m]}$')
        plt.ylabel(r'$z\mathrm{\ [m]}$')
        plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=1.0)
        plt.show()

    def make_ne_image(self, r, z):
        r_mesh, z_mesh = np.meshgrid(r, z)
        psi = np.array([list(map(lambda r, z : rt1.psi(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))
        #coilcase_truth_table = np.array([list(map(lambda r, z : rt1.check_coilcase(r, z), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))
        #psi[coilcase_truth_table == True] = 0
        psix  = rt1.psi(self.p_opt_best[3], 0.0, separatrix=True) # psi上のBが最小となる直線上の密度最大値
        psi0  = rt1.psi(1.0, 0.0, separatrix=True) # psi at the vacuum chamber
        ne = np.array([list(map(lambda r, z : ne_profile_r2.ne_single_gaussian(r, z, *self.p_opt_best, psix=psix, psi0=psi0), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))

        # For double peak profile
        p_opt_best_2ndPeak = [20, 17, 0.1, 0.75]
        psix  = rt1.psi(p_opt_best_2ndPeak[3], 0.0, separatrix=True) # psi上のBが最小となる直線上の密度最大値
        ne += np.array([list(map(lambda r, z : ne_profile_r2.ne_single_gaussian(r, z, *p_opt_best_2ndPeak, psix=psix, psi0=psi0), r_mesh.ravel(), z_mesh.ravel()))]).ravel().reshape(len(r), len(z))

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
        r_TM = (-(n_R**2 - n_I**2 + 2j*n_R*n_I)*np.cos(theta_i) + np.sqrt((n_R**2 - n_I**2 - np.sin(theta_i)**2) + 2j*n_I*n_R))\
              /((n_R**2 - n_I**2 + 2j*n_R*n_I)*np.cos(theta_i) + np.sqrt((n_R**2 - n_I**2 - np.sin(theta_i)**2) + 2j*n_I*n_R))

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

    def cal_injection_angle(self, dist_from_cam, z):
        cos_theta_i = (dist_from_cam**2 + (z**2 + 1) - self.R_cam**2)/(2*dist_from_cam*np.sqrt(z**2 + 1))

        return np.arccos(cos_theta_i)

    def ray_trace_3D(self, dist_from_cam, theta, phi):
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
        flg_reflection = 0
        try:
            if is_inside_vacuum_vessel_1[0][-1] > is_inside_vacuum_vessel_0[0][0]:
                flg_reflection += 1
                index_is_inside_vacuum_vessel = np.min(np.where(is_inside_vacuum_vessel_1 > is_inside_vacuum_vessel_0[0][0], is_inside_vacuum_vessel_1, np.inf)) - 1
                reflectionpoint = np.array([x[index_is_inside_vacuum_vessel], y[index_is_inside_vacuum_vessel], z[index_is_inside_vacuum_vessel]])
                vec_n = np.array([-reflectionpoint[0], -reflectionpoint[1], 0])
                sign_vec_ref = np.array([1, 1, 1])
            elif np.sum(is_inside_vacuum_vessel_2)>0:
                flg_reflection += 1
                index_is_inside_vacuum_vessel = np.min(np.argwhere((np.abs(z)>0.35) & (np.abs(z)<0.53) & (np.sqrt(x**2 + y**2)>(0.8 + np.sqrt(0.04 - (np.abs(z) - 0.35)**2))))) - 1
                reflectionpoint = np.array([x[index_is_inside_vacuum_vessel], y[index_is_inside_vacuum_vessel], z[index_is_inside_vacuum_vessel]])
                vec_n = np.array([-0.8*reflectionpoint[0], -0.8*reflectionpoint[1], 0.35*np.sign(reflectionpoint[2])])
                sign_vec_ref = np.array([1, 1, -1])
            elif np.sum(is_inside_vacuum_vessel_3)>0:
                flg_reflection += 1
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
            reflection_factor = self.cal_injection_angle_for2vector(vec_i, vec_n)
        except:
            pass

        #===============================================================================================================
        #   コイルとの接触を判定
        #===============================================================================================================
        r = np.sqrt(x**2 + y**2)
        is_inside_fcoil_0 = np.argwhere((r >= 0.185) & (r <= 0.1850 + 0.1930) & (z >= -.0200) & (z <= .0200))
        is_inside_fcoil_1 = np.argwhere((r >= 0.185 + 0.0550) & (r <= 0.1850 + 0.1930 - 0.0550) & (z >= -.0750) & (z <= .0750))
        is_inside_fcoil_2 = np.argwhere(((r - (0.185  + 0.055))**2 + (np.abs(z) - (0.020))**2 <= 0.055**2))
        is_inside_fcoil_3 = np.argwhere(((r - (0.185 + 0.193 - 0.055))**2 + (np.abs(z) - (0.020))**2 <= 0.055**2))
        #buf_index = dist_from_cam.__len__()
        buf_index = []
        if np.sum(is_inside_fcoil_0)>0:
            buf_index.append(np.min(is_inside_fcoil_0))
        if np.sum(is_inside_fcoil_1)>0:
            buf_index.append(np.min(is_inside_fcoil_1))
        if np.sum(is_inside_fcoil_2)>0:
            buf_index.append(np.min(is_inside_fcoil_2))
        if np.sum(is_inside_fcoil_3)>0:
            buf_index.append(np.min(is_inside_fcoil_3))
        try:
            index_is_inside_fcoil = np.min(buf_index)
            x[index_is_inside_fcoil+1:] = np.nan
            y[index_is_inside_fcoil+1:] = np.nan
            z[index_is_inside_fcoil+1:] = np.nan
            x_1ref[index_is_inside_fcoil+1:] = np.nan
            y_1ref[index_is_inside_fcoil+1:] = np.nan
            z_1ref[index_is_inside_fcoil+1:] = np.nan
        except:
            pass
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
        #===============================================================================================================
        #   センタースタックとの接触を判定
        #===============================================================================================================
        try:
            r = np.sqrt(x**2 + y**2)
            index_is_inside_vacuum_vessel = np.min(np.argwhere((r <= 0.0826) & (z >= -.6600) & (z <= .6600)))-1
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
        #return x*is_inside_vacuum_vessel, y*is_inside_vacuum_vessel, z*is_inside_vacuum_vessel

    def cal_injection_angle_for2vector(self, vec_i, vec_n):
        inner = np.inner(vec_i, -vec_n)
        if inner < 0:
            inner = np.inner(vec_i, vec_n)
        norm = np.linalg.norm(vec_i) * np.linalg.norm(vec_n)
        c = inner/norm
        #injection_angle = np.arccos(np.clip(c, 0, 1.0))
        injection_angle = np.arccos(c)
        R_s, R_p = self.cal_refractive_indices_metal(injection_angle, 2.6580, 2.8125)
        reflection_factor = (R_s + R_p)/2

        return reflection_factor

    def cal_reflection_vector(self, vec_i, vec_n):
        """
        入射ベクトルと法線ベクトルを入力として，反射ベクトルを返します
        :param vec_i:入射ベクトル
        :param vec_n: 反射ベクトル
        :return:
        """
        norm_vec_n = vec_n/np.linalg.norm(vec_n)
        return vec_i - 2*np.dot(vec_i, norm_vec_n)*norm_vec_n

    def plot_3Dto2D(self):
        dist_from_cam = np.linspace(0, 3, 100)
        theta = self.theta_max*np.arange(self.r_num)/self.r_num
        phi = self.phi_max - 2*self.phi_max*np.arange(self.z_num)/self.z_num
        r_image = np.linspace(0, 1, self.r_num)
        z_image = np.linspace(-0.5, 0.5, self.z_num)
        image = self.make_ne_image(r_image, z_image)
        interp_image = RectBivariateSpline(z_image, r_image, image)
        image_buf = np.zeros((self.r_num, self.z_num, dist_from_cam.__len__()))
        image_1ref_buf = np.zeros((self.r_num, self.z_num, dist_from_cam.__len__()))
        for i in range(self.r_num):
            for j in range(self.z_num):
                x, y, z, x_1ref, y_1ref, z_1ref, reflection_factor = self.ray_trace_3D(dist_from_cam, theta[i], phi[j])
                r = np.sqrt(x**2 + y**2)
                r_1ref = np.sqrt(x_1ref**2 + y_1ref**2)
                #plt.plot(np.sqrt(x**2 + y**2), z)
                #plt.plot(np.sqrt(x_1ref**2 + y_1ref**2), z_1ref)
                image_buf[i, j, :] = interp_image(z, r, grid=False)
                image_1ref_buf[i, j, :] = reflection_factor*interp_image(z_1ref, r_1ref, grid=False)
                #image_1ref_buf[i, j, :] = interp_image(z_1ref, r_1ref, grid=False)
                #image_1ref_buf[i, j, :] = reflection_factor

        image_buf[np.isnan(image_buf)] = 0
        image_1ref_buf[np.isnan(image_1ref_buf)] = 0
        image = np.sum(image_buf, axis=2)
        image_1ref = np.sum(image_1ref_buf, axis=2)
        plt.imshow((image + image_1ref).T, cmap='jet')
        #plt.imshow(image_1ref.T, cmap='jet')

        #plt.show()
        plt.savefig("integrated_image_3Dto2D.png")

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

    def plot_3D_ray(self, frame=None, showRay=True, showVV=False, showFC=False, showLC=False, showCS=False):
        # (x, y, z)
        dist_from_cam = np.linspace(0, 4, 100)
        fig = plt.figure()
        # 3Dでプロット
        ax = Axes3D(fig)
        if showRay:
            theta = self.theta_max*np.arange(self.r_num)/self.r_num
            phi = self.phi_max - 2*self.phi_max*np.arange(self.z_num)/self.z_num
            for i in range(self.r_num):
                for j in range(self.z_num):
                    x, y, z, x_1ref, y_1ref, z_1ref,_ = self.ray_trace_3D(dist_from_cam, theta[i], phi[j])
                    #x, y, z, x_1ref, y_1ref, z_1ref,_ = self.ray_trace_3D(dist_from_cam, theta[i], np.pi/20)
                    #x, y, z, x_1ref, y_1ref, z_1ref, _ = self.ray_trace_3D(dist_from_cam, 1*np.pi/10, phi[j])
                    #x, y, z, x_1ref, y_1ref, z_1ref, _ = self.ray_trace_3D(dist_from_cam, frame*np.pi/20, phi[j])
                    ax.plot(x, y, z, "-", color="#00aa00", ms=4, mew=0.5)
                    ax.plot(x_1ref, y_1ref, z_1ref, "-", color="#aa0000", ms=4, mew=0.5)
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
            ax.plot_surface(xs, ys, zs, color='b', alpha=0.3)

            zs = np.linspace(0.35, 0.53, 10)
            us, zs = np.meshgrid(us, zs)
            xs = (0.8 + np.sqrt(0.04 - (zs - 0.35)**2)) * np.cos(us)
            ys = (0.8 + np.sqrt(0.04 - (zs - 0.35)**2)) * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='b', alpha=0.3)

            zs = np.linspace(-0.35, -0.53, 10)
            us, zs = np.meshgrid(us, zs)
            xs = (0.8 + np.sqrt(0.04 - (zs + 0.35)**2)) * np.cos(us)
            ys = (0.8 + np.sqrt(0.04 - (zs + 0.35)**2)) * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='b', alpha=0.3)

            zs = np.linspace(0.53, 0.66, 2)
            us, zs = np.meshgrid(us, zs)
            xs = (0.8888889 - 2.7838*(zs-0.53)) * np.cos(us)
            ys = (0.8888889 - 2.7838*(zs-0.53)) * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='b', alpha=0.3)

            zs = np.linspace(-0.53, -0.66, 2)
            us, zs = np.meshgrid(us, zs)
            xs = (0.8888889 + 2.7838*(zs+0.53)) * np.cos(us)
            ys = (0.8888889 + 2.7838*(zs+0.53)) * np.sin(us)
            ax.plot_surface(xs, ys, zs, color='b', alpha=0.3)

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

if __name__ == '__main__':
    start = time.time()

    imrec = ImageReconstruction()
    #imrec.projection_poroidally(1.2, np.pi/4, np.pi/4)
    #imrec.projection_poroidally(0.9, 0, 0)
    #imrec.plot_projection()
    #imrec.plot_projection_image()
    #imrec.bokeh_local_image()
    #imrec.plot_local_image()
    #imrec.spline_image()
    #imrec.plot_projection_image_spline()
    #imrec.plot_projection_image_spline_wrt_1reflection(reflection_factor=0.5)
    #imrec.plot_projection_image_spline_wrt_1reflection_v3(reflection_factor=0.5)
    #imrec.run()
    #imrec.plot_refractive_indices(2.6580, 2.8125)
    #imrec.plot_3D_ray(showRay=False, showFC=False, showLC=False, showVV=True, showCS=False)
    #imrec.show_animation()
    imrec.plot_3Dto2D()

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

