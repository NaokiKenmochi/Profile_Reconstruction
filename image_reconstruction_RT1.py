import numpy as np
import matplotlib.pyplot as plt
import ne_profile_r2
import rt1mag as rt1
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RectBivariateSpline

class ImageReconstruction:
    def __init__(self):
        #self.p_opt_best = [35.210, 6.341, 1.501, 0.544]
        self.p_opt_best = [35, 10, 3.0, 0.45]
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
        p_opt_best_2ndPeak = [20, 10, 0.2, 0.75]
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
        for i in range(self.r_num):
            for j in range(self.z_num):
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
        for i in range(self.r_num):
            for j in range(self.z_num):
                #_,_,image[i, j] = self.projection_poroidally(self.theta_max*j/self.r_num, self.phi_max - 2*self.phi_max*i/self.z_num)
                _,_,image[i, j] = self.projection_poroidally_wrt_1reflection(self.theta_max*j/self.r_num, self.phi_max - 2*self.phi_max*i/self.z_num)

        plt.imshow(image, cmap='jet')
        plt.show()

    def plot_projection_image_spline_wrt_1reflection_v2(self, reflection_factor=1.0):
        dist_from_cam = np.linspace(0, 4.0, 400)

        r_image = np.linspace(0, 1, self.r_num)
        #z_image = np.linspace(-1, 1, self.z_num)
        z_image = np.linspace(-0.5, 0.5, self.z_num)
        image = self.make_ne_image(r_image, z_image)
        plt.imshow(image, cmap='jet')
        plt.show()
        interp_image = RectBivariateSpline(z_image, r_image, image)
        image = np.zeros((self.r_num, self.z_num))
        theta = self.theta_max*np.range(self.z_num)/self.r_num
        phi = self.phi_max - 2*self.phi_max*np.range(self.r_num)/self.z_num
        r_sightline = np.sqrt(self.R_cam**2 + (dist_from_cam*np.cos(phi))**2 - 2*self.R_cam*dist_from_cam*np.cos(phi)*np.cos(theta))
        z_sightline = dist_from_cam*np.sin(phi)
        for i in range(self.r_num):
            for j in range(self.z_num):
                #theta = self.theta_max*j/self.r_num
                #phi = self.phi_max - 2*self.phi_max*i/self.z_num
                #r_sightline = np.sqrt(self.R_cam**2 + (dist_from_cam*np.cos(phi))**2 - 2*self.R_cam*dist_from_cam*np.cos(phi)*np.cos(theta))
                #z_sightline = dist_from_cam*np.sin(phi)
                i_sightline = 0
                j_sightline = 0
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
                    buf += interp_image(z_sightline[i_sightline], r_sightline[j_sightline-(i_sightline-j_sightline)-1]) * reflection_factor
                else:
                    r_sightline[i_sightline:] = self.R_cam
                    z_sightline[i_sightline:] = 0.0
                image[i, j] = buf

        plt.imshow(image, cmap='jet')
        plt.title('Reflection factor = %.1f' % reflection_factor)
        plt.show()

    def plot_projection_image_spline_wrt_1reflection(self, reflection_factor=1.0):
        dist_from_cam = np.linspace(0, 4.0, 400)

        r_image = np.linspace(0, 1, self.r_num)
        #z_image = np.linspace(-1, 1, self.z_num)
        z_image = np.linspace(-0.5, 0.5, self.z_num)
        image = self.make_ne_image(r_image, z_image)
        plt.imshow(image, cmap='jet')
        plt.show()
        interp_image = RectBivariateSpline(z_image, r_image, image)
        image = np.zeros((self.r_num, self.z_num))
        for i in range(self.r_num):
            for j in range(self.z_num):
                theta = self.theta_max*j/self.r_num
                phi = self.phi_max - 2*self.phi_max*i/self.z_num
                r_sightline = np.sqrt(self.R_cam**2 + (dist_from_cam*np.cos(phi))**2 - 2*self.R_cam*dist_from_cam*np.cos(phi)*np.cos(theta))
                z_sightline = dist_from_cam*np.sin(phi)
                i_sightline = 0
                j_sightline = 0
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
                    buf += interp_image(z_sightline[i_sightline], r_sightline[j_sightline-(i_sightline-j_sightline)-1]) * reflection_factor
                else:
                    r_sightline[i_sightline:] = self.R_cam
                    z_sightline[i_sightline:] = 0.0
                image[i, j] = buf

        plt.imshow(image, cmap='jet')
        plt.title('Reflection factor = %.1f' % reflection_factor)
        plt.show()

    def plot_projection_image_spline(self):
        dist_from_cam = np.linspace(0, 2.0, 200)

        r_image = np.linspace(0, 1, self.r_num)
        #z_image = np.linspace(-1, 1, self.z_num)
        z_image = np.linspace(-0.5, 0.5, self.z_num)
        image = self.make_ne_image(r_image, z_image)
        interp_image = RectBivariateSpline(z_image, r_image, image)
        image = np.zeros((self.r_num, self.z_num))
        for i in range(self.r_num):
            for j in range(self.z_num):
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
if __name__ == '__main__':
    start = time.time()

    imrec = ImageReconstruction()
    #imrec.projection_poroidally(1.2, np.pi/4, np.pi/4)
    #imrec.projection_poroidally(0.9, 0, 0)
    #imrec.plot_projection()
    #imrec.plot_projection_image()
    #imrec.plot_local_image()
    #imrec.spline_image()
    #imrec.plot_projection_image_spline()
    imrec.plot_projection_image_spline_wrt_1reflection(reflection_factor=0.44)

    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time) + "[sec]")

