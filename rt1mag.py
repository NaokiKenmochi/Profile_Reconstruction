import numpy as np 
import scipy.special as spe


#-----------------------------------------#
#         RT-1 coil vals (globals)
#-----------------------------------------#
cic  = 250.0e3
cvf2 = 28.8e3
#-----------------------------------------#
xcc = 2.e-7



#     ============================================================================
def check_coilcase(r, z):
#     ============================================================================
#                                                             is inside coil case?
#                                                             ====================
	if r >= 0.185 and r <= 0.1850 + 0.1930 and z >= -.0200 and z <= .0200:
		return True
	elif r >= 0.185 + 0.0550 and r <= 0.1850 + 0.1930 - 0.0550 and z >= -.0750 and z <= .0750:
		return True
	elif (r - (0.185  + 0.055))**2 + (z - (0.020))**2 <= 0.055**2:
		return True
	elif (r - (0.185 + 0.055))**2 + (z + (0.020))**2 <= 0.055**2:
		return True
	elif (r - (0.185 + 0.193 - 0.055))**2 + (z - (0.020))**2 <= 0.055**2:
		return True
	elif (r - (0.185 + 0.193 - 0.055))**2 + (z + (0.020))**2 <= 0.055**2:
		return True
	else:
		return False


#     ============================================================================
def check_vacuum_vessel(r, z):
	#     ============================================================================
	#                                                         is inside Vacuum vessel?
	#                                                             ====================
	if r <= 1.000:
		return True
	else:
		return False


#     ============================================================================
def rathe(rc, r, z, ci):
#     calculate "atheta" produced by a loop current
#     rc:loop radius 
#     r, z:position 
#     ci:coil current 
#     Calculate magnetic surface created by the ring current. calculate ra_theta.h
#     ============================================================================
#                                                              biot-savalt formula
#                                                              ===================
    zk = 4.0*rc*r/((rc + r)*(rc + r) + z*z)
    
    if(zk == 0.):
        return 0.0

    a0 = 2.0*xcc*ci*r/np.sqrt(zk)
    a1 = np.sqrt(rc/r)
    a2 = 1.0 - 0.5*zk

    fk = spe.ellipk(zk)
    fe = spe.ellipe(zk)

    return a0*a1*(a2*fk - fe)



#     ============================================================================
def psi(r, z, separatrix=True):
#     ============================================================================
#     magnetic flux surface
#     ============================================================================
#                                                                    calculate psi
#                                                                    =============

    #    internal coil
    atv = rathe(0.25, r, z, cic)

    #    levitating coil
    at = 0.0
    if separatrix:
        at = rathe(0.4, r, z - 0.612, cvf2)

    psi = atv + at

    # #    set zero inside the coil case
    # if check_coilcase(r, z):
    # psi = 0.0

    return psi



#     ============================================================================
def bloop(rc, r, z, ci):
#     calculate the br and bz produced by a loop current
#     rc:loop radius r, z:position ci:coil current 
#     ============================================================================
#                                                              biot-savalt formula
#                                                              ===================
	zk = 4.0*rc*r/((rc + r)*(rc + r) + z*z)
	fk = spe.ellipk(zk)
	fe = spe.ellipe(zk)

	a  = xcc*ci/np.sqrt((rc + r)*(rc + r) + z*z)
	g  = rc*rc + r*r + z*z
	ff = (rc - r)*(rc - r) + z*z
	e  = a*z/r
	h  = rc*rc - r*r - z*z
	bz = a*(h*fe/ff + fk)
	br = e*(g*fe/ff - fk)

	if(zk == 0.0):
		br = 0.0

	return br, bz



#     ============================================================================
def bvec(r, z, separatrix=True):
#     ============================================================================
#     magnetic field vector
#     ============================================================================
#                                                                 calculate br, bz
#                                                                 ================
	#     floating coil
	br_flt, bz_flt = bloop(0.25, r, z, cic)

	#     levitating coil
	br_lev = bz_lev = 0.0
	if separatrix:
		br_lev, bz_lev = bloop(0.40, r, z - 0.612, cvf2)

	return br_flt + br_lev, bz_flt + bz_lev



#     ============================================================================
def b0(r, z, separatrix=True):
#     ============================================================================
#     magnetic field strength on the equatorial plane with the same psi
#     ============================================================================
#                                                                     calculate b0
#                                                                     ============
	p = psi(r, z, separatrix)

	#    calcuate r with same psi on z=0
	if separatrix:
		rifz= \
		5.048895851e0 \
		-1737.177125e0*p \
		+333629.588e0*p**2 \
		-41308926.8e0*p**3 \
		+3516463158.0e0*p**4 \
		-2.105742336e+11*p**5 \
		+8.863092318e+12*p**6 \
		-2.565289537e+14*p**7 \
		+4.854867131e+15*p**8 \
		-5.400623266e+16*p**9 \
		+2.673380473e+17*p**10 
	else:
		rifz = \
		4.366448e0\
		-1667.72e0*p\
		+363041.84e0*p**2\
		-49517859.0e0*p**3\
		+4.4418832e09*p**4\
		-2.659374e11*p**5\
		+1.0515442e13*p**6\
		-2.633817e14*p**7\
		+3.7816742e15*p**8\
		-2.3686326e16*p**9


	#     floating coil
	br_flt, bz_flt = bloop(0.25, rifz, 0.0, cic)

	#     levitating coil
	br_lev = bz_lev = 0.0
	if separatrix:
		br_lev, bz_lev = bloop(0.40, rifz, 0.0 - 0.612, cvf2)

	return np.sqrt( (br_flt + br_lev)**2 + (bz_flt + bz_lev)**2 )



#     ============================================================================
def l(r, z):
#     ============================================================================
#     magnetic field line length
#     ============================================================================
#                                                                      calculate l
#                                                                      ===========
  dl = 1.e-3
  if z < 0.0:
    dl = -dl
  l   = 0.0
  r_ = r
  z_ = z
  while  z_*z > 0.0: 
    br, bz = bvec(r_, z_)
    bb = np.sqrt(br**2 + bz**2)
    r_ = r_ + br/bb*dl
    z_ = z_ + bz/bb*dl
    l = l + dl
  return l
