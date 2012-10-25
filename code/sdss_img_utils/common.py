import os
from pyfits_utils import fits_table
from miscutils import get_overlapping_region
import numpy as np
import pyfits

try:
	import cutils
except:
	cutils = None

cas_flags = dict(
	CANONICAL_CENTER = 0x0000000000000001,
	BRIGHT = 0x0000000000000002,
	EDGE = 0x0000000000000004,
	BLENDED = 0x0000000000000008,
	CHILD = 0x0000000000000010,
	PEAKCENTER = 0x0000000000000020,
	NODEBLEND = 0x0000000000000040,
	NOPROFILE = 0x0000000000000080,
	NOPETRO = 0x0000000000000100,
	MANYPETRO = 0x0000000000000200,
	NOPETRO_BIG = 0x0000000000000400,
	DEBLEND_TOO_MANY_PEAKS = 0x0000000000000800,
	COSMIC_RAY = 0x0000000000001000,
	MANYR50 = 0x0000000000002000,
	MANYR90 = 0x0000000000004000,
	BAD_RADIAL = 0x0000000000008000,
	INCOMPLETE_PROFILE = 0x0000000000010000,
	INTERP = 0x0000000000020000,
	SATURATED = 0x0000000000040000,
	NOTCHECKED = 0x0000000000080000,
	SUBTRACTED = 0x0000000000100000,
	NOSTOKES = 0x0000000000200000,
	BADSKY = 0x0000000000400000,
	PETROFAINT = 0x0000000000800000,
	TOO_LARGE = 0x0000000001000000,
	DEBLENDED_AS_PSF = 0x0000000002000000,
	DEBLEND_PRUNED = 0x0000000004000000,
	ELLIPFAINT = 0x0000000008000000,
	BINNED1 = 0x0000000010000000,
	BINNED2 = 0x0000000020000000,
	BINNED4 = 0x0000000040000000,
	MOVED = 0x0000000080000000,
	DEBLENDED_AS_MOVING = 0x0000000100000000,
	NODEBLEND_MOVING = 0x0000000200000000,
	TOO_FEW_DETECTIONS = 0x0000000400000000,
	BAD_MOVING_FIT = 0x0000000800000000,
	STATIONARY = 0x0000001000000000,
	PEAKS_TOO_CLOSE = 0x0000002000000000,
	MEDIAN_CENTER = 0x0000004000000000,
	LOCAL_EDGE = 0x0000008000000000,
	BAD_COUNTS_ERROR = 0x0000010000000000,
	BAD_MOVING_FIT_CHILD = 0x0000020000000000,
	DEBLEND_UNASSIGNED_FLUX = 0x0000040000000000,
	SATUR_CENTER = 0x0000080000000000,
	INTERP_CENTER = 0x0000100000000000,
	DEBLENDED_AT_EDGE = 0x0000200000000000,
	DEBLEND_NOPEAK = 0x0000400000000000,
	PSF_FLUX_INTERP = 0x0000800000000000,
	TOO_FEW_GOOD_DETECTIONS = 0x0001000000000000,
	CENTER_OFF_AIMAGE = 0x0002000000000000,
	DEBLEND_DEGENERATE = 0x0004000000000000,
	BRIGHTEST_GALAXY_CHILD = 0x0008000000000000,
	CANONICAL_BAND = 0x0010000000000000,
	AMOMENT_FAINT = 0x0020000000000000,
	AMOMENT_SHIFT = 0x0040000000000000,
	AMOMENT_MAXITER = 0x0080000000000000,
	MAYBE_CR = 0x0100000000000000,
	MAYBE_EGHOST = 0x0200000000000000,
	NOTCHECKED_CENTER = 0x0400000000000000,
	OBJECT2_HAS_SATUR_DN = 0x0800000000000000,
	OBJECT2_DEBLEND_PEEPHOLE = 0x1000000000000000,
	GROWN_MERGED = 0x2000000000000000,
	HAS_CENTER = 0x4000000000000000,
	RESERVED = 0x8000000000000000,
	)
	

def band_names():
	return ['u','g','r','i','z']

def band_name(b):
	if b in band_names():
		return b
	if b in [0,1,2,3,4]:
		return 'ugriz'[b]
	raise Exception('Invalid SDSS band: "' + str(b) + '"')

def band_index(b):
	if b in band_names():
		return 'ugriz'.index(b)
	if b in [0,1,2,3,4]:
		return b
	raise Exception('Invalid SDSS band: "' + str(b) + '"')

class SdssDR(object):
	def __init__(self, curl=False, basedir=None):
		self.curl = curl
		self.basedir = basedir
		self.filenames = {}

	def getFilename(self, filetype, *args, **kwargs):
		for k,v in zip(['run', 'camcol', 'field', 'band'], args):
			kwargs[k] = v
		# convert band number to band character.
		if 'band' in kwargs and kwargs['band'] is not None:
			kwargs['band'] = band_name(kwargs['band'])
		if not filetype in self.filenames:
			return None
		pat = self.filenames[filetype]
		#print 'pat', pat, 'kwargs', kwargs
		fn = pat % kwargs
		return fn

	def getPath(self, *args, **kwargs):
		fn = self.getFilename(*args, **kwargs)
		if fn is None:
			return None
		if self.basedir is not None:
			fn = os.path.join(self.basedir, fn)
		return fn

	def setBasedir(self, dirnm):
		self.basedir = dirnm

	def _open(self, fn):
		if self.basedir is not None:
			path = os.path.join(self.basedir, fn)
		else:
			path = fn
		return pyfits.open(path)


class SdssFile(object):
	def __init__(self, run=None, camcol=None, field=None, band=None, rerun=None,
				 **kwargs):
		'''
		band: string ('u', 'g', 'r', 'i', 'z')
		'''
		self.run = run
		self.camcol = camcol
		self.field = field
		if band is not None:
			self.band = band_name(band)
			self.bandi = band_index(band)
		if rerun is not None:
			self.rerun = rerun
		self.filetype = 'unknown'

	def getRun(self):
		return self.__dict__.get('run', 0)
	def getCamcol(self):
		return self.__dict__.get('camcol', 0)
	def getField(self):
		return self.__dict__.get('field', 0)

	def __str__(self):
		s = 'SDSS ' + self.filetype
		s += ' %i-%i-%i' % (self.getRun(), self.getCamcol(), self.getField())
		if hasattr(self, 'band'):
			s += '-%s' % self.band
		return s


def munu_to_radec_rad(mu, nu, node, incl):
	ra = node + np.arctan2(np.sin(mu - node) * np.cos(nu) * np.cos(incl) -
						   np.sin(nu) * np.sin(incl),
						   np.cos(mu - node) * np.cos(nu))
	dec = np.arcsin(np.sin(mu - node) * np.cos(nu) * np.sin(incl) +
					np.sin(nu) * np.cos(incl))
	return ra,dec

def munu_to_radec_deg(mu, nu, node, incl):
	mu, nu = np.deg2rad(mu), np.deg2rad(nu)
	node, incl = np.deg2rad(node), np.deg2rad(incl)
	ra,dec = munu_to_radec_rad(mu, nu, node, incl)
	ra, dec = np.rad2deg(ra), np.rad2deg(dec)
	ra += (360. * (ra < 0))
	ra -= (360. * (ra > 360))
	return (ra, dec)


class AsTrans(SdssFile):
	'''
	In DR7, asTrans structures can appear in asTrans files (for a
	whole run) or in tsField files (in astrom/ or fastrom/).

	http://www.sdss.org/dr7/dm/flatFiles/asTrans.html

	In DR8, they are in asTrans files, or in the "frames".

	http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/astrom/asTrans.html
	'''
	def __init__(self, *args, **kwargs):
		'''
		node, incl: in radians

		astrans: must be an object with fields:
		 {a,b,c,d,e,f}[band]
		 {ricut}[band]
		 {drow0, drow1, drow2, drow3, dcol0, dcol1, dcol2, dcol3}[band]
		 {csrow, cscol, ccrow, cccol}[band]

		cut_to_band: in DR8 frames files, the astrans elements are not arrays;
		in DR7 tsField files they are.

		Note about units in this class:

		mu,nu are in degrees (great circle coords)

		a,d are in degrees (mu0, nu0)
		b,c,e,f are in degrees/pixel (dmu,dnu/drow,dcol)
		drow0,dcol0 are in pixels (distortion coefficients order 0); dpixels
		drow1,dcol1 are unitless dpixels / pixel (distortion coefficients order 1)
		drow2,dcol2 are in 1/pixels (dpixels/pixel**2) (distortion coefficients order 2)
		drow3,dcol3 are in 1/pixels**2 (dpixels/pixel**3) (distortion coefficients order 3)
		csrow,cscol are in pixels/mag (color-dependent shift)
		ccrow,cccol are in pixels (non-color-dependent shift)
		'''
		super(AsTrans, self).__init__(*args, **kwargs)
		self.filetype = 'asTrans'
		self.node = kwargs.get('node', None)
		self.incl = kwargs.get('incl', None)
		astrans = kwargs.get('astrans', None)
		self.trans = {}
		cut = kwargs.get('cut_to_band', True)
		if astrans is not None and hasattr(self, 'bandi'):
			for f in ['a','b','c','d','e','f', 'ricut',
					  'drow0', 'drow1', 'drow2', 'drow3',
					  'dcol0', 'dcol1', 'dcol2', 'dcol3',
					  'csrow', 'cscol', 'ccrow', 'cccol']:
				try:
					if hasattr(astrans, f):
						el = getattr(astrans, f)
						if cut:
							el = el[self.bandi]
						self.trans[f] = el
				except:
					print 'failed to get astrans.' + f
					import traceback
					traceback.print_exc()
					pass

	def __str__(self):
		return (SdssFile.__str__(self) +
				' (node=%g, incl=%g)' % (self.node, self.incl))

	def _get_abcdef(self):
		return tuple(self.trans[x] for x in 'abcdef')

	def _get_drow(self):
		return tuple(self.trans[x] for x in ['drow0', 'drow1', 'drow2', 'drow3'])

	def _get_dcol(self):
		return tuple(self.trans[x] for x in ['dcol0', 'dcol1', 'dcol2', 'dcol3'])

	def _get_cscc(self):
		return tuple(self.trans[x] for x in ['csrow', 'cscol', 'ccrow', 'cccol'])

	def _get_ricut(self):
		return self.trans['ricut']

	def cd_at_pixel(self, x, y, color=0):
		'''
		(x,y) to numpy array (2,2) -- the CD matrix at pixel x,y:

		[ [ dRA/dx * cos(Dec), dRA/dy * cos(Dec) ],
		  [ dDec/dx          , dDec/dy           ] ]

		in FITS these are called:
		[ [ CD11             , CD12              ],
		  [ CD21             , CD22              ] ]

		  Note: these statements have not been verified by the FDA.
		'''
		ra0,dec0 = self.pixel_to_radec(x, y, color)
		step = 10. # pixels
		rax,decx = self.pixel_to_radec(x+step, y, color)
		ray,decy = self.pixel_to_radec(x, y+step, color)
		cosd = np.cos(np.deg2rad(dec0))
		return np.array([ [ (rax-ra0)/step * cosd, (ray-ra0)/step * cosd ],
						  [ (decx-dec0)/step     , (decy-dec0)/step      ] ])

	def pixel_to_radec(self, x, y, color=0):
		mu, nu = self.pixel_to_munu(x, y, color)
		return self.munu_to_radec(mu, nu)

	def radec_to_pixel_single(self, ra, dec, color=0):
		'''RA,Dec -> x,y for scalar RA,Dec.'''
		# RA,Dec -> mu,nu -> prime -> pixel
		mu, nu = self.radec_to_munu_single(ra, dec)
		return self.munu_to_pixel_single(mu, nu, color)

	def radec_to_pixel(self, ra, dec, color=0):
		mu, nu = self.radec_to_munu(ra, dec)
		return self.munu_to_pixel(mu, nu, color)
	
	def munu_to_pixel(self, mu, nu, color=0):
		xprime, yprime = self.munu_to_prime(mu, nu, color)
		return self.prime_to_pixel(xprime, yprime)

	munu_to_pixel_single = munu_to_pixel

	def munu_to_prime(self, mu, nu, color=0):
		'''
		mu = a + b * rowm + c * colm
		nu = d + e * rowm + f * colm

		So

		[rowm; colm] = [b,c; e,f]^-1 * [mu-a; nu-d]

		[b,c; e,f]^1 = [B,C; E,F] in the code below, so

		[rowm; colm] = [B,C; E,F] * [mu-a; nu-d]

		'''
		a, b, c, d, e, f = self._get_abcdef()
		#print 'mu,nu', mu, nu, 'a,d', a,d
		determinant = b * f - c * e
		#print 'det', determinant
		B =  f / determinant
		C = -c / determinant
		E = -e / determinant
		F =  b / determinant
		#print 'B', B, 'mu-a', mu-a, 'C', C, 'nu-d', nu-d
		#print 'E', E, 'mu-a', mu-a, 'F', F, 'nu-d', nu-d
		mua = mu - a
		# in field 6955, g3, 809 we see a~413
		#if mua < -180.:
		#	mua += 360.
		mua += 360. * (mua < -180.)
		yprime = B * mua + C * (nu - d)
		xprime = E * mua + F * (nu - d)
		return xprime,yprime

	def pixel_to_munu(self, x, y, color=0):
		(xprime, yprime) = self.pixel_to_prime(x, y, color)
		a, b, c, d, e, f = self._get_abcdef()
		mu = a + b * yprime + c * xprime
		nu = d + e * yprime + f * xprime
		return (mu, nu)

	def pixel_to_prime(self, x, y, color=0):
		# Secret decoder ring:
		#  http://www.sdss.org/dr7/products/general/astrometry.html
		# (color)0 is called riCut;
		# g0, g1, g2, and g3 are called
		#    dRow0, dRow1, dRow2, and dRow3, respectively;
		# h0, h1, h2, and h3 are called
		#    dCol0, dCol1, dCol2, and dCol3, respectively;
		# px and py are called csRow and csCol, respectively;
		# and qx and qy are called ccRow and ccCol, respectively.
		color0 = self._get_ricut()
		g0, g1, g2, g3 = self._get_drow()
		h0, h1, h2, h3 = self._get_dcol()
		px, py, qx, qy = self._get_cscc()

		# #$(%*&^(%$%*& bad documentation.
		(px,py) = (py,px)
		(qx,qy) = (qy,qx)

		yprime = y + g0 + g1 * x + g2 * x**2 + g3 * x**3
		xprime = x + h0 + h1 * x + h2 * x**2 + h3 * x**3

		# The code below implements this, vectorized:
		# if color < color0:
		#	xprime += px * color
		#	yprime += py * color
		# else:
		#	xprime += qx
		#	yprime += qy
		qx = qx * np.ones_like(x)
		qy = qy * np.ones_like(y)
		#print 'color', color.shape, 'px', px.shape, 'qx', qx.shape
		xprime += np.where(color < color0, px * color, qx)
		yprime += np.where(color < color0, py * color, qy)

		return (xprime, yprime)

	def prime_to_pixel(self, xprime, yprime,  color=0):
		color0 = self._get_ricut()
		g0, g1, g2, g3 = self._get_drow()
		h0, h1, h2, h3 = self._get_dcol()
		px, py, qx, qy = self._get_cscc()

		# #$(%*&^(%$%*& bad documentation.
		(px,py) = (py,px)
		(qx,qy) = (qy,qx)

		qx = qx * np.ones_like(xprime)
		qy = qy * np.ones_like(yprime)
		#print 'color', color.shape, 'px', px.shape, 'qx', qx.shape
		xprime -= np.where(color < color0, px * color, qx)
		yprime -= np.where(color < color0, py * color, qy)

		# Now invert:
		#   yprime = y + g0 + g1 * x + g2 * x**2 + g3 * x**3
		#   xprime = x + h0 + h1 * x + h2 * x**2 + h3 * x**3
		x = xprime - h0
		# dumb-ass Newton's method
		dx = 1.
		# FIXME -- should just update the ones that aren't zero
		# FIXME -- should put in some failsafe...
		while max(np.abs(np.atleast_1d(dx))) > 1e-10:
			xp    = x + h0 + h1 * x + h2 * x**2 + h3 * x**3
			dxpdx = 1 +      h1     + h2 * 2*x +  h3 * 3*x**2
			dx = (xprime - xp) / dxpdx
			#print 'Max Newton dx', max(abs(dx))
			x += dx
		y = yprime - (g0 + g1 * x + g2 * x**2 + g3 * x**3)
		return (x, y)

	def radec_to_munu_single_c(self, ra, dec):
		''' Compute ra,dec to mu,nu for a single RA,Dec, calling C code'''
		mu,nu = cutils.radec_to_munu(ra, dec, self.node, self.incl)
		#mu2,nu2 = self.radec_to_munu(ra, dec)
		#print 'mu,mu2', mu, mu2
		#print 'nu,nu2', nu, nu2
		return mu,nu

	def radec_to_munu(self, ra, dec):
		'''
		RA,Dec in degrees

		mu,nu (great circle coords) in degrees
		'''
		node,incl = self.node, self.incl
		assert(ra is not None)
		assert(dec is not None)
		ra, dec = np.deg2rad(ra), np.deg2rad(dec)
		mu = node + np.arctan2(np.sin(ra - node) * np.cos(dec) * np.cos(incl) +
							   np.sin(dec) * np.sin(incl),
							   np.cos(ra - node) * np.cos(dec))
		nu = np.arcsin(-np.sin(ra - node) * np.cos(dec) * np.sin(incl) +
					   np.sin(dec) * np.cos(incl))
		mu, nu = np.rad2deg(mu), np.rad2deg(nu)
		mu += (360. * (mu < 0))
		mu -= (360. * (mu > 360))
		return (mu, nu)

	def munu_to_radec(self, mu, nu):
		node,incl = self.node, self.incl
		assert(mu is not None)
		assert(nu is not None)
		# just in case you thought we needed *more* rad/deg conversions...
		return munu_to_radec_deg(mu, nu, np.rad2deg(node), np.rad2deg(incl))


if cutils is not None:
	AsTrans.radec_to_munu_single = AsTrans.radec_to_munu_single_c
else:
	AsTrans.radec_to_munu_single = AsTrans.radec_to_munu


class TsField(SdssFile):
	def __init__(self, *args, **kwargs):
		super(TsField, self).__init__(*args, **kwargs)
		self.filetype = 'tsField'
		self.exptime = 53.907456
	def setHdus(self, p):
		self.hdus = p
		self.table = fits_table(self.hdus[1].data)[0]
		T = self.table
		self.aa = T.aa.astype(float)
		self.kk = T.kk.astype(float)
		self.airmass = T.airmass

	def getAsTrans(self, band):
		bandi = band_index(band)
		band = band_name(band)
		#node,incl = self.getNode(), self.getIncl()
		hdr = self.hdus[0].header
		node = np.deg2rad(hdr.get('NODE'))
		incl = np.deg2rad(hdr.get('INCL'))
		asTrans = AsTrans(self.run, self.camcol, self.field, band=band,
						  node=node, incl=incl, astrans=self.table)
		return asTrans

	#magL = -(2.5/ln(10))*[asinh((f/f0)/2b)+ln(b)]
	# luptitude == arcsinh mag
	# band: int
	def luptitude_to_counts(self, L, band):
		# from arcsinh softening parameters table
		#   http://www.sdss.org/dr7/algorithms/fluxcal.html#counts2mag
		b = [1.4e-10, 0.9e-10, 1.2e-10, 1.8e-10, 7.4e-10]

		b = b[band]
		maggies = 2.*b * np.sinh(-0.4 * np.log(10.) * L - np.log(b))
		dlogcounts = -0.4 * (self.aa[band] + self.kk[band] * self.airmass[band])
		return (maggies * self.exptime) * 10.**dlogcounts
		
	# band: int
	def mag_to_counts(self, mag, band):
		# log_10(counts)
		logcounts = (-0.4 * mag + np.log10(self.exptime)
					 - 0.4*(self.aa[band] + self.kk[band] * self.airmass[band]))
		#logcounts = np.minimum(logcounts, 308.)
		#print 'logcounts', logcounts
		#olderrs = np.seterr(all='print')
		rtn = 10.**logcounts
		#np.seterr(**olderrs)
		return rtn

	def counts_to_mag(self, counts, band):
		# http://www.sdss.org/dr5/algorithms/fluxcal.html#counts2mag
		# f/f0 = counts/exptime * 10**0.4*(aa + kk * airmass)
		# mag = -2.5 * log10(f/f0)
		return -2.5 * (np.log10(counts / self.exptime) +
					   0.4 * (self.aa[band] + self.kk[band] * self.airmass[band]))


	
class FpObjc(SdssFile):
	def __init__(self, *args, **kwargs):
		super(FpObjc, self).__init__(*args, **kwargs)
		self.filetype = 'fpObjc'

class FpM(SdssFile):
	def __init__(self, *args, **kwargs):
		super(FpM, self).__init__(*args, **kwargs)
		self.filetype = 'fpM'
		self.maskmap = None

	def setHdus(self, p):
		self.hdus = p

	def getMaskPlane(self, name):
		# Mask planes are described in HDU 11 (the last HDU)
		if self.maskmap is None:
			self.maskmap = {}
			T = fits_table(self.hdus[-1].data)
			#print 'Got mask definition table'
			#T.about()
			T.cut(T.defname == 'S_MASKTYPE')
			for k,v in zip(T.attributename, T.value):
				k = k.replace('S_MASK_', '')
				if k == 'S_NMASK_TYPES':
					continue
				#print '  Mask', k, '=', v
				self.maskmap[k] = v
		if not name in self.maskmap:
			raise RuntimeError('Unknown mask plane \"%s\"' % name)

		return fits_table(self.hdus[1 + self.maskmap[name]].data)

	def setMaskedPixels(self, name, img, val, roi=None):
		M = self.getMaskPlane(name)
		if M is None:
			return
		if roi is not None:
			x0,x1,y0,y1 = roi

		for (c0,c1,r0,r1,coff,roff) in zip(M.cmin,M.cmax,M.rmin,M.rmax,
										   M.col0, M.row0):
			assert(coff == 0)
			assert(roff == 0)
			if roi is not None:
				(outx,nil) = get_overlapping_region(c0-x0, c1+1-x0, 0, x1-x0)
				(outy,nil) = get_overlapping_region(r0-y0, r1+1-y0, 0, y1-y0)
				# print 'Mask col [%i, %i], row [%i, %i]' % (c0, c1, r0, r1)
				# print  '  outx', outx, 'outy', outy
				img[outy,outx] = val
			else:
				img[r0:r1, c0:c1] = val
		

class FpC(SdssFile):
	def __init__(self, *args, **kwargs):
		super(FpC, self).__init__(*args, **kwargs)
		self.filetype = 'fpC'
	def getImage(self):
		return self.image
	def getHeader(self):
		return self.header

class PsField(SdssFile):
	def __init__(self, *args, **kwargs):
		super(PsField, self).__init__(*args, **kwargs)
		self.filetype = 'psField'

	def setHdus(self, p):
		self.hdus = p
		t = fits_table(p[6].data)
		# the table has only one row...
		assert(len(t) == 1)
		t = t[0]
		#self.table = t
		self.gain = t.gain
		self.dark_variance = t.dark_variance
		self.sky = t.sky
		self.skyerr = t.skyerr
		self.psp_status = t.status
		# Double-Gaussian PSF params
		self.dgpsf_s1 = t.psf_sigma1_2g
		self.dgpsf_s2 = t.psf_sigma2_2g
		self.dgpsf_b  = t.psf_b_2g
		# summary PSF width (sigmas)
		self.psf_fwhm = t.psf_width * (2.*np.sqrt(2.*np.log(2.)))

	def getPsfFwhm(self, bandnum):
		return self.psf_fwhm[bandnum]

	def getDoubleGaussian(self, bandnum):
		# http://www.sdss.org/dr7/dm/flatFiles/psField.html
		# good = PSP_FIELD_OK
		status = self.psp_status[bandnum]
		if status != 0:
			print 'Warning: PsField status[band=%s] =' % (bandnum), status
		a  = 1.0
		s1 = self.dgpsf_s1[bandnum]
		s2 = self.dgpsf_s2[bandnum]
		b  = self.dgpsf_b[bandnum]
		return (float(a), float(s1), float(b), float(s2))

	def getPsfAtPoints(self, bandnum, x, y):
		'''
		Reconstruct the SDSS model PSF from KL basis functions.

		x,y can be scalars or 1-d numpy arrays.

		Return value:
		if x,y are scalars: a PSF image
		if x,y are arrays:  a list of PSF images
		'''
		rtnscalar = np.isscalar(x) and np.isscalar(y)
		x = np.atleast_1d(x)
		y = np.atleast_1d(y)
		psf = fits_table(self.hdus[bandnum+1].data)
		psfimgs = None
		(outh, outw) = (None,None)

		# From the IDL docs:
		# http://photo.astro.princeton.edu/photoop_doc.html#SDSS_PSF_RECON
		#   acoeff_k = SUM_i{ SUM_j{ (0.001*ROWC)^i * (0.001*COLC)^j * C_k_ij } }
		#   psfimage = SUM_k{ acoeff_k * RROWS_k }
		for k in range(len(psf)):
			nrb = psf.nrow_b[k]
			ncb = psf.ncol_b[k]
			c = psf.c[k].reshape(5, 5)
			c = c[:nrb,:ncb]
			(gridi,gridj) = np.meshgrid(range(nrb), range(ncb))

			if psfimgs is None:
				psfimgs = [np.zeros_like(psf.rrows[k]) for xy
						   in np.broadcast(x,y)]
				(outh,outw) = (psf.rnrow[k], psf.rncol[k])
			else:
				assert(psf.rnrow[k] == outh)
				assert(psf.rncol[k] == outw)

			for i,(xi,yi) in enumerate(np.broadcast(x,y)):
				#print 'xi,yi', xi,yi
				acoeff_k = np.sum(((0.001 * xi)**gridi * (0.001 * yi)**gridj * c))
				if False: # DEBUG
					print 'coeffs:', (0.001 * xi)**gridi * (0.001 * yi)**gridj
					print 'c:', c
					for (coi,ci) in zip(((0.001 * xi)**gridi * (0.001 * yi)**gridj).ravel(), c.ravel()):
						print 'co %g, c %g' % (coi,ci)
					print 'acoeff_k', acoeff_k

				#print 'acoeff_k', acoeff_k.shape, acoeff_k
				#print 'rrows[k]', psf.rrows[k].shape, psf.rrows[k]
				psfimgs[i] += acoeff_k * psf.rrows[k]

		psfimgs = [img.reshape((outh,outw)) for img in psfimgs]
		if rtnscalar:
			return psfimgs[0]
		return psfimgs


	def getGain(self, band=None):
		if band is not None:
			return self.gain[band]
		return self.gain

	def getDarkVariance(self, band=None):
		if band is not None:
			return self.dark_variance[band]
		return self.dark_variance

	def getSky(self, band=None):
		if band is not None:
			return self.sky[band]
		return self.sky

	def getSkyErr(self, band=None):
		if band is not None:
			return self.skyerr[band]
		return self.skyerr


