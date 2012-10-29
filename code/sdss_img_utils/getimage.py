import os
from math import pi, sqrt, ceil, floor

import pyfits
import numpy as np

from dr9 import *
from classutils import *
from scipy.ndimage.morphology import binary_dilation


def get_image_dr8(run, camcol, field, bandname, sdss=None,
		  roi=None, psf='kl-gm', roiradecsize=None,
		  savepsfimg=None, curl=False,zrange=[-3,10]):
	'''
        Chopped Tractor get_tractor_image_dr8
	'''
	valid_psf = ['dg', 'kl-gm']
	if psf not in valid_psf:
		raise RuntimeError('PSF must be in ' + str(valid_psf))

	if sdss is None:
		sdss = DR8(curl=curl)

	bandnum = band_index(bandname)

	for ft in ['psField', 'fpM']:
		fn = sdss.retrieve(ft, run, camcol, field, bandname)
	fn = sdss.retrieve('frame', run, camcol, field, bandname)

	# http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/frames/RERUN/RUN/CAMCOL/frame.html
	frame = sdss.readFrame(run, camcol, field, bandname, filename=fn)
	image = frame.getImage().astype(np.float32)
	(H,W) = image.shape
	
	info = dict()

	astrans = frame.getAsTrans()
	wcs = SdssWcs(astrans)
	print 'Created SDSS Wcs:', wcs
	print '(x,y) = 1,1 -> RA,Dec', wcs.pixelToPosition(1,1)

	x0 = y0 = 0
	# Mysterious half-pixel shift.  asTrans pixel coordinates?
	wcs.setX0Y0(x0 + 0.5, y0 + 0.5)

	print 'Band name:', bandname
	photocal = SdssNanomaggiesPhotoCal(bandname)

	sky = 0.
	skyobj = ConstantSky(sky)

	calibvec = frame.getCalibVec()

	print 'sky', #frame.sky
	print frame.sky.shape
	print 'x', len(frame.skyxi), frame.skyxi
	print frame.skyxi.shape
	print 'y', len(frame.skyyi), frame.skyyi
	print frame.skyyi.shape

	skyim = frame.sky
	(sh,sw) = skyim.shape
	print 'Skyim shape', skyim.shape
	if sw != 256:
		skyim = skyim.T
	(sh,sw) = skyim.shape
	xi = np.round(frame.skyxi).astype(int)
	print 'xi:', xi.min(), xi.max(), 'vs [0,', sw, ']'
	yi = np.round(frame.skyyi).astype(int)
	print 'yi:', yi.min(), yi.max(), 'vs [0,', sh, ']'
	if ((all(xi >= 0) and all(xi < sw))!=True):
		print 'xi fail on field,run,camcol',field,run,camcol
		return None,None
	if ((all(yi >= 0) and all(yi < sh))!=True):
                print 'yi fail on field,run,camcol',field,run,camcol
		return None,None
	XI,YI = np.meshgrid(xi, yi)
	# Nearest-neighbour interpolation -- we just need this for approximate invvar.
	bigsky = skyim[YI,XI]
	assert(bigsky.shape == image.shape)

	dn = (image / calibvec) + bigsky

	# Could get this from photoField instead
	# http://data.sdss3.org/datamodel/files/BOSS_PHOTOOBJ/RERUN/RUN/photoField.html
	psfield = sdss.readPsField(run, camcol, field)
	gain = psfield.getGain(bandnum)
	darkvar = psfield.getDarkVariance(bandnum)
	dnvar = (dn / gain) + darkvar
	invvar = 1./(dnvar * calibvec**2)
	invvar = invvar.astype(np.float32)
	assert(invvar.shape == image.shape)

	meansky = np.mean(frame.sky)
	meancalib = np.mean(calibvec)
	skysig = sqrt((meansky / gain) + darkvar) * meancalib

	info.update(sky=sky, skysig=skysig)
	zr = np.array(zrange)*skysig + sky
	info.update(zr=zr)

	# http://data.sdss3.org/datamodel/files/PHOTO_REDUX/RERUN/RUN/objcs/CAMCOL/fpM.html
	fpM = sdss.readFpM(run, camcol, field, bandname)

	if roi is not None:
		roislice = (slice(y0,y1), slice(x0,x1))
		image = image[roislice].copy()
		invvar = invvar[roislice].copy()

	for plane in [ 'INTERP', 'SATUR', 'CR', 'GHOST' ]:
		fpM.setMaskedPixels(plane, invvar, 0, roi=roi)


	timg = Image(data=image, invvar=invvar, psf=None, wcs=wcs,
				 sky=skyobj, photocal=photocal,
				 name=('SDSS (r/c/f/b=%i/%i/%i/%s)' %
					   (run, camcol, field, bandname)))
	timg.zr = zr
	return timg,info


def get_image_dr9(*args, **kwargs):
	sdss = kwargs.get('sdss', None)
	if sdss is None:
		curl = kwargs.pop('curl', False)
		kwargs['sdss'] = DR9(curl=curl)
	return get_image_dr8(*args, **kwargs)




class SdssWcs(ParamList):
	pnames = ['a', 'b', 'c', 'd', 'e', 'f',
			  'drow0', 'drow1', 'drow2', 'drow3',
			  'dcol0', 'dcol1', 'dcol2', 'dcol3',
			  'csrow', 'cscol', 'ccrow', 'cccol',
			  'x0', 'y0']

	@staticmethod
	def getNamedParams():
		# node and incl are properties of the survey geometry, not params.
		# riCut... not clear.
		# Note that we omit x0,y0 from this list
		return dict([(k,i) for i,k in enumerate(SdssWcs.pnames[:-2])])

	def __init__(self, astrans):
		self.x0 = 0
		self.y0 = 0
		super(SdssWcs, self).__init__(self.x0, self.y0, astrans)
		# ParamList keeps its params in a list; we don't want to do that.
		del self.vals
		self.astrans = astrans

	def _setThing(self, i, val):
		N = len(SdssWcs.pnames)
		if i == N-2:
			self.x0 = val
		elif i == N-1:
			self.y0 = val
		else:
			t = self.astrans.trans
			t[SdssWcs.pnames[i]] = val
	def _getThing(self, i):
		N = len(SdssWcs.pnames)
		if i == N-2:
			return self.x0
		elif i == N-1:
			return self.y0
		t = self.astrans.trans
		return t[SdssWcs.pnames[i]]
	def _getThings(self):
		t = self.astrans.trans
		return [t[nm] for nm in SdssWcs.pnames[:-2]] + [self.x0, self.y0]
	def _numberOfThings(self):
		return len(SdssWcs.pnames)

	def getStepSizes(self, *args, **kwargs):
		deg = 0.396 / 3600. # deg/pix
		P = 2000. # ~ image size
		ss = [ deg, deg/P, deg/P, deg, deg/P, deg/P,
			   1., 1./P, 1./P**2, 1./P**3,
			   1., 1./P, 1./P**2, 1./P**3,
			   1., 1., 1., 1.,
			   1., 1.]
		return list(self._getLiquidArray(ss))

	def setX0Y0(self, x0, y0):
		self.x0 = x0
		self.y0 = y0

	# This function is not used by the tractor, and it works in
	# *original* pixel coords (no x0,y0 offsets)
	# (x,y) to RA,Dec in deg
	def pixelToRaDec(self, x, y):
		ra,dec = self.astrans.pixel_to_radec(x, y)
		return ra,dec

	def cdAtPixel(self, x, y):
		return self.astrans.cd_at_pixel(x + self.x0, y + self.y0)

	# RA,Dec in deg to pixel x,y.
	def positionToPixel(self, pos, src=None):
		## FIXME -- color.
		x,y = self.astrans.radec_to_pixel_single(pos.ra, pos.dec)
		return x - self.x0, y - self.y0

	# (x,y) to RA,Dec in deg
	def pixelToPosition(self, x, y, src=None):
		## FIXME -- color.
		ra,dec = self.pixelToRaDec(x + self.x0, y + self.y0)
		return RaDecPos(ra, dec)

class SdssNanomaggiesPhotoCal(BaseParams):
	def __init__(self, bandname):
		self.bandname = bandname
	def __str__(self):
		return self.__class__.__name__
	def hashkey(self):
		return ('SdssNanomaggiesPhotoCal', self.bandname)
	def brightnessToCounts(self, brightness):
		mag = brightness.getMag(self.bandname)
		if not np.isfinite(mag):
			return 0.
		# MAGIC
		if mag > 50.:
			return 0.

		if mag < -50:
			print 'Warning: mag', mag, ': clipping'
			mag = -50

		nmgy = 10. ** ((mag - 22.5) / -2.5)
		return nmgy
              


class RaDecPos(ParamList):
	'''
	A Position implementation using RA,Dec positions, in degrees.

	Attributes:
	  * ``.ra``
	  * ``.dec``
	'''
	@staticmethod
	def getName():
		return "RaDecPos"
	@staticmethod
	def getNamedParams():
		return dict(ra=0, dec=1)
	def __str__(self):
		return '%s: RA, Dec = (%.5f, %.5f)' % (self.getName(), self.ra, self.dec)
	def getDimension(self):
		return 2
	def getStepSizes(self, *args, **kwargs):
		return [1e-4, 1e-4]

	def distanceFrom(self, pos):
		from astrometry.util.starutil_numpy import degrees_between
		return degrees_between(self.ra, self.dec, pos.ra, pos.dec)


class ConstantSky(ScalarParam):
	'''
	In counts
	'''
	def getParamDerivatives(self, img):
		p = Patch(0, 0, np.ones(img.shape))
		p.setName('dsky')
		return [p]
	def addTo(self, img):
		img += self.val
	def getParamNames(self):
		return ['sky']



class Image(MultiParams):
	'''
	An image plus its calibration information.  An ``Image`` has
	pixels, inverse-variance map, WCS, PSF, photometric calibration
	information, and sky level.  All these things are ``Params``
	instances, and ``Image`` is a ``MultiParams`` so that the Tractor
	can optimize them.
	'''
	def __init__(self, data=None, invvar=None, psf=None, wcs=None, sky=None,
				 photocal=None, name=None, **kwargs):
		'''
		Args:
		  * *data*: numpy array: the image pixels
		  * *invvar*: numpy array: the image inverse-variance
		  * *psf*: a :class:`tractor.PSF` duck
		  * *wcs*: a :class:`tractor.WCS` duck
		  * *sky*: a :class:`tractor.Sky` duck
		  * *photocal*: a :class:`tractor.PhotoCal` duck
		  * *name*: string name of this image.
		  * *zr*: plotting range ("vmin"/"vmax" in matplotlib.imshow)

		'''
		self.data = data
		self.origInvvar = 1. * np.array(invvar)
		self.setMask()
		self.setInvvar(self.origInvvar)
		self.name = name
		self.starMask = np.ones_like(self.data)
		self.zr = kwargs.pop('zr', None)
		super(Image, self).__init__(psf, wcs, photocal, sky)

	def getMask(self):
		return self.mask

	def setMask(self):
		self.mask = (self.origInvvar <= 0.)
		self.mask = binary_dilation(self.mask,iterations=3)

	def setInvvar(self,invvar):
		self.invvar = 1. * invvar
		self.invvar[self.mask] = 0. 
		self.inverr = np.sqrt(self.invvar)

