import os
import pyfits
import numpy as np

from common import *
from dr8 import *

class DR9(DR8):

	def __init__(self, **kwargs):
		'''
		Useful kwargs:
		
		basedir : (string) - local directory where data will be stored.
		'''
		DR8.__init__(self, **kwargs)
		self.dasurl = 'http://data.sdss3.org/sas/dr9/boss/'
		
	def _get_runlist_filename(self):
		return self._get_data_file('runList-dr9.par')
