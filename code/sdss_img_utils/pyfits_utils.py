import os
import pyfits
import numpy
from numpy import array, isscalar, ndarray

def pyfits_writeto(p, filename, **kwargs):
	'''
	*p*: HDUList object
	*filename*: uh, the filename to write to
	'''
	# old pyfits versions (eg the one in Ubuntu 10.04)
	# fail when used with python2.7 (warning.showwarning changed)
	# so work-around pyfits printing a warning when it overwrites an
	# existing file.
	if os.path.exists(filename):
		os.remove(filename)
	p.writeto(filename, **kwargs)

def merge_tables(TT):
	assert(len(TT) > 0)
	cols = set(TT[0].get_columns())
	for T in TT[1:]:
		# They must have the same set of columns
		assert(len(cols.symmetric_difference(T.get_columns())) == 0)
	N = sum([len(T) for T in TT])
	cols = TT[0].get_columns()
	td = tabledata()
	#print 'merge_tables: columns =', cols
	for col in cols:
		#print 'merging column', col
		if col.startswith('_'):
			continue
		v0 = TT[0].getcolumn(col)
		if isinstance(v0, numpy.ndarray):
			V = numpy.concatenate([T.getcolumn(col) for T in TT])
		elif type(v0) is list:
			V = v0
			for T in TT[1:]:
				V.extend(T.getcolumn(col))
		elif numpy.isscalar(v0):
			#print 'merge_tables: copying scalar from first table:', col, '=', v0
			V = v0
		else:
			raise RuntimeError("pyfits_utils.merge_tables: Don't know how to concatenate type: %s" % str(type(v0)))
			
		td.set(col, V)
	#td._columns = cols
	assert(td._length == N)
	return td
	

def add_nonstructural_headers(fromhdr, tohdr):
	for card in fromhdr.ascardlist():
		if ((card.key in ['SIMPLE','XTENSION', 'BITPIX', 'END', 'PCOUNT', 'GCOUNT',
						  'TFIELDS',]) or
			card.key.startswith('NAXIS') or
			card.key.startswith('TTYPE') or
			card.key.startswith('TFORM')):
			#card.key.startswith('TUNIT') or
			#card.key.startswith('TDISP')):
			#print 'skipping card', card.key
			continue
		#if tohdr.has_key(card.key):
		#	#print 'skipping existing card', card.key
		#	continue
		#print 'adding card', card.key
		#tohdr.update(card.key, card.value, card.comment, before='END')
		#tohdr.ascardlist().append(
		cl = tohdr.ascardlist()
		if 'END' in cl.keys():
			i = cl.index_of('END')
		else:
			i = len(cl)
		cl.insert(i, pyfits.Card(card.key, card.value, card.comment))

def cut_array(val, I, name=None):
	if type(I) is slice:
		return val[I]

	if type(val) in [numpy.ndarray, numpy.core.defchararray.chararray]:
		#print 'slicing numpy array "%s": val shape' % name, val.shape
		#print 'slice shape:', I.shape
		# You can't slice a two-dimensional, length-zero, numpy array,
		# with an empty array.
		if len(val) == 0:
			return val
		return val[I]

	if type(val) in [list,tuple] and type(I) in [int, numpy.int64]:
		return val[I]

	# HACK -- emulate numpy's boolean and int array slicing
	# (when "val" is a normal python list)
	if type(I) is numpy.ndarray and hasattr(I, 'dtype') and ((I.dtype.type in [bool, numpy.bool])
															 or (I.dtype == bool)):
		try:
			return [val[i] for i,b in enumerate(I) if b]
		except:
			print 'Failed to slice field', name
			#setattr(rtn, name, val)
			#continue

	inttypes = [int, numpy.int64, numpy.int32, numpy.int]
	if type(I) is numpy.ndarray and all(I.astype(int) == I):
		return [val[i] for i in I]

	if (numpy.isscalar(I) and hasattr(I, 'dtype') and
		I.dtype in inttypes):
		return val[int(I)]

	if hasattr(I, '__len__') and len(I) == 0:
		return []

	print 'Error slicing array:'
	print 'array is'
	print '  type:', type(val)
	print '  ', val
	print 'cut is'
	print '  type:', type(I)
	print '  ', I
	raise Exception('Error in cut_array')

class tabledata(object):

	class td_iter(object):
		def __init__(self, td):
			self.td = td
			self.i = 0
		def __iter__(self):
			return self
		def next(self):
			if self.i >= len(self.td):
				raise StopIteration
			X = self.td[self.i]
			self.i += 1
			return X


	def __init__(self, header=None):
		self._length = 0
		self._header = header
		self._columns = []
	def __str__(self):
		return 'tabledata object with %i rows and %i columns' % (len(self), len([k for k in self.__dict__.keys() if not k.startswith('_')]))
	def about(self):
		keys = [k for k in self.__dict__.keys() if not k.startswith('_')]
		print 'tabledata object with %i rows and %i columns:' % (len(self),	 len(keys))
		keys.sort()
		for k in keys:
			print '	 ', k,
			v = self.get(k)
			print '(%s)' % (str(type(v))),
			if numpy.isscalar(v):
				print v,
			elif hasattr(v, 'shape'):
				print 'shape', v.shape,
			elif hasattr(v, '__len__'):
				print 'length', len(v),
			else:
				print v,

			if hasattr(v, 'dtype'):
				print 'dtype', v.dtype,
			print

	def __setattr__(self, name, val):
		object.__setattr__(self, name, val)
		#print 'set', name, 'to', val
		if (self._length == 0) and (not (name.startswith('_'))) and hasattr(val, '__len__') and len(val) != 0 and type(val) != str:
			self._length = len(val)
		if hasattr(self, '_columns') and not name in self._columns:
			self._columns.append(name)
	def set(self, name, val):
		self.__setattr__(name, val)
	def getcolumn(self, name):
		return self.__dict__[name]
		#except:
		#	return self.__dict__[name.lower()]
	def get(self, name):
		return self.getcolumn(name)
	# Returns the list of column names, as they were ordered in the input FITS or text table.
	def get_columns(self):
		return self._columns
	# Returns the original FITS header.
	def get_header(self):
		return self._header

	def columns(self):
		return [k for k in self.__dict__.keys() if not k.startswith('_')]
	def __len__(self):
		return self._length
	def delete_column(self, c):
		del self.__dict__[c]
		self._columns.remove(c)
	def __setitem__(self, I, O):
		if type(I) is slice:
			print 'I:', I
			# HACK... "[:]" -> slice(None, None, None)
			if I.start is None and I.stop is None and I.step is None:
				I = numpy.arange(len(self))
			else:
				I = numpy.arange(I.start, I.stop, I.step)
		for name,val in self.__dict__.items():
			if name.startswith('_'):
				continue
			# ?
			if numpy.isscalar(val):
				self.set(name, O.get(name))
				continue
			try:
				val[I] = O.get(name)
			except Exception:
				# HACK -- emulate numpy's boolean and int array slicing...
				ok = False
				#if type(I) == numpy.ndarray and hasattr(I, 'dtype') and I.dtype == bool:
				#	for i,b in enumerate(I):
				#		if b:
				#			val[i] = O.get(val)
				#	ok = True
				#if type(I) == numpy.ndarray and hasattr(I, 'dtype') and I.dtype == 'int':
				#	rtn.set(name, [val[i] for i in I])
				#	ok = True
				#if len(I) == 0:
				#	rtn.set(name, [])
				#	ok = True
				if not ok:
					print 'Error in slicing an astrometry.util.pyfits_utils.table_data object:'
					#print '  -->', e

					import pdb; pdb.set_trace()

					print 'While setting member:', name
					print ' setting elements:', I
					print ' from obj', O
					print ' target type:', type(O.get(name))
					print ' dest type:', type(val)
					print 'index type:', type(I)
					#if hasattr(val, 'shape'):
					#	print ' shape:', val.shape
					#if hasattr(I, 'shape'):
					#	print ' index shape:', I.shape
					if hasattr(I, 'dtype'):
						print '	 index dtype:', I.dtype
					print 'my length:', self._length
					raise Exception('error in fits_table indexing')

	def copy(self):
		rtn = tabledata()
		for name,val in self.__dict__.items():
			if name.startswith('_'):
				continue
			if numpy.isscalar(val):
				#print 'copying scalar', name
				rtn.set(name, val)
				continue
			if type(val) is numpy.ndarray:
				#print 'copying numpy array', name
				rtn.set(name, val.copy())
				continue
			if type(val) in [list,tuple]:
				#print 'copying list', name
				rtn.set(name, val[:])
				continue
			print 'in pyfits_utils: copy(): can\'t copy', name, '=', val
		rtn._header = self._header
		if hasattr(self, '_columns'):
			rtn._columns = [c for c in self._columns]
		return rtn

	def cut(self, I):
		for name,val in self.__dict__.items():
			if name.startswith('_'):
				continue
			if numpy.isscalar(val):
				continue
			#print 'cutting', name
			C = cut_array(val, I, name)
			self.set(name, C)
			self._length = len(C)

	def __getitem__(self, I):
		rtn = tabledata()
		for name,val in self.__dict__.items():
			if name.startswith('_'):
				continue
			if numpy.isscalar(val):
				rtn.set(name, val)
				continue
			try:
				C = cut_array(val, I, name)
			except:
				print 'Error in cut_array() via __getitem__, name', name
				raise
			rtn.set(name, C)

			if isscalar(I):
				rtn._length = 1
			else:
				rtn._length = len(getattr(rtn, name))
		rtn._header = self._header
		if hasattr(self, '_columns'):
			rtn._columns = [c for c in self._columns]
		return rtn
	def __iter__(self):
		return tabledata.td_iter(self)

	def append(self, X):
		for name,val in self.__dict__.items():
			if name.startswith('_'):
				continue
			if numpy.isscalar(val):
				continue
			try:
				val2 = X.getcolumn(name)
				if type(val) is list:
					newX = val + val2
				else:
					newX = numpy.append(val, val2, axis=0)
				self.set(name, newX)
				self._length = len(newX)
			except Exception as e:
				print 'exception appending element "%s"' % name
				#print 'exception:', e
                #raise Exception('exception appending element "%s"' % name)

	def write_to(self, fn, columns=None, header='default', primheader=None):
		if columns is None and hasattr(self, '_columns'):
			columns = self._columns
		T = pyfits.new_table(self.to_fits_columns(columns))
		if header == 'default':
			header = self._header
		if header is not None:
			add_nonstructural_headers(header, T.header)
		if primheader is not None:
			P = pyfits.PrimaryHDU()
			add_nonstructural_headers(primheader, P.header)
			pyfits.HDUList([P, T]).writeto(fn, clobber=True)
		else:
			pyfits_writeto(T, fn)

	writeto = write_to

	def to_fits_columns(self, columns=None):
		cols = []

		fmap = {numpy.float64:'D',
				numpy.float32:'E',
				numpy.int32:'J',
				numpy.int64:'K',
				numpy.uint8:'B', #
				numpy.int16:'I',
				#numpy.bool:'X',
				#numpy.bool_:'X',
				numpy.bool:'L',
				numpy.bool_:'L',
				numpy.string_:'A',
				}

		if columns is None:
			columns = self.__dict__.keys()
				
		for name in columns:
			if name.startswith('_'):
				continue
			if not name in self.__dict__:
				continue
			val = self.__dict__.get(name)
			#print 'col', name, 'type', val.dtype, 'descr', val.dtype.descr
			#print repr(val.dtype)
			#print val.dtype.type
			#print repr(val.dtype.type)
			#print val.shape
			#print val.size
			#print val.itemsize
			if type(val) is list:
				val = array(val)
			try:
				fitstype = fmap.get(val.dtype.type, 'D')
			except:
				print 'Table column "%s" has no "dtype"; skipping' % name
				continue

			if fitstype == 'X':
				# pack bits...
				pass
			if len(val.shape) > 1:
				fitstype = '%i%s' % (val.shape[1], fitstype)
			elif fitstype == 'A' and val.itemsize > 1:
				# strings
				fitstype = '%i%s' % (val.itemsize, fitstype)
			else:
				fitstype = '1'+fitstype
			#print 'fits type', fitstype
			try:
				col = pyfits.Column(name=name, array=val, format=fitstype)
			except:
				print 'Error converting column', name, 'to a pyfits column:'
				print 'fitstype:', fitstype
				try:
					print 'numpy dtype:'
					print val.dtype
					print val.dtype.type
				except:
					pass
				print 'value:', val
				raise
			cols.append(col)
			#print 'fits type', fitstype, 'column', col
			#print repr(col)
			#print 'col', name, ': data length:', val.shape
		return cols
		

def fits_table(dataorfn, rows=None, hdunum=1, header='default',
			   columns=None):
	'''
	If 'columns' (a list of strings) is passed, only those columns
	will be read; otherwise all columns will be read.
	'''
	pf = None
	hdr = None
	if isinstance(dataorfn, str):
		pf = pyfits.open(dataorfn)
		data = pf[hdunum].data
		if header == 'default':
			hdr = pf[hdunum].header
	else:
		data = dataorfn

	if data is None:
		return None
	fields = tabledata(header=hdr)
	if columns is None:
		columns = data.dtype.names

	for c in columns:
		col = data.field(c)
		if rows is not None:
			col = col[rows]
		fields.set(c.lower(), col)
	fields._length = len(data)
	fields._columns = [c.lower() for c in columns]
	if pf:
		pf.close()
	return fields

table_fields = fits_table

# ultra-brittle text table parsing.
def text_table_fields(forfn, text=None, skiplines=0, split=None, trycsv=True, maxcols=None):
	if text is None:
		f = None
		if isinstance(forfn, str):
			f = open(forfn)
			data = f.read()
			f.close()
		else:
			data = forfn.read()
	else:
		data = text

	# replace newline variations with a single newline character
	data = data.replace('\r\n','\n') # windows
	data = data.replace('\r','\n') # mac os

	txtrows = data.split('\n')

	txtrows = txtrows[skiplines:]

	# column names are in the first (un-skipped) line.
	txt = txtrows.pop(0)
	header = txt
	if header[0] == '#':
		header = header[1:]
	header = header.split()
	if len(header) == 0:
		raise Exception('Expected to find column names in the first row of text; got \"%s\".' % txt)
	#assert(len(header) >= 1)
	if trycsv and (split is None) and (len(header) == 1) and (',' in header[0]):
		# try CSV
		header = header[0].split(',')
	colnames = header

	fields = tabledata()
	txtrows = [r for r in txtrows if not r.startswith('#')]
	coldata = [[] for x in colnames]
	for i,r in enumerate(txtrows):
		if maxcols is not None:
			r = r[:maxcols]
		if split is None:
			cols = r.split()
		else:
			cols = r.split(split)
		if len(cols) == 0:
			continue
		if trycsv and (split is None) and (len(cols) != len(colnames)) and (',' in r):
			# try to parse as CSV.
			cols = r.split(',')
			
		if len(cols) != len(colnames):
			raise Exception('Expected to find %i columns of data to match headers (%s) in row %i; got %i\n	"%s"' % (len(colnames), ', '.join(colnames), i, len(cols), r))
		#assert(len(cols) == len(colnames))
		for i,c in enumerate(cols):
			coldata[i].append(c)

	for i,col in enumerate(coldata):
		isint = True
		isfloat = True
		for x in col:
			try:
				float(x)
			except:
				isfloat = False
				#isint = False
				#break
			try:
				int(x, 0)
			except:
				isint = False
				#break
			if not isint and not isfloat:
				break
		if isint:
			isfloat = False

		if isint:
			vals = [int(x, 0) for x in col]
		elif isfloat:
			vals = [float(x) for x in col]
		else:
			vals = col

		fields.set(colnames[i].lower(), array(vals))
		fields._length = len(vals)

	fields._columns = [c.lower() for c in colnames]

	return fields
