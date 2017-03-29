#!/usr/bin/env python3
# Make it run in py3
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
from builtins import *
# Integer division may result in null if not run with Python >3.0



''' Analysis of Behavioral Data

Example input file (Line breaks represent CSV)

0	 ,
1	23,
2	f,
3	Subliminal CDT July 2016,
4	0,
5	lo,
6	3,
7	non-match,
8	left,
9	999999,
10	FA,
11	592,
12	"(['BLACK', 'RED', 'BLUE'],
		['BLACK', 'BLACK', 'GREEN'])",
13	"([-4.833454718581354, -2.8824553987581445, -2.3081588355080522],
		[-0.0037853209621609984, -2.8078914856435437, -0.31239484357072067],
		[3.86559432251519, 4.653214540852498, 1.83731757692552],
		[-1.3117916221511483, 1.866184035822822, -3.2871611286588642])",
14	"(['BLACK', 'RED', 'GREEN'],
		['BLACK', 'BLACK', 'GREEN'])"

'''

import numpy as np
import numpy.ma as ma
import pandas as pd

import inspect
from os import path

# Experiment Configuration
items = [2, 3, 4, 5]

# TODO WARNING Notice the error of the header for coordinates and colors

# TODO setup converters
# Setup Converters
# converters : variable, optional
# Example converters = {3: lambda s: float(s or 0)}.

print(60 * '=')
curdir = inspect.stack()[0][1]
current_dir = path.dirname(curdir)
print('Current directory is ' + current_dir)

name = 'ewabre_py_2016-05-16 13-41-12.756000.log'

filename = path.join(current_dir, 'data', name)

file_data = np.genfromtxt(
	filename,
	dtype = None,
	names = True,
	delimiter = ',',
	missing_values = (999999),
	usecols = (1,2,4,5,6,7,8,9,10) # TODO Ignoring stimulus data for now
)

d = file_data.ndim
file_data = np.reshape(file_data,(file_data.size, 1))

print('Dim reshaped from {} to {}'.format(d, file_data.ndim))

print('File Preview: ')
# Print the first 4 rows
#print(file_data[0:4])
print(60 * '-')
# Remove illegal responses
l = len(file_data)
file_data = file_data[file_data['RT'] < 7000]
#print(file_data)
print('Missing Values Removed: {}'.format(l - len(file_data)))
#print(file_data[0:4])
print(60 * '^')

# Compute Signal Detection Theory (SDT) measures
print('Level: Subject')
r = file_data['Response']
print('{} responses'.format(len(r)))
hit = len(r[r == b'HIT'])
miss = len(r[r == b'MISS'])
fa = len(r[r == b'FA'])
cr = len(r[r == b'CR'])
total = hit + miss + fa + cr
assert len(r) == total

print('Total Valid Trials {}'.format(total))
hitrate = 100 * hit / total
farate = 100 * fa / total
print('Hits: {}\nMisses: {}\nFalse Alarms: {}\nCorrect Responses: {}'.format(hit,miss,fa,cr))

# Rounding
decimals = 2

print('Hit{2}: {0}%\nFalse Alarm{2}: {1}%\n[of valid responses]'.format(round(hitrate, decimals),round(farate, decimals),'rate'))

print('Level: Per Item in Subject')
print('Items: {}'.format(items))

#  Compute Cowan's K for all items
 
file_data = pd.DataFrame(file_data)

#print(file_data)

'''TODO Groupby Subject ID... Accumulate'''
counts = file_data.groupby('Items').aggregate(sum)
counts




# item_mask = [file_data['Items' == i] for i in items]
# hit_mask = file_data['Response' == b'HIT']
# fa_mask = file_data['Response' == b'FA']

# k = 0

# for im in item_mask:
# 	k += 1
# 	hi = len(file_data[item_mask[im]][hit_mask])
# 	fai = len(file_data[item_mask[im]][fa_mask])
# 	totali = sum(item_mask[im])
# 	k_Cowans[k] = [i * ((hi - fai) / totali)]

# print([len(items) * '{}'].format(*tuple(k_Cowans)))


#for a in np.nditer(file_data, op_flags=['readwrite']):
#	 b[...] = ma.masked_equal(b, 999999)
#file_data[0].mask

# listwise exclusion
#file_data_compressed = ma.compress_rows(file_data)

#file_data.data[-file_data.mask]

#  Remove the mask
#file_data = file_data.data[-file_data.mask]


#print('Masked File Preview - Missing Values Excluded: ')
# Print the first 2 rows
#print(file_data[0:4]) #. WARNING indexing may refer to outside the soft mask


'''for x, y in np.nditer([a,b]):'''
# Print using Name
#print(file_data['Trial'])



#zip('name', header)


# TODO ID needs to be iterated for each file. Postpone

