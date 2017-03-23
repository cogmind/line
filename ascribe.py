
''' Document as much as possible. Keep it small and simple.'''

import numpy as np
from enum import Enum #  Requires Python 3.4

'''Using generics'''

# Simple Data Types
dt_byte = np.dtype('u1')
dt_int = np.dtype(int)
dt_sint = np.dtype(np.signedinteger)
dt_double = np.dtype('d')
#dt_float = np.dtype(np.number)
dt_string = np.dtype(np.character)
dt_str = str # String array
dt_date = np.dtype('M')
dt_bool = np.dtype('b')
#dt_str3 = np.dtype('a3') # A 3-character string
dt_str1 = np.dtype('a1') # A 1-character string
dt_str2 = np.dtype('a2') # A 2-character string
#dt_delta_t = np.dtype('m')
#dt_date = np.dtype('U') # unicode
#dt_pyobj = np.dtype(object)



# Complex Data Types
# TODO Add more types for stimulus data

# DEBUGGABLE
#TODO

dt_coordinates = np.dtype('o')  # Treat tuple of rand(n) arrays as a Python object
dt_colors = dt_double # Array of doubles

# This ordered header and type_field must correspond to the Data class (and the file input)
# TODO WARNING Notice the error of the header versus example below ---- colors versus coordinates
header = ['ID', 'Age', 'Gender', 'Session', 'Trial', 'Contrast', 'Items', 'Match', 'Cue', 'RT', 'Response', 'SOA', 'MemCoord', 'MemColor', 'TestColor'] # SIC!!! TODO Debuggable.
type_field = [dt_int, dt_byte, dt_str1, dt_string, dt_int, dt_str2, dt_byte, dt_str, dt_str, dt_double, dt_string, dt_int, dt_colors, dt_coordinates, dt_colors]

''' Example
 ,
23,
f,
Subliminal CDT July 2016,
0,
lo,
3,
non-match,
left,
999999,
FA,
592,
"(['BLACK', 'RED', 'BLUE'],
	['BLACK', 'BLACK', 'GREEN'])",
"([-4.833454718581354, -2.8824553987581445, -2.3081588355080522],
	[-0.0037853209621609984, -2.8078914856435437, -0.31239484357072067],
	[3.86559432251519, 4.653214540852498, 1.83731757692552],
	[-1.3117916221511483, 1.866184035822822, -3.2871611286588642])",
"(['BLACK', 'RED', 'GREEN'],
['BLACK', 'BLACK', 'GREEN'])"

'''

class Data(Enum):
	ID = 1
	AGE = 2
	GENDER = 3
	SESSION = 4
	TRIAL = 5
	CONTRAST = 6
	ITEMS = 7
	MATCH = 8
	CUE = 9
	RT = 10
	RESPONSE = 11
	SOA = 12
	MEMCOORD = 13 #  SIC
	MEMCOLOR = 14 #  SIC
	TESTCOLOR = 15 #  SIC

# Parameter style: [(field_name, field_dtype, field_shape), ...]. Shape omitted
dt_header = np.dtype(zip(len(header) * [dt_str], header))


#def map_type(name, type):
#	return {str(name): type}
#dt_fields = np.dtype([map_type(d, type_field) for d in Data])

# The TMAP
TMAP = dict(d for d in Data, type_field)
field_shape = (12 * [1] ).append()
dt_fields = np.dtype(TMAP, field_shape)



# Back to the drawing board! :)


zip('name', header)



fromfile(filename, data_type_array)



# TODO ID needs to be iterated for each file

