
''' Document as much as possible. Keep it small and simple.


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
import inspect
from enum import Enum #  Requires Python 3.4
from os import path

# Simple Data Types
#dt_int = np.dtype(int)
#dt_sint = np.dtype(np.signedinteger)
#dt_byte = np.dtype('u1')
#dt_double = np.dtype('d')
#dt_float = np.dtype(np.number)
#dt_string = np.dtype(np.character)
#dt_str = np.dtype(str) # String array
#dt_str1 = np.dtype('a1') # A 1-character string
#dt_str2 = np.dtype('a2') # A 2-character string
#dt_date = np.dtype('M')
#dt_bool = np.dtype('b')
#dt_delta_t = np.dtype('m')
#dt_date = np.dtype('U') # unicode
#'dt_pyobj = np.dtype(object)



# Complex Data Types

# Objects of variable lengths
#dt_coordinates = np.dtype('O')  # Treat tuple of rand(n) arrays as a Python object
#dt_colors = np.dtype('O') 

# Define the data types
# This ordered header and type_field must correspond to the Data class (and the file input)
# TODO WARNING Notice the error of the header versus example below ---- colors versus coordinates
header = ['ID', 'Age', 'Gender', 'Session', 'Trial', 'Contrast', 'Items', 'Match', 'Cue', 'RT', 'Response', 'SOA', 'MemCoord', 'MemColor', 'TestColor'] # SIC!!! TODO Debuggable.
#type_field = [dt_int, dt_byte, dt_str1, dt_string, dt_int, dt_str2, dt_byte, dt_str, dt_str, dt_double, dt_string, dt_int, dt_colors, dt_coordinates, dt_colors]

''' Don't know if I need this enum
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
'''

# Format Header
# Parameter style: [(field_name, field_dtype, field_shape), ...]. Shape omitted
# does not work
#dt_header = np.dtype(zip(len(header) * [dt_str], header))

#def map_type(name, type):
#	return {str(name): type}
#dt_fields = np.dtype([map_type(d, type_field) for d in Data])

#TMAP = dict.fromkeys([d for d in header], type_field)

# Format the main fields of the CSV file
#TMAP  = zip(header, type_field) #  Generate an iterable of 2-tuples
#assert type(TMAP)

#a = [t2 for t2 in TMAP]
#print(a)

#dt_fields = np.dtype(a) # field_shape is variable and excluded
#field_shape = (12 * [1] ).append()
# Consider using 'name:' as field


# TODO setup converters
# Setup Converters
# converters : variable, optional
# Example converters = {3: lambda s: float(s or 0)}.

a = inspect.stack()[0][1]
print(a)
current_dir = path.dirname(a)
#os.path.dirname(ascribe.__file__)
name = 'ewabre_py_2016-05-16 13-41-12.756000.log'
filename = path.join(current_dir, 'data', name)
file_data = np.genfromtxt(
	filename,
	dtype = None,#dt_fields DEPRECATE
	names = True,
	delimiter = ',',
	missing_values = (999999),
	usecols = (1,2,4,5,6,7,8,9,10) # TODO Ignore stimulus data
)

# Print the first 12 rows
print(file_data[0:12])

# Print using Name
print(file_data['Trial'])

# Back to the drawing board! :)


#zip('name', header)




# TODO ID needs to be iterated for each file. Postpone

