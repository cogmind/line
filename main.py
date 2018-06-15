'''
Verfasser: Carina Rose, Daniel Labbe
Datum: 2018-05-22

Analyze der Behavorialen Daten
'''

import pandas as pd
from os import getcwd, path, walk
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
import numpy as np
from itertools import repeat, chain
#from sklearn.decomposition import PCA
from sklearn.manifold import MDS

from computations import compute_Cowans_K, compute_centroid_euclidian_distances, convert2color_switch_series, roundrobin, compute_visual_grouping_index, heatmap

# SETTINGS
PLOT_COWANS_K = False

def plot_k(k):
	y = k
	x = np.arange(2, 6)
	print(x)
	print(y)

	# Plot by item for each individual
	fig, ax = plt.subplots()
	y2 = [(yy + 1) for yy in y]
	plt.plot(x, y, x, y2) # BEISPIEL
	plt.ylabel('Cowan\'s K')
	plt.minorticks_off()
	#plt.set(xlabel='Items', ylabel='Cowan\'s K',
	#       title='Cowan\'s K')

	plt.show()


print('Analyze der Behavorialen Daten')
print(60 * '-')
current_dir = 'D:\\analysis\\pomo\\_PythonPanda\\BehavioralAnalysis'
current_dir = current_dir is '' and getcwd() or current_dir
current_dir = current_dir + path.sep + 'data' + path.sep
print(current_dir + '\\n')


f = [] # Stackoverflow Recipe
for (dirpath, dirname, filename) in walk(current_dir):
    f.extend(filename)
    break

files = pd.Series(f)
print('### Files')
print(files)
print('### Subjects')
print(*[file[0:6] for file in files])

iter_files = iter(f) #np.nditer(f)

ks = []
corrected_ks = []
group_level_items = []
vg_group_level_items = []
group_level_color_switch = []

for filename in iter_files:
	print('Current file is {}...'.format(filename))
	filename = path.join(current_dir, filename)

	df = pd.read_csv(filename, usecols=['Age', 'Gender', 'Trial', 'Items', 'Match', 'Cue', 'RT', 'Response', 'MemCoord', 'MemColor', 'TestColor'])
	#print(df.head)

	# Remove extreme values, such as missing data
	df = df[df['RT'] < 7000.0]
	# Compute Centroid Euclidian Distance 
	# Compute Visual grouping index
	df['Visual Grouping'] = compute_visual_grouping_index(df.MemCoord, df.Cue)
	#print(df['Visual Grouping'])
	print('Computing Centroid Euclidian Distances...')
	df['Centroid Euclidian Distances'] = compute_centroid_euclidian_distances(df)

	# Compute Cowan's K
	print('Computing Cowan\'s K...')
	#k, corrected_k = compute_CowansK_within_subject(filename)
	#if PLOT_COWANS_K:
	#plot_k(k)
	#plot_k(corrected_k)
	#ks.append(k)
	#corrected_ks.append(corrected_k)

	df['Visual Grouping'] = compute_visual_grouping_index(df.MemCoord, df.Cue)
	df['Colors'] = df['Items'] - df['Visual Grouping']
	centroid_analysis_corrected = df.copy()
	standard_k = compute_Cowans_K(centroid_analysis_corrected.Items, centroid_analysis_corrected.Response)
	corrected_k = compute_Cowans_K(centroid_analysis_corrected.Colors, centroid_analysis_corrected.Response)
	print(standard_k)
	print(corrected_k)


	centroid_analysis_corrected = df.groupby('Colors')




	centroid_analysis = df.copy()
	centroid_analysis = centroid_analysis.groupby('Items')

	print('centroid_analysis:')
	for name, items in centroid_analysis:
		print(name, items)

	centroid_analysis['K'] = k

	centroid_analysis_corrected['corrected_k'] = corrected_k

	grouping_analysis_corrected = grouping_analysis_corrected.groupby('Colors')
	grouping_analysis_corrected['corrected_k'] = corrected_k

	centroid_analysis = centroid_analysis.agg(np.mean)
	grouping_analysis_corrected = grouping_analysis_corrected.agg(np.mean)
	centroid_analysis_corrected = centroid_analysis_corrected.agg(np.mean)
	#ITEMS = [2, 3, 4, 5]
	#colors = pd.Categorical.from_array(df.Response).labels
	#color_label = pd.Categorical.from_array(df.Response).categories.values
	#alpha = 0.45
#	plt.scatter(y=df['RT'][df.Items == 2], x=df[df.Items == 2].index, s= 8** (df['Centroid Euclidian Distances']), c='r', label='Two', alpha=alpha) #c=colors
#	plt.scatter(y=df['RT'][df.Items == 3], x=df[df.Items == 3].index, s= 8** (df['Centroid Euclidian Distances']), c='g', label='Three', alpha=alpha) #c=colors
#	plt.scatter(y=df['RT'][df.Items == 4], x=df[df.Items == 4].index, s= 8** (df['Centroid Euclidian Distances']), c='b', label='Four', alpha=alpha) #c=colors
#	plt.scatter(y=df['RT'][df.Items == 5], x=df[df.Items == 5].index, s= 8** (df['Centroid Euclidian Distances']), c='c', label='Five', alpha=alpha) #c=colors
#	plt.legend()
#	plt.show()

	#print(centroid_analysis)



	#TODO centroid_analysis_corrected
	group_level_items.append(centroid_analysis)
	vg_group_level_items.append(grouping_analysis_corrected)

	'''
		plt.scatter(y=df['RT'][df.Items == 2], x=df[df.Items == 2].index, s= 12** (df['Visual Grouping']), c='r', label='Two', alpha=alpha) #c=colors
		plt.scatter(y=df['RT'][df.Items == 3], x=df[df.Items == 3].index, s= 12** (df['Visual Grouping']), c='g', label='Three', alpha=alpha) #c=colors
		plt.scatter(y=df['RT'][df.Items == 4], x=df[df.Items == 4].index, s= 12** (df['Visual Grouping']), c='b', label='Four', alpha=alpha) #c=colors
		plt.scatter(y=df['RT'][df.Items == 5], x=df[df.Items == 5].index, s= 12** (df['Visual Grouping']), c='c', label='Five', alpha=alpha) #c=colors
		plt.legend()
		plt.show()
	'''
		
	#	for label, group in centroid_analysis:
	#		y = centroid_analysis['RT']
	#		area = centroid_analysis['Centroid Euclidian Distances']
	#		print(type(y), type(area))
		


	#print(df['Centroid Euclidian Distances'])

	

	# Compute Color Switches
	non_match = df[df['Match'] == 'non-match']
	print('Computing Color Switches...')
	non_match['ColorSwitch'], non_match['PreColor'], non_match['PostColor'] = convert2color_switch_series(non_match.MemCoord, non_match.TestColor, non_match.Cue)

	df['ColorSwitch'] = non_match['ColorSwitch']
	df['PreColor'] = non_match['PreColor']
	df['PostColor'] = non_match['PostColor']
	
	#print(df)
	group_level_color_switch.append(non_match)
	#print(grouped['RT'].agg([np.mean, np.std, np.median]))

	non_match['CR'] = pd.get_dummies(non_match.Response)['CR']
	non_match['FA'] = pd.get_dummies(non_match.Response)['FA']

	#print(non_match.groupby(['CR', 'ColorSwitch']).agg({'CR': np.sum, 'FA': np.sum, 'RT': [np.mean, np.std, np.median]}))
	
	#print('describe')
	#print(df.describe())
	
	#del df
	#del non_match
	
	print(60 * '-')

print(ks)
print(corrected_ks)

group_level_items = pd.concat(group_level_items)
vg_group_level_items = pd.concat(vg_group_level_items)
group_level_color_switch = pd.concat(group_level_color_switch)

ITEMS = 4
COLORS = 5

indexes = []
for i in range(0, int(len(group_level_items) / ITEMS)):
	index = repeat(i + 1, ITEMS)
	indexes.append(list(index))
indexes = list(chain.from_iterable(indexes))

print(indexes)	
group_level_items['Subject'] = indexes
print(group_level_items)


print(vg_group_level_items)
# Without Visual Grouping
vg_indexes = []
for i in range(0, int(len(vg_group_level_items) / COLORS)):
	index = repeat(i + 1, COLORS)
	vg_indexes.append(list(index))
vg_indexes = list(chain.from_iterable(vg_indexes))

print(vg_indexes)	
vg_group_level_items['Subject'] = vg_indexes
print(vg_group_level_items)

#fig, axes = plt.subplots(nrows=2, ncols=2)

#c = plt.cm.Spectral(np.linspace(0, 1, 256))
#ax = plt.gca()
#ax.set_color_cycle(['black', 'blue', 'red', 'green'])
#plt.plot(y=k, x=k)#, ax=axes[0, 0])


plt.subplot(311)

for n in range(1, max(group_level_items['Subject'].values)):
	x = group_level_items.index[group_level_items['Subject'] == n]
	y = group_level_items[group_level_items['Subject'] == n]['K']
	plt.plot(x, y, label=n)
plt.ylabel('Cowan\'s K')
plt.xlabel('Items')
plt.minorticks_off()
#plt.show()


plt.subplot(312)

for n in range(1, max(vg_group_level_items['Subject'].values)):
	x = vg_group_level_items.index[vg_group_level_items['Subject'] == n]
	y = vg_group_level_items[vg_group_level_items['Subject'] == n]['Corrected K']
	plt.plot(x, y, label=n)
plt.ylabel('Cowan\'s K')
plt.xlabel('Items')
plt.minorticks_off()
#plt.show()


plt.subplot(313)

for n in range(1, max(group_level_items['Subject'].values)):
	x = group_level_items.index[group_level_items['Subject'] == n]
	y = group_level_items[group_level_items['Subject'] == n]['RT']
	plt.plot(x, y, label=n)
plt.ylabel('RT(ms)')
plt.xlabel('Items')
plt.minorticks_off()
plt.show()


#print('All Cowan\'s K:s ', ks)
#print(group_level_color_switch)


colors = ['RED', 'BLACK', 'BLUE', 'WHITE', 'YELLOW', 'GREEN']


color_switch_RT_means = []
for pre in colors:
	for post in colors:
		color_switch_RT_mean = group_level_color_switch[(group_level_color_switch['PreColor'] == pre) & (group_level_color_switch['PostColor'] == post)]['RT'].agg(np.mean)
		color_switch_RT_means.append(color_switch_RT_mean)
#print(color_switch_RT_means)

color_switch_RT_means = color_switch_RT_means - np.nanmean(color_switch_RT_means)
#print(color_switch_RT_means)
color_switch_RT_means = color_switch_RT_means.reshape((len(colors), len(colors)))
#color_switch_RT_means = np.split(color_switch_RT_means, len(colors))
print(color_switch_RT_means)

# Heat Map: Whitened Response Times
fig, ax = plt.subplots()
im, cbar = heatmap(color_switch_RT_means, colors, colors, ax=ax,
                   cmap="RdBu", cbarlabel="Whitened Mean RT per Color Switch (ms)")
#texts = annotate_heatmap(im, valfmt="{x:.1f} t")

fig.tight_layout()
plt.show()



mds_color_switches = MDS.fit(color_switch_RT_means, color_switch_RT_means)
print(mds_color_switches)
plt.scatter(mds_color_switches[:, 0], mds_color_switches[:, 1])

'''
# PCA Color switches
mean_match_rt = df[df['Match'] == 'match']['RT'].agg(np.mean)
print(mean_match_rt)

nans = np.where(np.isnan(color_switch_RT_means))
print(nans)

color_switch_RT_means[nans] = 0
print(color_switch_RT_means)

pca = PCA(color_switch_RT_means)
plt.scatter(pca.Y[:,0], pca.Y[:,1])
plt.show()
'''