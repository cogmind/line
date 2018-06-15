import pandas as pd
import numpy as np
from itertools import product, cycle, islice
from collections import Counter
import matplotlib.pyplot as plt


'''
def compute_CowansK(pre_CowansK, item):

    try:
        FA = pre_CowansK.loc[item, 'FA']
    except:
        FA = 0
        print('ERROR in computing FA for ' + str(item) + ' items. Replacing with zero (0)')

    try:
        MISS = pre_CowansK.loc[item, 'MISS']
    except:
        MISS = 0
        print('ERROR in computing MISS for ' + str(item) + ' items. Replacing with zero (0)')

    hit_rate = pre_CowansK.loc[item, 'HIT'] / (pre_CowansK.loc[item, 'HIT'] + MISS)
    cr_rate = pre_CowansK.loc[item, 'CR'] / (pre_CowansK.loc[item, 'CR'] + FA)

    return item * (hit_rate + cr_rate - 1)
'''

def compute_Cowans_K(unit_series, response_series):

    df = pd.DataFrame({'Unit': unit_series, 'Response': response_series})
    print(df)

    START = min(unit_series.unique())
    units = range(START, max(unit_series.unique()) + 1)
    responses = ['HIT', 'FA', 'MISS', 'CR']
    hit = []
    fa = []
    miss = []
    cr = []
    ks = []
    for u in units:
        for r in responses:
            print('u inside loop:', u)
            resp = df[(df.Response == r) & (df.Unit == u)].count().iat[0]
            if r == 'HIT':
                hit.append(resp)
            if r == 'FA':
                fa.append(resp)
            if r == 'MISS':
                miss.append(resp)
            if r == 'CR':
                cr.append(resp)
            print(hit, fa, miss, cr)

    for un in units:
        u = un - START
        hit_rate = hit[u] / (hit[u] + miss[u])
        print('hit_rate ', hit_rate)
        cr_rate = cr[u] / (cr[u] + fa[u])
        print('cr_rate ', cr_rate)
        k = un * (hit_rate + cr_rate - 1)
        ks.append(k)

    return ks


def compute_CowansK_within_subject(dateiname):

    df = pd.read_csv(dateiname, usecols=['ID', 'Age', 'Gender', 'Trial', 'Items', 'Match', 'Response', 'RT', 'MemCoord', 'Cue'])
    df['Visual Grouping'] = compute_visual_grouping_index(df.MemCoord, df.Cue)
    response = ['HIT', 'MISS', 'CR', 'FA']
    item = [2, 3, 4, 5]

    df['Colors'] = df['Items'] - df['Visual Grouping']

    df2 = df.copy()
    pre_CowansK = df.groupby(['Items', 'Response'])['Response'].count()

    #print(pre_CowansK)

    k = []
    for i in item:
            k.append(compute_CowansK(pre_CowansK, i))

    #print(k)

    pre_CowansK2 = df2.groupby(['Colors', 'Response'])['Response'].count()
    print(pre_CowansK2)

    k_corrected = []
    colors = [2, 3, 4, 5]
    for i in colors:
            k_corrected.append(compute_CowansK(pre_CowansK2, i))

    return k, k_corrected


def roundrobin(*iterables):
    '''Used by compute centroid_euclidian_distance()'''
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def _compute_centroid_euclidian_distance(p):
    punkte = len(p)
    xs = p[::2]
    ys = p[1::2]

    cx = np.mean(xs)
    cy = np.mean(ys)
    #print(cx)
    #print(cy)
    mittelwerte = np.array([cx, cy])

    points = list(zip(xs, ys))
    #print(points)

    # Calculate Euclidean distance for each data point assigned to centroid
    distances = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for (x, y) in points]
    # return the mean value
    return np.mean(distances)


def compute_centroid_euclidian_distances(df):

    distances = []
    for _, coordinates in enumerate(df.MemColor):
        coord = eval(coordinates)
        #print(coord)
        xs = coord[0]
        ys = coord[1]
        points = list(roundrobin(xs, ys))
        #print(p)
        distances.append(_compute_centroid_euclidian_distance(points))

    return distances


def convert2color_switch_series(mem_colors, test_colors, cues):
    
    color_switches = []
    pre_color = []
    post_color = []
    for row, _ in enumerate(mem_colors):

        ##print(row)
        #print(cues)
        c = cues.iat[row]
        ##print(c)

        if c == 'left':
            cue = 0
        elif c == 'right':
            cue = 1
        else:
            print('WARNING: No left or right cue, in convert2color_switch_series()')

        mem_color = eval(mem_colors.iat[row])
        test_color = eval(test_colors.iat[row])

        #print(mem_color)
        #print(test_color)
        for m, t in zip(mem_color[cue], test_color[cue]):
            if not m == t:
                target = t
                color = m

        color_switches.append(color + target)
        pre_color.append(color)
        post_color.append(target)

    return color_switches, pre_color, post_color
    '''
    original_colors, target_colors = mem_colors.map(eval), test_colors.map(eval)
    #print(colors_df.MemCoord, '-----------------', colors_df.TestColor)
    cue = cues.isin(['right'])

    targets = []
    originals = []

    for mem_color, test_color, c in zip(mem_colors, test_colors, cue):

        target = list(set(mem_color[c]) - set(test_color[c]))
        original = list(set(test_color[c]) - set(mem_color[c]))
        targets.append(target)
        originals.append(originals)

    print('-----------TARGET-----------')
    print(targets)
    print('-----------ORIGINAL-----------')
    print(originals)

    return target
    '''

def _visual_grouping(array_colors):

    counts = Counter(array_colors)
    #print(counts)
    counts = {k: v for k, v in counts.items() if v > 1}
    #print(counts)
    return len(counts)

# TEST visual_grouping():
#print(visual_grouping(['RED', 'BLUE', 'BLUE', 'GREEN', 'GREEN'])) # Expect 2
#print(visual_grouping(['RED', 'BLUE', 'GREEN', 'GREEN'])) # Expect 1
#print(visual_grouping(['RED', 'BLUE', 'GREEN',])) # Expect 0

def compute_visual_grouping_index(mem_colors, cues):

    cued_colors = []
    for row, _ in enumerate(mem_colors):

            ##print(row)
            #print(cues)
            c = cues.iat[row]
            ##print(c)

            if c == 'left':
                cue = 0
            elif c == 'right':
                cue = 1
            else:
                print('WARNING: No left or right cue, in compute_visual_grouping_index()')

            mem_color = eval(mem_colors.iat[row])[cue]
            cued_colors.append(_visual_grouping(mem_color))

    return cued_colors


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar