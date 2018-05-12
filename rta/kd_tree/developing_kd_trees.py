import numpy as np


self = kdtree(points)
box = np.array([[600, 1000], 
			  	[25, 	27],
			  	[30, 	40]])

%%timeit
self.box_search(box)

def linear_box_search(points, box):
	return [p for p in points if 
			all(bs <= c <= be for c, (bs, be) in zip(p, box))]

%%timeit
linear_box_search(points, box)