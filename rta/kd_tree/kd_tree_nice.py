def build_kdtree(points):
    # preprocessing
    k = len(points[0])

    # main function
    def __build_kdtree(points, depth=0):
        n = len(points)

        if n <= 0:
            return None

        axis = depth % k
        sorted_points = sorted(points, key=lambda point: point[axis])

        return dict(point = sorted_points[int(n / 2)],
                    left  = __build_kdtree(sorted_points[          :int(n/2)], depth + 1),
                    right = __build_kdtree(sorted_points[int(n/2)+1: ]       , depth + 1))

    return __build_kdtree(points)



class kdtree(object):
    """Simple kdtree implementation for fast box search."""

    def __init__(self, points):
        """Instantiate the class."""
        # the dimension k
        self.k = len(points[0][0])

        # recursively build a kd-tree
        def __build_kdtree(points, depth=0):
            n = len(points)

            if n <= 0:
                return None

            axis = depth % self.k
            sorted_points = sorted(points, key=lambda point: point[0][axis])

            return dict(point = sorted_points[int(n / 2)],
                        left  = __build_kdtree(sorted_points[          :int(n/2)], depth + 1),
                        right = __build_kdtree(sorted_points[int(n/2)+1:        ], depth + 1))

        self.tree = __build_kdtree(points)

    def box_search(self, box):
        """Recursive box search."""
        
        assert len(box) == self.k
        res = []
        
        def __box_search(subtree, res, depth=0):
            if subtree is not None:
                point   = subtree['point']
                axis    = depth % self.k
                bs, be  = box[axis]
                if point[0][axis] >= bs:
                   __box_search(subtree['left'], res, depth + 1)
                
                if point[0][axis] <= be:
                    __box_search(subtree['right'], res, depth + 1)
                
                if all(bs <= p <= be for p, (bs, be) in zip(point[0], box)):
                    res.append(point[1])                     

            
        __box_search(self.tree, res)

        return res


# import pprint
# import numpy as np
# points = np.random.rand(5,2)
# pp = pprint.PrettyPrinter(indent=4)
# tree = build_kdtree(points)
# pp.pprint(tree)
