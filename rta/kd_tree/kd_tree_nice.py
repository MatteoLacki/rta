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
                    left  = __build_kdtree(sorted_points[:int(n/2)], depth + 1),
                    right = __build_kdtree(sorted_points[int(n/2)+1:], depth + 1))

    return __build_kdtree(points)


# import pprint
# import numpy as np
# points = np.random.rand(5,2)
# pp = pprint.PrettyPrinter(indent=4)
# tree = build_kdtree(points)
# pp.pprint(tree)
