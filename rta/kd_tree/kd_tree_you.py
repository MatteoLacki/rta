import pprint
pp = pprint.PrettyPrinter(indent=4)

def build_kdtree(points, depth=0):
    n = len(points)
    
    if n <= 0:
        return None

    axis = depth % k

    sorted_points = sorted(points, key=lambda point: point[axis])

    return dict(point = sorted_points[n / 2],
                left  = build_kdtree(sorted_points[: n/2], depth + 1),
                right = build_kdtree(sorted_points[n/2 + 1:], depth + 1)

