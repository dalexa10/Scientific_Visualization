def halton_sequence(index, base):
    """Return the Halton sequence value for the given index and base."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i = i // base
        f /= base
    return result

def generate_halton_points(dimensions, num_points):
    """Generate Halton sequence points in the given dimensions."""
    points = []
    for i in range(num_points):
        point = [halton_sequence(i, p) for p in range(2, dimensions + 2)]
        points.append(point)
    return points

if __name__ == '__main__':

    # Example usage:
    num_dimensions = 2
    num_points = 6
    halton_points = generate_halton_points(num_dimensions, num_points)
    print(halton_points)
