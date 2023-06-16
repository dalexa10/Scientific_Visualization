import numpy as np

def linear_interpolation(x1, f1, x2, f2, x):
    t = (x - x1) / (x2 - x1)
    f_int = (1 - t) * f1 + t * f2
    return f_int

def interpolate(v1,v2,t):
    int_val = (t - v1) / (v2 - v1)
    return int_val
def getContourCase(top, left, thres, cells):
    lookup_dict = {'0000': 0,
                   '0001': 1,
                   '0010': 2,
                   '0011': 3,
                   '0100': 4,
                   '0101': 5,
                   '0110': 6,
                   '0111': 7,
                   '1000': 8,
                   '1001': 9,
                   '1010': 10,
                   '1011': 11,
                   '1100': 12,
                   '1101': 13,
                   '1110': 14,
                   '1111': 15}

    try:
        comp_f = lambda val, thres: '1' if val >= thres else '0'
        cell_code = comp_f(cells[top, left], thres) + comp_f(cells[top, left + 1], thres) + comp_f(cells[top + 1, left + 1], thres) + comp_f(cells[top + 1, left], thres)
        case = lookup_dict[cell_code]
#         print(case)
        return case
    except Exception as e:
        return 0

def disambiguateSaddle(top,left,thres,cells):
    avg = (cells[top, left] + cells[top, left + 1] + cells[top + 1, left + 1] + cells[top + 1, left]) / 4
    # print(avg)
    if avg >= thres:
        return True
    else:
        return False


def getCellSegments(top,left,thres,cells):
    try:
        saddle_ls = [5, 10]
        cd_ls = []
        case = getContourCase(top, left, thres, cells)
#         print(case)

        if case == 0 or case == 15:
            return []

        elif case == 1 or case == 14:
            y1, x1 = top + interpolate(cells[top, left], cells[top + 1, left], thres), left
            y2, x2 = top + 1, left + interpolate(cells[top + 1, left], cells[top + 1, left + 1], thres)
            cd_ls.append([(x1, y1), (x2, y2)])
            return cd_ls

        elif case == 2 or case == 13:
            y1, x1 = top + 1, left + interpolate(cells[top + 1, left], cells[top + 1, left + 1], thres)
            y2, x2 = top + interpolate(cells[top, left + 1], cells[top + 1, left + 1], thres), left + 1
            cd_ls.append([(x1, y1), (x2, y2)])
            return cd_ls

        elif case == 3 or case == 12:
            y1, x1 = top + interpolate(cells[top, left], cells[top + 1, left], thres), left
            y2, x2 = top + interpolate(cells[top, left + 1], cells[top + 1, left + 1], thres), left + 1
            cd_ls.append([(x1, y1), (x2, y2)])
            return cd_ls

        elif case == 4 or case == 11:
            y1, x1 = top, left + interpolate(cells[top, left], cells[top, left + 1], thres)
            y2, x2 = top + interpolate(cells[top, left + 1], cells[top + 1, left + 1], thres), left + 1
            cd_ls.append([(x1, y1), (x2, y2)])
            return cd_ls

        elif case == 6 or case == 9:
            y1, x1 = top, left + interpolate(cells[top, left], cells[top, left + 1], thres)
            y2, x2 = top + 1, left + interpolate(cells[top + 1, left], cells[top + 1, left + 1], thres)
            cd_ls.append([(x1, y1), (x2, y2)])
            return cd_ls

        elif case == 7 or case == 8:
            y1, x1 = top + interpolate(cells[top, left], cells[top + 1, left], thres), left
            y2, x2 = top, left + interpolate(cells[top, left], cells[top, left + 1], thres)
            cd_ls.append([(x1, y1), (x2, y2)])
            return cd_ls

        elif case == 5 or case == 10:
            dis_cell = disambiguateSaddle(top,left,thres,cells)

            if (case == 5 and dis_cell == True) or (case == 10 and dis_cell == False):
                y1, x1 = top + interpolate(cells[top, left], cells[top + 1, left], thres), left
                y2, x2 = top, left + interpolate(cells[top, left], cells[top, left + 1], thres)
                y3, x3 = top + 1, left + interpolate(cells[top + 1, left], cells[top + 1, left + 1], thres)
                y4, x4 = top + interpolate(cells[top, left + 1], cells[top + 1, left + 1], thres), left + 1
                cd_ls.append([(x1, y1), (x2, y2)])
                cd_ls.append([(x3, y3), (x4, y4)])
                return cd_ls
            else:
                y1, x1 = top, left + interpolate(cells[top, left], cells[top, left + 1], thres)
                y2, x2 = top + interpolate(cells[top, left + 1], cells[top + 1, left + 1], thres), left + 1
                y3, x3 = top + interpolate(cells[top, left], cells[top + 1, left], thres), left
                y4, x4 = top + 1, left + interpolate(cells[top + 1, left], cells[top + 1, left + 1], thres)
                cd_ls.append([(x1, y1), (x2, y2)])
                cd_ls.append([(x3, y3), (x4, y4)])
                return cd_ls
    except Exception as e:
        return []

def getContourSegments(thres,cells):
    #Hint: this function should accumulate a list of line segments by looping
    #      over each cell in "cells" and calling getCellSegments() on each cell
    coord_ls = []
    for j in range(cells.shape[1]):
        for i in range(cells.shape[0]):
            cell_ls = getCellSegments(j,i,thres,cells)
            for cell_i in cell_ls:
                coord_ls.append(cell_i)
    return coord_ls


def get_mid_point_segment(sgmt):
    mid_cd = ((sgmt[0][0] + sgmt[1][0])/2, (sgmt[0][1] + sgmt[1][1])/2)
    return mid_cd


if __name__ == '__main__':
    top_pos = (8, 6)
    thres = 73
    cells = np.array([[22, 90],
                      [22, 64]])

    case = getContourCase(0, 0, thres, cells)
    print('This cell array corresponds to case: {}'.format(case))

    cell_segments = getCellSegments(0, 0, thres, cells)
    print('The segments for this cell array are: \n')
    for i, seg in enumerate(cell_segments):
        print('Segment {}'.format(i))
        print(seg)
        v_pos_rel = get_mid_point_segment(seg)
        v_pos = (v_pos_rel[0] + top_pos[0], top_pos[1] - v_pos_rel[1])
        print('Vertex position for dual marching square is: \n')
        print(v_pos)
