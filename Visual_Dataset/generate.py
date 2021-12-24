from PIL import Image, ImageDraw
from joblib import Parallel, delayed
import errno, os
import random
from collections import Counter
from math import pi, sin, tan, cos, radians
import json

distractor_count = 6
sample_count = 100
draw_att_map = False

image_width = 256
n_grid = 2 # n_grid on a side
max_n_obj_in_grid = 4
grid_size = image_width / n_grid

large_object_size = (grid_size / 4) ** 2  # 2986
small_object_size = large_object_size / 3

n_core = 8
dirname = 'images'

ellipse_ratio = 1.6
loop_ratio = 0.6



positions = [(grid_size / 2 + grid_size * i, grid_size / 2 + grid_size * j) for i in range(n_grid) for j in range(n_grid)]
colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
sizes = [large_object_size, small_object_size]
half_grid = grid_size / 2
assert(type(image_width) == int)
assert(type(n_grid) == int)
assert(max_n_obj_in_grid <= 4)


# class game_generator:
#     def __init__(self, distractor_count, sample_count, image_width, n_grid_per_side, max_n_obj_in_grid, dirname=None, n_core=-1, ellipse_ratio=1.6, loop_ratio=0.6):
#         assert(type(image_width) == int)
#         assert(type(n_grid_per_side) == int)
#         assert(max_n_obj_in_grid <= 4)

#         self.distractor_count = distractor_count
#         self.sample_count = sample_count
#         self.image_width = image_width
#         self.n_grid = n_grid_per_side
#         self.max_n_obj_in_grid = max_n_obj_in_grid
#         self.dirname = dirname
#         self.n_core = n_core
#         self.ellipse_ratio = ellipse_ratio
#         self.loop_ratio = loop_ratio

#         self.grid_size = self.image_width / self.n_grid
#         self.large_object_size = (self.grid_size / 4) ** 2  # 2986
#         self.small_object_size = self.large_object_size / 3
#         self.positions = [(self.grid_size / 2 + self.grid_size * i, self.grid_size / 2 + self.grid_size * j) for i in range(self.n_grid) for j in range(self.n_grid)]
#         self.colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
#         self.sizes = [large_object_size, small_object_size]
#         self.half_grid = self.grid_size / 2



def round_xy(positions):
    l = []
    for pos in positions:
        l.append((round(pos[0]), round(pos[1])))
    return l

def draw_board(draw):
    white_grid_poses = []
    for i in range(n_grid):
        for j in range(n_grid):
            if (i + j) % 2 == 0:
                x0, y0 = i * grid_size, j * grid_size
                x1, y1 = x0 + grid_size, y0 + grid_size
                white_grid_poses.append([(x0, y0), (x1, y1)])
    for pos in white_grid_poses:
        draw.rectangle(round_xy(pos), 'white')


def circle(draw, color, size, pos, att=False):
    radius = (size / pi) ** 0.5
    x, y = pos
    upper_left = (x - radius, y - radius)
    lower_right = (x + radius, y + radius)
    if not att:
        draw.ellipse(round_xy([upper_left, lower_right]), fill=color)
    else:
        draw.ellipse(round_xy([upper_left, lower_right]), fill='white')


def rect(draw, color, size, pos, att=False):
    half_len = size ** 0.5 / 2
    x, y = pos
    upper_left = (x - half_len, y - half_len)
    lower_right = (x + half_len, y + half_len)
    if not att:
        draw.rectangle(round_xy([upper_left, lower_right]), fill=color)
    else:
        draw.rectangle(round_xy([upper_left, lower_right]), fill='white')


def tri(draw, color, size, pos, att=False):
    x = (size / (3 * (3 ** 0.5))) ** 0.5
    x2 = x * 2
    xsqrt3 = x * (3 * 0.5)
    top = (pos[0], pos[1] - x2)
    left = (pos[0] - xsqrt3, pos[1] + x)
    right = (pos[0] + xsqrt3, pos[1] + x)
    if not att:
        draw.polygon(round_xy([top, left, right]), fill=color)
    else:
        draw.polygon(round_xy([top, left, right]), fill='white')


def ellipse(draw, color, size, pos, att=False):
    '''
    area = pi * a * b
    a: long half axis
    b: short half axis
    '''
    b = (size / pi / ellipse_ratio)**0.5
    a = ellipse_ratio * b
    x, y = pos
    upper_left = (x - a, y - b)
    lower_right = (x + a, y + b)
    if not att:
        draw.ellipse(round_xy([upper_left, lower_right]), fill=color)
    else:
        draw.ellipse(round_xy([upper_left, lower_right]), fill='white')


def star(draw, color, size, pos, att=False):
    center_to_pentagon_edge = ((size / 5) / (1 / tan(radians(54)) + tan(radians(72)) * ((1 / tan(radians(54))) ** 2))) ** 0.5
    pentagon_edge_to_vertex =  tan(radians(72)) * (1 / tan(radians(54))) * center_to_pentagon_edge
    edge_len = ((1 / tan(radians(54))) / cos(radians(72))) * center_to_pentagon_edge
    half_pentage_edge_len = (1 / tan(radians(54))) * center_to_pentagon_edge
    center_to_vertex = center_to_pentagon_edge + pentagon_edge_to_vertex
    center_to_pentagon_vertex = center_to_pentagon_edge / sin(radians(54))
    x, y = pos
    top = (x, y - center_to_vertex)
    upper_left_inner = (x - half_pentage_edge_len, y - center_to_pentagon_edge)
    left = (x - half_pentage_edge_len - edge_len, y - center_to_pentagon_edge)
    left_inner = (x - cos(radians(18)) * center_to_pentagon_vertex, y + sin(radians(18)) * center_to_pentagon_vertex)
    bottom_left = (x - sin(radians(36)) * center_to_vertex, y + cos(radians(36)) * center_to_vertex)
    bottom_inner = (x, y + center_to_pentagon_vertex)
    bottom_right = (x + sin(radians(36)) * center_to_vertex, y + cos(radians(36)) * center_to_vertex)
    right_inner = (x + cos(radians(18)) * center_to_pentagon_vertex, y + sin(radians(18)) * center_to_pentagon_vertex)
    right = (x + half_pentage_edge_len + edge_len, y - center_to_pentagon_edge)
    upper_right_inner = (x + half_pentage_edge_len, y - center_to_pentagon_edge)
    if not att:
        draw.polygon(round_xy([top, upper_left_inner, left, left_inner, bottom_left, bottom_inner, bottom_right, right_inner, right, upper_right_inner]), fill=color)
    else:
        draw.polygon(round_xy([top, upper_left_inner, left, left_inner, bottom_left, bottom_inner, bottom_right, right_inner, right, upper_right_inner]), fill='white')


def loop(draw, color, size, pos, att=False):
    outer_radius = (size / pi) ** 0.5
    inner_radius = outer_radius * loop_ratio
    x, y = pos
    outer_upper_left = (x - outer_radius, y - outer_radius)
    outer_lower_right = (x + outer_radius, y + outer_radius)
    inner_upper_left = (x - inner_radius, y - inner_radius)
    inner_lower_right = (x + inner_radius, y + inner_radius)
    if not att:
        draw.ellipse(round_xy([outer_upper_left, outer_lower_right]), fill=color)
        if (x // grid_size + y // grid_size) % 2 == 0: 
            # white grid
            draw.ellipse(round_xy([inner_upper_left, inner_lower_right]), fill='white')
        else:
            draw.ellipse(round_xy([inner_upper_left, inner_lower_right]), fill='black')
    else:
        draw.ellipse(round_xy([outer_upper_left, outer_lower_right]), fill='white')
        draw.ellipse(round_xy([inner_upper_left, inner_lower_right]), fill='black')




def generate(core_index):
    for i in range(round(core_index * sample_count / n_core), round((core_index + 1) * sample_count / n_core)):
        im = Image.new('RGB', (image_width, image_width))
        draw = ImageDraw.Draw(im)
        draw_board(draw)

        while True:
            shape_fn_chosens = random.choices(shape_fns, k=distractor_count)
            color_chosens = random.choices(colors, k=distractor_count)
            size_chosens = random.choices(sizes, k=distractor_count)
            position_chosens = random.choices(positions, k=distractor_count)

            resample = False
            for pos, count in Counter(position_chosens).items():
                if count > max_n_obj_in_grid:
                    resample = True
                    break
                if count == 2:
                    find1, find2 = [k for k, position in enumerate(position_chosens) if position == pos]
                    if shape_fn_chosens[find1] == shape_fn_chosens[find2] and color_chosens[find1] == color_chosens[find2]:
                        resample = True
                        break
                    x, y = pos
                    fourth_grid = grid_size / 4
                    if random.randrange(2) == 0:
                        x1 = x - fourth_grid
                        x2 = x + fourth_grid
                        y1 = y - fourth_grid
                        y2 = y + fourth_grid
                    else:
                        x1 = x + fourth_grid
                        x2 = x - fourth_grid
                        y1 = y - fourth_grid
                        y2 = y + fourth_grid
                    position_chosens[find1] = (x1, y1)
                    position_chosens[find2] = (x2, y2)
                    
                elif count == 3:
                    find = find1, find2, find3 = [k for k, position in enumerate(position_chosens) if position == pos]
                    l = [(shape_fn_chosens[f], color_chosens[f]) for f in find]
                    s = set(l)
                    if len(l) != len(s):
                        resample = True
                        break
                    x, y = pos
                    fourth_grid = grid_size / 4
                    possible_positions = [(x - fourth_grid, y - fourth_grid), (x - fourth_grid, y + fourth_grid), (x + fourth_grid, y - fourth_grid), (x + fourth_grid, y + fourth_grid)]
                    pos1, pos2, pos3 = random.sample(possible_positions, k=3)
                    position_chosens[find1] = pos1
                    position_chosens[find2] = pos2
                    position_chosens[find3] = pos3
                
                elif count == 4:
                    find = find1, find2, find3, find4 = [k for k, position in enumerate(position_chosens) if position == pos]
                    l = [(shape_fn_chosens[f], color_chosens[f]) for f in find]
                    s = set(l)
                    if len(l) != len(s):
                        resample = True
                        break
                    x, y = pos
                    fourth_grid = grid_size / 4
                    x1 = x - fourth_grid
                    x2 = x + fourth_grid
                    x3 = x + fourth_grid
                    x4 = x - fourth_grid
                    y1 = y - fourth_grid
                    y2 = y + fourth_grid
                    y3 = y - fourth_grid
                    y4 = y + fourth_grid
                    position_chosens[find1] = (x1, y1)
                    position_chosens[find2] = (x2, y2)
                    position_chosens[find3] = (x3, y3)
                    position_chosens[find4] = (x4, y4)
            
            if not resample:
                break

        unique_objects = set(zip(shape_fn_chosens, color_chosens, size_chosens, position_chosens))
        m = {}
        for j, info in enumerate(unique_objects):
            fn, color, size, pos = info
            fn(draw, color, size, pos)
            m[j] = {'shape': fn.__name__, 'color': color, 'size': size, 'pos': pos}
        
        with open('{}/{}.json'.format(dirname, i), 'w') as f:
            json.dump(m, f)
        im.save('{}/{}.png'.format(dirname, i), "png")

        if draw_att_map:
            im_att = Image.new('1', (image_width, image_width))
            draw_att = ImageDraw.Draw(im_att)
            target_ind = random.randrange(len(unique_objects))
            fn_att, color_att, size_att, pos_att = list(unique_objects)[target_ind]
            fn_att(draw_att, color_att, size_att, pos_att, True)
        
            dscp = '{}_{}_{}_{}_{}'.format(fn_att.__name__, color_att, 'l' if size_att == large_object_size else 's', pos_att[0], pos_att[1])
            im_att.save('{}/{}_{}.png'.format(dirname, i, dscp), "png")


import numpy as np
def bayesian_update(old_belief, concepts, info):
    likelihood = []
    for concept in concepts:
        prob = 1.0 * (info in concept) / len(concept)
        likelihood.append(prob)
    new_belief = old_belief * np.array(likelihood)
    new_belief /= np.sum(new_belief) + 1e-9
    return new_belief

if __name__ == "__main__":
    try:
        os.mkdir(dirname)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    global shape_fns
    shape_fns = [circle, rect, tri, ellipse, star, loop]
    # Parallel(n_core)(delayed(generate)(i) for i in range(n_core))

    np.set_printoptions(threshold=np.nan)


    game_set = [[4, 8, 12, 14],
                [2, 11, 13, 15],
                [3, 6, 12, 15],
                [0, 6, 12, 15],
                [2, 10, 12, 17],
                [0, 8, 13, 17]]
    
    im0 = Image.new('1', (image_width, image_width))
    star(ImageDraw.Draw(im0), 'blue', large_object_size, (64.0, 64.0), True)
    im0_2d = np.asarray(im0, np.float32) * 255
    im1 = Image.new('1', (image_width, image_width))
    tri(ImageDraw.Draw(im1), 'magenta', small_object_size, (160, 32), True)
    im1_2d = np.asarray(im1, np.float32) * 255
    im2 = Image.new('1', (image_width, image_width))
    ellipse(ImageDraw.Draw(im2), 'red', large_object_size, (160, 96), True)
    im2_2d = np.asarray(im2, np.float32) * 255
    im3 = Image.new('1', (image_width, image_width))
    circle(ImageDraw.Draw(im3), 'red', large_object_size, (224, 96), True)
    im3_2d = np.asarray(im3, np.float32) * 255
    im4 = Image.new('1', (image_width, image_width))
    tri(ImageDraw.Draw(im4), 'cyan', large_object_size, (160, 160), True)
    im4_2d = np.asarray(im4, np.float32) * 255
    im5 = Image.new('1', (image_width, image_width))
    circle(ImageDraw.Draw(im5), 'blue', small_object_size, (224, 224), True)
    im5_2d = np.asarray(im5, np.float32) * 255

    im_2d = [im0_2d, im1_2d, im2_2d, im3_2d, im4_2d, im5_2d]
    im = [im0, im1, im2, im3, im4, im5]
    prior = np.ones(6) / 6
    belief1 = bayesian_update(prior, game_set, 0)
    belief2 = bayesian_update(belief1, game_set, 8)
    belief3 = bayesian_update(belief2, game_set, 13)
    belief4 = bayesian_update(belief3, game_set, 17)
    import matplotlib.pyplot as plt
    for num, belief in enumerate([prior, belief1, belief2, belief3, belief4]):
        map = np.zeros_like(im0_2d)
        for i, i_2d in zip(belief, im_2d):
            map += i * i_2d
        img = Image.fromarray(map.astype(np.uint8), mode='L')
        img.save('{}.png'.format(num))
    
