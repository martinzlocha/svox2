import numpy as np

resolution = 128
side_len = 2 / resolution

init_loc = np.array([0., 0., -side_len * (resolution / 2)])
locations = [init_loc]
for i in range(resolution):
    locations.append(init_loc + i * np.array([0., 0., side_len]))
locations = np.stack(locations)
with open('./grid.npy', 'wb') as f:
    np.save(f, {'locations':locations,
                'side_length':side_len})