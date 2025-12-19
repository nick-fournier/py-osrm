import os

data_dir = os.path.dirname(os.path.abspath(__file__)) + "/data/"

# Constants and fixtures for Python tests on our Monaco dataset.

# Somewhere in Monaco
# http://www.openstreetmap.org/#map=18/43.73185/7.41772
three_test_coordinates = [(7.41337, 43.72956),
                          (7.41546, 43.73077),
                          (7.41862, 43.73216)]

two_test_coordinates = three_test_coordinates[0:2]

test_tile = {'at': [17059, 11948, 15], 'size': 159125}

# Explicit data paths for different algorithms
ch_data_path = data_dir + "ch/monaco.osrm"
mld_data_path = data_dir + "mld/monaco.osrm"
test_memory_path = data_dir + "test_memory"

# Legacy alias for backward compatibility with test_index.py
data_path = ch_data_path