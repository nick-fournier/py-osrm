import pytest
import osrm
import constants

ch_data_path = constants.ch_data_path
mld_data_path = constants.mld_data_path
test_tile = constants.test_tile

class TestTile:
    @classmethod
    def setup_class(cls):
        cls.py_osrm = osrm.OSRM(
            storage_config = ch_data_path, 
            use_shared_memory = False
        )

    def test_tile(self):
        res = self.py_osrm.Tile(test_tile["at"])
        assert(len(res) == test_tile["size"])

    def test_tile_preconditions(self):
        with pytest.raises(Exception):
            # Must be an array
            tile_params = osrm.TileParameters(17059, 11948, -15)
        with pytest.raises(Exception):
            # Must be unsigned
            tile_params = osrm.TileParameters([17059, 11948, -15])
        tile_params = osrm.TileParameters([17059, 11948, 15])
        res = self.py_osrm.Tile(tile_params)
