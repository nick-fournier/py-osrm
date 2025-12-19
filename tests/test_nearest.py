import osrm
import constants

mld_data_path = constants.mld_data_path
two_test_coordinates = constants.two_test_coordinates

class TestNearest:
    py_osrm = osrm.OSRM(
        storage_config = mld_data_path, 
        algorithm = "MLD",
        use_shared_memory = False
    )

    def test_nearest(self):
        res = self.py_osrm.Nearest(
            coordinates=[two_test_coordinates[0]],
            exclude=["motorway"]
        )
        assert(len(res["waypoints"]) == 1)
