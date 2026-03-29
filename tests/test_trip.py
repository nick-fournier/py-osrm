import osrm
import constants

ch_data_path = constants.ch_data_path
mld_data_path = constants.mld_data_path
three_test_coordinates = constants.three_test_coordinates
two_test_coordinates = constants.two_test_coordinates

class TestTrip:
    @classmethod
    def setup_class(cls):
        cls.py_osrm = osrm.OSRM(
            storage_config = ch_data_path, 
            use_shared_memory = False
        )

    def test_trip_manylocations(self):
        res = self.py_osrm.Trip(
            coordinates = three_test_coordinates[0:5]
        )
        for trip in res["trips"]:
            assert(trip["geometry"])

    def test_trip_invalidargs(self):
        py_osrm = osrm.OSRM()
        res = py_osrm.Trip(
            coordinates = two_test_coordinates
        )
        for trip in res["trips"]:
            assert(trip["geometry"])

    # def test_trip_hints(self):
    #     trip_parameters = osrm.TripParameters(
    #         coordinates = two_test_coordinates,
    #         steps = False
    #     )
    #     res = self.py_osrm.Trip(trip_parameters)
    #  
    #     for trip in res["trips"]:
    #         assert(trip["geometry"])
    #     assert(res["waypoints"]["map"])
    #     for h in res["waypoints"]["map"]:
    #         assert(isinstance(h, str))

    def test_trip_geometrycompression(self):
        py_osrm = osrm.OSRM()
        res = py_osrm.Trip(
            coordinates = [three_test_coordinates[0], three_test_coordinates[1]]
        )
        for trip in res["trips"]:
            assert(isinstance(trip["geometry"], str))

    def test_trip_nogeometrycompression(self):
        py_osrm = osrm.OSRM()
        res = py_osrm.Trip(
            coordinates = two_test_coordinates,
            geometries = "geojson"
        )
        for trip in res["trips"]:
            assert(isinstance(trip["geometry"]["coordinates"], list))
    
    def test_trip_speedannotations(self):
        py_osrm = osrm.OSRM()
        res = py_osrm.Trip(
            coordinates = two_test_coordinates,
            steps = True,
            annotations = ["speed"],
            overview = "false"
        )
        for trip in res["trips"]:
            assert(trip)
            for l in trip["legs"]:
                assert(len(l["steps"]) > 0
                    and l["annotation"]
                    and l["annotation"]["speed"])
                assert("weight" not in l["annotation"]
                    and "datasources" not in l["annotation"]
                    and "duration" not in l["annotation"]
                    and "distance" not in l["annotation"]
                    and "nodes" not in l["annotation"])
                assert("geometry" not in l)

    def test_trip_severalannotations(self):
        res = self.py_osrm.Trip(
            coordinates = two_test_coordinates,
            steps = True,
            annotations = ["duration", "distance", "nodes"],     
            overview = "false"
        )
        assert(len(res["trips"]) == 1)
        for trip in res["trips"]:
            assert(trip)
            for l in trip["legs"]:
                assert(len(l["steps"]) > 0)
                assert(l["annotation"]
                    and l["annotation"]["distance"] 
                    and l["annotation"]["duration"] 
                    and l["annotation"]["nodes"])
                assert("weight" not in l["annotation"]
                    and "datasources" not in l["annotation"] 
                    and "speed" not in l["annotation"])
                assert("geometry" not in l)

    def test_trip_options(self):
        res = self.py_osrm.Trip(
            coordinates = two_test_coordinates,
            steps = True,
            annotations = ["all"],
            overview = "false"        
        )
        assert(len(res["trips"]) == 1)
        for trip in res["trips"]:
            assert(trip)
            for l in trip["legs"]:
                assert(len(l["steps"]) > 0
                       and l["annotation"])
            assert("geometry" not in trip)

    def test_trip_nomotorways(self):
        py_osrm = osrm.OSRM(
            algorithm = "MLD",
            storage_config = mld_data_path,
            use_shared_memory = False
        )
        res = py_osrm.Trip(
            coordinates = two_test_coordinates,
            exclude = ["motorway"]      
        )
        assert(len(res["waypoints"]) == 2)
        assert(len(res["trips"]) == 1)
