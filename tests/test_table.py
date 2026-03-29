import pytest
import osrm
import constants

ch_data_path = constants.ch_data_path
mld_data_path = constants.mld_data_path
three_test_coordinates = constants.three_test_coordinates
two_test_coordinates = constants.two_test_coordinates

class TestTable:
    @classmethod
    def setup_class(cls):
        cls.py_osrm = osrm.OSRM(
            storage_config = ch_data_path, 
            use_shared_memory = False
        )

    def test_table_annotations(self):
        res = self.py_osrm.Table(
            coordinates = [three_test_coordinates[0], three_test_coordinates[1]],
            annotations = ["distance"]
        )
        assert(res["distances"])
        assert("durations" not in res)

        res = self.py_osrm.Table(
            coordinates = [three_test_coordinates[0], three_test_coordinates[1]],
            annotations = ["duration"]
        )
        assert(res["durations"])
        assert("distances" not in res)

        res = self.py_osrm.Table(
            coordinates = [three_test_coordinates[0], three_test_coordinates[1]],
            annotations = ["duration", "distance"]
        )
        assert(res["durations"])
        assert(res["distances"])

        res = self.py_osrm.Table(
            coordinates = [three_test_coordinates[0], three_test_coordinates[1]]
        )
        assert(res["durations"])
        assert("distances" not in res)

    def test_table_snapping(self):
        res = self.py_osrm.Table(
            coordinates = [three_test_coordinates[0], three_test_coordinates[1]],
            snapping = "any"
        )
        assert(res["durations"])

    # def test_table_annotation(self):
    #     tables = ["distance", "duration"]

    #     for annotation in tables:
    #         table_params = osrm.TableParameters(
    #             coordinates = [three_test_coordinates[0], three_test_coordinates[1]],
    #             annotations = [annotation]
    #         )
    #         res = self.py_osrm.Table(table_params)
    #      
    #         rows = res[annotation]
    #         for i, col in enumerate(res[annotation]):
    #             assert(len(rows) == len(col))
    #             for j, row in enumerate(col):
    #                 if(i == j):
    #                     # check that diagonal is zero
    #                     assert(col[j] == 0)
    #                 else:
    #                     # everything else is non-zero and finite
    #                     assert(not col[j] == 0)
    #                     assert(math.isfinite(col[j]))
    #         assert(len(table_params.coordinates) == len(rows))

    #     for annotation in tables:
    #         table_params = osrm.TableParameters(
    #             coordinates = [three_test_coordinates[0], three_test_coordinates[1]],
    #             sources = [0],
    #             destinations = [0,1],
    #             annotations = [annotation]
    #         )
    #         res = self.py_osrm.Table(table_params)
    #      
    #         rows = res[annotation]
    #         for i, col in enumerate(res[annotation]):
    #             assert(len(rows) == len(col))
    #             for j, row in enumerate(col):
    #                 if(i == j):
    #                     # check that diagonal is zero
    #                     assert(col[j] == 0)
    #                 else:
    #                     # everything else is non-zero and finite
    #                     assert(not col[j] == 0)
    #                     assert(math.isfinite(col[j]))
    #         assert(len(table_params.sources) == len(rows))

    def test_table_withoutwaypoints(self):
        res = self.py_osrm.Table(
            coordinates = two_test_coordinates,
            annotations = ["duration"],
            skip_waypoints = True
        )
        assert("sources" not in res)
        assert("destinations" not in res)

    def test_table_fallbackspeeds(self):
        res = self.py_osrm.Table(
            coordinates = two_test_coordinates,
            annotations = ["duration"],
            fallback_speed = 1,
            fallback_coordinate_type = "input"
        )
        assert(len(res["destinations"]) == 2)
        assert(len(res["fallback_speed_cells"]) == 0)

    def test_table_invalidfallbackspeeds(self):
        py_osrm = osrm.OSRM(
            storage_config = mld_data_path,
            algorithm = "MLD", 
            use_shared_memory = False
        )
        with pytest.raises(RuntimeError) as ex:
            res = py_osrm.Table(
                coordinates = two_test_coordinates,
                annotations = ["duration"],
                fallback_speed = -1
            )
        assert(str(ex.value) == "Invalid Table Parameters")

        res = py_osrm.Table(
            coordinates = two_test_coordinates,
            annotations = ["duration"],
            fallback_speed = 10
        )

    def test_table_invalidscalefactor(self):
        py_osrm = osrm.OSRM(
            storage_config = mld_data_path,
            algorithm = "MLD", 
            use_shared_memory = False
        )
        with pytest.raises(RuntimeError) as ex:
            res = py_osrm.Table(
                coordinates = two_test_coordinates,
                annotations = ["duration"],
                scale_factor = -1
            )
        assert(str(ex.value) == "Invalid Table Parameters")

        res = py_osrm.Table(
            coordinates = two_test_coordinates,
            annotations = ["duration"],
            scale_factor = 1
        )
