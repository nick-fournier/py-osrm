"""Smoke tests for py-osrm - basic functionality without requiring OSRM data files."""
import pytest
import osrm


class TestImport:
    """Test that the module imports and basic classes are available."""
    
    def test_module_import(self):
        """Test that osrm module can be imported."""
        assert osrm is not None
        
    def test_osrm_class_exists(self):
        """Test that OSRM class is available."""
        assert hasattr(osrm, 'OSRM')
        
    def test_parameter_classes_exist(self):
        """Test that all parameter classes are available."""
        assert hasattr(osrm, 'RouteParameters')
        assert hasattr(osrm, 'TableParameters')
        assert hasattr(osrm, 'MatchParameters')
        assert hasattr(osrm, 'TripParameters')
        assert hasattr(osrm, 'NearestParameters')
        assert hasattr(osrm, 'TileParameters')
        
    def test_config_classes_exist(self):
        """Test that configuration classes are available."""
        assert hasattr(osrm, 'EngineConfig')
        
    def test_enum_types_exist(self):
        """Test that enum types are available."""
        assert hasattr(osrm, 'Algorithm')
        
    def test_json_types_exist(self):
        """Test that JSON container types are available."""
        assert hasattr(osrm, 'Object')
        assert hasattr(osrm, 'Array')


class TestEnums:
    """Test enum values are properly defined."""
    
    def test_algorithm_enum_values(self):
        """Test Algorithm enum has expected values."""
        assert hasattr(osrm.Algorithm, 'CH')
        assert hasattr(osrm.Algorithm, 'MLD')
        # CoreCH was removed in OSRM 6.0.0
        assert not hasattr(osrm.Algorithm, 'CoreCH')
        

class TestParameterInstantiation:
    """Test that parameter objects can be instantiated."""
    
    def test_route_parameters_creation(self):
        """Test RouteParameters can be created with default constructor."""
        params = osrm.RouteParameters()
        assert params is not None
        
    def test_table_parameters_creation(self):
        """Test TableParameters can be created with default constructor."""
        params = osrm.TableParameters()
        assert params is not None
        
    def test_match_parameters_creation(self):
        """Test MatchParameters can be created with default constructor."""
        params = osrm.MatchParameters()
        assert params is not None
        
    def test_trip_parameters_creation(self):
        """Test TripParameters can be created with default constructor."""
        params = osrm.TripParameters()
        assert params is not None
        
    def test_nearest_parameters_creation(self):
        """Test NearestParameters can be created with default constructor."""
        params = osrm.NearestParameters()
        assert params is not None
        
    def test_coordinate_assignment(self):
        """Test that coordinates can be assigned to parameters."""
        params = osrm.RouteParameters()
        coord1 = osrm.Coordinate((7.419758, 43.731142))
        coord2 = osrm.Coordinate((7.419505, 43.736825))
        params.coordinates = [coord1, coord2]
        assert len(params.coordinates) == 2
        
    def test_tile_parameters_creation(self):
        """Test TileParameters can be created."""
        params = osrm.TileParameters([17059, 11948, 15])
        assert params is not None


class TestEngineConfig:
    """Test EngineConfig functionality."""
    
    def test_engine_config_creation(self):
        """Test EngineConfig can be instantiated."""
        config = osrm.EngineConfig()
        assert config is not None
        
    def test_algorithm_assignment(self):
        """Test Algorithm can be assigned to config."""
        config = osrm.EngineConfig()
        config.algorithm = osrm.Algorithm.MLD
        # No assertion needed - just shouldn't raise an exception
        
    def test_use_shared_memory(self):
        """Test use_shared_memory can be set."""
        config = osrm.EngineConfig()
        config.use_shared_memory = False
        # No assertion needed - just shouldn't raise an exception


class TestJSONContainers:
    """Test JSON container types."""
    
    def test_object_creation(self):
        """Test Object can be created."""
        obj = osrm.Object()
        assert obj is not None
        assert len(obj) == 0
        
    def test_array_creation(self):
        """Test Array can be created."""
        arr = osrm.Array()
        assert arr is not None
        assert len(arr) == 0
