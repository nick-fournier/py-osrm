
from .osrm_ext import ( # type: ignore
    OSRM as _OSRM_Base,
    EngineConfig,
    Algorithm,

    Bearing,
    Coordinate,

    RouteParameters as _RouteParameters,
    RouteGeometriesType,
    RouteOverviewType,
    RouteAnnotationsType,
    NearestParameters,
    TableParameters,
    TileParameters,
    TripParameters,
    MatchParameters,
    
    MatchGapsType,
    OutputFormatType,
    SnappingType,
    TableAnnotationsType,
    TableFallbackCoordinateType,
    TripSourceType,
    TripDestinationType,

    Array,
    Object,
    
    # Preprocessing config classes
    ExtractorConfig,
    ContractorConfig,
    PartitionerConfig,
    CustomizationConfig,
)

# Preprocessing functions
from .preprocessing import (
    extract,
    contract,
    partition,
    customize,
)

# Bulk processing functions (optional - only if used)
try:
    from .bulk import bulk_route, bulk_nearest, bulk_match, bulk_table
    _BULK_AVAILABLE = True
except ImportError:
    _BULK_AVAILABLE = False
    bulk_route = None
    bulk_nearest = None
    bulk_match = None
    bulk_table = None

# Import-time validation of profile files
from pathlib import Path as _Path

_package_dir = _Path(__file__).parent
_profiles_dir = _package_dir / 'profiles'
_required_profiles = ['car.lua', 'bicycle.lua', 'foot.lua']
_lib_dir = _profiles_dir / 'lib'

# Check profiles directory exists
if not _profiles_dir.exists() or not _profiles_dir.is_dir():
    raise ImportError(
        f"Profile files missing: {_profiles_dir} not found.\n"
        f"This indicates a broken package installation. "
        f"Try reinstalling: pip install --force-reinstall py-osrm"
    )

# Check main profile files exist
for profile in _required_profiles:
    _profile_path = _profiles_dir / profile
    if not _profile_path.exists():
        raise ImportError(
            f"Profile file missing: {_profile_path} not found.\n"
            f"This indicates a broken package installation. "
            f"Try reinstalling: pip install --force-reinstall py-osrm"
        )

# Check lib directory exists (required by all profiles)
if not _lib_dir.exists() or not _lib_dir.is_dir():
    raise ImportError(
        f"Profile library directory missing: {_lib_dir} not found.\n"
        f"This indicates a broken package installation. "
        f"Try reinstalling: pip install --force-reinstall py-osrm"
    )

# String to enum mappings
_GEOMETRIES_MAP = {
    'polyline': RouteGeometriesType.Polyline,
    'polyline6': RouteGeometriesType.Polyline6,
    'geojson': RouteGeometriesType.GeoJSON,
}

_OVERVIEW_MAP = {
    'simplified': RouteOverviewType.Simplified,
    'full': RouteOverviewType.Full,
    'false': getattr(RouteOverviewType, "False"),  # False is a keyword, use getattr
}

# Annotations are bitflags, can be combined
_ANNOTATIONS_MAP = {
    'none': getattr(RouteAnnotationsType, "None"),  # None is a keyword, use getattr
    'duration': RouteAnnotationsType.Duration,
    'nodes': RouteAnnotationsType.Nodes,
    'distance': RouteAnnotationsType.Distance,
    'weight': RouteAnnotationsType.Weight,
    'datasources': RouteAnnotationsType.Datasources,
    'speed': RouteAnnotationsType.Speed,
    'all': RouteAnnotationsType.All,
}

_GAPS_MAP = {
    'split': MatchGapsType.Split,
    'ignore': MatchGapsType.Ignore,
}

_FORMAT_MAP = {
    'json': OutputFormatType.JSON,
    'flatbuffers': OutputFormatType.FLATBUFFERS,
}

_SNAPPING_MAP = {
    'default': SnappingType.Default,
    'any': SnappingType.Any,
}

_TABLE_ANNOTATIONS_MAP = {
    'none': getattr(TableAnnotationsType, "None"),
    'duration': TableAnnotationsType.Duration,
    'distance': TableAnnotationsType.Distance,
    'all': TableAnnotationsType.All,
}

_TABLE_FALLBACK_COORDINATE_MAP = {
    'input': TableFallbackCoordinateType.Input,
    'snapped': TableFallbackCoordinateType.Snapped,
}

_TRIP_SOURCE_MAP = {
    'any': TripSourceType.Any,
    'first': TripSourceType.First,
}

_TRIP_DESTINATION_MAP = {
    'any': TripDestinationType.Any,
    'last': TripDestinationType.Last,
}

# Helper function to convert strings to enums
def _convert_enum(name, value, enum_map):
    """Convert string value to enum using provided map."""
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower not in enum_map:
            raise ValueError(f"Invalid {name}: '{value}'. Must be one of {list(enum_map.keys())}")
        return enum_map[value_lower]
    return value

def _convert_annotations(value, annotations_map):
    """Convert string or list of strings to annotation enum(s)."""
    if isinstance(value, str):
        return _convert_enum('annotations', value, annotations_map)
    elif isinstance(value, list):
        # Convert list of strings to list of enums (C++ will OR them)
        return [_convert_enum('annotation', item, annotations_map) if isinstance(item, str) else item 
                for item in value]
    return value

# Helper to set parameters with automatic enum conversion
def _set_param(params, key, value):
    """Set parameter attribute with automatic string-to-enum conversion."""
    # Map friendly names to actual attribute names
    if key == 'annotations':
        # Check which attribute name the parameter type uses
        if hasattr(params, 'annotations_type'):
            key = 'annotations_type'
        # else it's 'annotations' which TableParameters uses
    
    # Special handling for annotations
    if key == 'annotations' or key == 'annotations_type':
        converted = _convert_annotations(value, _ANNOTATIONS_MAP if key == 'annotations_type' else _TABLE_ANNOTATIONS_MAP)
        
        # If it's a list, use set_annotations method (C++ will OR them)
        if isinstance(converted, list):
            if hasattr(params, 'set_annotations'):
                params.set_annotations(converted)
            return
        
        # Single enum value - set directly
        if key == 'annotations_type':
            params.annotations_type = converted
        else:
            params.annotations = converted
        return
    
    # Convert known enum types
    if key == 'geometries':
        value = _convert_enum('geometries', value, _GEOMETRIES_MAP)
    elif key == 'overview':
        value = _convert_enum('overview', value, _OVERVIEW_MAP)
    elif key == 'gaps':
        value = _convert_enum('gaps', value, _GAPS_MAP)
    elif key == 'format':
        value = _convert_enum('format', value, _FORMAT_MAP)
    elif key == 'snapping':
        value = _convert_enum('snapping', value, _SNAPPING_MAP)
    elif key == 'fallback_coordinate_type':
        value = _convert_enum('fallback_coordinate_type', value, _TABLE_FALLBACK_COORDINATE_MAP)
    elif key == 'source':
        value = _convert_enum('source', value, _TRIP_SOURCE_MAP)
    elif key == 'destination':
        value = _convert_enum('destination', value, _TRIP_DESTINATION_MAP)
    
    # Set the attribute (skip if not supported)
    if hasattr(params, key):
        setattr(params, key, value)
    else:
        # Some parameters like 'waypoints' are constructor-only
        pass

class RouteParameters(_RouteParameters):
    """RouteParameters wrapper that handles string-to-enum conversion."""
    
    def __init__(self, **kwargs):
        # Create empty parameters
        super().__init__()
        
        # Set all parameters
        for key, value in kwargs.items():
            self.__setattr__(key, value)
    
    def __setattr__(self, name, value):
        # Intercept attribute setting to handle string-to-enum conversion
        if name in ['geometries', 'overview', 'snapping']:
            value = _convert_enum(name, value, {
                'geometries': _GEOMETRIES_MAP,
                'overview': _OVERVIEW_MAP,
                'snapping': _SNAPPING_MAP
            }.get(name, {})) if isinstance(value, str) else value
            super().__setattr__(name, value)
        elif name in ['annotations', 'annotations_type']:
            converted = _convert_annotations(value, _ANNOTATIONS_MAP)
            if isinstance(converted, list):
                # Use set_annotations for lists
                if hasattr(self, 'set_annotations'):
                    self.set_annotations(converted)
            else:
                super().__setattr__('annotations_type', converted)
        else:
            super().__setattr__(name, value)


class OSRM:
    """Python wrapper for OSRM with convenience methods."""
    
    def __init__(self, *args, **kwargs):
        """Initialize OSRM engine - same as C++ OSRM constructor."""
        self._engine = _OSRM_Base(*args, **kwargs)
    
    def Route(self, coordinates=None, **kwargs):
        """
        Compute route between coordinates.
        
        Args:
            coordinates: List of (lon, lat) tuples or RouteParameters object
            steps: Return route steps for each route leg (default False)
            alternatives: Search for alternative routes (default False)
            number_of_alternatives: Number of alternative routes (default 0)
            annotations: Annotation type string or list (e.g., 'speed', ['distance', 'duration'])
            geometries: Geometry format: 'polyline', 'polyline6', or 'geojson'
            overview: Overview detail: 'simplified', 'full', or 'false'
            continue_straight: Force straight at waypoints
            waypoints: Waypoint indices to use
            radiuses: Search radius in meters for each coordinate
            bearings: Bearing constraints
            hints: Hints from previous request
            approaches: Approach constraints
            exclude: Road classes to avoid (e.g., ['motorway'])
            generate_hints: Generate hints for response
            snapping: Snapping mode: 'default' or 'any'
        
        Returns:
            Route result as osrm.Object
            
        Examples:
            # Simple usage
            result = osrm_instance.Route([(7.41, 43.73), (7.42, 43.74)])
            
            # With options
            result = osrm_instance.Route(
                coordinates=[(7.41, 43.73), (7.42, 43.74)],
                steps=True,
                annotations='speed',
                geometries='geojson'
            )
            
            # Traditional usage with RouteParameters object
            params = osrm.RouteParameters()
            params.coordinates = [(7.41, 43.73), (7.42, 43.74)]
            params.steps = True
            result = osrm_instance.Route(params)
        """
        # Support traditional usage with RouteParameters object
        if isinstance(coordinates, RouteParameters):
            return self._engine.Route(coordinates).to_dict()
        
        # Convenience method: build parameters from kwargs
        params = RouteParameters()
        
        # Set coordinates
        if coordinates is not None:
            params.coordinates = coordinates
        
        # Set other parameters with automatic enum conversion
        for key, value in kwargs.items():
            _set_param(params, key, value)
        
        return self._engine.Route(params).to_dict()
    
    def Nearest(self, coordinates=None, **kwargs):
        """
        Find nearest road segment to coordinates.
        
        Args:
            coordinates: List of (lon, lat) tuples or NearestParameters object
            number: Number of nearest segments to return (default 1)
            radiuses: Search radius in meters for each coordinate
            bearings: Bearing constraints
            hints: Hints from previous request
            approaches: Approach constraints
            exclude: Road classes to avoid (e.g., ['motorway'])
            generate_hints: Generate hints for response
            snapping: Snapping mode: 'default' or 'any'
        
        Returns:
            Nearest result as osrm.Object
        """        
        # Support traditional usage with NearestParameters object
        if isinstance(coordinates, NearestParameters):
            return self._engine.Nearest(coordinates).to_dict()
        
        # Convenience method: build parameters from kwargs
        params = NearestParameters()
        
        # Set coordinates
        if coordinates is not None:
            params.coordinates = coordinates
        
        # Set other parameters with automatic enum conversion
        for key, value in kwargs.items():
            _set_param(params, key, value)
        
        return self._engine.Nearest(params).to_dict()
    
    def Match(self, coordinates=None, **kwargs):
        """
        Match GPS trace to road network.
        
        Args:
            coordinates: List of (lon, lat) tuples or MatchParameters object
            timestamps: Timestamps for each coordinate
            steps: Return route steps for each route leg
            geometries: Geometry format: 'polyline', 'polyline6', or 'geojson'
            overview: Overview detail: 'simplified', 'full', or 'false'
            annotations: Annotation type string or list
            radiuses: Search radius in meters for each coordinate
            bearings: Bearing constraints
            hints: Hints from previous request
            approaches: Approach constraints
            exclude: Road classes to avoid
            generate_hints: Generate hints for response
            snapping: Snapping mode
            gaps: Gap handling: 'split' or 'ignore'
            tidy: Clean up trace
            waypoints: Waypoint indices to use
        
        Returns:
            Match result as osrm.Object
        """        
        # Support traditional usage with MatchParameters object
        if isinstance(coordinates, MatchParameters):
            return self._engine.Match(coordinates).to_dict()
        
        # Convenience method: build parameters from kwargs
        params = MatchParameters()
        
        # Set coordinates
        if coordinates is not None:
            params.coordinates = coordinates
        
        # Set other parameters with automatic enum conversion
        for key, value in kwargs.items():
            _set_param(params, key, value)
        
        return self._engine.Match(params).to_dict()
    
    def Table(self, coordinates=None, **kwargs):
        """
        Compute distance/duration table between coordinates.
        
        Args:
            coordinates: List of (lon, lat) tuples or TableParameters object
            sources: Indices of source coordinates
            destinations: Indices of destination coordinates
            annotations: 'duration', 'distance', or both
            fallback_speed: Fallback speed in km/h
            fallback_coordinate: Coordinate fallback: 'input' or 'snapped'
            scale_factor: Scale factor for durations
            radiuses: Search radius in meters for each coordinate
            bearings: Bearing constraints
            hints: Hints from previous request
            approaches: Approach constraints
            exclude: Road classes to avoid
            generate_hints: Generate hints for response
            snapping: Snapping mode
        
        Returns:
            Table result as osrm.Object
        """
        # Support traditional usage with TableParameters object
        if isinstance(coordinates, TableParameters):
            return self._engine.Table(coordinates).to_dict()
        
        # Convenience method: build parameters from kwargs
        params = TableParameters()
        
        # Set coordinates
        if coordinates is not None:
            params.coordinates = coordinates
        
        # Set other parameters with automatic enum conversion
        for key, value in kwargs.items():
            _set_param(params, key, value)
        
        return self._engine.Table(params).to_dict()
    
    def Trip(self, coordinates=None, **kwargs):
        """
        Solve traveling salesman problem.
        
        Args:
            coordinates: List of (lon, lat) tuples or TripParameters object
            roundtrip: Return to start point (default True)
            source: Source constraint: 'any' or 'first'
            destination: Destination constraint: 'any' or 'last'
            steps: Return route steps for each route leg
            geometries: Geometry format: 'polyline', 'polyline6', or 'geojson'
            overview: Overview detail: 'simplified', 'full', or 'false'
            annotations: Annotation type string or list
            radiuses: Search radius in meters for each coordinate
            bearings: Bearing constraints
            hints: Hints from previous request
            approaches: Approach constraints
            exclude: Road classes to avoid
            generate_hints: Generate hints for response
            snapping: Snapping mode
        
        Returns:
            Trip result as osrm.Object
        """
        # Support traditional usage with TripParameters object
        if isinstance(coordinates, TripParameters):
            return self._engine.Trip(coordinates).to_dict()
        
        # Convenience method: build parameters from kwargs
        params = TripParameters()
        
        # Set coordinates
        if coordinates is not None:
            params.coordinates = coordinates
        
        # Set other parameters with automatic enum conversion
        for key, value in kwargs.items():
            _set_param(params, key, value)
        
        return self._engine.Trip(params).to_dict()
    
    def Tile(self, params):
        """Get vector tile data."""
        result = self._engine.Tile(params)
        # Tile returns bytes (vector tiles), not JSON
        if isinstance(result, bytes):
            return result
        return result.to_dict()
