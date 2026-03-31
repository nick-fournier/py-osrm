"""
py-osrm: Python bindings for OSRM routing engine

This package provides native Python bindings to the OSRM (Open Source Routing Machine)
backend, enabling high-performance routing directly from Python without HTTP overhead.

Quick Start:
    >>> import osrm
    >>> 
    >>> # Load preprocessed OSRM data
    >>> engine = osrm.OSRM("path/to/map.osrm")
    >>> 
    >>> # Calculate route
    >>> result = engine.Route([(7.41, 43.73), (7.42, 43.74)])
    >>> print(result["routes"][0]["distance"])

Main Components:
    OSRM: Native routing engine (loads .osrm files)
    OSRM_HTTP: HTTP client for remote OSRM servers
    
    Bulk Processing:
        bulk_route: Parallel route calculations
        bulk_nearest: Parallel nearest road queries
        bulk_match: Parallel GPS trace matching
    
    Preprocessing:
        extract: Extract road network from OSM data
        contract: Contract graph for CH algorithm
        partition: Partition graph for MLD algorithm
        customize: Customize graph for specific profile

For detailed documentation, visit: https://github.com/gis-ops/py-osrm
"""

# ruff: noqa: F401

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

# Bulk processing functions
from .bulk import bulk_route, bulk_nearest, bulk_match

# HTTP client
from .http_client import OSRM_HTTP

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

# String to enum mappings — defined in _params.py to avoid circular imports
from ._params import (
    GEOMETRIES_MAP as _GEOMETRIES_MAP,
    OVERVIEW_MAP as _OVERVIEW_MAP,
    ANNOTATIONS_MAP as _ANNOTATIONS_MAP,
    GAPS_MAP as _GAPS_MAP,
    FORMAT_MAP as _FORMAT_MAP,
    SNAPPING_MAP as _SNAPPING_MAP,
    TABLE_ANNOTATIONS_MAP as _TABLE_ANNOTATIONS_MAP,
    TABLE_FALLBACK_COORDINATE_MAP as _TABLE_FALLBACK_COORDINATE_MAP,
    TRIP_SOURCE_MAP as _TRIP_SOURCE_MAP,
    TRIP_DESTINATION_MAP as _TRIP_DESTINATION_MAP,
    convert_enum as _convert_enum,
    convert_annotations as _convert_annotations,
    set_param as _set_param,
    RouteParameters,
)


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
    
    def BatchRoute(self, params_list):
        """Route multiple OD pairs in parallel using native C++ TBB threading.

        Args:
            params_list: List of RouteParameters objects.

        Returns:
            List of route result dicts (None for failed routes).
        """
        raw = self._engine.BatchRoute(params_list)
        return [r.to_dict() if r is not None else None for r in raw]
    
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


# Public API
__all__ = [
    # Main classes
    "OSRM",
    "OSRM_HTTP",
    "EngineConfig",
    "Algorithm",
    
    # Data types
    "Bearing",
    "Coordinate",
    "Array",
    "Object",
    
    # Parameter classes
    "RouteParameters",
    "NearestParameters",
    "TableParameters",
    "TileParameters",
    "TripParameters",
    "MatchParameters",
    
    # Enums
    "RouteGeometriesType",
    "RouteOverviewType",
    "RouteAnnotationsType",
    "MatchGapsType",
    "OutputFormatType",
    "SnappingType",
    "TableAnnotationsType",
    "TableFallbackCoordinateType",
    "TripSourceType",
    "TripDestinationType",
    
    # Preprocessing functions
    "extract",
    "contract",
    "partition",
    "customize",
    
    # Preprocessing config classes
    "ExtractorConfig",
    "ContractorConfig",
    "PartitionerConfig",
    "CustomizationConfig",
    
    # Bulk processing functions
    "bulk_route",
    "bulk_nearest",
    "bulk_match",
]
