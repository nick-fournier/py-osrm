"""Shared parameter helpers — imported by both __init__.py and bulk.py."""

from .osrm_ext import (  # type: ignore
    RouteParameters as _RouteParameters,
    RouteGeometriesType,
    RouteOverviewType,
    RouteAnnotationsType,
    MatchGapsType,
    OutputFormatType,
    SnappingType,
    TableAnnotationsType,
    TableFallbackCoordinateType,
    TripSourceType,
    TripDestinationType,
)

# ── String-to-enum maps ──────────────────────────────────────────────

GEOMETRIES_MAP = {
    'polyline': RouteGeometriesType.Polyline,
    'polyline6': RouteGeometriesType.Polyline6,
    'geojson': RouteGeometriesType.GeoJSON,
}

OVERVIEW_MAP = {
    'simplified': RouteOverviewType.Simplified,
    'full': RouteOverviewType.Full,
    'false': getattr(RouteOverviewType, "False"),
}

ANNOTATIONS_MAP = {
    'none': getattr(RouteAnnotationsType, "None"),
    'duration': RouteAnnotationsType.Duration,
    'nodes': RouteAnnotationsType.Nodes,
    'distance': RouteAnnotationsType.Distance,
    'weight': RouteAnnotationsType.Weight,
    'datasources': RouteAnnotationsType.Datasources,
    'speed': RouteAnnotationsType.Speed,
    'all': RouteAnnotationsType.All,
}

GAPS_MAP = {
    'split': MatchGapsType.Split,
    'ignore': MatchGapsType.Ignore,
}

FORMAT_MAP = {
    'json': OutputFormatType.JSON,
    'flatbuffers': OutputFormatType.FLATBUFFERS,
}

SNAPPING_MAP = {
    'default': SnappingType.Default,
    'any': SnappingType.Any,
}

TABLE_ANNOTATIONS_MAP = {
    'none': getattr(TableAnnotationsType, "None"),
    'duration': TableAnnotationsType.Duration,
    'distance': TableAnnotationsType.Distance,
    'all': TableAnnotationsType.All,
}

TABLE_FALLBACK_COORDINATE_MAP = {
    'input': TableFallbackCoordinateType.Input,
    'snapped': TableFallbackCoordinateType.Snapped,
}

TRIP_SOURCE_MAP = {
    'any': TripSourceType.Any,
    'first': TripSourceType.First,
}

TRIP_DESTINATION_MAP = {
    'any': TripDestinationType.Any,
    'last': TripDestinationType.Last,
}

# ── Conversion helpers ────────────────────────────────────────────────

def convert_enum(name, value, enum_map):
    """Convert string value to enum using provided map."""
    if isinstance(value, str):
        value_lower = value.lower()
        if value_lower not in enum_map:
            raise ValueError(f"Invalid {name}: '{value}'. Must be one of {list(enum_map.keys())}")
        return enum_map[value_lower]
    return value


def convert_annotations(value, annotations_map):
    """Convert string or list of strings to annotation enum(s)."""
    if isinstance(value, str):
        return convert_enum('annotations', value, annotations_map)
    elif isinstance(value, list):
        return [convert_enum('annotation', item, annotations_map) if isinstance(item, str) else item
                for item in value]
    return value


def set_param(params, key, value):
    """Set parameter attribute with automatic string-to-enum conversion."""
    if key == 'annotations':
        if hasattr(params, 'annotations_type'):
            key = 'annotations_type'

    if key in ('annotations', 'annotations_type'):
        converted = convert_annotations(
            value,
            ANNOTATIONS_MAP if key == 'annotations_type' else TABLE_ANNOTATIONS_MAP,
        )
        if isinstance(converted, list):
            if hasattr(params, 'set_annotations'):
                params.set_annotations(converted)
            return
        if key == 'annotations_type':
            params.annotations_type = converted
        else:
            params.annotations = converted
        return

    if key == 'geometries':
        value = convert_enum('geometries', value, GEOMETRIES_MAP)
    elif key == 'overview':
        value = convert_enum('overview', value, OVERVIEW_MAP)
    elif key == 'gaps':
        value = convert_enum('gaps', value, GAPS_MAP)
    elif key == 'format':
        value = convert_enum('format', value, FORMAT_MAP)
    elif key == 'snapping':
        value = convert_enum('snapping', value, SNAPPING_MAP)
    elif key == 'fallback_coordinate_type':
        value = convert_enum('fallback_coordinate_type', value, TABLE_FALLBACK_COORDINATE_MAP)
    elif key == 'source':
        value = convert_enum('source', value, TRIP_SOURCE_MAP)
    elif key == 'destination':
        value = convert_enum('destination', value, TRIP_DESTINATION_MAP)

    if hasattr(params, key):
        setattr(params, key, value)


# ── RouteParameters wrapper ──────────────────────────────────────────

class RouteParameters(_RouteParameters):
    """RouteParameters wrapper that handles string-to-enum conversion."""

    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            self.__setattr__(key, value)

    def __setattr__(self, name, value):
        if name in ('geometries', 'overview', 'snapping'):
            value = convert_enum(name, value, {
                'geometries': GEOMETRIES_MAP,
                'overview': OVERVIEW_MAP,
                'snapping': SNAPPING_MAP,
            }.get(name, {})) if isinstance(value, str) else value
            super().__setattr__(name, value)
        elif name in ('annotations', 'annotations_type'):
            converted = convert_annotations(value, ANNOTATIONS_MAP)
            if isinstance(converted, list):
                if hasattr(self, 'set_annotations'):
                    self.set_annotations(converted)
            else:
                super().__setattr__('annotations_type', converted)
        else:
            super().__setattr__(name, value)
