"""Traffic assignment module for py-osrm.

Provides network state tracking, volume-delay functions, density smoothing,
and segment speed CSV generation for iterative traffic assignment using OSRM.
"""

from osrm.assignment.vdf import BiParabolicVDF
from osrm.assignment.network_state import NetworkState
from osrm.assignment.density_smoothing import DensitySmoothing, DensitySmoothingConfig
from osrm.assignment.fractional_loading import FractionalLoader
from osrm.assignment.segment_speed_writer import SegmentSpeedWriter
from osrm.assignment.od_matrix import ODMatrixAdapter, DemandTrip
from osrm.assignment.trip_stream import TripBatch, TripStreamAdapter
from osrm.assignment.assignment_loop import (
    AssignmentLoop, AssignmentConfig, AssignmentResult, StopReason,
)
from osrm.assignment.solvers import (
    HillClimberBatchResult,
    HillClimberResult,
    MSAIterationResult,
    TrafficAssignmentSolver,
    MatrixAssignmentSolver,
    MatrixFreeHillClimber,
    ODLedger,
    ODLedgerEntry,
    RouteAssignment,
)
from osrm.assignment import plots

__all__ = [
    "BiParabolicVDF",
    "NetworkState",
    "DensitySmoothing",
    "DensitySmoothingConfig",
    "FractionalLoader",
    "SegmentSpeedWriter",
    "ODMatrixAdapter",
    "DemandTrip",
    "TripBatch",
    "TripStreamAdapter",
    "AssignmentLoop",
    "AssignmentConfig",
    "AssignmentResult",
    "StopReason",
    "HillClimberBatchResult",
    "HillClimberResult",
    "MSAIterationResult",
    "TrafficAssignmentSolver",
    "MatrixAssignmentSolver",
    "MatrixFreeHillClimber",
    "ODLedger",
    "ODLedgerEntry",
    "RouteAssignment",
    "plots",
]
