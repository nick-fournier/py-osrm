from osrm.assignment import (
    AssignmentConfig,
    AssignmentSolver,
    DemandTrip,
    TripStreamAdapter,
)


def _trip(dep_s: float, volume: float = 1.0) -> DemandTrip:
    return DemandTrip(
        origin=(7.41, 43.73),
        destination=(7.42, 43.74),
        volume=volume,
        departure_time_s=dep_s,
    )


def test_trip_stream_adapter_batches_and_slices():
    stream = TripStreamAdapter(
        [_trip(7200), _trip(0), _trip(3500), _trip(3700), _trip(10800)],
        sort_by_departure=True,
    )

    batches = list(stream.iter_batches(2))
    assert [len(b.trips) for b in batches] == [2, 2, 1]
    assert batches[0].trips[0].departure_time_s == 0

    slices = list(stream.iter_time_slices(bin_width_s=3600, max_batch_size=2))
    assert [s.departure_bin for s in slices] == [0, 1, 2, 3]
    assert [len(s.trips) for s in slices] == [2, 1, 1, 1]


def test_snap_preserves_departure_time():
    solver = AssignmentSolver("network.osrm", AssignmentConfig())

    class FakeEngine:
        def Nearest(self, coordinates, number=1):
            lon, lat = coordinates[0]
            return {"code": "Ok", "waypoints": [{"location": [lon + 0.001, lat + 0.001]}]}

    trips = [_trip(123.0, volume=4.0)]
    snapped = solver._snap_trips(FakeEngine(), trips)
    assert snapped[0].departure_time_s == 123.0
    assert snapped[0].volume == 4.0
