# microsegments

Where do buses and trams linger? `microsegments` cuts each line into short stretches (30 m by default)
and counts how many times a vehicle position is observed in each one, per hour and per day of the week.
Works with GTFS + any vehicle position feed: GTFS-RT VehiclePositions (lat/lon, CSV or parquet) or
linear-referenced AVL (last stop + distance, like STIB).

Work in progress.
