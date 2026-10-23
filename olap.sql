-- Roll-up: City -> Region
SELECT l.region, SUM(f.cost_inr) AS total_cost
FROM fact_service f
JOIN dim_location l ON f.location_key = l.location_key
GROUP BY l.region ORDER BY total_cost DESC;

-- Drill-down: Year -> Quarter
SELECT t.year, t.quarter, SUM(f.cost_inr) AS total_cost
FROM fact_service f JOIN dim_time t ON f.time_key = t.time_key
GROUP BY t.year, t.quarter ORDER BY t.year, t.quarter;

-- Slice: Year = 2023
SELECT v.vehicle_type, SUM(f.cost_inr) AS total_cost
FROM fact_service f
JOIN dim_time t ON f.time_key = t.time_key
JOIN dim_vehicle v ON f.vehicle_key = v.vehicle_key
WHERE t.year = 2023
GROUP BY v.vehicle_type ORDER BY total_cost DESC;

-- Dice: Region and vehicle type sub-cube
SELECT l.region, v.vehicle_type, SUM(f.cost_inr) AS total_cost
FROM fact_service f
JOIN dim_location l ON f.location_key = l.location_key
JOIN dim_vehicle v ON f.vehicle_key = v.vehicle_key
WHERE l.region IN ('South', 'West')
  AND v.vehicle_type IN ('Truck', 'Bus')
GROUP BY l.region, v.vehicle_type;

-- Pivot: regions as columns
SELECT v.vehicle_type,
  SUM(CASE WHEN l.region = 'South' THEN f.cost_inr END) AS south,
  SUM(CASE WHEN l.region = 'West' THEN f.cost_inr END) AS west,
  SUM(CASE WHEN l.region = 'North' THEN f.cost_inr END) AS north,
  SUM(CASE WHEN l.region = 'East' THEN f.cost_inr END) AS east
FROM fact_service f
JOIN dim_location l ON f.location_key = l.location_key
JOIN dim_vehicle v ON f.vehicle_key = v.vehicle_key
GROUP BY v.vehicle_type;
