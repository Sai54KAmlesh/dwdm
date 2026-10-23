DROP TABLE IF EXISTS bridge_service_part; DROP TABLE IF EXISTS fact_service;
DROP TABLE IF EXISTS dim_time; DROP TABLE IF EXISTS dim_vehicle; DROP TABLE IF EXISTS dim_location;
DROP TABLE IF EXISTS dim_service_type; DROP TABLE IF EXISTS dim_part;

CREATE TABLE dim_time (time_key INT PRIMARY KEY, full_date DATE, day INT, month INT,
  month_name TEXT, quarter INT, year INT);
CREATE TABLE dim_vehicle (vehicle_key INT PRIMARY KEY, vehicle_id TEXT,
  vehicle_type TEXT, make TEXT, purchase_year INT);
CREATE TABLE dim_location (location_key INT PRIMARY KEY, city TEXT, state TEXT, region TEXT);
CREATE TABLE dim_service_type (service_type_key INT PRIMARY KEY, service_type TEXT);
CREATE TABLE dim_part (part_key INT PRIMARY KEY, part_name TEXT);

CREATE TABLE fact_service (
  service_key INT PRIMARY KEY,
  time_key INT REFERENCES dim_time(time_key),
  vehicle_key INT REFERENCES dim_vehicle(vehicle_key),
  location_key INT REFERENCES dim_location(location_key),
  service_type_key INT REFERENCES dim_service_type(service_type_key),
  odometer_km INT, avg_daily_km REAL, idle_hours REAL, harsh_events INT, engine_temp_c REAL,
  days_since_last_service INT, days_to_next_service INT, parts_count INT,
  cost_inr REAL, downtime_hours REAL, maintenance_need TEXT
);
CREATE TABLE bridge_service_part (
  service_key INT REFERENCES fact_service(service_key),
  part_key INT REFERENCES dim_part(part_key)
);
