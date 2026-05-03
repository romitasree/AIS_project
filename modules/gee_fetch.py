#gee_fetch.py
import ee
ee.Authenticate()
ee.Initialize(project="terratrace-488618")
TEXAS_CITIES = [
    {"label": "Celina, TX", "lat": 33.314479, "lon": -96.776550},
    {"label": "Falcon Lake, TX", "lat": 26.667417, "lon": -99.159667},
    {"label": "Dallas, TX", "lat": 32.7767, "lon": -96.7970},
    {"label": "Austin, TX", "lat": 30.2672, "lon": -97.7431},
    {"label": "Houston, TX", "lat": 29.7604, "lon": -95.3698},
    {"label": "San Antonio, TX", "lat": 29.4241, "lon": -98.4936},
    {"label": "Fort Worth, TX", "lat": 32.7555, "lon": -97.3308},
    {"label": "Frisco, TX", "lat": 33.1507, "lon": -96.8236},
    {"label": "McKinney, TX", "lat": 33.1976, "lon": -96.6153},
]
DATES = [
    {"start": "2016-05-01", "end": "2016-08-01"},
    {"start": "2017-05-01", "end": "2017-08-01"},
    {"start": "2018-05-01", "end": "2018-08-01"},
    {"start": "2019-05-01", "end": "2019-08-01"},
    {"start": "2020-05-01", "end": "2020-08-01"},
    {"start": "2021-05-01", "end": "2021-08-01"},
    {"start": "2022-05-01", "end": "2022-08-01"},
    {"start": "2023-05-01", "end": "2023-08-01"},
    {"start": "2024-05-01", "end": "2024-08-01"},
    {"start": "2025-05-01", "end": "2025-08-01"},
]

def get_dw_data(locations: list[dict], dates: list[dict], buffer_meters: int = 1500) -> list[dict]:
    results = []
    for coords in locations:
        label = coords.get("label", f"{coords['lat']},{coords['lon']}")
        lat = coords["lat"]
        lon = coords["lon"]
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(buffer_meters).bounds()
        for date in dates:
            start = date["start"]
            end = date["end"]
            try:
                collection = (
                    ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                    .filterBounds(region)
                    .filterDate(start, end)
                )
                count = collection.size().getInfo()
                if count == 0:
                    print(f"[gee_fetch] No images for ({lat}, {lon}) {start}→{end}")
                    continue
                # Mode composite — most common land-cover class per pixel
                composite = collection.select("label").mode().clip(region)
                # Count pixels per class using a histogram (light GEE call)
                histogram = composite.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=region,
                    scale=10,           # Dynamic World native resolution
                    maxPixels=1e6,
                ).getInfo()
                label_hist = histogram.get("label", {})
                if not label_hist:
                    print(f"[gee_fetch] Empty histogram for ({lat}, {lon}) {start}→{end}")
                    continue
                # Convert histogram → synthetic "bands" format _class_percentages expects
                pixels = []
                for class_idx, pixel_count in label_hist.items():
                    pixels.extend([int(class_idx)] * int(pixel_count))
                results.append({
                    "label":      label,
                    "lat":        lat,
                    "lon":        lon,
                    "start":      start,
                    "end":        end,
                    "count":      count,
                    "collection": collection,
                    "region":     region,
                    "bands": [
                        {
                            "id": "label",
                            "data": pixels,
                        }
                    ],
                })
            except Exception as exc:
                print(f"[gee_fetch] Error ({lat}, {lon}) {start}→{end}: {exc}")
    return results
def get_city_coords() -> list[dict]:
    return TEXAS_CITIES
if __name__ == "__main__":
    results = get_dw_data(locations=TEXAS_CITIES, dates=DATES)
    print(results)
