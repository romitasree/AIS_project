import ee

TEXAS_CITIES = [
    {"lat": 33.314479, "lon": -96.776550},  # Celina
    {"lat": 26.667417, "lon": -99.159667}   # Falcon Lake
]

dates = [
    {"start": "2016-05-01", "end": "2016-08-01"},
    {"start": "2025-05-01", "end": "2025-08-01"}
]

def get_dw_data(lon, lat, start_date, end_date, buffer_meters=1500):
    try:
        ee.Initialize(project='terratrace-488618')
    except Exception:
        pass  # already initialized

    point = ee.Geometry.Point([lon, lat]).buffer(buffer_meters).bounds()

    collection = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
                .filterBounds(point)
                .filterDate(start_date, end_date))

    count = collection.size().getInfo()
    if count == 0:
        print("No images found")
        return None

    return collection

def get_city_coords():
    return TEXAS_CITIES

if __name__ == "__main__":
    ee.Authenticate()
    ee.Initialize(project='terratrace-488618')
    print(get_dw_data(lon=-96.776550, lat=33.314479,
                      start_date="2016-05-01", end_date="2016-08-01"))