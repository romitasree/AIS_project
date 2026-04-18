from modules.gee_fetch import get_dw_data
import ee

landcover_classes = {
  "0": {
    "name": "water",
    "color": "#419bdf",
    "impact": "Oceans, rivers, lakes; absorbs light; may resemble shadows or dark surfaces"
  },
  "1": {
    "name": "trees",
    "color": "#397d49",
    "impact": "Dense woody vegetation; forests and woodlands; strong canopy structure; may resemble tall shrubs"
  },
  "2": {
    "name": "grass",
    "color": "#88b053",
    "impact": "Low herbaceous vegetation; croplands and lawns; seasonal variation; may resemble sparse shrubs"
  },
  "3": {
    "name": "flooded_vegetation",
    "color": "#7a87c6",
    "impact": "Vegetation mixed with standing water; wetlands and mangroves; variable reflectance; may resemble water or marsh"
  },
  "4": {
    "name": "crops",
    "color": "#e49635",
    "impact": "Cultivated agricultural land; seasonal growth cycles; may resemble grass or bare soil"
  },
  "5": {
    "name": "shrub_and_scrub",
    "color": "#dfc35a",
    "impact": "Low woody vegetation; semi-arid ecosystems; may resemble grass or sparse trees"
  },
  "6": {
    "name": "built",
    "color": "#c4281b",
    "impact": "Human-made surfaces; urban development; may resemble bare soil or dry fields"
  },
  "7": {
    "name": "bare",
    "color": "#a59b8f",
    "impact": "Exposed soil or rock; minimal vegetation; may resemble built areas"
  },
  "8": {
    "name": "snow_and_ice",
    "color": "#b39fe1",
    "impact": "Frozen surfaces; high reflectance; may resemble clouds or bright rooftops"
  }
}

def get_landcover_analysis(lon, lat, start_date, end_date, buffer_meters=1500):
    collection = get_dw_data(lon, lat, start_date, end_date, buffer_meters)  # ← fixed: 4 spaces now
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(buffer_meters).bounds()
    classification = collection.select('label')
    composite = classification.reduce(ee.Reducer.mode())
    stats = composite.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=region,
        scale=10,
        maxPixels=1e9
    )
    histogram = stats.getInfo()
    if len(histogram) == 1:
        histogram = list(histogram.values())[0]
    total_pixels = sum(histogram.values())
    percentages = {}
    confidence_scores = {}
    for key, value in histogram.items():
        class_info = landcover_classes.get(str(key))
        if class_info:
            class_name = class_info["name"]
        else:
            class_name = str(key)
        percentage = (value / total_pixels) * 100
        percentages[class_name] = round(percentage, 2)
        confidence_scores[class_name] = round(percentage / 100, 2)
    return {
        'percentages': percentages,
        'confidence': confidence_scores
    }
