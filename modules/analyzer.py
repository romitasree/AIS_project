from gee_fetch import get_dw_data, get_city_coords, DATES
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


def get_landcover_analysis(entry: dict) -> dict | None:
    if entry["count"] == 0 or entry["collection"] is None:
        print(f"[landcover] Skipping ({entry['lat']}, {entry['lon']}) {entry['start']}→{entry['end']} — no imagery")
        return None
 
    collection: ee.ImageCollection = entry["collection"]
    region: ee.Geometry = entry["region"]
 
    try:
        composite = collection.select("label").reduce(ee.Reducer.mode()).clip(region)
 
        stats = composite.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=region,
            scale=10,
            maxPixels=1e9,
        )
 
        histogram: dict = stats.getInfo()
 
        if len(histogram) == 1:
            histogram = list(histogram.values())[0]
 
        if not histogram:
            print(f"[landcover] Empty histogram for ({entry['lat']}, {entry['lon']})")
            return None
 
        total_pixels = sum(histogram.values())
        percentages: dict[str, float] = {}
        confidence: dict[str, float] = {}
 
        for key, count in histogram.items():
            class_info = landcover_classes.get(str(key))
            class_name = class_info["name"] if class_info else f"class_{key}"
            pct = (count / total_pixels) * 100
            percentages[class_name] = round(pct, 2)
            confidence[class_name] = round(pct / 100, 4)
 
        return {
            "lat": entry["lat"],
            "lon": entry["lon"],
            "start": entry["start"],
            "end": entry["end"],
            "percentages": percentages,
            "confidence": confidence,
        }
 
    except Exception as exc:
        print(f"[landcover] Analysis failed ({entry['lat']}, {entry['lon']}) {entry['start']}→{entry['end']}: {exc}")
        return None
 
 
def run_all(locations: list[dict] | None = None, dates: list[dict] | None = None, buffer_meters: int = 1500) -> list[dict]:
    locations = locations or get_city_coords()
    dates = dates or DATES
 
    raw_entries = get_dw_data(locations, dates, buffer_meters)
    results = []
 
    for entry in raw_entries:
        analysis = get_landcover_analysis(entry)
        if analysis is not None:
            results.append(analysis)
 
    return results
 
 
if __name__ == "__main__":
    results = run_all()
    for r in results:
        print(f"\n({r['lat']}, {r['lon']})  {r['start']} → {r['end']}")
        for cls, pct in sorted(r["percentages"].items(), key=lambda x: -x[1]):
            print(f"  {cls:<20}  {pct:6.2f}%   confidence={r['confidence'][cls]:.4f}")
