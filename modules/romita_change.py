# romita_change.py
# Calculate percentage change in land cover types between historical and recent data.
# Output is used for Plotly visualisation and Gemini prompting.

def calculate_percentage_change(historical: dict, recent: dict) -> dict:
    """
    Calculate the percentage-point difference for each land-cover class.

    Parameters
    ----------
    historical : dict
        Land-cover percentages for the earlier period.
    recent : dict
        Land-cover percentages for the more recent period.

    Returns
    -------
    dict
        {class_name: delta_pp, ...}  — positive means the class grew.
    """
    all_keys = set(historical) | set(recent)
    change_dict = {}
    for land_type in all_keys:
        old_value = historical.get(land_type, 0)
        new_value = recent.get(land_type, 0)
        change_dict[land_type] = round(new_value - old_value, 4)
    return change_dict


# ── Quick self-test with dummy data ───────────────────────────────────────────
if __name__ == "__main__":
    from dummy_landcover_data import historical_percentages, recent_percentages

    changes = calculate_percentage_change(historical_percentages, recent_percentages)
    print("Land Cover Percentage Changes:")
    for cls, delta in sorted(changes.items(), key=lambda x: -abs(x[1])):
        bar = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
        print(f"  {bar} {cls:<25} {delta:+.2f} pp")