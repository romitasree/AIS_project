# Calculate percentage change in land cover types between historical and recent data
# Output will be used later for Plotly visualization

from dummy_landcover_data import historical_percentages, recent_percentages

# Calculates percentage change for each land cover type
# Arguments:
# historical (dict): earlier percentages
# recent (dict): recent percentages
# Returns:
# dict: percentage change for each land cover type
def calculate_percentage_change(historical, recent):

    change_dict = {}

    for land_type in historical:
        old_value = historical.get(land_type, 0)
        new_value = recent.get(land_type, 0)

        change = new_value - old_value
        change_dict[land_type] = change

    return change_dict


# Run test with dummy data
if __name__ == "__main__":

    changes = calculate_percentage_change(
        historical_percentages,
        recent_percentages
    )

    print("Land Cover Percentage Changes:")
    print(changes)