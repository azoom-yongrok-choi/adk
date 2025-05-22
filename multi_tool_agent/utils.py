import json
import os

# Recursively extract all field paths from data_type.json properties
def extract_field_paths(properties, parent=""):
    paths = []
    for k, v in properties.items():
        path = f"{parent}.{k}" if parent else k
        # If nested or object, go deeper
        if v.get("type") == "nested" or "properties" in v:
            child_props = v.get("properties", {})
            paths.extend(extract_field_paths(child_props, path))
        else:
            paths.append(path)
    return paths

# Extract only the paths corresponding to main fields (location, fee, security, space)
def get_default_parking_fields(data_type_path=None):
    if data_type_path is None:
        # Relative path from agent.py
        data_type_path = os.path.join(os.path.dirname(__file__), "data_type.json")
    with open(data_type_path, encoding="utf-8") as f:
        data_type = json.load(f)
    all_paths = extract_field_paths(data_type["properties"])
    # Main field keywords
    location_keywords = ["address", "addressView", "location", "city.name", "prefecture.name", "region.name", "nearbyStations.name"]
    fee_keywords = ["payment.fee", "spaces.rent", "spaces.rentMin", "spaces.rentTaxClass", "referralFeeTotal", "storageDocument.issuingFee"]
    security_keywords = ["securityFacilities.status", "spaces.facility"]
    space_keywords = ["spaces", "capacity", "hasDivisionDrawing"]
    # Filter only paths that match the keywords
    def match_keywords(path, keywords):
        for kw in keywords:
            if path.endswith(kw):
                return True
        return False
    default_fields = [
        p for p in all_paths if (
            match_keywords(p, location_keywords)
            or match_keywords(p, fee_keywords)
            or match_keywords(p, security_keywords)
            or match_keywords(p, space_keywords)
        )
    ]
    # Order: location, fee, security, space
    ordered = []
    for group in [location_keywords, fee_keywords, security_keywords, space_keywords]:
        for kw in group:
            for p in default_fields:
                if p.endswith(kw) and p not in ordered:
                    ordered.append(p)
    return ordered

# Recursively extract all nested field paths from data_type.json properties
def extract_nested_field_paths(properties, parent=""):
    nested_paths = []
    for k, v in properties.items():
        path = f"{parent}.{k}" if parent else k
        if v.get("type") == "nested":
            nested_paths.append(path)
        if "properties" in v:
            nested_paths.extend(extract_nested_field_paths(v["properties"], path))
    return nested_paths

# Get all nested field paths from data_type.json
def get_nested_fields(data_type_path=None):
    if data_type_path is None:
        data_type_path = os.path.join(os.path.dirname(__file__), "data_type.json")
    with open(data_type_path, encoding="utf-8") as f:
        data_type = json.load(f)
    return extract_nested_field_paths(data_type["properties"]) 