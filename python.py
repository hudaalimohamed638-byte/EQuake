
import streamlit as st
import folium
import requests
import pandas as pd
from streamlit_folium import st_folium
from pathlib import Path

# -----------------------------
# 1. PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Dancing with the SARs",
    page_icon="🌏",
    layout="wide"
)

# -----------------------------
# 2. DESIGN
# -----------------------------
st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
}
section[data-testid="stSidebar"] {
    background-color: #F7FBF9;
}
.block-container {
    padding-top: 2rem;
}
.section-title {
    background-color: #287D68;
    color: white;
    padding: 12px 18px;
    border-radius: 9px;
    font-size: 20px;
    font-weight: 700;
    margin: 18px 0 12px 0;
}
.subtitle {
    color: #78AEE8;
    font-size: 15px;
    font-weight: 500;
    /* Light blue subtitles */
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #78AEE8 !important;

}

}
</style>
""", unsafe_allow_html=True)


def section_title(title):
    st.markdown(
        f'<div class="section-title">{title}</div>',
        unsafe_allow_html=True
    )


# -----------------------------
# 3. DATA FILE
# -----------------------------
DATA_FOLDER = Path("data")
DATA_FOLDER.mkdir(exist_ok=True)

SAR_FILE = DATA_FOLDER / "sar_observations.csv"

# Create an empty template if it does not exist
columns = [
    "event_id",
    "satellite",
    "acquisition_before",
    "acquisition_after",
    "before_image_url",
    "after_image_url",
    "interferogram_url",
    "source_url"
]

if not SAR_FILE.exists():
    pd.DataFrame(columns=columns).to_csv(
        SAR_FILE,
        index=False
    )

sar_data = pd.read_csv(SAR_FILE).fillna("")


# -----------------------------
# 4. JAPAN REGIONS
# Approximate geographic boxes
# -----------------------------
REGIONS = {
    "All Japan": {
        "lat": [24, 46],
        "lon": [123, 146],
        "center": [36.2, 138.2],
        "zoom": 5
    },
    "Hokkaido": {
        "lat": [41.3, 45.6],
        "lon": [139.3, 148.9],
        "center": [43.5, 143.5],
        "zoom": 6
    },
    "Tohoku": {
        "lat": [36.8, 41.6],
        "lon": [139.5, 142.5],
        "center": [39.0, 140.8],
        "zoom": 7
    },
    "Kanto (Tokyo / Koto)": {
        "lat": [34.8, 37.2],
        "lon": [138.5, 141.2],
        "center": [35.7, 139.8],
        "zoom": 8
    },
    "Chubu": {
        "lat": [34.5, 38.5],
        "lon": [136.0, 139.5],
        "center": [36.3, 137.7],
        "zoom": 7
    },
    "Kansai": {
        "lat": [33.5, 36.5],
        "lon": [134.0, 137.5],
        "center": [34.8, 135.5],
        "zoom": 7
    },
    "Chugoku": {
        "lat": [33.8, 35.7],
        "lon": [130.8, 134.5],
        "center": [34.7, 132.5],
        "zoom": 7
    },
    "Shikoku": {
        "lat": [32.7, 34.5],
        "lon": [132.0, 134.5],
        "center": [33.7, 133.5],
        "zoom": 7
    },
    "Kyushu": {
        "lat": [30.8, 34.2],
        "lon": [129.0, 132.5],
        "center": [32.5, 130.7],
        "zoom": 7
    },
    "Okinawa": {
        "lat": [24.0, 27.5],
        "lon": [122.5, 131.5],
        "center": [26.2, 127.7],
        "zoom": 7
    }
}


# -----------------------------
# 5. GET EARTHQUAKE INFORMATION
# Only retrieve events linked to
# verified SAR catalog records.
# -----------------------------
@st.cache_data(ttl=3600)
def get_earthquakes(event_ids):
    if not event_ids:
        return []

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    response = requests.get(
        url,
        params={
            "format": "geojson",
            "eventid": ",".join(event_ids)
        },
        timeout=40
    )

    response.raise_for_status()

    return response.json().get("features", [])


# -----------------------------
# 6. EARTHQUAKE DETAILS WINDOW
# -----------------------------
@st.dialog("Earthquake and SAR Observation", width="large")
def show_event_details(event, sar):
    properties = event["properties"]
    coordinates = event["geometry"]["coordinates"]

    longitude, latitude, depth = coordinates

    magnitude = properties.get("mag")
    place = properties.get("place", "Unknown location")
    timestamp = properties.get("time")

    st.subheader(place)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Magnitude",
        magnitude if magnitude is not None else "N/A"
    )

    col2.metric(
        "Depth (km)",
        round(depth, 2) if depth is not None else "N/A"
    )

    col3.metric(
        "Latitude",
        round(latitude, 4)
    )

    if timestamp:
        st.write(
            "**Date and time (UTC):**",
            pd.to_datetime(timestamp, unit="ms", utc=True)
        )

    st.write("**Longitude:**", round(longitude, 4))
    st.write("**USGS Event ID:**", event["id"])

    st.divider()

    st.subheader("SAR Observation Information")

    st.write("**Satellite:**", sar["satellite"] or "Not specified")
    st.write(
        "**Before acquisition:**",
        sar["acquisition_before"] or "Not available"
    )
    st.write(
        "**After acquisition:**",
        sar["acquisition_after"] or "Not available"
    )

    st.caption(
        "An acquisition date confirms an image observation, "
        "not necessarily measured ground deformation."
    )

    st.divider()

    st.subheader("Satellite Images")

    before = sar["before_image_url"]
    after = sar["after_image_url"]
    interferogram = sar["interferogram_url"]

    image_col1, image_col2 = st.columns(2)

    with image_col1:
        st.write("**Before the event**")
        if before:
            st.image(before, use_container_width=True)
        else:
            st.info("Before image is not available.")

    with image_col2:
        st.write("**After the event**")
        if after:
            st.image(after, use_container_width=True)
        else:
            st.info("After image is not available.")

    if interferogram:
        st.write("**Processed Interferogram (InSAR)**")
        st.image(interferogram, use_container_width=True)
    else:
        st.info("No interferogram image has been linked yet.")

    source = sar["source_url"]
    if source:
        st.link_button("Open SAR Data Source", source)

    st.link_button(
        "Open USGS Event Record",
        f"https://earthquake.usgs.gov/earthquakes/eventpage/{event['id']}"
    )


# -----------------------------
# 7. APP HEADER
# -----------------------------
st.markdown("""
<h1 style="color:#287D68; margin-bottom:0;">
Dancing with the SARs
</h1>
<p style="color:#64748B;">
Japan Satellite-Based Earthquake Monitoring Dashboard
</p>
""", unsafe_allow_html=True)

st.write(
    "Explore earthquake events associated with verified "
    "satellite radar observations across Japan."
)


# -----------------------------
# 8. AREA SELECTION
# -----------------------------
section_title("Study Area Selection")

selected_region = st.selectbox(
    "Choose a region in Japan",
    list(REGIONS.keys())
)

region = REGIONS[selected_region]

st.caption(
    "Regional boundaries are approximate geographic filters, "
    "not official administrative borders."
)


# -----------------------------
# 9. LOAD VERIFIED SAR EVENTS
# -----------------------------
event_ids = (
    sar_data["event_id"]
    .astype(str)
    .str.strip()
    .replace("", pd.NA)
    .dropna()
    .unique()
    .tolist()
)

st.markdown("")

metric1, metric2 = st.columns(2)

metric1.metric(
    "Verified SAR-linked events",
    len(event_ids)
)

metric2.metric(
    "Selected region",
    selected_region
)


# -----------------------------
# 10. MAP
# -----------------------------
section_title("Japan Radar Observation Map")

earthquake_map = folium.Map(
    location=region["center"],
    zoom_start=region["zoom"],
    tiles="CartoDB positron"
)

events = []
error_message = None

try:
    events = get_earthquakes(tuple(event_ids))
except requests.RequestException as error:
    error_message = str(error)


# Match USGS events with SAR records
event_lookup = {
    event["id"]: event
    for event in events
}

matched_rows = []

for _, sar in sar_data.iterrows():

    event_id = str(sar["event_id"]).strip()

    if event_id not in event_lookup:
        continue

    event = event_lookup[event_id]

    longitude, latitude, depth = event["geometry"]["coordinates"]

    # Geographic region filter
    if not (
        region["lat"][0] <= latitude <= region["lat"][1]
        and region["lon"][0] <= longitude <= region["lon"][1]
    ):
        continue

    properties = event["properties"]

    folium.CircleMarker(
        location=[latitude, longitude],
        radius=8,
        color="#D94841",
        fill=True,
        fill_color="#D94841",
        fill_opacity=0.85,
        tooltip=f"SAR-linked earthquake: {event_id}",
        popup=folium.Popup(event_id, max_width=250)
    ).add_to(earthquake_map)

    matched_rows.append({
        "event": event,
        "sar": sar,
        "latitude": latitude,
        "longitude": longitude,
        "place": properties.get("place", "Unknown"),
        "magnitude": properties.get("mag"),
        "depth": depth
    })


map_result = st_folium(
    earthquake_map,
    width=None,
    height=580,
    key="japan_radar_map"
)


# -----------------------------
# 11. OPEN DETAILS ON MARKER CLICK
# -----------------------------
if map_result and map_result.get("last_object_clicked_popup"):

    clicked_id = str(map_result["last_object_clicked_popup"]).strip()

    for item in matched_rows:
        if item["event"]["id"] == clicked_id:
            st.session_state["selected_event_id"] = clicked_id
            break


# -----------------------------
# 12. EVENT DETAILS
# -----------------------------
if "selected_event_id" in st.session_state:

    selected_id = st.session_state["selected_event_id"]

    selected_item = next(
        (
            item for item in matched_rows
            if item["event"]["id"] == selected_id
        ),
        None
    )

    if selected_item:
        show_event_details(
            selected_item["event"],
            selected_item["sar"]
        )
        del st.session_state["selected_event_id"]


# -----------------------------
# 13. EVENT TABLE
# -----------------------------
section_title("Verified Earthquake Records")

if error_message:
    st.error(f"Could not retrieve USGS information: {error_message}")

elif not event_ids:
    st.info(
        "No verified SAR-linked earthquake records have been added yet. "
        "The map will display events after the SAR catalog is populated."
    )

elif not matched_rows:
    st.info(
        "No linked SAR earthquake events were found in this selected region."
    )

else:
    table_data = []

    for item in matched_rows:
        event = item["event"]
        sar = item["sar"]

        timestamp = event["properties"].get("time")

        table_data.append({
            "Event ID": event["id"],
            "Location": item["place"],
            "Magnitude": item["magnitude"],
            "Depth (km)": item["depth"],
            "Satellite": sar["satellite"],
            "Before Date": sar["acquisition_before"],
            "After Date": sar["acquisition_after"]
        })

    st.dataframe(
        pd.DataFrame(table_data),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Click a red marker on the map to open its event and SAR details."
    )


# -----------------------------
# 14. DATA TRANSPARENCY
# -----------------------------
section_title("Data Sources and Limitations")

st.markdown("""
- **Earthquake information:** USGS Earthquake Catalog.
- **Satellite observations:** JAXA ALOS / ALOS-2 catalog records.
- **Ground deformation:** Only shown when a processed InSAR product
  is available and linked to the event.
- **Important:** A satellite acquisition does not automatically prove
  that an earthquake-related deformation was detected.
""")
