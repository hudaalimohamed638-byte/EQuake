import calendar
import json
from datetime import date
from pathlib import Path

import folium
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_folium import st_folium

# ==========================================================
# Dancing with the SARs
# Japan earthquake catalog + documented damage + SAR records
# No invented damage or deformation values.
# ==========================================================

st.set_page_config(
    page_title="Dancing with the SARs",
    page_icon="🌏",
    layout="wide",
)

st.markdown("""
<style>
/* Dark interface */
.stApp, [data-testid="stAppViewContainer"] {background:#050505;color:#F5F5F5;}
[data-testid="stHeader"] {background:rgba(5,5,5,0.95);}
section[data-testid="stSidebar"] {background:#101010;border-right:1px solid #292929;}
.block-container {padding-top:1.5rem;}
.stApp p, .stApp label, .stApp span, .stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp li, .stApp [data-testid="stMarkdownContainer"] {color:#F1F1F1;}
.section-title {
 background:#173D35;color:#FFFFFF;padding:11px 16px;border-radius:9px;
 font-size:19px;font-weight:700;margin:18px 0 12px 0;
}
div[data-testid="stMetric"] {
 background:#171717;border:1px solid #333333;padding:12px;border-radius:10px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {background:#151515;}
.stButton button, .stDownloadButton button {background:#173D35;color:#FFFFFF;border:1px solid #3D7668;}
</style>
""", unsafe_allow_html=True)


def section_title(text):
    st.markdown(
        f'<div class="section-title">{text}</div>',
        unsafe_allow_html=True,
    )


# -------------------- DATA FILES --------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SAR_FILE = DATA_DIR / "sar_observations.csv"
DAMAGE_FILE = DATA_DIR / "damage_assessments.csv"
ZONES_FILE = DATA_DIR / "damage_zones.geojson"

SAR_COLUMNS = [
    "event_id", "satellite", "acquisition_before", "acquisition_after",
    "before_image_url", "after_image_url", "interferogram_url", "source_url",
]
DAMAGE_COLUMNS = [
    "zone_id", "event_id", "zone_name", "impact_level",
    "affected_area_km2", "buildings_affected", "impact_types",
    "findings", "methodology", "confidence", "source_url",
]

if not SAR_FILE.exists():
    pd.DataFrame(columns=SAR_COLUMNS).to_csv(SAR_FILE, index=False)
if not DAMAGE_FILE.exists():
    pd.DataFrame(columns=DAMAGE_COLUMNS).to_csv(DAMAGE_FILE, index=False)
if not ZONES_FILE.exists():
    ZONES_FILE.write_text(
        json.dumps({"type": "FeatureCollection", "features": []}, indent=2),
        encoding="utf-8",
    )


def read_csv(path, columns):
    try:
        df = pd.read_csv(path).fillna("")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        df = pd.DataFrame(columns=columns)
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns]


sar_data = read_csv(SAR_FILE, SAR_COLUMNS)
damage_data = read_csv(DAMAGE_FILE, DAMAGE_COLUMNS)


def number_or_none(value):
    try:
        if value == "" or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def color_for(level):
    return {
        "high": "#D94841",
        "moderate": "#E6A23C",
        "low": "#3A9D75",
        "unknown": "#87939E",
        "unassessed": "#87939E",
    }.get(str(level).strip().lower(), "#87939E")


def load_geojson():
    try:
        with ZONES_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("type") == "FeatureCollection":
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"type": "FeatureCollection", "features": []}


geojson = load_geojson()
features = geojson.get("features", [])

# Join assessment information to zone features using zone_id.
assessment_by_zone = {}
for _, row in damage_data.iterrows():
    zid = str(row["zone_id"]).strip()
    if zid:
        assessment_by_zone[zid] = row.to_dict()

for feature in features:
    props = feature.setdefault("properties", {})
    zid = str(props.get("zone_id", "")).strip()
    if zid in assessment_by_zone:
        for key, value in assessment_by_zone[zid].items():
            if value != "":
                props[key] = value


# -------------------- JAPAN REGIONS --------------------
# These bounding boxes are approximate browsing filters, not official borders.
REGIONS = {
    "All Japan": (24.0, 46.0, 122.0, 146.0),
    "Hokkaido": (41.0, 46.0, 139.0, 146.0),
    "Tohoku": (37.0, 41.5, 139.0, 142.5),
    "Kanto (Tokyo area)": (34.5, 37.5, 138.0, 141.5),
    "Chubu": (34.0, 38.5, 135.0, 140.5),
    "Kansai": (33.5, 36.5, 134.0, 137.5),
    "Chugoku": (33.5, 36.0, 130.5, 134.5),
    "Shikoku": (32.5, 34.5, 132.0, 134.5),
    "Kyushu": (30.0, 34.5, 129.0, 132.5),
    "Okinawa": (24.0, 27.5, 122.0, 131.5),
}


# -------------------- USGS API --------------------
USGS_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"


@st.cache_data(ttl=3600, show_spinner=False)
def get_one_usgs_event(event_id):
    response = requests.get(
        USGS_URL,
        params={"format": "geojson", "eventid": event_id},
        timeout=30,
    )
    response.raise_for_status()
    items = response.json().get("features", [])
    return items[0] if items else None


@st.cache_data(ttl=3600, show_spinner=False)
def get_japan_events(start_date, end_date, min_magnitude, max_magnitude,
                     min_lat, max_lat, min_lon, max_lon):
    """Query USGS month by month to avoid one oversized multi-year request."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError("Start date must be on or before end date.")

    collected = {}
    year, month = start.year, start.month

    while (year, month) <= (end.year, end.month):
        last_day = calendar.monthrange(year, month)[1]
        month_start = date(year, month, 1)
        month_end = date(year, month, last_day)
        chunk_start = max(start, month_start)
        chunk_end = min(end, month_end)

        params = {
            "format": "geojson",
            "starttime": f"{chunk_start.isoformat()}T00:00:00",
            "endtime": f"{chunk_end.isoformat()}T23:59:59",
            "minlatitude": min_lat,
            "maxlatitude": max_lat,
            "minlongitude": min_lon,
            "maxlongitude": max_lon,
            "minmagnitude": min_magnitude,
            "maxmagnitude": max_magnitude,
            "orderby": "time",
            "limit": 20000,
        }
        response = requests.get(USGS_URL, params=params, timeout=60)
        response.raise_for_status()

        for event in response.json().get("features", []):
            event_id = event.get("id")
            if event_id:
                collected[event_id] = event

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    return list(collected.values())


# -------------------- LOOKUPS --------------------
event_ids = sorted({
    str(value).strip()
    for value in sar_data["event_id"].tolist()
    if str(value).strip() and str(value).lower() != "nan"
})

sar_by_event = {
    str(row["event_id"]).strip(): row.to_dict()
    for _, row in sar_data.iterrows()
    if str(row["event_id"]).strip()
}

damage_by_event = {}
for _, row in damage_data.iterrows():
    eid = str(row["event_id"]).strip()
    if eid:
        damage_by_event.setdefault(eid, []).append(row.to_dict())

linked_events = {}
for eid in event_ids:
    try:
        event = get_one_usgs_event(eid)
        if event:
            linked_events[eid] = event
    except requests.RequestException:
        continue

# -------------------- SIDEBAR --------------------
st.sidebar.title("🌏 Dancing with the SARs")
st.sidebar.caption("Japan Earthquake Impact & Satellite Monitoring")

def go_to_sar():
    st.session_state["page"] = "SAR Analysis"
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview", "Explore Map", "Event Details",
        "Earthquake Image Gallery", "SAR Analysis",
        "Impact Analytics", "Timeline & Data",
    ],
    key="page",
)

st.sidebar.divider()
selected_region = st.sidebar.selectbox("Japan region", list(REGIONS.keys()))
region_bounds = REGIONS[selected_region]
st.sidebar.caption(
    "Regional boundaries are approximate geographic filters, not official prefecture borders."
)

st.sidebar.markdown("### Earthquake filters")
min_magnitude = st.sidebar.slider(
    "Minimum magnitude", 0.0, 9.5, 4.0, 0.5
)
max_magnitude = st.sidebar.slider(
    "Maximum magnitude", 0.5, 10.0, 10.0, 0.5
)
if min_magnitude > max_magnitude:
    st.sidebar.warning("Minimum magnitude cannot exceed maximum magnitude.")
    min_magnitude = max_magnitude

st.sidebar.caption(
    "Choose the date range and load catalog records in Timeline & Data. "
    "Loaded records will also appear on the maps."
)

# -------------------- EVENT HELPERS --------------------
def event_coordinates(event):
    coords = event.get("geometry", {}).get("coordinates", [])
    if len(coords) < 2:
        return None
    return coords[1], coords[0]


def event_in_region(event, bounds):
    location = event_coordinates(event)
    if location is None:
        return False
    lat, lon = location
    min_lat, max_lat, min_lon, max_lon = bounds
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def get_loaded_events():
    # Catalog events are independent of SAR records.
    events = st.session_state.get("timeline_events", [])
    if events:
        return events
    # Before the user loads a timeline, show only events with SAR catalog links.
    return list(linked_events.values())


def get_event_id(event):
    return str(event.get("id", ""))


def get_event_label(event):
    p = event.get("properties", {})
    return f'{p.get("place", "Unknown location")} | M {p.get("mag", "N/A")} | {event.get("id", "")}'


# -------------------- MAP --------------------
def build_map(map_key):
    bounds = region_bounds
    center_lat = (bounds[0] + bounds[1]) / 2
    center_lon = (bounds[2] + bounds[3]) / 2

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5 if selected_region == "All Japan" else 6,
        tiles=None,
        control_scale=True,
    )
    folium.TileLayer("CartoDB positron", name="Light map").add_to(m)
    folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri",
        name="Satellite basemap",
        overlay=False,
    ).add_to(m)

    zone_layer = folium.FeatureGroup(name="Affected areas", show=True)
    for feature in features:
        props = feature.get("properties", {})
        zid = str(props.get("zone_id", "Unknown"))
        name = str(props.get("zone_name", props.get("name", zid)))
        level = str(props.get("impact_level", "Unassessed"))
        color = color_for(level)
        area = props.get("affected_area_km2", "")
        popup_html = f"""
        <div style="width:230px;font-family:Arial">
          <b>{name}</b><br>Zone ID: {zid}<br>
          Impact: {level}<br>
          Affected area (km²): {area if area != '' else 'Not available'}
        </div>
        """
        folium.GeoJson(
            feature,
            style_function=lambda f, c=color: {
                "fillColor": c, "color": c, "weight": 2, "fillOpacity": 0.4
            },
            highlight_function=lambda f: {"weight": 4, "fillOpacity": 0.65},
            tooltip=folium.Tooltip(f"{name} | {level}", sticky=True),
            popup=folium.Popup(popup_html, max_width=280),
        ).add_to(zone_layer)
    zone_layer.add_to(m)

    quake_layer = folium.FeatureGroup(name="USGS earthquake epicenters", show=True)
    visible_events = [
        event for event in get_loaded_events()
        if event_in_region(event, bounds)
        and number_or_none(event.get("properties", {}).get("mag")) is not None
        and min_magnitude <= float(event["properties"]["mag"]) <= max_magnitude
    ]

    for event in visible_events:
        location = event_coordinates(event)
        if location is None:
            continue
        lat, lon = location
        p = event.get("properties", {})
        eid = get_event_id(event)
        mag = p.get("mag")
        place = p.get("place", "Unknown location")
        has_sar = eid in sar_by_event
        folium.CircleMarker(
            [lat, lon],
            radius=5 + min(float(mag or 0), 8) * 0.7,
            color="#287D68" if has_sar else "#263B53",
            fill=True,
            fill_color="#287D68" if has_sar else "#263B53",
            fill_opacity=0.85,
            tooltip=f"{place} | Magnitude: {mag}",
            popup=(
                f"Event ID: {eid}<br>Magnitude: {mag}<br>"
                f"{'SAR record linked' if has_sar else 'No SAR record linked'}"
            ),
        ).add_to(quake_layer)

    quake_layer.add_to(m)

    # -------------------- FIXED HISTORICAL EARTHQUAKES --------------------
    # These markers are independent of the date and magnitude filters.
    # Values are based on the listed USGS ComCat event records.
    historical_layer = folium.FeatureGroup(
        name="Fixed historical earthquakes (blue)",
        show=True,
    )

    historical_earthquakes = [
        {
            "name": "2011 Great Tohoku Earthquake",
            "latitude": 38.297,
            "longitude": 142.373,
            "magnitude": 9.1,
            "depth": 29.0,
            "date": "2011-03-11 05:46:24 UTC",
            "event_id": "usc0001xgp",
        },
        {
            "name": "1933 Sanriku Earthquake",
            "latitude": 39.209,
            "longitude": 144.590,
            "magnitude": 8.4,
            "depth": 15.0,
            "date": "1933-03-02 17:31:00 UTC",
            "event_id": "official19330302173100_30",
        },
        {
            # USGS catalog place-name is used to avoid overstating the
            # historical event's formal name.
            "name": "1946 Japan Earthquake (68 km S of Shingu)",
            "latitude": 33.123,
            "longitude": 135.905,
            "magnitude": 8.3,
            "depth": 15.0,
            "date": "1946-12-20 19:19:10 UTC",
            "event_id": "iscgem898698",
        },
        {
            "name": "2016 Kumamoto Earthquake",
            "latitude": 32.790,
            "longitude": 130.754,
            "magnitude": 7.0,
            "depth": 10.0,
            "date": "2016-04-15 16:25:06 UTC (April 16 in Japan)",
            "event_id": "us20005iis",
        },
    ]

    for quake in historical_earthquakes:
        lat = quake["latitude"]
        lon = quake["longitude"]

        # Show each fixed marker only when it is inside the selected region.
        if bounds[0] <= lat <= bounds[1] and bounds[2] <= lon <= bounds[3]:
            usgs_url = (
                "https://earthquake.usgs.gov/earthquakes/eventpage/"
                + quake["event_id"]
            )
            popup_html = f"""
            <div style="width:260px;font-family:Arial">
                <b>{quake['name']}</b><br>
                <b>Magnitude:</b> {quake['magnitude']}<br>
                <b>Depth:</b> {quake['depth']} km<br>
                <b>Date/time:</b> {quake['date']}<br>
                <b>Latitude:</b> {lat}<br>
                <b>Longitude:</b> {lon}<br>
                <a href="{usgs_url}" target="_blank">Open USGS event record</a>
            </div>
            """

            folium.CircleMarker(
                location=[lat, lon],
                radius=9,
                color="#1769FF",
                weight=2,
                fill=True,
                fill_color="#1769FF",
                fill_opacity=0.95,
                tooltip=f"{quake['name']} | M {quake['magnitude']}",
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(historical_layer)

    historical_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m, len(visible_events)



# -------------------- REPORT RENDERER --------------------
def render_event_report(event_id):
    event = linked_events.get(event_id)
    if event is None:
        for item in st.session_state.get("timeline_events", []):
            if get_event_id(item) == event_id:
                event = item
                break

    sar = sar_by_event.get(event_id, {})
    assessments = damage_by_event.get(event_id, [])

    st.subheader("Earthquake information")
    if event:
        props = event.get("properties", {})
        coords = event.get("geometry", {}).get("coordinates", [None, None, None])
        lon, lat, depth = (coords + [None, None, None])[:3]
        c1, c2, c3 = st.columns(3)
        c1.metric("Magnitude", props.get("mag") if props.get("mag") is not None else "N/A")
        c2.metric("Depth (km)", round(depth, 2) if depth is not None else "N/A")
        c3.metric("Latitude", round(lat, 4) if lat is not None else "N/A")
        st.write("**Location:**", props.get("place", "Unknown"))
        st.write("**Longitude:**", round(lon, 4) if lon is not None else "N/A")
        if props.get("time"):
            st.write(
                "**Date/time (UTC):**",
                pd.to_datetime(props["time"], unit="ms", utc=True).strftime("%Y-%m-%d %H:%M"),
            )
        st.link_button(
            "Open USGS event",
            f"https://earthquake.usgs.gov/earthquakes/eventpage/{event_id}",
        )
    else:
        st.warning("USGS information is unavailable for this event ID.")

    st.divider()
    st.subheader("Damage assessment results")
    if assessments:
        for record in assessments:
            st.markdown(f"### {record.get('zone_name') or record.get('zone_id')}")
            a, b, c = st.columns(3)
            a.metric("Impact level", record.get("impact_level") or "Not assessed")
            area = number_or_none(record.get("affected_area_km2"))
            b.metric("Affected area (km²)", f"{area:g}" if area is not None else "N/A")
            buildings = number_or_none(record.get("buildings_affected"))
            c.metric("Buildings affected", f"{buildings:g}" if buildings is not None else "N/A")
            st.write("**Impact types:**", record.get("impact_types") or "Not recorded")
            st.write("**Findings:**", record.get("findings") or "No findings entered.")
            st.write("**Methodology:**", record.get("methodology") or "Not documented.")
            st.write("**Confidence:**", record.get("confidence") or "Not specified.")
            if record.get("source_url"):
                st.link_button("Open assessment source", record["source_url"])
            st.divider()
    else:
        st.info(
            "No damage assessment is linked to this event yet. "
            "An epicenter does not prove that nearby locations were damaged."
        )

    st.subheader("SAR observations")
    if not sar:
        st.info("No SAR observation is linked to this earthquake in the SAR CSV.")
    st.write("**Satellite:**", sar.get("satellite") or "Not specified")
    st.write("**Before acquisition:**", sar.get("acquisition_before") or "N/A")
    st.write("**After acquisition:**", sar.get("acquisition_after") or "N/A")
    left, right = st.columns(2)
    with left:
        st.caption("Before the event")
        if sar.get("before_image_url"):
            st.image(sar["before_image_url"], use_container_width=True)
        else:
            st.info("No before image linked.")
    with right:
        st.caption("After the event")
        if sar.get("after_image_url"):
            st.image(sar["after_image_url"], use_container_width=True)
        else:
            st.info("No after image linked.")
    if sar.get("interferogram_url"):
        st.subheader("InSAR interferogram")
        st.image(sar["interferogram_url"], use_container_width=True)
    if sar.get("source_url"):
        st.link_button("Open SAR source", sar["source_url"])
    st.caption(
        "Image acquisition alone does not prove ground deformation or damage. "
        "Report displacement only when a processed and validated InSAR result exists."
    )


# -------------------- HEADER --------------------
st.markdown(
    '<h1 style="color:#287D68;margin-bottom:0;">Dancing with the SARs</h1>'
    '<p style="color:#64748B;">Japan Earthquake Impact & Satellite Monitoring Platform</p>',
    unsafe_allow_html=True,
)
st.write(
    "Explore Japan earthquake catalog events, mapped affected areas, "
    "satellite observations, and documented results."
)


# -------------------- OVERVIEW --------------------
if page == "Overview":
    section_title("Japan Impact Overview")
    total_zones = len(features)
    assessed = sum(
        str(f.get("properties", {}).get("impact_level", "")).lower()
        not in ("", "unknown", "unassessed")
        for f in features
    )
    loaded_events = get_loaded_events()
    catalog_count = sum(event_in_region(e, region_bounds) for e in loaded_events)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mapped affected areas", total_zones)
    c2.metric("Assessed areas", assessed)
    c3.metric("Loaded earthquake events", catalog_count)
    c4.metric("SAR-linked events", len(event_ids))
    st.caption(
        "Counts refer to currently loaded catalog and assessment records, "
        "not all historical earthquakes or all affected locations."
    )

    section_title("Interactive Japan Earthquake & Damage Map")
    st.write(
        "The map can show catalog earthquakes even when no SAR record is linked. "
        "Affected-area polygons appear only when verified GeoJSON is supplied."
    )
    map_obj, map_count = build_map("overview_map")
    st.caption(f"Earthquake epicenters currently shown in this region: {map_count}")
    st_folium(map_obj, width=None, height=610, key="overview_map")

    zone_options = {}
    for feature in features:
        p = feature.get("properties", {})
        zid = str(p.get("zone_id", "")).strip()
        name = str(p.get("zone_name", p.get("name", zid)))
        if zid:
            zone_options[f"{name} ({zid})"] = zid

    if zone_options:
        label = st.selectbox("Choose an affected area", list(zone_options.keys()))
        zid = zone_options[label]
        feature = next(
            f for f in features
            if str(f.get("properties", {}).get("zone_id", "")).strip() == zid
        )
        p = feature.get("properties", {})
        a, b, c = st.columns(3)
        a.metric("Impact level", p.get("impact_level") or "Not assessed")
        area = number_or_none(p.get("affected_area_km2"))
        b.metric("Affected area (km²)", f"{area:g}" if area is not None else "N/A")
        c.metric("Linked event", p.get("event_id") or "Not linked")
        st.write("**Findings:**", p.get("findings") or "No findings entered.")
        st.write("**Impact types:**", p.get("impact_types") or "Not recorded.")
        linked = str(p.get("event_id", "")).strip()
        if linked and st.button("Open full earthquake and area report", type="primary"):
            st.session_state["report_event_id"] = linked
            st.session_state["open_report_dialog"] = True
            st.rerun()
    else:
        st.info(
            "No affected-area polygons yet. Add verified polygon features to "
            "data/damage_zones.geojson to populate this map."
        )

    section_title("Map legend")
    st.markdown(
        "- Red: high impact assessment\n"
        "- Orange: moderate impact assessment\n"
        "- Green: low impact assessment\n"
        "- Grey: unknown / unassessed\n"
        "- Dark circles: USGS epicenters without a linked SAR row\n"
        "- Green circles: USGS epicenters with a linked SAR row\n"
        "- Blue circles: fixed historical earthquakes (independent of magnitude/date filters)"
    )


# -------------------- EXPLORE MAP --------------------
elif page == "Explore Map":
    section_title("Explore Map")
    map_obj, map_count = build_map("explore_map")
    st.caption(f"Earthquake epicenters currently shown: {map_count}")
    st_folium(map_obj, width=None, height=650, key="explore_map")

    visible_events = [
        event for event in get_loaded_events()
        if event_in_region(event, region_bounds)
        and number_or_none(event.get("properties", {}).get("mag")) is not None
        and min_magnitude <= float(event["properties"]["mag"]) <= max_magnitude
    ]
    if visible_events:
        event_labels = {get_event_label(e): get_event_id(e) for e in visible_events}
        selected_label = st.selectbox("Select an earthquake to inspect", list(event_labels))
        if st.button("Open selected earthquake report"):
            st.session_state["report_event_id"] = event_labels[selected_label]
            st.session_state["open_report_dialog"] = True
            st.rerun()

    options = {}
    for feature in features:
        p = feature.get("properties", {})
        zid = str(p.get("zone_id", "")).strip()
        name = str(p.get("zone_name", p.get("name", zid)))
        if zid:
            options[f"{name} ({zid})"] = zid

    if options:
        label = st.selectbox("Select an affected area", list(options.keys()))
        zid = options[label]
        feature = next(
            f for f in features
            if str(f.get("properties", {}).get("zone_id", "")).strip() == zid
        )
        p = feature.get("properties", {})
        st.subheader(p.get("zone_name", p.get("name", zid)))
        x, y, z = st.columns(3)
        x.metric("Impact level", p.get("impact_level") or "Not assessed")
        area = number_or_none(p.get("affected_area_km2"))
        y.metric("Affected area (km²)", f"{area:g}" if area is not None else "N/A")
        z.metric("Event ID", p.get("event_id") or "Not linked")
        st.write("**Findings:**", p.get("findings") or "Not entered.")
        st.write("**Methodology:**", p.get("methodology") or "Not documented.")
        st.write("**Source:**", p.get("source_url") or "Not provided.")
        eid = str(p.get("event_id", "")).strip()
        if eid and st.button("Open linked earthquake report"):
            st.session_state["report_event_id"] = eid
            st.session_state["open_report_dialog"] = True
            st.rerun()
    else:
        st.info("No GeoJSON damage zones are loaded yet.")


# -------------------- EVENT DETAILS --------------------
elif page == "Event Details":
    section_title("Earthquake Reports")
    available = [
        e for e in get_loaded_events()
        if event_in_region(e, region_bounds)
    ]
    labels = {get_event_label(e): get_event_id(e) for e in available}
    for eid, event in linked_events.items():
        labels.setdefault(get_event_label(event), eid)

    if not labels:
        st.info(
            "No earthquake records are loaded. Go to Timeline & Data, "
            "choose a date range, and load records from USGS."
        )
    else:
        selected_label = st.selectbox("Choose an earthquake", list(labels.keys()))
        render_event_report(labels[selected_label])


# -------------------- EARTHQUAKE IMAGE GALLERY --------------------
elif page == "Earthquake Image Gallery":

    section_title("Tohoku Earthquake 2011 | Before & After")

    before_image = Path(
        r"C:\Users\Timo Store\Pictures\Screenshots\Screenshot 2026-11-15 140003.png"
    )

    after_image = Path(
        r"C:\Users\Timo Store\Pictures\Screenshots\Screenshot 2026-11-15 140015.png"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("BEFORE EARTHQUAKE")

        if before_image.exists():
            st.image(str(before_image), use_container_width=True)
        else:
            st.error(f"Before image not found: {before_image}")

        st.markdown("**Image Description**")
        st.write(
            "This image represents the pre-earthquake reference scene. "
            "It provides a baseline for examining the Earth's surface "
            "before the 11 March 2011 Tohoku earthquake."
        )

        st.button(
            "Explore SAR Analysis →",
            key="before_sar",
            on_click=go_to_sar,
        )

    with right:
        st.subheader("AFTER EARTHQUAKE")

        if after_image.exists():
            st.image(str(after_image), use_container_width=True)
        else:
            st.error(f"After image not found: {after_image}")

        st.markdown("**Image Description**")
        st.write(
            "This image represents the post-earthquake scene. "
            "It can be compared with the pre-earthquake reference "
            "to investigate possible changes in the observed surface."
        )

        st.button(
            "Explore SAR Analysis →",
            key="after_sar",
            on_click=go_to_sar,
        )

    st.caption("Tohoku Earthquake — 11 March 2011")


# -------------------- SAR ANALYSIS --------------------
elif page == "SAR Analysis":

    section_title("Rikuzentakata | 2011 Tohoku Earthquake")

    st.markdown(
        "This section compares Rikuzentakata before and after the "
        "11 March 2011 earthquake. The comparison focuses on the "
        "settlement, landscape, vegetation, damage, and reconstruction."
    )

    st.subheader("Earthquake Event Information")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Magnitude", "9.0")
    e2.metric("Depth", "≈ 24 km")
    e3.metric("Date", "11 March 2011")
    e4.metric("Local time", "2:46 PM")
    st.caption(
        "Reported seismic intensity in Rikuzentakata: 6-lower "
        "on Japan's seismic intensity scale."
    )

    st.divider()

    before_col, after_col = st.columns(2, gap="large")

    with before_col:
        st.markdown("### BEFORE THE EARTHQUAKE")
        st.markdown("**Location and physical geography**")
        st.write(
            "Rikuzentakata is a coastal city in Iwate Prefecture, "
            "northeastern Japan, facing the Pacific Ocean and Hirota Bay."
        )
        st.write(
            "The city includes low-lying coastal plains, agricultural "
            "fields, and residential areas near the sea, with hills and "
            "mountains surrounding the coastal area."
        )
        st.markdown("**Population and land use**")
        st.metric("Population before the disaster", "≈ 24,246")
        st.write(
            "Many homes and buildings were located on the low coastal "
            "plain, close to the shoreline."
        )
        st.markdown("**Coastal vegetation**")
        st.metric("Pine trees in Takata Matsubara Forest", "≈ 70,000")
        st.write(
            "Takata Matsubara was a major coastal pine forest before "
            "the 2011 disaster."
        )

    with after_col:
        st.markdown("### AFTER THE EARTHQUAKE")
        st.markdown("**Human and building impacts**")
        a1, a2 = st.columns(2)
        a1.metric("Completely destroyed houses", "3,805")
        a2.metric("Seriously damaged houses", "236")
        a3, a4 = st.columns(2)
        a3.metric("Recorded deaths", "1,550")
        a4.metric("Missing people", "207")
        st.write(
            "The city hall, medical facilities, and several public "
            "buildings also suffered severe damage."
        )
        st.markdown("**Environmental change**")
        st.write(
            "Most of the pine trees in Takata Matsubara Forest were lost. "
            "One surviving tree became known as the Miracle Pine and was "
            "preserved as a memorial."
        )
        st.markdown("**Reconstruction and protection**")
        st.write(
            "- Some residential areas were relocated to higher ground.\n"
            "- Certain areas were raised by approximately 10 m.\n"
            "- A concrete seawall approximately 12.5 m high and 2 km long "
            "was constructed.\n"
            "- Coastal restoration included planting approximately "
            "40,000 tree seedlings.\n"
            "- Roads, public facilities, and essential services were "
            "repaired or rebuilt."
        )

    st.divider()
    st.subheader("Before-and-After Landscape Comparison")
    comparison_rows = [
        {
            "Feature": "Settlement pattern",
            "Before the earthquake": "Homes and buildings on low coastal plains",
            "After the earthquake": "Some residential areas relocated to higher ground",
        },
        {
            "Feature": "Coastal vegetation",
            "Before the earthquake": "Approximately 70,000 pine trees",
            "After the earthquake": "Most trees lost; restoration planting undertaken",
        },
        {
            "Feature": "Ground elevation",
            "Before the earthquake": "Low-lying coastal land",
            "After the earthquake": "Some areas raised by approximately 10 m",
        },
        {
            "Feature": "Coastal protection",
            "Before the earthquake": "Original coastal landscape",
            "After the earthquake": "Seawall approximately 12.5 m high and 2 km long",
        },
        {
            "Feature": "Public infrastructure",
            "Before the earthquake": "City facilities serving the community",
            "After the earthquake": "Damaged facilities repaired or reconstructed",
        },
    ]
    st.dataframe(
        pd.DataFrame(comparison_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "The figures above describe reported impacts and reconstruction "
        "following the 2011 earthquake disaster. They should not be "
        "interpreted as measurements of ground displacement caused by "
        "earthquake shaking alone."
    )

    st.divider()
    st.subheader("How SAR Helps Compare the Area")
    st.write(
        "Synthetic Aperture Radar (SAR) is an active remote-sensing "
        "technology that transmits microwave signals and records their "
        "reflections from the Earth's surface. It can collect observations "
        "during both day and night and is less affected by cloud cover."
    )
    st.write(
        "Comparing pre-event and post-event SAR images can reveal changes "
        "in radar backscatter. These differences may be associated with "
        "changes in surface roughness, moisture, land cover, vegetation, "
        "or built-up areas. A visual difference alone does not identify "
        "the exact cause of every change."
    )

    st.subheader("Ground Deformation and InSAR")
    st.write(
        "Interferometric Synthetic Aperture Radar (InSAR) uses phase "
        "differences between SAR acquisitions to investigate ground "
        "displacement. Quantitative displacement values require suitable "
        "images, acquisition metadata, processing, and validation."
    )

    st.subheader("Scientific Limitations")
    st.write(
        "The reliability of a SAR comparison depends on acquisition dates, "
        "satellite viewing geometry, spatial resolution, processing quality, "
        "and environmental conditions. Without verified acquisition "
        "metadata and processed SAR products, the displayed images cannot "
        "establish the amount or direction of ground displacement."
    )

    st.divider()
    st.subheader("Available SAR Dataset")

    if sar_data.empty:
        st.info(
            "No SAR records yet. Add documented rows to sar_observations.csv."
        )
    else:
        ids = sorted(
            set(
                str(value).strip()
                for value in sar_data["event_id"]
                if str(value).strip()
            )
        )
        eid = st.selectbox("Select event ID", ids)
        sar = sar_by_event.get(eid, {})

        st.write("**Satellite:**", sar.get("satellite") or "Not specified")
        st.write("**Before date:**", sar.get("acquisition_before") or "N/A")
        st.write("**After date:**", sar.get("acquisition_after") or "N/A")

        if sar.get("before_image_url"):
            st.subheader("Before-event SAR image")
            st.image(sar["before_image_url"], use_container_width=True)

        if sar.get("after_image_url"):
            st.subheader("After-event SAR image")
            st.image(sar["after_image_url"], use_container_width=True)

        if sar.get("interferogram_url"):
            st.subheader("Interferogram")
            st.image(sar["interferogram_url"], use_container_width=True)
            st.caption(
                "Interpretation requires processing metadata and documented analysis."
            )

        if sar.get("source_url"):
            st.link_button("Open SAR source", sar["source_url"])


# -------------------- IMPACT ANALYTICS --------------------
elif page == "Impact Analytics":
    section_title("Impact Analytics")
    if damage_data.empty:
        st.info("Add verified assessment records to damage_assessments.csv.")
    else:
        df = damage_data.copy()
        df["area_num"] = pd.to_numeric(df["affected_area_km2"], errors="coerce")
        df["buildings_num"] = pd.to_numeric(df["buildings_affected"], errors="coerce")

        total_area = df["area_num"].sum(min_count=1)
        total_buildings = df["buildings_num"].sum(min_count=1)
        a, b, c = st.columns(3)
        a.metric("Assessment records", len(df))
        b.metric(
            "Reported area (km²)",
            f"{total_area:,.2f}" if pd.notna(total_area) else "N/A",
        )
        c.metric(
            "Reported buildings affected",
            f"{total_buildings:,.0f}" if pd.notna(total_buildings) else "N/A",
        )

        levels = df[df["impact_level"].astype(str).str.strip() != ""]
        if not levels.empty:
            counts = (
                levels["impact_level"].value_counts()
                .rename_axis("Impact level").reset_index(name="Records")
            )
            st.subheader("Assessment records by impact level")
            st.plotly_chart(
                px.bar(counts, x="Impact level", y="Records"),
                use_container_width=True,
            )

        areas = df.dropna(subset=["area_num"])
        if not areas.empty:
            st.subheader("Reported affected area by zone")
            st.plotly_chart(
                px.bar(
                    areas, x="zone_name", y="area_num", color="impact_level",
                    labels={
                        "zone_name": "Affected zone",
                        "area_num": "Area (km²)",
                    },
                ),
                use_container_width=True,
            )

        st.subheader("Assessment data")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(
            "Check for overlapping zones or repeated assessments before summing totals."
        )


# -------------------- TIMELINE & DATA --------------------
elif page == "Timeline & Data":
    section_title("Japan Earthquake Timeline")
    st.write(
        "USGS catalog records are separate from SAR observations. "
        "Loading an earthquake here does not mean that SAR imagery or damage data exists for it."
    )

    col1, col2 = st.columns(2)
    with col1:
        start = st.date_input(
            "Start date",
            value=date(2011, 1, 1),
            min_value=date(1900, 1, 1),
            max_value=date.today(),
        )
    with col2:
        end = st.date_input(
            "End date",
            value=date.today(),
            min_value=date(1900, 1, 1),
            max_value=date.today(),
        )

    if st.button("Load earthquake records from USGS", type="primary"):
        if start > end:
            st.error("Start date must be before or equal to end date.")
        else:
            min_lat, max_lat, min_lon, max_lon = region_bounds
            try:
                with st.spinner(
                    "Loading USGS records month by month. A long date range may take time..."
                ):
                    st.session_state["timeline_events"] = get_japan_events(
                        start.isoformat(),
                        end.isoformat(),
                        float(min_magnitude),
                        float(max_magnitude),
                        min_lat, max_lat, min_lon, max_lon,
                    )
                    st.session_state["timeline_range"] = (start, end)
                    st.session_state["timeline_region"] = selected_region
                st.success(
                    f"Loaded {len(st.session_state['timeline_events']):,} "
                    "USGS earthquake records."
                )
            except (requests.RequestException, ValueError) as error:
                st.error(f"Could not load USGS data: {error}")

    timeline = st.session_state.get("timeline_events", [])
    if timeline:
        rows = []
        for event in timeline:
            p = event.get("properties", {})
            coords = event.get("geometry", {}).get("coordinates", [None, None, None])
            rows.append({
                "Date (UTC)": (
                    pd.to_datetime(p["time"], unit="ms", utc=True)
                    if p.get("time") else None
                ),
                "Location": p.get("place"),
                "Magnitude": p.get("mag"),
                "Depth (km)": coords[2] if len(coords) > 2 else None,
                "Latitude": coords[1] if len(coords) > 1 else None,
                "Longitude": coords[0] if len(coords) > 0 else None,
                "Event ID": event.get("id"),
                "USGS URL": p.get("url"),
            })

        timeline_df = pd.DataFrame(rows)
        st.subheader("Earthquake records")
        st.dataframe(timeline_df, use_container_width=True, hide_index=True)

        st.subheader("Magnitude over time")
        plot_df = timeline_df.dropna(subset=["Date (UTC)", "Magnitude"])
        if not plot_df.empty:
            st.plotly_chart(
                px.scatter(
                    plot_df,
                    x="Date (UTC)",
                    y="Magnitude",
                    hover_name="Location",
                    hover_data=["Event ID", "Depth (km)"],
                    labels={"Date (UTC)": "Date (UTC)", "Magnitude": "Magnitude"},
                ),
                use_container_width=True,
            )

        st.download_button(
            "Download loaded earthquake records (CSV)",
            data=timeline_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="japan_earthquake_catalog.csv",
            mime="text/csv",
        )
        st.caption(
            "These are earthquake catalog events, not necessarily damaging earthquakes. "
            "Historical catalog completeness varies by time, location, and magnitude."
        )
    else:
        st.info(
            "No timeline records loaded in this session yet. Choose a date range "
            "and select 'Load earthquake records from USGS'."
        )

    section_title("Download project data files")
    st.write("Files are stored in the project's `data` folder.")
    for path in [SAR_FILE, DAMAGE_FILE, ZONES_FILE]:
        mime = (
            "application/geo+json"
            if path.suffix == ".geojson"
            else "text/csv"
        )
        st.download_button(
            f"Download {path.name}",
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            key=f"download_{path.name}",
        )

    section_title("Data sources and limitations")
    st.markdown("""
    - **Earthquake catalog:** USGS FDSN Event API.
    - **SAR observations:** original provider/product links recorded in `sar_observations.csv`.
    - **Affected-area boundaries:** verified GeoJSON from a documented source.
    - **Damage statistics:** sourced assessment values entered in `damage_assessments.csv`.
    - **Ground displacement:** processed and validated InSAR measurements only.

    A nearby epicenter does not automatically mean an area was damaged.
    Magnitude alone is not used to assign impact severity. SAR acquisition alone
    does not establish ground displacement. Historical USGS catalog completeness
    varies, and very large date ranges can take longer to retrieve.
    """)


# -------------------- OPTIONAL REPORT DIALOG --------------------
if st.session_state.get("open_report_dialog"):
    selected_id = st.session_state.get("report_event_id", "")

    @st.dialog("Earthquake and affected-area report", width="large")
    def show_report():
        render_event_report(selected_id)
        if st.button("Close report"):
            st.session_state["open_report_dialog"] = False
            st.rerun()

    show_report()

st.divider()
st.caption("Dancing with the SARs | Japan Earthquake Impact & Satellite Monitoring")
