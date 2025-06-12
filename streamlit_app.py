import streamlit as st
import pandas as pd
import yfinance as yf
import re
import plotly.graph_objects as go
from datetime import datetime
from datetime import timedelta
from sqlalchemy import create_engine
import base64
from itertools import combinations
import swisseph as swe
import hashlib

DATABASE_URL = "postgresql://numeroniq-db_owner:npg_EWIGjD91LKxP@ep-muddy-boat-a15emu03-pooler.ap-southeast-1.aws.neon.tech/numeroniq-db?sslmode=require"
engine = create_engine(DATABASE_URL)

# Utility function to hash passwords
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Format: username: hashed_password
USER_CREDENTIALS = {
    "admin": hash_password("admin123"),
    "vin": hash_password("vin69"),
    "transleads": hash_password("leads27"),
}

# Check login status
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login():
    st.title("🔐 Secure Access")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == hash_password(password):
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

if not st.session_state.authenticated:
    login()
    st.stop()

st.set_page_config(page_title="Numeroniq", layout="wide")

# Inject CSS and JS to disable text selection and right-click
st.markdown("""
    <style>
    * {
        -webkit-user-select: none !important;
        -moz-user-select: none !important;
        -ms-user-select: none !important;
        user-select: none !important;
    }

    /* Specifically target tables */
    div[data-testid="stTable"] {
        -webkit-user-select: none !important;
        -moz-user-select: none !important;
        -ms-user-select: none !important;
        user-select: none !important;
    }

    /* Also target the scrollable dataframe area */
    .css-1wmy9hl, .css-1xarl3l {
        user-select: none !important;
    }
    </style>

    <script>
    document.addEventListener('contextmenu', event => event.preventDefault());
    </script>
    """, unsafe_allow_html=True)

# Disable right click with JavaScript
st.markdown("""
    <script>
    document.addEventListener('contextmenu', event => event.preventDefault());
    </script>
    """, unsafe_allow_html=True)

st.markdown("""
    <style>
        .stApp {
            background: radial-gradient(circle at top left, #e6cbb6, #fde6ef, #dcf7fc, #c2f0f7);
        }
        .block-container {
            background: radial-gradient(circle at top left, #e6cbb6, #fde6ef, #dcf7fc, #c2f0f7);
            padding: 2rem;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# Define custom CSS for the table background
custom_css = """
<style>
.scroll-table {
    overflow-x: auto;
    max-height: 500px;
    border: 1px solid #ccc;
}

.scroll-table table {
    width: 100%;
    border-collapse: collapse;
    background-color: #f0f8ff; /* Light blue background */
}

.scroll-table th, .scroll-table td {
    padding: 8px;
    border: 1px solid #ddd;
    text-align: left;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

def render_scrollable_table_with_arrow(df):
    html_rows = []
    for label, row in df.iterrows():
        is_arrow = row.get("→", "") == "◀️"
        row_id = ' id="match"' if is_arrow else ''
        level_val = row["Level"]
        arrow_val = row["→"]
        html_rows.append(f"<tr{row_id}><td>{arrow_val}</td><td>{label}</td><td>{level_val}</td></tr>")

    html_table = f"""
    <div class="scroll-table" id="scroll-container" style="max-height: 500px; overflow-y: auto;">
        <table>
            <thead><tr><th></th><th>Label</th><th>Level</th></tr></thead>
            <tbody>
                {''.join(html_rows)}
            </tbody>
        </table>
    </div>

    <script>
        setTimeout(function() {{
            var matchRow = document.getElementById("match");
            if (matchRow) {{
                matchRow.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
            }}
        }}, 300);
    </script>
    """

    st.markdown(html_table, unsafe_allow_html=True)

@st.cache_data
def load_stock_data():
    query = "SELECT * FROM companies"
    df = pd.read_sql(query, engine)

    # These are case-sensitive and must exactly match your DB column names
    date_cols = ["NSE LISTING DATE", "BSE LISTING DATE", "DATE OF INCORPORATION"]

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')  # convert only if column exists
        else:
            st.warning(f"Column '{col}' not found in stock data!")

    return df

def highlight_range_levels(row):
    label = str(row.name).lower()  # Access label from the index
    if "high" in label and "sp" not in label:
        return ['background-color: #87bd74'] * len(row)  # green
    elif "low" in label and "sp" not in label:
        return ['background-color: #e7191f'] * len(row)  # red
    elif "midpoint" in label or (label.startswith("mp") and "+" not in label and "-" not in label):
        return ['background-color: #fff858'] * len(row)  # yellow
    else:
        return ['background-color: #d9d9d9'] * len(row)  # gray

@st.cache_data(ttl=3600)
def load_excel_data(file):
    index_name_map = {
        "nifty.xlsx": "Nifty",
        "banknifty.xlsx": "BankNifty"
    }

    index_name = index_name_map.get(file)
    if not index_name:
        st.error(f"Unsupported file: {file}")
        return pd.DataFrame()

    query = """
        SELECT "Date", "Open", "High", "Low", "Close", "Vol(in M)"
        FROM ohlc_index
        WHERE index_name = %s
        ORDER BY "Date"
    """
    df = pd.read_sql(query, engine, params=(index_name,))
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns={"Vol(in M)": "Volume"}, inplace=True)
    df.set_index('Date', inplace=True)
    return df

# Load numerology data
@st.cache_data
def load_numerology_data():
    query = "SELECT * FROM numerology ORDER BY date"
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    return df

# Load moon data
@st.cache_data
def load_moon_data():
    query = 'SELECT * FROM moon_phases ORDER BY "Date"'
    df = pd.read_sql(query, engine)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.rename(columns={"a_or_p": "A/P", "paksh": "Paksh"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce') 
    return df

# Load Mercury data
@st.cache_data
def load_mercury_data():
    query = 'SELECT * FROM mercury_phases ORDER BY "Date"'
    df = pd.read_sql(query, engine)
    df.rename(columns={"dr": "D/R"}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    return df

# Load Combust data
@st.cache_data
def load_combust_data():
    query = 'SELECT * FROM mercury_combust ORDER BY "Start Date"'
    df = pd.read_sql(query, engine)
    df['Start Date'] = pd.to_datetime(df['Start Date'], dayfirst=True, errors='coerce')
    df['End Date'] = pd.to_datetime(df['End Date'], dayfirst=True, errors='coerce')
    return df

# Load Mercury Ingress data
@st.cache_data
def load_mercury_ingress():
    query = "SELECT * FROM mercury_ingress ORDER BY date"
    df = pd.read_sql(query, engine)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    return df

def highlight_mercury_ingress_rows(row, ingress_dates):
    d = row['Date'].date() if isinstance(row['Date'], pd.Timestamp) else None
    if d in ingress_dates:
        return ['background-color: #ffdd57'] * len(row)  # Light yellow
    return [''] * len(row)

# Load Panchak Data
@st.cache_data
def load_panchak_data():
    query = 'SELECT * FROM panchak_periods ORDER BY "Start Date"'
    df = pd.read_sql(query, engine)
    df['Start Date'] = pd.to_datetime(df['Start Date'], dayfirst=True, errors='coerce')
    df['End Date'] = pd.to_datetime(df['End Date'], dayfirst=True, errors='coerce')
    return df

panchak_df = load_panchak_data()

def highlight_moon_and_ingress_rows(row, amavasya_dates, poornima_dates, ingress_dates):
    d = row['Date'].date() if isinstance(row['Date'], pd.Timestamp) else None
    if d in amavasya_dates:
        return ['background-color: #ffcccc'] * len(row)  # Light red
    elif d in poornima_dates:
        return ['background-color: #ccf2ff'] * len(row)  # Sky blue
    elif d in ingress_dates:
        return ['background-color: #ffdd57'] * len(row)  # Light yellow
    return [''] * len(row)

def calculate_destiny_number(date_obj):
    if pd.isnull(date_obj):
        return None, None
    digits = [int(ch) for ch in date_obj.strftime('%Y%m%d')]
    total = sum(digits)
    reduced = reduce_to_single_digit(total)
    return total, reduced

def get_stock_data(ticker, start_date, end_date):
    """
    Fetch historical stock data using yfinance.
    """
    stock_data = yf.download(ticker, start=start_date, end=end_date, multi_level_index = False)
    return stock_data

def get_combined_index_data(index_name, start_date, end_date):
    import yfinance as yf
    full_range = pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1))

    ticker = "^NSEI" if index_name == "Nifty" else "^NSEBANK"

    # Step 1: Fetch from PostgreSQL
    query = '''
        SELECT "Date", "Open", "High", "Low", "Close", "Vol(in M)"
        FROM ohlc_index
        WHERE index_name = %s AND "Date" BETWEEN %s AND %s
        ORDER BY "Date"
    '''
    df = pd.read_sql(query, engine, params=(index_name, start_date, end_date))
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()

    # ✅ Step 2: Aggregate by Date
    df = df.groupby(df.index).agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Vol(in M)": "sum"
    })

    # Step 3: Check for missing dates and fetch from Yahoo
    missing_dates = full_range.difference(df.index)
    if not missing_dates.empty:
        fetch_start = missing_dates.min()
        fetch_end = end_date + pd.Timedelta(days=1)

        yf_data = yf.download(ticker, start=fetch_start, end=fetch_end, progress=False, multi_level_index=False)
        if not yf_data.empty:
            append_df = yf_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            append_df.reset_index(inplace=True)
            append_df.rename(columns={"Date": "Date", "Volume": "Vol(in M)"}, inplace=True)
            append_df["index_name"] = index_name
            append_df["Vol(in M)"] = append_df["Vol(in M)"] / 1_000_000
            append_df.drop_duplicates(subset=["Date", "index_name"], inplace=True)
            append_df["Date"] = pd.to_datetime(append_df["Date"]).dt.normalize()
            unique_dates = append_df["Date"].unique()

            if len(unique_dates) > 0:
                date_tuple = tuple(pd.to_datetime(unique_dates).tolist())
                placeholder = "(" + ",".join(["%s"] * len(date_tuple)) + ")"
                query = f'''
                    SELECT "Date"
                    FROM ohlc_index
                    WHERE index_name = %s AND "Date" IN {placeholder}
                '''
                params = (index_name, *date_tuple)
                existing_df = pd.read_sql(query, engine, params=params)
                existing_df["Date"] = pd.to_datetime(existing_df["Date"], errors="coerce")
                existing_dates = existing_df["Date"].dt.normalize()

                append_df = append_df[~append_df["Date"].isin(existing_dates)]

            if not append_df.empty:
                append_df["Date"] = pd.to_datetime(append_df["Date"]).dt.normalize()
                new_keys = list(zip(append_df["Date"], append_df["index_name"]))
                placeholders = ",".join(["(%s, %s)"] * len(new_keys))
                flat_params = [item for tup in new_keys for item in tup]

                existing_query = f'''
                    SELECT "Date", index_name
                    FROM ohlc_index
                    WHERE (Date, index_name) IN ({placeholders})
                '''
                existing_df = pd.read_sql(existing_query, engine, params=flat_params)
                existing_keys = set(zip(existing_df["Date"], existing_df["index_name"]))
                append_df = append_df[~append_df.apply(lambda x: (x["Date"], x["index_name"]) in existing_keys, axis=1)]

                if not append_df.empty:
                    append_df.to_sql("ohlc_index", engine, if_exists="append", index=False)

    # ✅ FINAL: Always return reindexed dataframe
    return df.reindex(full_range)

def get_index_ohlc(index_name, ticker, start_date, end_date):
    import yfinance as yf
    full_range = pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1))

    query = '''
        SELECT "Date", "Open", "High", "Low", "Close", "Vol(in M)"
        FROM ohlc_index
        WHERE index_name = %s AND "Date" BETWEEN %s AND %s
        ORDER BY "Date"
    '''
    df = pd.read_sql(query, engine, params=(index_name, start_date, end_date))
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df.set_index('Date', inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()

    missing_dates = full_range.difference(df.index)

    if not missing_dates.empty:
        fetch_start = missing_dates.min()
        fetch_end = end_date + pd.Timedelta(days=1)
        yf_data = yf.download(ticker, start=fetch_start, end=fetch_end, progress=False, multi_level_index=False)

        if not yf_data.empty:
            append_df = yf_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
            append_df.reset_index(inplace=True)
            append_df.rename(columns={'Date': 'Date', 'Volume': 'Vol(in M)'}, inplace=True)
            append_df['index_name'] = index_name
            append_df['Vol(in M)'] = append_df['Vol(in M)'] / 1_000_000
            append_df['Date'] = pd.to_datetime(append_df['Date']).dt.normalize()

            # Fetch existing keys to avoid inserting duplicates
            existing_query = '''
                SELECT "Date" FROM ohlc_index WHERE index_name = %s AND "Date" IN ({})
            '''
            placeholders = ",".join(["%s"] * len(append_df))
            existing_dates = pd.read_sql(
                existing_query.format(placeholders),
                engine,
                params=(index_name, *append_df['Date'].tolist())
            )["Date"].dt.normalize()

            append_df = append_df[~append_df['Date'].isin(existing_dates)]

            if not append_df.empty:
                append_df.to_sql("ohlc_index", engine, if_exists="append", index=False, method="multi")

                append_df.set_index('Date', inplace=True)
                df = pd.concat([df, append_df])
                df = df[~df.index.duplicated(keep='last')]

    return df.reindex(full_range)

def plot_candlestick_chart(stock_data, vertical_lines=None):

    import plotly.graph_objects as go
    import pandas as pd
    import streamlit as st

    # ✅ Normalize index for consistent date comparison
    stock_data.index = pd.to_datetime(stock_data.index).normalize()


    """
    Generate and return a candlestick chart using Plotly,
    with optional vertical lines on specific dates.
    """
    fig = go.Figure(data=[go.Candlestick(
        x=stock_data.index,
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    
    stock_data.index = pd.to_datetime(stock_data.index).normalize()

    for date_str in vertical_lines:
        try:
            date_obj = pd.to_datetime(date_str).normalize()
            fig.add_vline(
                x=date_obj,
                line_width=2,
                line_dash="solid",
                line_color="black",
               
            )
        except Exception as e:
            print(f"Could not plot vertical line for {date_str}: {e}")


    
    fig.update_layout(
        title="Candlestick chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )
    
    return fig

from collections import defaultdict

# Load data
stock_df = load_stock_data()
numerology_df = load_numerology_data()

# === Chaldean Numerology Setup ===
chaldean_map = {
    1: 'A I J Q Y',
    2: 'B K R',
    3: 'C G L S',
    4: 'D M T',
    5: 'E H N X',
    6: 'U V W',
    7: 'O Z',
    8: 'F P'
}

# === Pythagorean Numerology Setup ===
pythagorean_map = {
    1: 'A J S',
    2: 'B K T',
    3: 'C L U',
    4: 'D M V',
    5: 'E N W',
    6: 'F O X',
    7: 'G P Y',
    8: 'H Q Z',
    9: 'I R'
}

char_to_num = {letter: num for num, letters in chaldean_map.items() for letter in letters.split()}

pythagorean_char_to_num = {
    letter: num for num, letters in pythagorean_map.items() for letter in letters.split()
}

def calculate_pythagorean_numerology(name):
    clean_name = re.sub(r'[^A-Za-z ]+', '', name)
    words = re.findall(r'\b[A-Za-z]+\b', clean_name)

    word_parts = []
    original_values = []

    for word in words:
        word_val = sum(pythagorean_char_to_num.get(char.upper(), 0) for char in word)
        reduced_val = reduce_to_single_digit(word_val)
        word_parts.append(f"{word_val}({reduced_val})")
        original_values.append(word_val)

    if not original_values:
        return None, None

    total_sum = sum(original_values)
    final_reduced = reduce_to_single_digit(total_sum)
    equation = f"{' + '.join(word_parts)} = {total_sum}({final_reduced})"
    return final_reduced, equation

def get_word_value(word):
    return sum(char_to_num.get(char.upper(), 0) for char in word)

def reduce_to_single_digit(n):
    while n > 9:
        n = sum(int(d) for d in str(n))
    return n

def calculate_numerology(name):
    clean_name = re.sub(r'[^A-Za-z ]+', '', name)
    words = re.findall(r'\b[A-Za-z]+\b', clean_name)

    word_parts = []
    original_values = []

    for word in words:
        word_val = get_word_value(word)
        reduced_val = reduce_to_single_digit(word_val)
        word_parts.append(f"{word_val}({reduced_val})")
        original_values.append(word_val)

    if not original_values:
        return None, None

    total_sum = sum(original_values)
    final_reduced = reduce_to_single_digit(total_sum)

    equation = f"{' + '.join(word_parts)} = {total_sum}({final_reduced})"
    return final_reduced, equation

def calculate_chaldean_isin_numerology(isin):
    """
    Calculates Chaldean numerology for ISIN.
    Returns total and reduced value in format: 34(7)
    """
    if not isin:
        return None, None

    total = 0
    for char in isin:
        if char.isdigit():
            total += int(char)
        elif char.upper() in char_to_num:
            total += char_to_num[char.upper()]

    if total == 0:
        return None, None

    reduced = reduce_to_single_digit(total)
    return reduced, f"{total}({reduced})"

def calculate_pythagorean_isin_numerology(isin):
    """
    Calculates Pythagorean numerology for ISIN.
    Returns total and reduced value in format: 34(7)
    """
    if not isin:
        return None, None

    total = 0
    for char in isin:
        if char.isdigit():
            total += int(char)
        elif char.upper() in pythagorean_char_to_num:
            total += pythagorean_char_to_num[char.upper()]

    if total == 0:
        return None, None

    reduced = reduce_to_single_digit(total)
    return reduced, f"{total}({reduced})"

def get_day_type_conjunction(jd, planets, planet_rank):
    flag = swe.FLG_SIDEREAL | swe.FLG_SPEED
    planet_data = {}

    for name, pid in planets.items():
        lon, speed = swe.calc_ut(jd, pid, flag)[0][0:2]
        if name == "Ketu":
            lon = (swe.calc_ut(jd, swe.TRUE_NODE, flag)[0][0] + 180) % 360
            speed = -speed
        planet_data[name] = {"deg": round(lon, 2), "speed": round(speed, 4)}

    for p1, p2 in combinations(planet_data.keys(), 2):
        r1 = planet_rank.get(p1, 999)
        r2 = planet_rank.get(p2, 999)
        fast, slow = (p1, p2) if r1 < r2 else (p2, p1)

        d1 = planet_data[fast]["deg"]
        d2 = planet_data[slow]["deg"]
        diff = (d1 - d2 + 360) % 360
        if diff > 180:
            diff -= 360
        diff = round(diff, 2)

        if abs(diff) <= 1.0:
            return "Red Day" if diff < 0 else "Green Day"

    return "-"

aspect_config = {
    "Sun→Ketu": {"from": "Sun", "to": "Ketu", "angles": [0, 90, 120]},
    "Venus→Ketu": {"from": "Venus", "to": "Ketu", "angles": [0, 120]},
}

planet_map = {
    'Sun': swe.SUN, 'Moon': swe.MOON, 'Mercury': swe.MERCURY, 'Venus': swe.VENUS,
    'Mars': swe.MARS, 'Jupiter': swe.JUPITER, 'Saturn': swe.SATURN,
    'Rahu': swe.MEAN_NODE, 'Ketu': swe.TRUE_NODE
}

def angular_diff(from_deg, to_deg):
    return round((to_deg - from_deg) % 360, 2)

def check_aspects(from_deg, to_deg, angles, label):
    for angle in angles:
        if abs(angular_diff(from_deg, to_deg) - angle) <= 0.5:
            return f"{label} ≈ {angle}°"
    return None

def get_d9_longitude(lon):
    sign_index = int(lon // 30)
    pos_in_sign = lon % 30
    navamsa_index = int(pos_in_sign // (30 / 9))
    if sign_index in [0, 3, 6, 9]: start = sign_index
    elif sign_index in [1, 4, 7, 10]: start = (sign_index + 8) % 12
    else: start = (sign_index + 4) % 12
    d9_sign_index = (start + navamsa_index) % 12
    deg_in_navamsa = pos_in_sign % (30 / 9)
    return d9_sign_index * 30 + deg_in_navamsa * 9

def check_mm_aspects(from_deg, to_deg):
    angles = [0, 90, 180]
    matched = []
    diff1 = angular_diff(from_deg, to_deg)
    diff2 = angular_diff(to_deg, from_deg)
    for angle in angles:
        if abs(diff1 - angle) <= 1:
            matched.append(f"Moon→Mercury ≈ {angle}°")
        if abs(diff2 - angle) <= 1:
            matched.append(f"Mercury→Moon ≈ {angle}°")
    return ", ".join(matched) if matched else "0"

def get_d9_longitude(longitude_deg):
    sign_index = int(longitude_deg // 30)
    pos_in_sign = longitude_deg % 30
    navamsa_index = int(pos_in_sign // (30 / 9))
    if sign_index in [0, 3, 6, 9]:
        start = sign_index
    elif sign_index in [1, 4, 7, 10]:
        start = (sign_index + 8) % 12
    else:
        start = (sign_index + 4) % 12
    d9_sign_index = (start + navamsa_index) % 12
    deg_in_navamsa = pos_in_sign % (30 / 9)
    return d9_sign_index * 30 + deg_in_navamsa * 9

def classify_sign_type(sign_number):
        for k, v in sign_types.items():
            if sign_number in v:
                return k
        return "Unknown"

def get_planet_deg(jd, name):
    flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH
    lon = swe.calc_ut(jd, planets[name], flag)[0][0]
    if name == "Ketu":
        lon = (swe.calc_ut(jd, planets["Rahu"], flag)[0][0] + 180) % 360
    return lon

def get_day_type(jd):
    flag = swe.FLG_SIDEREAL | swe.FLG_SPEED
    data = {}
    for name in planets:
        lon, speed = swe.calc_ut(jd, planets[name], flag)[0][0:2]
        if name == "Ketu":
            lon = (swe.calc_ut(jd, planets["Rahu"], flag)[0][0] + 180) % 360
            speed = -speed
        data[name] = {"deg": lon, "speed": speed}

    for p1, p2 in combinations(data.keys(), 2):
        r1, r2 = planet_rank.get(p1, 999), planet_rank.get(p2, 999)
        fast, slow = (p1, p2) if r1 < r2 else (p2, p1)
        d1, d2 = data[fast]["deg"], data[slow]["deg"]
        diff = (d1 - d2 + 360) % 360
        if diff > 180:
            diff -= 360
        if abs(diff) <= 1:
            return "Red Day" if diff < 0 else "Green Day"
    return "-"

from itertools import combinations

def get_planet_data(jd, name, pid):
    flag = swe.FLG_SIDEREAL | swe.FLG_SPEED
    lon, speed = swe.calc_ut(jd, pid, flag)[0][0:2]
    if name == "Ketu":
        lon = (swe.calc_ut(jd, swe.TRUE_NODE, flag)[0][0] + 180) % 360
        speed = -speed
    return round(lon, 2), round(speed, 4)

def signed_diff(fast_deg, slow_deg):
    diff = (fast_deg - slow_deg + 360) % 360
    if diff > 180:
        diff -= 360
    return round(diff, 2)

nakshatras = [
        "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra", "Punarvasu",
        "Pushya", "Ashlesha", "Magha", "Purva Phalguni", "Uttara Phalguni", "Hasta",
        "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha",
        "Uttara Ashadha", "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada",
        "Uttara Bhadrapada", "Revati"
    ]

planets = {
    "Sun": swe.SUN,
    "Moon": swe.MOON,
    "Mars": swe.MARS,
    "Mercury": swe.MERCURY,
    "Jupiter": swe.JUPITER,
    "Venus": swe.VENUS,
    "Saturn": swe.SATURN,
    "Rahu": swe.TRUE_NODE,
    "Ketu": swe.TRUE_NODE,
    "Uranus": swe.URANUS,
    "Neptune": swe.NEPTUNE,
    "Pluto": swe.PLUTO
}

st.title("📊 Numeroniq")

st.html("""
<style>
[data-testid=stElementToolbarButton]:first-of-type {
    display: none;
}
</style>
""")

# === Role Permissions ===
allowed_users_for_astro = ["admin", "vin"]

# === Section Groups
numerology_sections = [
    "Company Overview", 
    "Numerology Date Filter", 
    "Filter by Sector/Symbol", 
    "Filter by Numerology",
    "Name Numerology", 
    "View Nifty/BankNifty OHLC", 
    "Equinox",
    "Moon",
    "Mercury",
    "Mercury Combust",
    "Sun Number Dates",
    "Panchak",
    "Range",
    "Daily Report"
]

astro_sections = [
    "Navamasa",
    "Planetary Conjunctions",
    "Planetary Report",
    "Moon–Mercury Aspects",
    "Planetary Aspects",
    "Swapt Nadi Chakra",
    "Planetary Ingress",
    "AOT Monthly Calendar"
]

bayers_rule_sections = [
    "Rule 1: Speed of Mercury",
    "Rule 2: Mars–Mercury 59-Min Rule",
    "Rule 3: 161° Mars–Mercury Bullish Signal",
    "test Mars–Mercury 59-Min Rule",
    "Rule 4: Mercury Retrograde Echo",
    "Rule 13: Neptune Log Distance",
    "Monthly OHLC Viewer",
    "Rule 13: All Planets Log Distance",
    "Rule 23: Saturn Latitude Differential",
    "Rule 27: Mercury's Speed Triggers (59′ and 1°58′)",
]

# === Sidebar Title
st.sidebar.title("📊 Navigation")

# === Setup session state tracker
if "active_section" not in st.session_state:
    st.session_state.active_section = "numerology"

# === Functions to set active section
def set_section(name):
    st.session_state.active_section = name

# === Sidebar Options
with st.sidebar.expander("🧮 Numerology", expanded=True):
    selected_numerology = st.radio("Select Numerology Mode:", numerology_sections, 
                                   key="numerology_radio", on_change=set_section, args=("numerology",))

if st.session_state.username in allowed_users_for_astro:
    with st.sidebar.expander("🔭 Astrology"):
        selected_astro = st.radio("Select Astro Mode:", astro_sections, 
                                  key="astro_radio", on_change=set_section, args=("astro",))

    with st.sidebar.expander("📜 Bayer’s Rules"):
        selected_bayer = st.radio("Select Rule:", bayers_rule_sections, 
                                  key="bayer_radio", on_change=set_section, args=("bayer",))

# === Choose based on active section
if st.session_state.active_section == "numerology":
    filter_mode = selected_numerology
elif st.session_state.active_section == "astro":
    filter_mode = selected_astro
elif st.session_state.active_section == "bayer":
    filter_mode = selected_bayer
else:
    filter_mode = None



if filter_mode == "Filter by Sector/Symbol":
    # === Sector Filter ===
    sectors = stock_df['SECTOR'].dropna().unique()
    selected_sector = st.selectbox("Select Sector:", ["All"] + sorted(sectors))

    show_all_in_sector = st.checkbox("Show all companies in this sector", value=True)

    if selected_sector != "All":
        sector_filtered_df = stock_df[stock_df['SECTOR'] == selected_sector]
    else:
        sector_filtered_df = stock_df.copy()

    if not show_all_in_sector:
        filtered_symbols = sector_filtered_df['Symbol'].dropna().unique()
        selected_symbol = st.selectbox("Select Symbol:", sorted(filtered_symbols))
        company_data = sector_filtered_df[sector_filtered_df['Symbol'] == selected_symbol]
    else:
        company_data = sector_filtered_df

    # === Display Company Data ===
    if not company_data.empty:
        st.write("### Company Info")
        display_cols = company_data.drop(columns=['Series', 'Pythagoras Eqn Symbol' , 'Chaldean Eqn Symbol', 'Chaldean Eqn Company name without Ltd',
                                                'Pythagoras Eqn Company name without Ltd', 'Pythagoras Eqn Company name with Ltd',
                                                'Chaldean Eqn Company name with Ltd' ,'Company Name', 'Pythagoras Eqn ISIN without IN',
                                                'Chaldean Eqn ISIN without IN', 'Pythagoras Eqn ISIN with IN', 'Chaldean Eqn ISIN with IN', 
                                                'ISIN Code', 'IPO TIMING ON NSE'], errors='ignore')
        for col in ['NSE LISTING DATE', 'BSE LISTING DATE', 'DATE OF INCORPORATION']:
            if col in display_cols.columns:
                display_cols[col] = display_cols[col].dt.strftime('%Y-%m-%d')
        # Convert DataFrame to HTML table
        html_table = display_cols.to_html(index=False, escape=False)

        # Embed HTML table in a scrollable container
        st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

        # Date choice: Single date or All Dates (NSE, BSE, Incorporation)
        date_choice = st.radio("Select Listing Date Source for Numerology:", 
                               ("NSE LISTING DATE", "BSE LISTING DATE", "DATE OF INCORPORATION", "By All Dates"))

        if date_choice == "By All Dates":
            # If "By All Dates" is selected, we show three rows for each symbol
            combined_numerology = []
            for idx, row in company_data.iterrows():
                for date_column in ['NSE LISTING DATE', 'BSE LISTING DATE', 'DATE OF INCORPORATION']:
                    date_val = row[date_column]
                    if pd.notnull(date_val):
                        numerology_row = numerology_df[numerology_df['date'] == pd.to_datetime(date_val)]
                        if not numerology_row.empty:
                            temp = numerology_row.copy()
                            temp['Symbol'] = row['Symbol']
                            temp['Date Type'] = date_column
                            temp['NSE age'] = row['NSE age']  # Add NSE AGE column
                            temp['BSE age'] = row['BSE age']  # Add BSE AGE column
                            temp['DOC age'] = row['DOC age']  # Add DOC AGE column
                            combined_numerology.append(temp)

            if combined_numerology:
                st.write(f"### Numerology Data for All Companies in {selected_sector} (Using All Dates)")
                all_numerology_df = pd.concat(combined_numerology, ignore_index=True)

                # Reorder columns: Symbol, Date Type, Date Used first
                cols = all_numerology_df.columns.tolist()
                cols = ['Symbol', 'Date Type', 'NSE age', 'BSE age', 'DOC age'] + [col for col in all_numerology_df.columns if col not in ['Symbol', 'Date Type', 'NSE age', 'BSE age', 'DOC age']]
                all_numerology_df = all_numerology_df[cols]

                # Convert DataFrame to HTML table
                html_table = all_numerology_df.to_html(index=False, escape=False)

                # Embed HTML table in a scrollable container
                st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

            else:
                st.warning("No numerology data found for selected dates across these companies.")
        
        else:
            # Handle the case where a single date type (NSE/BSE/Inc) is selected
            if len(company_data) == 1:
                # Single company selected — choose one date
                listing_date = pd.to_datetime(company_data[date_choice].values[0])
                if pd.notnull(listing_date):
                    st.write(f"### Numerology Data for {listing_date.strftime('%Y-%m-%d')}")
                    matched_numerology = numerology_df[numerology_df['date'] == listing_date].copy()
                    if not matched_numerology.empty:
                        matched_numerology['Symbol'] = company_data['Symbol'].values[0]
                        matched_numerology['Date Type'] = date_choice

                        if date_choice == "NSE LISTING DATE":
                            matched_numerology['NSE age'] = company_data['NSE age'].values[0]
                        elif date_choice == "BSE LISTING DATE":
                            matched_numerology['BSE age'] = company_data['BSE age'].values[0]
                        elif date_choice == "DATE OF INCORPORATION":
                            matched_numerology['DOC age'] = company_data['DOC age'].values[0]
                            
                        # Convert DataFrame to HTML table
                        html_table = matched_numerology.to_html(index=False, escape=False)

                        # Embed HTML table in a scrollable container
                        st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No numerology data found for this date.")
                else:
                    st.warning(f"{date_choice} is not available for this company.")

            elif show_all_in_sector:
                # Multiple companies shown (whole sector) — apply one date field to all
                combined_numerology = []

                for idx, row in company_data.iterrows():
                    date_val = row[date_choice]
                    if pd.notnull(date_val):
                        numerology_row = numerology_df[numerology_df['date'] == pd.to_datetime(date_val)]
                        if not numerology_row.empty:
                            temp = numerology_row.copy()
                            temp['Symbol'] = row['Symbol']
                            temp['Date Type'] = date_choice

                            if date_choice == "NSE LISTING DATE":
                                temp['NSE age'] = row['NSE age']
                            elif date_choice == "BSE LISTING DATE":
                                temp['BSE age'] = row['BSE age']
                            elif date_choice == "DATE OF INCORPORATION":
                                temp['DOC age'] = row['DOC age']

                            combined_numerology.append(temp)

                if combined_numerology:
                    st.write(f"### Numerology Data for All Companies in {selected_sector} (Using {date_choice})")
                    all_numerology_df = pd.concat(combined_numerology, ignore_index=True)

                    # Move Symbol, Date Type and relevant Age column to front
                    cols_to_front = ['Symbol', 'Date Type']
                    if date_choice == "NSE LISTING DATE":
                        cols_to_front.append('NSE age')
                    elif date_choice == "BSE LISTING DATE":
                        cols_to_front.append('BSE age')
                    elif date_choice == "DATE OF INCORPORATION":
                        cols_to_front.append('DOC age')

                    all_cols = cols_to_front + [col for col in all_numerology_df.columns if col not in cols_to_front]
                    
                    
                    # Convert DataFrame to HTML table
                    html_table = all_numerology_df[all_cols].to_html(index=False, escape=False)

                    # Embed HTML table in a scrollable container
                    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

                else:
                    st.warning("No numerology data found for selected date field across these companies.")

            else:
                # Multiple companies manually selected but show_all_in_sector is False
                st.info("Select a single symbol (uncheck the box) to see numerology data.")

    else:
        st.warning("No matching data found.")

elif filter_mode == "Numerology Date Filter":
    st.subheader("📅 Filter Numerology Data by Date")

    # Parse and clean the date column
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], dayfirst=True, errors='coerce')
    numerology_df = numerology_df.dropna(subset=['date'])

    # Get year range
    min_year = numerology_df['date'].dt.year.min()
    max_year = numerology_df['date'].dt.year.max()

    # Build decade periods
    periods = []
    for start in range((min_year // 10) * 10, (max_year // 10 + 1) * 10, 10):
        end = start + 9
        periods.append(f"{start}-{end}")

    # Step 1: Select Time Period
    selected_period = st.selectbox("Select Time Period", periods)
    start_year, end_year = map(int, selected_period.split('-'))
    period_start = pd.to_datetime(f"{start_year}-01-01")
    period_end = pd.to_datetime(f"{end_year}-12-31")

    # Step 2: Let user refine with date pickers within the selected period
    date_range = numerology_df[(numerology_df['date'] >= period_start) & (numerology_df['date'] <= period_end)]
    if date_range.empty:
        st.warning("No data available in this period.")
        st.stop()

    min_date = date_range['date'].min().date()
    max_date = date_range['date'].max().date()

    start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.date_input("End Date", value=max_date, min_value=start_date, max_value=max_date)

    # Step 3: Filter data based on custom date range within the period
    filtered = date_range[
        (date_range['date'] >= pd.to_datetime(start_date)) &
        (date_range['date'] <= pd.to_datetime(end_date))
    ]

    # Step 4: Additional filters
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        bn_filter = st.selectbox("BN", options=['All'] + numerology_df['BN'].dropna().unique().tolist(), index=0)
    with col2:
        dn_filter = st.selectbox("DN (Formatted)", options=['All'] + numerology_df['DN (Formatted)'].dropna().unique().tolist(), index=0)
    with col3:
        sn_filter = st.selectbox("SN", options=['All'] + numerology_df['SN'].dropna().unique().tolist(), index=0)
    with col4:
        hp_filter = st.selectbox("HP", options=['All'] + numerology_df['HP'].dropna().unique().tolist(), index=0)
    with col5:
        day_number_filter = st.selectbox("Day Number", options=['All'] + numerology_df['Day Number'].dropna().unique().tolist(), index=0)

    if bn_filter != 'All':
        filtered = filtered[filtered['BN'] == bn_filter]
    if dn_filter != 'All':
        filtered = filtered[filtered['DN (Formatted)'] == dn_filter]
    if sn_filter != 'All':
        filtered = filtered[filtered['SN'] == sn_filter]
    if hp_filter != 'All':
        filtered = filtered[filtered['HP'] == hp_filter]
    if day_number_filter != 'All':
        filtered = filtered[filtered['Day Number'] == day_number_filter]

    # Display filtered data
    st.write(f"Showing {len(filtered)} records from **{start_date}** to **{end_date}**")

    html_table = filtered.to_html(index=False, escape=False)
    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Filter by Numerology":
    st.markdown("### 🔢 Filter by Numerology Values (Live & Horizontal Layout)")

    # Step 1: Ask how to match date: NSE/BSE/Inc
    date_match_option = st.selectbox("Select Date Type to Match Companies:", 
                                 ["NSE LISTING DATE", "BSE LISTING DATE", "DATE OF INCORPORATION"])

    # Step 2: Start with full numerology data
    filtered_numerology = numerology_df.copy()
    # Calculate DN dynamically
    dn_values = filtered_numerology['date'].apply(calculate_destiny_number)
    filtered_numerology['DN Raw'] = dn_values.apply(lambda x: x[0])
    filtered_numerology['DN'] = dn_values.apply(lambda x: x[1])
    filtered_numerology['DN (Formatted)'] = filtered_numerology.apply(lambda row: f"({row['DN Raw']}){row['DN']}" if pd.notnull(row['DN Raw']) else None, axis=1)


    # Prepare layout
    col1, col2, col3, col4, col5 = st.columns(5)

    # === BN Filter ===
    with col1:
        bn_options = ["All"] + sorted(numerology_df['BN'].dropna().unique())
        selected_bn = st.selectbox("BN", bn_options)
        if selected_bn != "All":
            filtered_numerology = filtered_numerology[filtered_numerology['BN'] == selected_bn]

    # === DN Filter ===
    with col2:
        dn_options = ["All"] + sorted(filtered_numerology['DN'].dropna().unique())
        selected_dn = st.selectbox("DN", dn_options)
        if selected_dn != "All":
            filtered_numerology = filtered_numerology[filtered_numerology['DN'] == selected_dn]

    # === SN Filter ===
    with col3:
        sn_options = ["All"] + sorted(filtered_numerology['SN'].dropna().unique())
        selected_sn = st.selectbox("SN", sn_options)
        if selected_sn != "All":
            filtered_numerology = filtered_numerology[filtered_numerology['SN'] == selected_sn]

    # === HP Filter ===
    with col4:
        hp_options = ["All"] + sorted(filtered_numerology['HP'].dropna().unique())
        selected_hp = st.selectbox("HP", hp_options)
        if selected_hp != "All":
            filtered_numerology = filtered_numerology[filtered_numerology['HP'] == selected_hp]

    # === Day Number Filter ===
    with col5:
        dayn_options = ["All"] + sorted(filtered_numerology['Day Number'].dropna().unique())
        selected_dayn = st.selectbox("Day Number", dayn_options)
        if selected_dayn != "All":
            filtered_numerology = filtered_numerology[filtered_numerology['Day Number'] == selected_dayn]


    # Create a mapping of dates to numerology rows (after filter)
    filtered_numerology_map = filtered_numerology.set_index('date')

    # Loop through stock_df and match per company by selected date
    matching_records = []

    for _, row in stock_df.iterrows():
        match_date = row.get(date_match_option)
        if pd.notnull(match_date) and match_date in filtered_numerology_map.index:
            numerology_match = filtered_numerology_map.loc[match_date]
        
            # Handle multiple matches (if any) from numerology_df
            if isinstance(numerology_match, pd.DataFrame):
                numerology_match = numerology_match.iloc[0]

            combined_row = row.to_dict()
            combined_row.update(numerology_match.to_dict())
            combined_row['Matching Date Source'] = date_match_option
            matching_records.append(combined_row)

    # Create final DataFrame
    matching_stocks = pd.DataFrame(matching_records)

    st.markdown("### 🎯 Matching Companies")

    if not matching_stocks.empty:

        display_cols = matching_stocks.drop(columns=['Series', 'Company Name', 'ISIN Code', 'IPO TIMING ON NSE'], errors='ignore')

        # Format for display
        for col in ['NSE LISTING DATE', 'BSE LISTING DATE', 'DATE OF INCORPORATION']:
            if col in display_cols.columns:
                display_cols[col] = display_cols[col].dt.strftime('%Y-%m-%d')

        # Optional: format date columns
        for col in ['NSE LISTING DATE', 'BSE LISTING DATE', 'DATE OF INCORPORATION']:
            if col in matching_stocks.columns:
                matching_stocks[col] = pd.to_datetime(matching_stocks[col], errors='coerce').dt.strftime('%Y-%m-%d')

        # Reorder to show date source
        cols_order = ['Symbol', date_match_option, 'BN', 'DN (Formatted)', 'SN', 'HP', 'Day Number'] + \
            [col for col in matching_stocks.columns if col not in ['Symbol', 'Matching Date Source', date_match_option, 'BN', 'DN', 'DN (Formatted)', 'SN', 'HP', 'Day Number']]


        # Convert DataFrame to HTML table
        html_table = matching_stocks.to_html(index=False, escape=False)

        # Embed HTML table in a scrollable container
        st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

    else:
        st.info("No companies found with matching numerology dates.")

elif filter_mode == "Name Numerology":
    st.subheader("🔢 Name Numerology")
    
    col1, col2, col3 = st.columns(3)

    with col1:
        use_ltd = st.radio(
            "Include 'Ltd' or 'Limited'?",
            ["Yes", "No"],
            index=1
        )

    with col2:
        use_is_prefix = st.radio(
            "Include 'IN' prefix in ISIN numerology?",
            ["Yes", "No"],
            index=0
        )

    with col3:
        numerology_system = st.radio(
            "Numerology System:",
           ["Chaldean", "Pythagoras", "Both"]
        )


    numerology_data = []

    for _, row in stock_df.iterrows():
        company_original = row['Company Name']
        symbol = str(row['Symbol'])
        isin_code = str(row['ISIN Code']) 

        # Remove 'Ltd' or 'Limited' if user chose "No"
        if use_ltd == "No":
            company_clean = re.sub(r'\b(Ltd|Limited)\b', '', company_original, flags=re.IGNORECASE).strip()
        else:
            company_clean = company_original

        entry = {
            'Symbol': row['Symbol'],
            'Company Name': company_original,
            'ISIN Code': isin_code,  # Add ISIN code for display
        }

        if numerology_system in ["Chaldean", "Both"]:
            ch_company_num, ch_company_eq = calculate_numerology(company_clean)
            ch_symbol_num, ch_symbol_eq = calculate_numerology(symbol)
            isin_to_use = isin_code if use_is_prefix == "Yes" else isin_code[2:]
            ch_isin_num, ch_isin_eq = calculate_chaldean_isin_numerology(isin_to_use)


            entry['Chaldean Eqn (Company Name)'] = ch_company_eq
            entry['Chaldean Eqn (Symbol)'] = ch_symbol_eq
            entry['Chaldean Eqn (ISIN Code)'] = ch_isin_eq

        if numerology_system in ["Pythagoras", "Both"]:
            py_company_num, py_company_eq = calculate_pythagorean_numerology(company_clean)
            py_symbol_num, py_symbol_eq = calculate_pythagorean_numerology(symbol)
            isin_to_use = isin_code if use_is_prefix == "Yes" else isin_code[2:]
            py_isin_num, py_isin_eq = calculate_pythagorean_isin_numerology(isin_to_use) 
            entry['Pythagoras Eqn (Company Name)'] = py_company_eq
            entry['Pythagoras Eqn (Symbol)'] = py_symbol_eq
            entry['Pythagoras Eqn (ISIN Code)'] = py_isin_eq 

        numerology_data.append(entry)


    numerology_df_display = pd.DataFrame(numerology_data)


    # === Filters ===
    col1, col2 = st.columns(2)

    with col1:
        company_filter = st.selectbox(
            "Select Company (or choose All)",
            options=["All"] + sorted(numerology_df_display['Company Name'].unique())
        )


    filtered_df = numerology_df_display.copy()

    if company_filter != "All":
        filtered_df = filtered_df[filtered_df['Company Name'] == company_filter]

    if numerology_system in ["Chaldean", "Both"]:
        col1, col2, col3 = st.columns(3)

        with col1:
            ch_company_totals = numerology_df_display['Chaldean Eqn (Company Name)'].dropna().apply(
                lambda eq: int(re.search(r'=\s*(\d+)\(', eq).group(1)) if re.search(r'=\s*(\d+)\(', eq) else None
            )
            selected_ch_company = st.selectbox("Chaldean Total (Company Name)", ["All"] + sorted(ch_company_totals.dropna().unique()))

        with col2:
            ch_symbol_totals = numerology_df_display['Chaldean Eqn (Symbol)'].dropna().apply(
                lambda eq: int(re.search(r'=\s*(\d+)\(', eq).group(1)) if re.search(r'=\s*(\d+)\(', eq) else None
            )
            selected_ch_symbol = st.selectbox("Chaldean Total (Symbol)", ["All"] + sorted(ch_symbol_totals.dropna().unique()))

        with col3:
            ch_isin_totals = numerology_df_display['Chaldean Eqn (ISIN Code)'].dropna().apply(
                lambda eq: int(re.search(r'=\s*(\d+)\(', eq).group(1)) if re.search(r'=\s*(\d+)\(', eq) else None
            )
            selected_ch_isin = st.selectbox("Chaldean Total (ISIN Code)", ["All"] + sorted(ch_isin_totals.dropna().unique()))

        if selected_ch_company != "All":
            filtered_df = filtered_df[
                filtered_df['Chaldean Eqn (Company Name)'].str.extract(r'=\s*(\d+)\(')[0].astype(float) == float(selected_ch_company)
            ]

        if selected_ch_symbol != "All":
            filtered_df = filtered_df[
                filtered_df['Chaldean Eqn (Symbol)'].str.extract(r'=\s*(\d+)\(')[0].astype(float) == float(selected_ch_symbol)
            ]

        if selected_ch_isin != "All":
            filtered_df = filtered_df[
                filtered_df['Chaldean Eqn (ISIN Code)'].str.extract(r'=\s*(\d+)\(')[0].astype(float) == float(selected_ch_isin)
            ]

    if numerology_system in ["Pythagoras", "Both"]:
        col1, col2, col3 = st.columns(3)

        with col1:
            py_company_totals = numerology_df_display['Pythagoras Eqn (Company Name)'].dropna().apply(
                lambda eq: int(re.search(r'=\s*(\d+)\(', eq).group(1)) if re.search(r'=\s*(\d+)\(', eq) else None
            )
            selected_py_company = st.selectbox("Pythagoras Total (Company Name)", ["All"] + sorted(py_company_totals.dropna().unique()))

        with col2:
            py_symbol_totals = numerology_df_display['Pythagoras Eqn (Symbol)'].dropna().apply(
                lambda eq: int(re.search(r'=\s*(\d+)\(', eq).group(1)) if re.search(r'=\s*(\d+)\(', eq) else None
            )
            selected_py_symbol = st.selectbox("Pythagoras Total (Symbol)", ["All"] + sorted(py_symbol_totals.dropna().unique()))

        with col2:
            py_isin_totals = numerology_df_display['Pythagoras Eqn (ISIN Code)'].dropna().apply(
                lambda eq: int(re.search(r'=\s*(\d+)\(', eq).group(1)) if re.search(r'=\s*(\d+)\(', eq) else None
            )
            selected_py_isin = st.selectbox("Pythagoras Total (ISIN Code)", ["All"] + sorted(py_isin_totals.dropna().unique()))


        if selected_py_company != "All":
            filtered_df = filtered_df[
                filtered_df['Pythagoras Eqn (Company Name)'].str.extract(r'=\s*(\d+)\(')[0].astype(float) == float(selected_py_company)
            ]

        if selected_py_symbol != "All":
            filtered_df = filtered_df[
                filtered_df['Pythagoras Eqn (Symbol)'].str.extract(r'=\s*(\d+)\(')[0].astype(float) == float(selected_py_symbol)
            ]

        if selected_py_isin != "All":
            filtered_df = filtered_df[
                filtered_df['Pythagoras Eqn (ISIN Code)'].str.extract(r'=\s*(\d+)\(')[0].astype(float) == float(selected_py_isin)
            ]


    
    # Convert DataFrame to HTML table
    html_table = filtered_df.to_html(index=False, escape=False)

    # Embed HTML table in a scrollable container
    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Company Overview":
    st.title("🏠 Company Overview")

    # Prepare searchable list for suggestions
    search_options = stock_df['Symbol'].dropna().tolist() + stock_df['Company Name'].dropna().tolist()
    search_options = sorted(set(search_options))

    user_input = st.selectbox("Search by Symbol or Company Name:", options=[""] + search_options)

    if user_input:
        # Case-insensitive match
        company_info = stock_df[
            (stock_df['Symbol'].str.lower() == user_input.lower()) |
            (stock_df['Company Name'].str.lower() == user_input.lower())
        ]

        if not company_info.empty:
            row = company_info.iloc[0]

            # --- Line 1: Sector, Sub-sector, ISIN Code ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Sector:** {row.get('SECTOR', 'N/A')}")
            with col2:
                st.markdown(f"**Sub-Sector:** {row.get('SUB SECTOR', 'N/A')}")
            with col3:
                st.markdown(f"**ISIN Code:** {row.get('ISIN Code', 'N/A')}")

            # --- Line 2: Dates ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Date of Incorporation:** {row['DATE OF INCORPORATION'].date() if pd.notnull(row['DATE OF INCORPORATION']) else 'N/A'}")
            with col2:
                st.markdown(f"**NSE Listing Date:** {row['NSE LISTING DATE'].date() if pd.notnull(row['NSE LISTING DATE']) else 'N/A'}")
            with col3:
                st.markdown(f"**BSE Listing Date:** {row['BSE LISTING DATE'].date() if pd.notnull(row['BSE LISTING DATE']) else 'N/A'}")

            # --- Line 3: Ages ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**DOC Age:** {row.get('DOC age', 'N/A')}")
            with col2:
                st.markdown(f"**NSE Age:** {row.get('NSE age', 'N/A')}")
            with col3:
                st.markdown(f"**BSE Age:** {row.get('BSE age', 'N/A')}")

            # --- Line 4: Name Numerology ---
            st.markdown("### 🔢 Name Numerology")

            use_ltd_home = st.radio(
                "Include 'Ltd' or 'Limited' in company name (if present)?",
                ["Yes", "No"],
                index=1,
                key="home_ltd"
            )

            use_in_prefix_home = st.radio(
                "Include 'IN' prefix in ISIN code (if present)?",
                ["Yes", "No"],
                index=1,
                key="home_isin"
            )

            isin_code = str(row.get("ISIN Code", ""))

            company_name_original = str(row['Company Name'])
            symbol_name = str(row['Symbol'])
            isin_code = str(row['ISIN Code'])

            if use_ltd_home == "No":
                company_clean = re.sub(r'\b(Ltd|Limited)\b', '', company_name_original, flags=re.IGNORECASE).strip()
            else:
                company_clean = company_name_original

            if use_in_prefix_home == "Yes":
                isin_to_use = isin_code
            else:
                isin_to_use = isin_code[2:] if isin_code.startswith("IN") else isin_code

            # Chaldean system
            ch_company_num, ch_company_eq = calculate_numerology(company_clean)
            ch_symbol_num, ch_symbol_eq = calculate_numerology(symbol_name)
            ch_isin_num, ch_isin_eq = calculate_chaldean_isin_numerology(isin_to_use)

            # Pythagorean system
            py_company_num, py_company_eq = calculate_pythagorean_numerology(company_clean)
            py_symbol_num, py_symbol_eq = calculate_pythagorean_numerology(symbol_name)
            py_isin_num, py_isin_eq = calculate_pythagorean_isin_numerology(isin_to_use)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Chaldean Eqn (Company Name):** {ch_company_eq}")
                st.markdown(f"**Chaldean Eqn (Symbol):** {ch_symbol_eq}")
                st.markdown(f"**Chaldean Eqn (ISIN Code):** {ch_isin_eq}")

            with col2:
                st.markdown(f"**Pythagoras Eqn (Company Name):** {py_company_eq}")
                st.markdown(f"**Pythagoras Eqn (Symbol):** {py_symbol_eq}")
                st.markdown(f"**Pythagoras Eqn (ISIN Code):** {py_isin_eq}")

            # --- Line 5: Zodiac Signs ---
            st.markdown("### ♈ Zodiac Information (Based on Dates)")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**DOC Zodiac Sign:** {row.get('DOC zodiac sign', 'N/A')}")
                doc_zodiac_number = row.get('DOC zodiac number', 'N/A')
                if isinstance(doc_zodiac_number, float) and doc_zodiac_number.is_integer():
                    doc_zodiac_number = int(doc_zodiac_number)
                st.markdown(f"**DOC Zodiac Number:** {doc_zodiac_number}")


            with col2:
                st.markdown(f"**NSE Zodiac Sign:** {row.get('NSE zodiac sign', 'N/A')}")
                st.markdown(f"**NSE Zodiac Number:** {row.get('NSE zodiac number', 'N/A')}")

            with col3:
                st.markdown(f"**BSE Zodiac Sign:** {row.get('BSE zodiac sign', 'N/A')}")
                st.markdown(f"**BSE Zodiac Number:** {row.get('BSE zodiac number', 'N/A')}")

            # --- Numerology Selection for Home Page ---
            st.markdown("### 🔢 Numerology Data Based on Selected Date")

            # Step 1: Ask user for date type preference
            date_match_option = st.selectbox(
                "Select Date Type to View Numerology Data:",
                ["NSE LISTING DATE", "BSE LISTING DATE", "DATE OF INCORPORATION", "All Dates"]
            )

            selected_row = row  # Already fetched from earlier using user_input

            date_types = (
                ["NSE LISTING DATE", "BSE LISTING DATE", "DATE OF INCORPORATION"]
                if date_match_option == "All Dates"
                else [date_match_option]
            )

            vertical_lines = []

            for dt_type in date_types:
                match_date = selected_row.get(dt_type)
                st.markdown(f"#### 📅 Numerology for {dt_type}: {match_date.date() if pd.notnull(match_date) else 'N/A'}")

                if pd.notnull(match_date):
                    match_date = pd.to_datetime(match_date)
                    numerology_row = numerology_df[numerology_df['date'] == match_date]
                    if not numerology_row.empty:
                        row_data = numerology_row.iloc[0]

                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.markdown(f"**BN:** {row_data.get('BN', 'N/A')}")
                        with col2:
                            st.markdown(f"**DN (Formatted):** {row_data.get('DN (Formatted)', 'N/A')}")
                        with col3:
                            st.markdown(f"**SN:** {row_data.get('SN', 'N/A')}")
                        with col4:
                            st.markdown(f"**HP:** {row_data.get('HP', 'N/A')}")
                        with col5:
                            st.markdown(f"**Day Number:** {row_data.get('Day Number', 'N/A')}")

                        # SN-based vertical line mapping
                        sn_vertical_lines = {
                            1: ["2025-05-05", "2025-05-07", "2025-05-08", "2025-05-10"],
                            2: ["2025-05-03", "2025-05-08", "2025-05-09", "2025-05-13"],
                            3: ["2025-05-06", "2025-05-10", "2025-05-11"],
                            4: ["2025-05-01", "2025-05-04", "2025-05-11", "2025-05-12"],
                            5: ["2025-05-02", "2025-05-05", "2025-05-08", "2025-05-12"],
                            6: ["2025-05-01", "2025-05-03", "2025-05-11", "2025-05-13"],
                            7: ["2025-05-04", "2025-05-14"],
                            8: ["2025-05-05", "2025-05-07", "2025-05-09"],
                            9: ["2025-05-02", "2025-05-06", "2025-05-11"]
                        }

                        # Extract SN value from numerology row
                        sn_value = row_data.get('SN', None)
                        if sn_value in sn_vertical_lines:
                            vertical_lines.extend(sn_vertical_lines[sn_value])


                else:
                    st.info(f"No numerology data available for {dt_type}.")
            else:
                st.info(f"No date available for {dt_type}.")

            vertical_lines = [pd.to_datetime(d) for d in vertical_lines]

            # --- Candlestick Chart (After Zodiac Info) ---
            st.markdown("### 📈 Stock Price Candlestick Chart")

            start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
            end_date = st.date_input("End Date", value=pd.to_datetime("today").normalize())
            ticker = str(row['Symbol']).upper() + ".NS"  # Get the company's symbol

            if start_date and end_date:
                stock_data = get_stock_data(ticker, start_date, end_date)
                if not stock_data.empty:
                    chart = plot_candlestick_chart(stock_data, vertical_lines=vertical_lines)
                    st.plotly_chart(chart)
                else:
                    st.warning("No data available for the selected date range.")
        else:
            st.warning("No matching company found.")

elif filter_mode == "View Nifty/BankNifty OHLC":
    st.subheader("📈 Nifty & BankNifty OHLC Viewer")

    from datetime import datetime
    import yfinance as yf

    index_choice = st.selectbox("Select Index:", ["Nifty 50", "Bank Nifty"])
    symbol = "^NSEI" if index_choice == "Nifty 50" else "^NSEBANK"
    index_name = "Nifty" if index_choice == "Nifty 50" else "BankNifty"

    # Load numerology
    numerology_aligned = numerology_df.copy()
    numerology_aligned['date'] = pd.to_datetime(numerology_aligned['date'], errors='coerce')
    numerology_aligned = numerology_aligned.set_index('date')

    # Default date range
    today = pd.to_datetime("today").normalize()
    default_end = today
    default_start = today - pd.Timedelta(days=90)

    start_date = st.date_input("Start Date", value=default_start, max_value=today)
    end_date = st.date_input("End Date", value=default_end, min_value=start_date, max_value=today)

    if start_date > end_date:
        st.error("End Date must be after Start Date")
        st.stop()

    ohlc_data = get_index_ohlc(index_name, symbol, start_date, end_date)
    ohlc_data['Volatility %'] = ((ohlc_data['High'] - ohlc_data['Low']) / ohlc_data['Low']) * 100
    ohlc_data['Close %'] = ohlc_data['Close'].pct_change() * 100
    ohlc_data['Volatility %'] = ohlc_data['Volatility %'].round(2)
    ohlc_data['Close %'] = ohlc_data['Close %'].round(2)
    ohlc_data['Day'] = ohlc_data.index.day
    ohlc_data['Month'] = ohlc_data.index.month

    st.markdown("### 🔍 Filter OHLC Table")
    col1, col2, col3 = st.columns(3)
    with col1:
        vol_op = st.selectbox("Volatility Operator", ["All", "<", "<=", ">", ">=", "=="])
        vol_val = st.number_input("Volatility % Value", value=2.0, step=0.1)
    with col2:
        close_op = st.selectbox("Close % Operator", ["All", "<", "<=", ">", ">=", "=="])
        close_val = st.number_input("Close % Value", value=0.5, step=0.1)
    with col3:
        apply_day_month_filter = st.checkbox("🗓️ Filter by Day and Month", value=False)

    filtered_data = ohlc_data.copy()
    if vol_op != "All":
        filtered_data = filtered_data.query(f"`Volatility %` {vol_op} @vol_val")
    if close_op != "All":
        filtered_data = filtered_data.query(f"`Close %` {close_op} @close_val")
    if apply_day_month_filter:
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            filter_day = st.number_input("Day (1-31)", min_value=1, max_value=31, value=1)
        with dcol2:
            filter_month = st.selectbox("Month", list(range(1, 13)), format_func=lambda x: datetime(1900, x, 1).strftime('%B'))
        filtered_data = filtered_data[(filtered_data.index.day == filter_day) & (filtered_data.index.month == filter_month)]

    # Merge numerology
    full_data_merged = filtered_data.merge(numerology_aligned, left_index=True, right_index=True, how='left')

    # Numerology filters
    st.markdown("### 🧮 Numerology Filters")
    ncol1, ncol2, ncol3, ncol4, ncol5 = st.columns(5)
    with ncol1:
        bn_filter = st.selectbox("BN", ["All"] + sorted(numerology_df['BN'].dropna().unique()))
    with ncol2:
        dn_filter = st.selectbox("DN (Formatted)", ["All"] + sorted(numerology_df['DN (Formatted)'].dropna().unique()))
    with ncol3:
        sn_filter = st.selectbox("SN", ["All"] + sorted(numerology_df['SN'].dropna().unique()))
    with ncol4:
        hp_filter = st.selectbox("HP", ["All"] + sorted(numerology_df['HP'].dropna().unique()))
    with ncol5:
        dayn_filter = st.selectbox("Day Number", ["All"] + sorted(numerology_df['Day Number'].dropna().unique()))

    if bn_filter != "All":
        full_data_merged = full_data_merged[full_data_merged['BN'] == bn_filter]
    if dn_filter != "All":
        full_data_merged = full_data_merged[full_data_merged['DN (Formatted)'] == dn_filter]
    if sn_filter != "All":
        full_data_merged = full_data_merged[full_data_merged['SN'] == sn_filter]
    if hp_filter != "All":
        full_data_merged = full_data_merged[full_data_merged['HP'] == hp_filter]
    if dayn_filter != "All":
        full_data_merged = full_data_merged[full_data_merged['Day Number'] == dayn_filter]

    # Display final styled table
    st.markdown("### 🔢 OHLC + Numerology Alignment")
    final_df = full_data_merged.reset_index().rename(columns={"index": "Date"})
    primary_dates = {(3, 20),(3, 21), (6, 20), (6, 21), (9, 22), (9, 23), (12, 21), (12, 22)}
    secondary_dates = {(2, 4), (5, 6), (8, 8), (11, 7)}

    def highlight_rows(row):
        date = pd.to_datetime(row['Date'])
        month_day = (date.month, date.day)
        if month_day in primary_dates:
            return ['background-color: lightgreen'] * len(row)
        elif month_day in secondary_dates:
            return ['background-color: lightsalmon'] * len(row)
        else:
            return [''] * len(row)

    styled_df = final_df.style.apply(highlight_rows, axis=1).format(precision=2)
    st.markdown(f'<div class="scroll-table">{styled_df.to_html()}</div>', unsafe_allow_html=True)

    if st.checkbox("📊 Show Candlestick Chart"):
        if not filtered_data.empty:
            candlestick = go.Figure(data=[go.Candlestick(
                x=filtered_data.index,
                open=filtered_data['Open'],
                high=filtered_data['High'],
                low=filtered_data['Low'],
                close=filtered_data['Close'],
                increasing_line_color='green',
                decreasing_line_color='red'
            )])
            candlestick.update_layout(
                title='Candlestick Chart',
                xaxis_title='Date',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                height=600
            )
            st.plotly_chart(candlestick, use_container_width=True)
        else:
            st.warning("No data available for selected filters to display candlestick chart.")

elif filter_mode == "Equinox":
    st.subheader("📊 Nifty/BankNifty Report for Primary & Secondary Dates")

    index_choice = st.selectbox("Choose Index", ["Nifty 50", "Bank Nifty"], key="econ_index")
    index_name = "Nifty" if index_choice == "Nifty 50" else "BankNifty"
    ticker = "^NSEI" if index_name == "Nifty" else "^NSEBANK"

    # Recalculate volatility & close %
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], errors='coerce')
    numerology_data = numerology_df.set_index('date')

    all_dates = numerology_df['date'].dropna().dt.date.unique()
    all_dates = sorted(pd.to_datetime(all_dates))

    st.markdown("### 🗓️ Select Time Period")
    min_date = min(all_dates)
    max_date = max(all_dates)
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=pd.to_datetime("2024-04-01").date(), min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("End Date", value=pd.to_datetime("2025-06-01").date(), min_value=min_date, max_value=max_date)

    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
        st.stop()

    primary_dates = {(3, 21), (6, 20), (6, 21), (9, 22), (9, 23), (12, 21), (12, 22)}
    secondary_dates = {(2, 4), (5, 6), (8, 8), (11, 7)}

    def classify_date(dt):
        m, d = dt.month, dt.day
        if (m, d) in primary_dates:
            return "Primary"
        elif (m, d) in secondary_dates:
            return "Secondary"
        return None

    filtered_dates = [dt for dt in all_dates if start_date <= dt.date() <= end_date]
    ohlc_data = get_index_ohlc(index_name, ticker, start_date, end_date)

    # Recalculate % columns
    ohlc_data['Volatility %'] = ((ohlc_data['High'] - ohlc_data['Low']) / ohlc_data['Low']) * 100
    ohlc_data['Close %'] = ohlc_data['Close'].pct_change() * 100
    ohlc_data['Volatility %'] = ohlc_data['Volatility %'].round(2)
    ohlc_data['Close %'] = ohlc_data['Close %'].round(2)

    report_rows = []
    for date in filtered_dates:
        tag = classify_date(date)
        if tag:
            row = {"Date": date, "Category": tag}
            date_index = pd.to_datetime(date)

            if date_index in ohlc_data.index:
                row.update(ohlc_data.loc[date_index].to_dict())
            else:
                for col in ['Open', 'High', 'Low', 'Close', 'Vol(in M)', 'Volatility %', 'Close %']:
                    row[col] = float('nan')

            if date_index in numerology_data.index:
                row.update(numerology_data.loc[date_index].to_dict())

            report_rows.append(row)

    if report_rows:
        final_df = pd.DataFrame(report_rows)
        final_df = final_df.sort_values("Date", ascending=False).reset_index(drop=True)
        final_df['Date'] = final_df['Date'].dt.strftime('%Y-%m-%d')

        float_cols = ['Open', 'High', 'Low', 'Close', 'Vol(in M)', 'Volatility %', 'Close %']

        # ✅ Force conversion of suspected columns
        for col in float_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

        formatter_dict = {
            col: "{:.2f}" for col in float_cols
            if col in final_df.columns and pd.api.types.is_numeric_dtype(final_df[col])
        }


        def highlight_econ_rows(row):
            date = pd.to_datetime(row['Date'], errors='coerce')
            if pd.isna(date): return ''
            md = (date.month, date.day)
            if md in primary_dates:
                return 'background-color: #d1fab8'
            elif md in secondary_dates:
                return 'background-color: #ffa868'
            return ''

        styled_df = final_df.style \
            .apply(lambda row: [highlight_econ_rows(row)] * len(row), axis=1) \
            .format(formatter_dict)

        html_table = styled_df.to_html()
        st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)
    else:
        st.info("No primary or secondary dates found in selected range.")

elif filter_mode == "Moon":
    st.header("🌑 Moon Phase Analysis")

    # Load moon data
    moon_df = load_moon_data()
    moon_df = moon_df.sort_values('Date')
    doc_df = load_stock_data()
    available_symbols = sorted(doc_df['Symbol'].dropna().unique().tolist())
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], dayfirst=True)
    phase_choice = st.selectbox("Select Moon Phase:", ["Amavasya", "Poornima"])

    # Filter moon phase based on user selection
    phase_filtered = moon_df[moon_df['A/P'].str.lower() == phase_choice.lower()]
    available_dates = phase_filtered['Date'].dt.date.tolist()

    today = pd.Timestamp.today().normalize()
    one_month_ago = today - pd.Timedelta(days=30)

    recent_dates = [d for d in available_dates if d >= one_month_ago.date()]
    default_date = recent_dates[0] if recent_dates else available_dates[-1]  # fallback to latest

    # Create selectbox with default selection
    selected_date_str = st.selectbox(
        f"Select a {phase_choice} Date:",
        [d.strftime("%Y-%m-%d") for d in available_dates],
        index=available_dates.index(default_date)
    )
    selected_date = pd.to_datetime(selected_date_str)

    # Moon Info
    match = moon_df[moon_df['Date'].dt.date == selected_date.date()]
    if match.empty:
        st.error("Selected date not found in moon data.")
        st.stop()

    selected_row = match.iloc[0]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Degree:** {selected_row['Degree']}")
    with col2:
        st.markdown(f"**Time:** {selected_row['Time']}")
    with col3:
        st.markdown(f"**Paksh:** {selected_row['Paksh']}")

    # Find next moon date
    future_dates = moon_df[moon_df['Date'] > selected_date]
    next_date = future_dates.iloc[0]['Date'] if not future_dates.empty else selected_date + pd.Timedelta(days=15)
    st.markdown(f"### 📅 Period: {selected_date.date()} to {next_date.date()}")

    st.subheader("📈 Symbol OHLC + Numerology")

    # --- SYMBOL SECTION ---
    selected_symbol = st.selectbox("Select Stock Symbol:", available_symbols)

    listing_row = doc_df[doc_df['Symbol'] == selected_symbol]
    if listing_row.empty or pd.isnull(listing_row.iloc[0]['DATE OF INCORPORATION']):
        st.warning("Listing date unavailable.")
    else:
        listing_date = pd.to_datetime(listing_row.iloc[0]['DATE OF INCORPORATION'])

        if selected_date < listing_date:
            st.warning(f"{selected_symbol} was not listed on {selected_date.date()}")
        else:
            ticker = selected_symbol + ".NS"
            stock_data = get_stock_data(ticker, selected_date, next_date)

            # Generate full date range
            all_dates = pd.date_range(start=selected_date, end=next_date - pd.Timedelta(days=1))

            if stock_data.empty:
                stock_data = pd.DataFrame(index=all_dates)
                stock_data[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
            else:
                stock_data = stock_data.reindex(all_dates)

            # Merge with numerology
            numerology_subset = numerology_df.set_index('date')
            combined = stock_data.merge(numerology_subset, left_index=True, right_index=True, how='left')
            combined = combined.loc[all_dates]  # ensure consistent order

            # High/Low check
            if combined['High'].notna().any():
                high_val = combined['High'].max()
                low_val = combined['Low'].min()
                st.markdown(f"**📈 High:** {high_val} | 📉 Low:** {low_val}")
            else:
                st.info("No OHLC data available in this period — only numerology shown.")

            combined_reset = combined.reset_index()
            combined_reset.rename(columns={"index": "Date"}, inplace=True)

            # Render table
            html_table = combined_reset.to_html(index=False)
            st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

    # --- INDEX SECTION ---
    st.subheader("📊 Nifty / BankNifty OHLC + Numerology")

    index_choice = st.radio("Select Index:", ["Nifty 50", "Bank Nifty"])
    index_file = "nifty.xlsx" if index_choice == "Nifty 50" else "banknifty.xlsx"

    index_df = load_excel_data(index_file)
    all_dates = pd.date_range(start=selected_date, end=next_date - pd.Timedelta(days=1))
    index_range = index_df[(index_df.index >= selected_date) & (index_df.index < next_date)]

    if index_range.empty:
        index_range = pd.DataFrame(index=all_dates)
        index_range[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
    else:
        index_range = index_range[~index_range.index.duplicated(keep='last')]
        index_range = index_range.reindex(all_dates)

    numerology_subset = numerology_df.set_index('date')
    index_combined = index_range.merge(numerology_subset, left_index=True, right_index=True, how='left')
    index_combined = index_combined.loc[all_dates]

    if index_combined['High'].notna().any():
        high_val = index_combined['High'].max()
        low_val = index_combined['Low'].min()
        st.markdown(f"**📈 High:** {high_val} | 📉 Low:** {low_val}")
    else:
        st.info("No index OHLC data available in this period — only numerology shown.")

    index_combined_reset = index_combined.reset_index()
    index_combined_reset.rename(columns={"index": "Date"}, inplace=True)
    html_table = index_combined_reset.to_html(index=False)

    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Mercury":
    st.header("🪐Mercury Phase Analysis")

    # Load mercury data
    mercury_df = load_mercury_data()
    mercury_df['Date'] = pd.to_datetime(mercury_df['Date'], dayfirst=True)
    mercury_df = mercury_df.sort_values('Date')
    # Load moon phase data
    moon_df = load_moon_data()
    moon_df['Date'] = pd.to_datetime(moon_df['Date'], dayfirst=True)

    # Get dates for Amavasya and Poornima
    amavasya_dates = set(moon_df[moon_df['A/P'].str.lower() == "amavasya"]['Date'].dt.date)
    poornima_dates = set(moon_df[moon_df['A/P'].str.lower() == "poornima"]['Date'].dt.date)


    # Load stock symbols from doc.xlsx
    doc_df = load_stock_data()
    available_symbols = sorted(doc_df['Symbol'].dropna().unique().tolist())

    # Load numerology
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], dayfirst=True)

    # mercury Phase & Date
    phase_choice = st.selectbox("Select mercury Phase:", ["Direct", "Retrograde"])

    # Step 2: Filter dates based on the chosen phase
    phase_filtered = mercury_df[mercury_df['D/R'].str.lower() == phase_choice.lower()]
    available_dates = phase_filtered['Date'].dropna().dt.date.tolist()

    # Step 3: Compute the default date (last 30 days logic)
    today = pd.Timestamp.today().normalize()
    one_month_ago = today - pd.Timedelta(days=30)
    recent_dates = [d for d in available_dates if d >= one_month_ago.date()]

    # If recent date found, use the first one, else fallback to latest available
    default_date = recent_dates[0] if recent_dates else available_dates[-1]

    # Step 4: Show in selectbox with pre-selected default
    selected_date_str = st.selectbox(
        f"Select a {phase_choice} Date:",
        [d.strftime("%Y-%m-%d") for d in available_dates],
        index=available_dates.index(default_date)
    )

    # Step 5: Parse the selected date
    selected_date = pd.to_datetime(selected_date_str)

    future_dates = mercury_df[mercury_df['Date'] > selected_date]
    next_date = future_dates.iloc[0]['Date'] if not future_dates.empty else selected_date + pd.Timedelta(days=15)

    # mercury Info
    match = mercury_df[mercury_df['Date'].dt.date == selected_date.date()]
    if match.empty:
        st.error("Selected date not found in mercury data.")
        st.stop()

    # Mercury info at start date
    start_row = mercury_df[mercury_df['Date'].dt.date == selected_date.date()]
    start_degree = start_row.iloc[0]['Degree'] if not start_row.empty else "N/A"
    start_time = start_row.iloc[0]['Time'] if not start_row.empty else "N/A"

    # Mercury info at end date (if exists)
    end_row = mercury_df[mercury_df['Date'].dt.date == next_date.date()]
    end_degree = end_row.iloc[0]['Degree'] if not end_row.empty else "N/A"
    end_time = end_row.iloc[0]['Time'] if not end_row.empty else "N/A"

    # Display both
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Start Date:** {selected_date.date()}  \n**Degree:** {start_degree}  \n**Time:** {start_time}")
    with col2:
        st.markdown(f"**End Date:** {next_date.date()}  \n**Degree:** {end_degree}  \n**Time:** {end_time}")

    

    # Find next mercury date
    future_dates = mercury_df[mercury_df['Date'] > selected_date]
    next_date = future_dates.iloc[0]['Date'] if not future_dates.empty else selected_date + pd.Timedelta(days=15)
    st.markdown(f"### 📅 Period: {selected_date.date()} to {next_date.date()}")

    st.subheader("📈 Symbol OHLC + Numerology")

    # --- SYMBOL SECTION ---
    selected_symbol = st.selectbox("Select Stock Symbol:", available_symbols)

    listing_row = doc_df[doc_df['Symbol'] == selected_symbol]
    if listing_row.empty or pd.isnull(listing_row.iloc[0]['DATE OF INCORPORATION']):
        st.warning("Listing date unavailable.")
    else:
        listing_date = pd.to_datetime(listing_row.iloc[0]['DATE OF INCORPORATION'])

        if selected_date < listing_date:
            st.warning(f"{selected_symbol} was not listed on {selected_date.date()}")
        else:
            ticker = selected_symbol + ".NS"
            stock_data = get_stock_data(ticker, selected_date, next_date)

            # Generate full date range
            all_dates = pd.date_range(start=selected_date, end=next_date - pd.Timedelta(days=1))

            if stock_data.empty:
                stock_data = pd.DataFrame(index=all_dates)
                stock_data[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
            else:
                stock_data = stock_data.reindex(all_dates)

            # Merge with numerology
            numerology_subset = numerology_df.set_index('date')
            combined = stock_data.merge(numerology_subset, left_index=True, right_index=True, how='left')
            combined = combined.loc[all_dates]  # ensure consistent order

            # High/Low check
            if combined['High'].notna().any():
                high_val = combined['High'].max()
                low_val = combined['Low'].min()
                st.markdown(f"**📈 High:** {high_val} | 📉 Low:** {low_val}")
            else:
                st.info("No OHLC data available in this period — only numerology shown.")

            combined_reset = combined.reset_index()
            combined_reset.rename(columns={"index": "Date"}, inplace=True)

            # Step 7: Highlight rows based on moon phase
            def highlight_moon_rows(row):
                date = row['Date'].date() if isinstance(row['Date'], pd.Timestamp) else None
                if date in amavasya_dates:
                    return ['background-color: #ff2525'] * len(row)  # Light red
                elif date in poornima_dates:
                    return ['background-color: #7aceff'] * len(row)  # Sky blue
                else:
                    return [''] * len(row)

            # Render table
            styled_df = combined_reset.style.apply(highlight_moon_rows, axis=1).format(precision=2)
            html_table = styled_df.to_html()
            st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)


    # --- INDEX SECTION ---
    st.subheader("📊 Nifty / BankNifty OHLC + Numerology")

    index_choice = st.radio("Select Index:", ["Nifty 50", "Bank Nifty"])
    index_file = "nifty.xlsx" if index_choice == "Nifty 50" else "banknifty.xlsx"

    index_df = load_excel_data(index_file)
    all_dates = pd.date_range(start=selected_date, end=next_date - pd.Timedelta(days=1))
    index_range = index_df[(index_df.index >= selected_date) & (index_df.index < next_date)]

    if index_range.empty:
        index_range = pd.DataFrame(index=all_dates)
        index_range[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
    else:
        index_range = index_range.reindex(all_dates)

    numerology_subset = numerology_df.set_index('date')
    index_combined = index_range.merge(numerology_subset, left_index=True, right_index=True, how='left')
    index_combined = index_combined.loc[all_dates]

    if index_combined['High'].notna().any():
        high_val = index_combined['High'].max()
        low_val = index_combined['Low'].min()
        st.markdown(f"**📈 High:** {high_val} | 📉 Low:** {low_val}")
    else:
        st.info("No index OHLC data available in this period — only numerology shown.")

    index_combined_reset = index_combined.reset_index()
    index_combined_reset.rename(columns={"index": "Date"}, inplace=True)

    # Step 7: Highlight rows based on moon phase
    def highlight_moon_rows(row):
        date = row['Date'].date() if isinstance(row['Date'], pd.Timestamp) else None
        if date in amavasya_dates:
            return ['background-color: #ff2525'] * len(row)  # Light red
        elif date in poornima_dates:
            return ['background-color: #7aceff'] * len(row)  # Sky blue
        else:
            return [''] * len(row)

    # Step 8: Display styled table
    styled_df = index_combined_reset.style.apply(highlight_moon_rows, axis=1).format(precision=2)
    html_table = styled_df.to_html()
    

    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Sun Number Dates":
    st.title("🌞 Sun Number Dates")

    # Step 1: User selects which date to base SN on
    date_type = st.selectbox("Choose Date Type for Sun Number:", 
                             ["NSE LISTING DATE", "BSE LISTING DATE", "DATE OF INCORPORATION"])

    # Step 2: Map selected date to real SN using numerology_df
    stock_df['Selected Date'] = pd.to_datetime(stock_df[date_type], errors='coerce')
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], errors='coerce')

    # Drop duplicates to ensure clean mapping
    sn_lookup = numerology_df.drop_duplicates(subset='date').set_index('date')['SN']
    stock_df['Selected SN'] = stock_df['Selected Date'].map(sn_lookup)

    # Step 3: Filter by selected SN
    valid_sns = sorted(stock_df['Selected SN'].dropna().unique())
    selected_sn = st.selectbox("Filter by Sun Number:", valid_sns)

    matching_df = stock_df[stock_df['Selected SN'] == selected_sn]

    if matching_df.empty:
        st.warning("No companies found with this SN.")
        st.stop()

    # Step 4: User selects company from filtered list
    company_choice = st.selectbox("Select Company Symbol:", matching_df['Symbol'])
    selected_row = matching_df[matching_df['Symbol'] == company_choice].iloc[0]
    st.markdown(f"**Company Name:** {selected_row['Company Name']}")

    # Step 5: Date input
    default_end = pd.to_datetime("today").normalize()
    default_start = default_end - pd.Timedelta(days=30)
    start_date = st.date_input("Start Date", value=default_start)
    end_date = st.date_input("End Date", value=default_end)

    # Step 6: Get stock data
    ticker = company_choice + ".NS"
    stock_data = get_stock_data(ticker, start_date, end_date)

    # Step 7: Prepare vertical lines based on SN
    sn_vertical_lines = {
        1: ["2025-05-05", "2025-05-07", "2025-05-08", "2025-05-10"],
        2: ["2025-05-03", "2025-05-08", "2025-05-09", "2025-05-13"],
        3: ["2025-05-06", "2025-05-10", "2025-05-11"],
        4: ["2025-05-01", "2025-05-04", "2025-05-11", "2025-05-12"],
        5: ["2025-05-02", "2025-05-05", "2025-05-08", "2025-05-12"],
        6: ["2025-05-01", "2025-05-03", "2025-05-11", "2025-05-13"],
        7: ["2025-05-04", "2025-05-14"],
        8: ["2025-05-05", "2025-05-07", "2025-05-09"],
        9: ["2025-05-02", "2025-05-06", "2025-05-11"]
    }

    vertical_lines = [pd.to_datetime(d) for d in sn_vertical_lines.get(selected_sn, [])]
    vertical_lines = [d for d in vertical_lines if start_date <= d.date() <= end_date]

    # Step 8: Plot candlestick chart
    if not stock_data.empty:
        st.subheader("📈 Candlestick Chart")
        chart = plot_candlestick_chart(stock_data, vertical_lines=vertical_lines)
        st.plotly_chart(chart)
    else:
        st.warning("No stock data found for selected date range.")

    # Step 9: Merge with numerology and show OHLCV + numerology
    st.subheader("📊 OHLC + Numerology Data")
    if not stock_data.empty:
        stock_data.index = pd.to_datetime(stock_data.index)
        numerology_merge = numerology_df.set_index('date')
        merged = stock_data.merge(numerology_merge, left_index=True, right_index=True, how='left')
        
        # Render as HTML
        html_table = merged.reset_index().to_html()

        # Display
        st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Panchak":
    st.title("📅 Panchak Dates Analysis")

    # Load Panchak data
    panchak_df = load_panchak_data()
    panchak_df['Start Date'] = pd.to_datetime(panchak_df['Start Date'], errors='coerce', dayfirst=True)
    panchak_df['End Date'] = pd.to_datetime(panchak_df['End Date'], errors='coerce', dayfirst=True)
    panchak_df = panchak_df.dropna(subset=['Start Date', 'End Date']).sort_values('Start Date').reset_index(drop=True)

    # Load moon data
    moon_df = load_moon_data()
    moon_df['Date'] = pd.to_datetime(moon_df['Date'], errors='coerce')
    amavasya_dates = set(moon_df[moon_df['A/P'].str.lower() == "amavasya"]['Date'].dt.date)
    poornima_dates = set(moon_df[moon_df['A/P'].str.lower() == "poornima"]['Date'].dt.date)

    # Symbol list
    symbol_list = ["Nifty", "BankNifty"] + sorted(stock_df['Symbol'].dropna().unique().tolist())
    selected_symbol = st.selectbox("Select Symbol", symbol_list)

    # Define today's date and the cutoff for the last month
    today = pd.Timestamp.today().normalize()
    one_month_ago = today - pd.Timedelta(days=30)

    # Find the most recent Panchak starting within the last month
    recent_panchaks = panchak_df[panchak_df['Start Date'] >= one_month_ago]
    if not recent_panchaks.empty:
        default_start_date = recent_panchaks.iloc[0]['Start Date'].date()
    else:
        default_start_date = panchak_df['Start Date'].dt.date.max()

    # Selectbox with default
    selected_start_date = st.selectbox(
        "Select Panchak Start Date:",
        panchak_df['Start Date'].dt.date.unique(),
        index=panchak_df['Start Date'].dt.date.tolist().index(default_start_date)
    )

    # Get the corresponding row
    row = panchak_df[panchak_df['Start Date'].dt.date == selected_start_date].iloc[0]
    start_date = row['Start Date']
    end_date = row['End Date']

    st.markdown(f"### 🕒 Panchak Period: {start_date.date()} to {end_date.date()}")
    st.markdown(f"**Start Time:** {row['Start Time']} | **End Time:** {row['End Time']}")
    st.markdown(f"**Start Degree:** {row['Degree']:.4f}")

    def get_combined_index_data(symbol, start_date, end_date):

        index_name = "Nifty" if symbol == "Nifty" else "BankNifty"
        ticker = "^NSEI" if symbol == "Nifty" else "^NSEBANK"

        # Step 1: Try fetching from DB
        query = '''
            SELECT "Date", "Open", "High", "Low", "Close", "Vol(in M)"
            FROM ohlc_index
            WHERE index_name = %s AND "Date" BETWEEN %s AND %s
            ORDER BY "Date"
        '''
        df = pd.read_sql(query, engine, params=(index_name, start_date, end_date))
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[~df.index.duplicated(keep='last')]

        # Step 2: Find missing dates in range
        full_range = pd.date_range(start=start_date, end=end_date - pd.Timedelta(days=1))
        missing_dates = full_range.difference(df.index)

        if not missing_dates.empty:
            fetch_start = missing_dates.min()
            fetch_end = end_date + pd.Timedelta(days=1)

            yf_data = yf.download(ticker, start=fetch_start, end=fetch_end, progress=False, multi_level_index=False)

            if not yf_data.empty:
                append_df = yf_data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                append_df.reset_index(inplace=True)
                append_df['index_name'] = index_name
                append_df.rename(columns={"Date": "Date", "Volume": "Vol(in M)"}, inplace=True)

                # Insert new rows to DB
                append_df.to_sql("ohlc_index", engine, if_exists="append", index=False)

                # Add to current DataFrame
                append_df.set_index('Date', inplace=True)
                df = pd.concat([df, append_df])
                df = df[~df.index.duplicated(keep='last')]

        # Step 3: Return data reindexed over requested date range
        return df.reindex(full_range)


    # Get OHLC data
    if selected_symbol in ["Nifty", "BankNifty"]:
        ohlc = get_combined_index_data(selected_symbol, start_date, end_date)
    else:
        ticker = selected_symbol + ".NS"
        ohlc = get_stock_data(ticker, start_date, end_date)

    # Full date range
    all_dates = pd.date_range(start=start_date, end=end_date)

    if ohlc.empty:
        ohlc = pd.DataFrame(index=all_dates)
        ohlc[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
    else:
        ohlc = ohlc.reindex(all_dates)

    # Load and prepare numerology data
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], errors='coerce').dt.date
    numerology_subset = numerology_df.set_index('date')

    # Merge OHLC and numerology
    merged = ohlc.merge(numerology_subset, left_index=True, right_index=True, how='left')
    merged = merged.loc[all_dates]
    merged = merged.reset_index().rename(columns={"index": "Date"})

    # High/Low display
    if merged['High'].notna().any():
        high_val = merged['High'].max()
        low_val = merged['Low'].min()
        st.markdown(f"**📈 High:** {high_val:.2f} | 📉 Low:** {low_val:.2f}")
    else:
        st.warning("⚠ No OHLC data available for this period — only numerology is shown.")

    # Highlight Amavasya / Poornima
    def highlight_moon_rows(row):
        date = row['Date'].date() if isinstance(row['Date'], pd.Timestamp) else None
        if date in amavasya_dates:
            return ['background-color: #ffcccc'] * len(row)  # Light red
        elif date in poornima_dates:
            return ['background-color: #ccf2ff'] * len(row)  # Sky blue
        else:
            return [''] * len(row)

    # Display styled table
    styled_df = merged.style.apply(highlight_moon_rows, axis=1).format(precision=2)
    html_table = styled_df.to_html()
    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

    # --- 📊 Post-Panchak Period ---
    st.subheader("📊 Post-Panchak Period Analysis")

    # Find next Panchak start date
    future_rows = panchak_df[panchak_df['Start Date'] > end_date]
    if not future_rows.empty:
        next_start_date = future_rows.iloc[0]['Start Date']
    else:
        next_start_date = end_date + pd.Timedelta(days=10)

    post_start_date = end_date + pd.Timedelta(days=1)
    post_end_date = next_start_date

    st.markdown(f"### ⏭️ Period: {post_start_date.date()} to {(post_end_date).date()}")


    # Get OHLC for post-Panchak period
    if selected_symbol in ["Nifty", "BankNifty"]:
        post_ohlc = get_combined_index_data(selected_symbol, post_start_date, post_end_date)
    else:
        ticker = selected_symbol + ".NS"
        post_ohlc = get_stock_data(ticker, post_start_date, post_end_date)

    # Full date range
    post_dates = pd.date_range(start=post_start_date, end=post_end_date - pd.Timedelta(days=1))

    if post_ohlc.empty:
        post_ohlc = pd.DataFrame(index=post_dates)
        post_ohlc[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
    else:
        post_ohlc = post_ohlc.reindex(post_dates)

    # Merge with numerology
    post_merged = post_ohlc.merge(numerology_subset, left_index=True, right_index=True, how='left')
    post_merged = post_merged.loc[post_dates]
    post_merged = post_merged.reset_index().rename(columns={"index": "Date"})

    # High/Low display
    if post_merged['High'].notna().any():
        high_val = post_merged['High'].max()
        low_val = post_merged['Low'].min()
        st.markdown(f"**📈 High:** {high_val:.2f} | 📉 Low:** {low_val:.2f}")
    else:
        st.info("⚠ No OHLC data for post-Panchak period — only numerology shown.")
    
    
    # Highlight moon phases
    styled_post_df = post_merged.style.apply(highlight_moon_rows, axis=1).format(precision=2)
    post_html_table = styled_post_df.to_html()
    st.markdown(f'<div class="scroll-table">{post_html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Mercury Combust":
    st.header("🔥 Mercury Combust Period Analysis")

    # Load data from DB
    combust_df = load_combust_data()
    moon_df = load_moon_data()
    doc_df = load_stock_data()
    numerology_df['date'] = pd.to_datetime(numerology_df['date'], errors='coerce')
    numerology_subset = numerology_df.set_index('date')

    # Extract Moon Phase Dates
    amavasya_dates = set(moon_df[moon_df['A/P'].str.lower() == 'amavasya']['Date'].dt.date)
    poornima_dates = set(moon_df[moon_df['A/P'].str.lower() == 'poornima']['Date'].dt.date)

    # Prepare valid dates from combust periods
    valid_dates = sorted(set(
        date for _, row in combust_df.iterrows()
        for date in pd.date_range(row['Start Date'], row['End Date'])
    ))

    recent_cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)
    recent_dates = [d for d in valid_dates if d >= recent_cutoff]
    default_date = recent_dates[0] if recent_dates else valid_dates[-1]

    selected_date = st.selectbox(
        "Select a Date in Combust Period:",
        [d.date() for d in valid_dates],
        index=[d.date() for d in valid_dates].index(default_date.date())
    )

    # Filter to matching combust period
    match = combust_df[
        (combust_df['Start Date'] <= pd.Timestamp(selected_date)) &
        (combust_df['End Date'] >= pd.Timestamp(selected_date))
    ]
    if match.empty:
        st.warning("Selected date does not fall under any Mercury Combust period.")
        st.stop()

    # Get combust period range
    start_date = match.iloc[0]['Start Date']
    end_date = match.iloc[0]['End Date']

    st.markdown(f"📅 **Combust Period:** {start_date.date()} to {end_date.date()}")
    st.markdown(f"**Start Time:** {match.iloc[0]['Start Time']} | **End Time:** {match.iloc[0]['End Time']}")
    st.markdown(f"**Start Degree:** {match.iloc[0]['Start Degree']} | **End Degree:** {match.iloc[0]['End Degree']}")

    # --- SYMBOL SECTION ---
    st.subheader("📈 Symbol OHLC + Numerology")
    available_symbols = sorted(doc_df['Symbol'].dropna().unique().tolist())
    selected_symbol = st.selectbox("Select Stock Symbol:", available_symbols)

    listing_row = doc_df[doc_df['Symbol'] == selected_symbol]
    if listing_row.empty or pd.isnull(listing_row.iloc[0]['DATE OF INCORPORATION']):
        st.warning("Listing date unavailable.")
        st.stop()

    listing_date = pd.to_datetime(listing_row.iloc[0]['DATE OF INCORPORATION'])
    if start_date < listing_date:
        st.warning(f"{selected_symbol} was not listed during this combustion period.")
        st.stop()

    stock_data = get_stock_data(selected_symbol + ".NS", start_date, end_date)
    all_days = pd.date_range(start=start_date, end=end_date)

    if stock_data.empty:
        stock_data = pd.DataFrame(index=all_days)
        stock_data[['Open', 'High', 'Low', 'Close', 'Volume']] = float('nan')
    else:
        stock_data = stock_data.reindex(all_days)

    combined = stock_data.merge(numerology_subset, left_index=True, right_index=True, how='left')
    combined_reset = combined.reset_index().rename(columns={'index': 'Date'})

    def highlight_moon_rows(row):
        d = row['Date'].date() if isinstance(row['Date'], pd.Timestamp) else None
        if d in amavasya_dates:
            return ['background-color: #ffcccc'] * len(row)
        elif d in poornima_dates:
            return ['background-color: #ccf2ff'] * len(row)
        return [''] * len(row)

    styled_df = combined_reset.style.apply(highlight_moon_rows, axis=1).format(precision=2)
    st.markdown(f'<div class="scroll-table">{styled_df.to_html()}</div>', unsafe_allow_html=True)

    # --- INDEX SECTION ---
    st.subheader("📊 Nifty / BankNifty OHLC + Numerology")
    index_choice = st.radio("Select Index:", ["Nifty 50", "Bank Nifty"])
    index_name = "Nifty" if index_choice == "Nifty 50" else "BankNifty"

    index_data = get_combined_index_data(index_name, start_date, end_date)
    index_data = index_data.reindex(all_days)
    index_combined = index_data.merge(numerology_subset, left_index=True, right_index=True, how='left')

    if index_combined['High'].notna().any():
        high_val = index_combined['High'].max()
        low_val = index_combined['Low'].min()
        st.markdown(f"**📈 High:** {high_val:.2f} | 📉 Low:** {low_val:.2f}")
    else:
        st.info("No index OHLC data available in this period.")

    index_combined_reset = index_combined.reset_index().rename(columns={"index": "Date"})
    styled_index = index_combined_reset.style.apply(highlight_moon_rows, axis=1).format(precision=2)
    st.markdown(f'<div class="scroll-table">{styled_index.to_html()}</div>', unsafe_allow_html=True)

elif filter_mode == "Range":
    st.subheader("📊 Range Levels (Nifty)")

    def generate_custom_levels(high, low, sp_levels, levels_up=5, levels_down=5, current_price=None):
        sp1 = sp_levels[0]
        range_val = round(high - low, 2)
        levels_output = []

        # === Level 1 (Original)
        levels_output.append(("🔺 Level 1", ""))
        levels_output.append(("High", high))
        for i in reversed(range(1, len(sp_levels))):
            levels_output.append((f"High - sp{i+1}", round(high - sp_levels[i], 2)))
        midpoint = round(high - sp1, 2)
        levels_output.append(("Midpoint", midpoint))
        for i in range(1, len(sp_levels)):
            levels_output.append((f"Low + sp{i+1}", round(low + sp_levels[i], 2)))
        levels_output.append(("Low", low))

        # === Upper Levels (2+)
        current_high = high
        for level in range(2, levels_up + 2):
            prev_high = current_high
            current_high = round(prev_high + range_val, 2)
            current_low = prev_high
            mp = round(current_high - sp1, 2)

            levels_output.append((f"🔺 Level {level}", ""))
            levels_output.append((f"High {level}", current_high))
            for i in reversed(range(1, len(sp_levels))):
                levels_output.append((f"mp{level} - sp{i+1}", round(current_high - sp_levels[i], 2)))
            levels_output.append((f"mp{level}", mp))
            for i in range(1, len(sp_levels)):
                levels_output.append((f"mp{level} + sp{i+1}", round(current_low + sp_levels[i], 2)))
            levels_output.append((f"Low {level}", current_low))

        # === Lower Levels (Level 0, -1, -2)
        current_low = low
        for level in range(0, levels_down):
            level_id = f"0" if level == 0 else f"-{level}"
            prev_low = current_low
            current_low = round(prev_low - range_val, 2)
            current_high = prev_low
            mp = round(current_low + sp1, 2)

            levels_output.append((f"🔻 Level {level_id}", ""))
            levels_output.append((f"High {level_id}", current_high))
            for i in reversed(range(1, len(sp_levels))):
                levels_output.append((f"mp{level_id} - sp{i+1}", round(current_high - sp_levels[i], 2)))
            levels_output.append((f"mp{level_id}", mp))
            for i in range(1, len(sp_levels)):
                levels_output.append((f"mp{level_id} + sp{i+1}", round(current_low + sp_levels[i], 2)))
            levels_output.append((f"Low {level_id}", current_low))
    
        df = pd.DataFrame(levels_output, columns=["Label", "Level"]).set_index("Label")

        # Keep only numeric level rows
        df = df[pd.to_numeric(df['Level'], errors='coerce').notna()]


        # Separate numeric and non-numeric
        df_numeric = df[pd.to_numeric(df['Level'], errors='coerce').notna()].copy()
        df_non_numeric = df[pd.to_numeric(df['Level'], errors='coerce').isna()].copy()

        # Sort numeric rows
        df_numeric_sorted = df_numeric.sort_values(by="Level", ascending=False)

        
        # ➡️ Add arrow for the closest level to current_price
        if current_price is not None:
            diffs = (df_numeric_sorted['Level'] - current_price).abs()
            closest_label = diffs.idxmin()
            df_numeric_sorted["→"] = ""
            df_numeric_sorted.loc[closest_label, "→"] = "◀️"
            
        else:
            df_numeric_sorted["→"] = ""

        # Add arrow column to non-numeric rows too (blank)
        df_non_numeric["→"] = ""

        # Combine back
        df_sorted = pd.concat([df_non_numeric, df_numeric_sorted])

        return df_sorted


    # === MONTHLY RANGE ===
    st.markdown("## 🗓️ Monthly")
    today = pd.Timestamp.today()
    first_of_month = today.replace(day=1)
    panchak_df = load_panchak_data()
    

    recent_panchak = panchak_df[
        (panchak_df['Start Date'] <= today)
    ].sort_values("Start Date", ascending=False)

    if not recent_panchak.empty:
        row = recent_panchak.iloc[0]
        start_date = row['Start Date']
        end_date = row['End Date']

        ohlc = get_combined_index_data("Nifty", start_date, end_date)
        if ohlc is None or ohlc.empty:
            st.warning(f"No OHLC data available from {start_date.date()} to {end_date.date()}")
            df_monthly = pd.DataFrame()
        else:
            eod_close = ohlc['Close'].dropna().iloc[-1]
            high = ohlc['High'].max()
            low = ohlc['Low'].min()
            range_val = round(high - low, 2)

            sp_levels = []
            current = range_val
            while True:
                sp = round(current / 2, 2)
                if sp < 10:
                    break
                sp_levels.append(sp)
                current = sp

            if len(sp_levels) >= 2:
                st.markdown(f"**High:** {high:.2f} | **Low:** {low:.2f} | **Range:** {range_val:.2f}")
                df_monthly = generate_custom_levels(high, low, sp_levels, current_price=eod_close)
            else:
                st.warning("Not enough SP levels to calculate for Monthly Range.")
                df_monthly = pd.DataFrame()

            st.info("No Panchak range found in current month.")

    # === FORTNIGHTLY RANGE ===
    st.markdown("## 🌕 Fortnightly")
    moon_df = load_moon_data()
    moon_df = moon_df.sort_values("Date")
    today = pd.Timestamp.today()

    recent_moon = moon_df[moon_df['Date'] <= today].iloc[-1]
    next_moon = moon_df[moon_df['Date'] > recent_moon['Date']].iloc[0]
    start_date = recent_moon['Date']
    end_date = next_moon['Date']

    ohlc = get_combined_index_data("Nifty", start_date, end_date)
    eod_close = ohlc['Close'].dropna().iloc[-1]
    high = ohlc['High'].max()
    low = ohlc['Low'].min()
    range_val = round(high - low, 2)

    sp_levels = []
    current = range_val
    while True:
        sp = round(current / 2, 2)
        if sp < 10:
            break
        sp_levels.append(sp)
        current = sp

    if len(sp_levels) >= 2:
        st.markdown(f"**Period:** {start_date.date()} to {end_date.date()}")
        st.markdown(f"**High:** {high:.2f} | **Low:** {low:.2f} | **Range:** {range_val:.2f}")
        df_fortnight = generate_custom_levels(high, low, sp_levels, current_price=eod_close)

    else:
        st.warning("Not enough SP levels for Fortnightly Range.")

    # === WEEKLY RANGE ===
    st.markdown("## 📅 Weekly")
    today = pd.Timestamp.today()
    # Create list of past Mondays (last 12 weeks)
    mondays = [today - pd.Timedelta(days=today.weekday() + 7*i) for i in range(12)]
    mondays = sorted([d.date() for d in mondays], reverse=True)

    selected_monday = st.selectbox("Select Monday:", mondays)
    monday_date = pd.to_datetime(selected_monday)

    start_date = monday_date
    end_date = monday_date + pd.Timedelta(days=1)

    ohlc = get_combined_index_data("Nifty", start_date, end_date)
    eod_close = ohlc['Close'].dropna().iloc[-1]


    # Group/aggregate Monday OHLC
    ohlc_day = ohlc.groupby(ohlc.index.date).agg({'High': 'max', 'Low': 'min'})
    monday_only = monday_date.date()

    if monday_only not in ohlc_day.index:
        st.warning(f"No data for Monday: {monday_only}")
    else:
        row = ohlc_day.loc[monday_only]
        high = row['High']
        low = row['Low']
        range_val = round(high - low, 2)

        sp_levels = []
        current = range_val
        while True:
            sp = round(current / 2, 2)
            if sp < 10:
                break
            sp_levels.append(sp)
            current = sp

        if len(sp_levels) >= 2:
            st.markdown(f"**Monday Date:** {monday_date.date()}")
            st.markdown(f"**High:** {high:.2f} | **Low:** {low:.2f} | **Range:** {range_val:.2f}")
            df_weekly = generate_custom_levels(high, low, sp_levels, current_price=eod_close)

        else:
            st.warning("Not enough SP levels for Weekly Range.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🗓️ Monthly")
        styled_df1 = df_monthly.style.hide(axis="index").apply(highlight_range_levels, axis=1).format(precision=2)
        html_table = styled_df1.to_html()
        st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### 🌕 Fortnightly")
        styled_df2 = df_fortnight.style.hide(axis="index").apply(highlight_range_levels, axis=1).format(precision=2)
        html_table = styled_df2.to_html()
        st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)


    with col3:
        st.markdown("### 📅 Weekly")
        styled_df3 = df_weekly.style.hide(axis="index").apply(highlight_range_levels, axis=1).format(precision=2)
        html_table = styled_df3.to_html()
        st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Daily Report":
    st.markdown("## 📆 Daily Numerology Report")

    numerology_df = load_numerology_data()
    numerology_df['date'] = pd.to_datetime(numerology_df['date'])

    today = pd.Timestamp.today().normalize()
    selected_date = st.date_input("Select Date", value=today)
    row = numerology_df[numerology_df['date'] == pd.to_datetime(selected_date)]

    if row.empty:
        st.warning("No numerology data available for this date.")
        st.stop()

    row = row.iloc[0]

    bn = row['BN']
    dn_formatted = row['DN (Formatted)']
    sn = row['SN']
    hp = row['HP']
    dayn = row['Day Number']

    html = f"""
    <div style="font-family:'Segoe UI', sans-serif; border:1px solid #ccc; box-shadow: 0px 4px 15px rgba(0,0,0,0.1); border-radius:10px; padding:40px; margin:auto; max-width:750px; background:#fff;">
        <h2 style="text-align:center; margin-bottom:10px;">📄 Daily Report</h2>
        <h4 style="text-align:center; color:#666; margin-top:0;">Date: {selected_date.strftime('%Y-%m-%d')}</h4>
        <hr style="margin:20px 0;">
        <p> ➤ Birth Number (BN): <span style="color:#000;">{bn}</span></p>
        <p> ➤ DN - SN: <span style="color:#000;">{dn_formatted} - {sn}</span></p>
        <p> ➤ Sun Number (SN): <span style="color:#000;">{sn}</span></p>
        <p> ➤ Hidden Personality (HP): <span style="color:#000;">{hp}</span></p>
        <p> ➤ BN - DN: <span style="color:#000;">{bn} - {dn_formatted}</span></p>
        <hr style="margin:30px 0;">
        <h6>Day Number Comparisons: <span style="color:#000;">{dayn}</span></h6>
        <ul style="font-size:14px; padding-left:25px;">
            <li><strong>Day Number - BN:</strong> {dayn} - {bn}</li>
            <li><strong>Day Number - DN:</strong> {dayn} - {dn_formatted}</li>
            <li><strong>Day Number - SN:</strong> {dayn} -  {sn}</li>
            <li><strong>Day Number - HP:</strong> {dayn} - {hp}</li>
            <li><strong>HP - SN:</strong> {hp} - {sn}</li>
        </ul>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

if filter_mode == "Navamasa":
    if st.session_state.username not in allowed_users_for_astro:
        st.stop()

    st.header("🌌 Navamasa (D1 & D9 Planetary Chart)")

    import swisseph as swe
    import datetime
    import pandas as pd

    # === Setup Swiss Ephemeris ===
    swe.set_ephe_path("C:/ephe")  # Replace with actual path
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)

    # === User Input ===
    birth_dt = st.date_input("Select Date", value=datetime.date.today())
    birth_time = st.time_input("Select Time", value=datetime.time(9, 0))
    datetime_obj = datetime.datetime.combine(birth_dt, birth_time)

    timezone_offset = 5.5  # IST
    latitude = 19.076
    longitude = 72.8777

    utc_dt = datetime_obj - datetime.timedelta(hours=timezone_offset)
    jd = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour + utc_dt.minute / 60)

    # === Constants ===
    signs = [
        'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
        'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'
    ]

    sign_lords = [
        'Mars', 'Venus', 'Mercury', 'Moon', 'Sun', 'Mercury',
        'Venus', 'Mars', 'Jupiter', 'Saturn', 'Saturn', 'Jupiter'
    ]    

    custom_d1_map = {sign: i+1 for i, sign in enumerate(signs)}
    custom_d9_map = {
        "Aries": 1, "Taurus": 2, "Gemini": 3, "Leo": 4, "Cancer": 5, "Virgo": 6,
        "Libra": 7, "Scorpio": 8, "Sagittarius": 9, "Capricorn": 10, "Aquarius": 11, "Pisces": 12
    }

    # === Helper: Navamsa D9 Sign Logic ===
    def get_d9_sign_index(longitude_deg):
        sign_index = int(longitude_deg // 30)
        pos_in_sign = longitude_deg % 30
        navamsa_index = int(pos_in_sign // (30 / 9))
        if sign_index in [0, 3, 6, 9]: start = sign_index
        elif sign_index in [1, 4, 7, 10]: start = (sign_index + 8) % 12
        else: start = (sign_index + 4) % 12
        return (start + navamsa_index) % 12

    # === Planetary Calculations ===
    rows = []
    longitudes = {}
    flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH
    navamsa_size = 30 / 9

    for name, pid in planets.items():
        lon = swe.calc_ut(jd, pid, flag)[0][0]
        if name == "Ketu":
            lon = (swe.calc_ut(jd, swe.TRUE_NODE, flag)[0][0] + 180) % 360

        longitudes[name] = round(lon, 2)

        sign_index = int(lon // 30)
        rashi = signs[sign_index]
        d1_sign_number = custom_d1_map[rashi]
        rashi_lord = sign_lords[sign_index]

        pos_in_sign = lon % 30
        deg_in_sign = f"{int(pos_in_sign)}° {int((pos_in_sign % 1) * 60):02d}'"

        nak_index = int((lon % 360) // (360 / 27))
        nakshatra = nakshatras[nak_index]

        d9_sign_index = get_d9_sign_index(lon)
        d9_sign = signs[d9_sign_index]
        d9_sign_number = custom_d9_map[d9_sign]
        deg_in_d9_total = (pos_in_sign % navamsa_size) * 9
        deg_in_d9_sign = f"{int(deg_in_d9_total)}° {int((deg_in_d9_total % 1) * 60):02d}'"
        d9_long = d9_sign_index * 30 + deg_in_d9_total

        rows.append({
            "Planet": name,
            "Longitude (deg)": round(lon, 2),
            "Sign": rashi,
            "D1 Sign #": d1_sign_number,
            "Sign Lord": rashi_lord,
            "Nakshatra": nakshatra,
            "Degrees in Sign": deg_in_sign,
            "Navamsa (D9) Sign": d9_sign,
            "D9 Sign #": d9_sign_number,
            "Degrees in D9 Sign": deg_in_d9_sign,
            "D9 Longitude (deg)": round(d9_long, 2)
        })

    df = pd.DataFrame(rows)

    # === D1/D9 Degree Difference Tables ===
    def angular_diff(a, b): return round((b - a) % 360, 2)
    def create_diff_table(deg_dict): return pd.DataFrame([
        [angular_diff(deg_dict[p1], deg_dict[p2]) for p2 in deg_dict]
        for p1 in deg_dict
    ], index=deg_dict.keys(), columns=deg_dict.keys())

    d1_table = create_diff_table(longitudes)
    d9_table = create_diff_table({row["Planet"]: row["D9 Longitude (deg)"] for row in rows})

    # === Type Classification (Movable/Fixed/Dual) ===
    def classify(num):
        if num in [1, 4, 7, 10]: return "Movable"
        elif num in [2, 5, 8, 11]: return "Fixed"
        elif num in [3, 6, 9, 12]: return "Dual"
        return "Unknown"

    df["D1 Type"] = df["D1 Sign #"].apply(classify)
    df["D9 Type"] = df["D9 Sign #"].apply(classify)

    def summary(df, col):
        cats = ["Movable", "Fixed", "Dual"]
        return pd.DataFrame([{
            "Category": cat,
            "Planets": ", ".join(df[df[col] == cat]["Planet"]),
            "Total": len(df[df[col] == cat])
        } for cat in cats])

    d1_class = summary(df, "D1 Type")
    d9_class = summary(df, "D9 Type")

    # === Render All as Styled HTML Tables ===
    st.markdown("### 🪐 Planetary Positions (D1 + D9)")
    st.markdown(f'<div class="scroll-table">{df.to_html(index=False)}</div>', unsafe_allow_html=True)

    st.markdown("### 📘 D1 Longitudinal Differences (0–360°)")
    st.markdown(f'<div class="scroll-table">{d1_table.to_html()}</div>', unsafe_allow_html=True)

    st.markdown("### 📙 D9 Longitudinal Differences (0–360°)")
    st.markdown(f'<div class="scroll-table">{d9_table.to_html()}</div>', unsafe_allow_html=True)

    st.markdown("### 📊 D1 Sign Type Classification")
    st.markdown(f'<div class="scroll-table">{d1_class.to_html(index=False)}</div>', unsafe_allow_html=True)

    st.markdown("### 📊 D9 Sign Type Classification")
    st.markdown(f'<div class="scroll-table">{d9_class.to_html(index=False)}</div>', unsafe_allow_html=True)

elif filter_mode == "Planetary Conjunctions":
    st.header("🪐 Planetary Conjunctions (±1°) with Nakshatra, Pada & Zodiac")

    import swisseph as swe
    import datetime
    import pandas as pd
    from itertools import combinations

    # Setup
    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)

    # Date Range Input
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.date(2025, 6, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.date(2025, 6, 30))

    # Planet order by speed
    planet_order = ["Moon", "Mercury", "Venus", "Sun", "Mars",
                    "Jupiter", "Rahu", "Ketu", "Saturn",
                    "Uranus", "Neptune", "Pluto"]
    planet_rank = {planet: i for i, planet in enumerate(planet_order)}

    planets = {
        "Sun": swe.SUN,
        "Moon": swe.MOON,
        "Mars": swe.MARS,
        "Mercury": swe.MERCURY,
        "Jupiter": swe.JUPITER,
        "Venus": swe.VENUS,
        "Saturn": swe.SATURN,
        "Rahu": swe.TRUE_NODE,
        "Ketu": swe.TRUE_NODE,
        "Uranus": swe.URANUS,
        "Neptune": swe.NEPTUNE,
        "Pluto": swe.PLUTO
    }

    nakshatras = [
        "Ashwini", "Bharani", "Krittika", "Rohini", "Mrigashira", "Ardra",
        "Punarvasu", "Pushya", "Ashlesha", "Magha", "Purva Phalguni",
        "Uttara Phalguni", "Hasta", "Chitra", "Swati", "Vishakha",
        "Anuradha", "Jyeshtha", "Mula", "Purva Ashadha", "Uttara Ashadha",
        "Shravana", "Dhanishta", "Shatabhisha", "Purva Bhadrapada",
        "Uttara Bhadrapada", "Revati"
    ]

    zodiacs = [
        "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
        "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
    ]

    def get_nakshatra_and_pada(degree):
        nak_index = int(degree // (360 / 27))
        pada = int((degree % (360 / 27)) // 3.3333) + 1
        return nakshatras[nak_index], pada

    def get_zodiac(degree):
        return zodiacs[int(degree // 30)]

    def signed_diff(fast_deg, slow_deg):
        diff = (fast_deg - slow_deg + 360) % 360
        if diff > 180:
            diff -= 360
        return round(diff, 2)

    def get_planet_data(jd, name, pid):
        flag = swe.FLG_SIDEREAL | swe.FLG_SPEED
        lon, speed = swe.calc_ut(jd, pid, flag)[0][0:2]
        if name == "Ketu":
            lon = (swe.calc_ut(jd, swe.TRUE_NODE, flag)[0][0] + 180) % 360
            speed = -speed
        return round(lon, 2), round(speed, 4)

    # Iterate over days
    current = start_date
    results = []

    while current <= end_date:
        jd = swe.julday(current.year, current.month, current.day, 0)
        planet_data = {}

        for name, pid in planets.items():
            lon, speed = get_planet_data(jd, name, pid)
            planet_data[name] = {"deg": lon, "speed": speed}

        for p1, p2 in combinations(planet_data.keys(), 2):
            r1, r2 = planet_rank.get(p1, 999), planet_rank.get(p2, 999)
            fast, slow = (p1, p2) if r1 < r2 else (p2, p1)

            d1 = planet_data[fast]["deg"]
            d2 = planet_data[slow]["deg"]
            diff = signed_diff(d1, d2)

            if abs(diff) <= 1.0:
                midpoint = (d1 + d2) / 2 % 360
                nakshatra, pada = get_nakshatra_and_pada(midpoint)
                zodiac = get_zodiac(midpoint)

                results.append({
                    "Date": current.strftime("%Y-%m-%d"),
                    "Planet 1": f"{fast} ({d1}° / {planet_data[fast]['speed']}°/day)",
                    "Planet 2": f"{slow} ({d2}° / {planet_data[slow]['speed']}°/day)",
                    "Degree Difference": diff,
                    "Nakshatra": nakshatra,
                    "Pada": pada,
                    "Zodiac": zodiac,
                    "Day Type": "Red Day" if diff < 0 else "Green Day"
                })

        current += datetime.timedelta(days=1)

    df = pd.DataFrame(results)

    if df.empty:
        st.warning("No conjunctions found within ±1° for the selected date range.")
    else:
        st.markdown("### 🔭 Conjunction Report")

        def styled_html_table(df):
            rows = []
            for _, row in df.iterrows():
                bg_color = "#ffe6e6" if row["Day Type"] == "Red Day" else "#e6ffe6"
                row_html = "<tr style='background-color:{}'>".format(bg_color)
                for val in row:
                    row_html += f"<td>{val}</td>"
                row_html += "</tr>"
                rows.append(row_html)

            header = "".join([f"<th>{col}</th>" for col in df.columns])
            html_table = f"""
            <div class="scroll-table">
                <table>
                    <thead><tr>{header}</tr></thead>
                    <tbody>
                        {''.join(rows)}
                    </tbody>
                </table>
            </div>
            """
            return html_table

        st.markdown(styled_html_table(df), unsafe_allow_html=True)

elif filter_mode == "Planetary Report":
    st.header("📅 Daily Planetary Summary Report (D1 + D9 + Conjunctions)")

    import swisseph as swe
    import datetime
    import pandas as pd
    from itertools import combinations

    # === Input Range ===
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.date(2025, 6, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.date(2025, 6, 30))

    swe.set_ephe_path("C:/ephe")  
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)
    timezone_offset = 5.5  

    planets = {
        "Sun": swe.SUN,
        "Moon": swe.MOON,
        "Mars": swe.MARS,
        "Mercury": swe.MERCURY,
        "Jupiter": swe.JUPITER,
        "Venus": swe.VENUS,
        "Saturn": swe.SATURN,
        "Rahu": swe.TRUE_NODE,
        "Ketu": swe.TRUE_NODE,
        "Uranus": swe.URANUS,
        "Neptune": swe.NEPTUNE,
        "Pluto": swe.PLUTO
    }

    signs = [
        'Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
        'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces'
    ]
    custom_d1_map = {sign: i + 1 for i, sign in enumerate(signs)}
    custom_d9_map = {
        "Aries": 1, "Taurus": 2, "Gemini": 3, "Leo": 4,
        "Cancer": 5, "Virgo": 6, "Libra": 7, "Scorpio": 8,
        "Sagittarius": 9, "Capricorn": 10, "Aquarius": 11, "Pisces": 12
    }
    sign_types = {
        "Movable": [1, 4, 7, 10],
        "Fixed":   [2, 5, 8, 11],
        "Dual":    [3, 6, 9, 12]
    }

    planet_order = [
        "Moon", "Mercury", "Venus", "Sun", "Mars",
        "Jupiter", "Rahu", "Ketu", "Saturn",
        "Uranus", "Neptune", "Pluto"
    ]
    planet_rank = {planet: i for i, planet in enumerate(planet_order)}

    def get_d9_sign_index(longitude_deg):
        sign_index = int(longitude_deg // 30)
        pos_in_sign = longitude_deg % 30
        navamsa_index = int(pos_in_sign // (30 / 9))
        if sign_index in [0, 3, 6, 9]:
            start = sign_index
        elif sign_index in [1, 4, 7, 10]:
            start = (sign_index + 8) % 12
        else:
            start = (sign_index + 4) % 12
        return (start + navamsa_index) % 12
    
    def classify_sign_type(sign_number):
        if sign_number in sign_types["Movable"]:
            return "Movable"
        elif sign_number in sign_types["Fixed"]:
            return "Fixed"
        elif sign_number in sign_types["Dual"]:
            return "Dual"
        return "Unknown"

    def get_conjunction_day_info(jd):
        """
        Returns (day_type, reason). If any two planets are within ±1° on this JD:
         - day_type = "Red Day" if (fast_lon - slow_lon) < 0,
                      "Green Day" if > 0.
         - reason = "<Fast>-<Slow>: <signed_diff>°"
        Otherwise returns ("-", "").
        """
        flag = swe.FLG_SIDEREAL | swe.FLG_SPEED

        planet_data = {}
        for name, pid in planets.items():
            lon, speed = swe.calc_ut(jd, pid, flag)[0][0:2]
            if name == "Ketu":
                true_node_lon = swe.calc_ut(jd, swe.TRUE_NODE, flag)[0][0]
                lon = (true_node_lon + 180) % 360
                speed = -speed
            planet_data[name] = {"lon": lon, "speed": speed}

        # 2) Check each pair for |difference| ≤ 1°
        for p1, p2 in combinations(planet_data.keys(), 2):
            r1 = planet_rank.get(p1, 999)
            r2 = planet_rank.get(p2, 999)
            fast, slow = (p1, p2) if r1 < r2 else (p2, p1)

            d1 = planet_data[fast]["lon"]
            d2 = planet_data[slow]["lon"]
            diff = (d1 - d2 + 360) % 360
            if diff > 180:
                diff -= 360
            diff = round(diff, 2)

            if abs(diff) <= 1.0:
                day_type = "Red Day" if diff < 0 else "Green Day"
                reason = f"{fast}-{slow}: {diff}°"
                return day_type, reason

        return "-", ""

    # === Helper: Minimal Absolute Circular Difference ===
    def minimal_abs_diff(a, b):
        """
        Given two angles a, b (0–360), return the minimal |a-b| around the circle (0–180).
        """
        raw = abs((a - b + 360) % 360)
        return min(raw, 360 - raw)

    # === Build the Report Rows ===
    report_rows = []
    current_date = start_date

    while current_date <= end_date:
        jd = swe.julday(current_date.year, current_date.month, current_date.day, 0)

        # 2) Conjunction “Day Type” & “Reason”
        day_type, reason = get_conjunction_day_info(jd)

        # 3) Collect D1 & D9 classification and longitudes for all planets
        d1_types = []
        d9_types = []
        moon_d1_type = mercury_d1_type = None
        moon_d1_lon = mercury_d1_lon = None
        moon_d9_lon = mercury_d9_lon = None

        for name, pid in planets.items():
            # 3a) D1 (sidereal) longitude at 00:00 UTC
            lon = swe.calc_ut(jd, pid, swe.FLG_SIDEREAL | swe.FLG_SWIEPH)[0][0]
            if name == "Ketu":
                true_node_lon = swe.calc_ut(jd, swe.TRUE_NODE, swe.FLG_SIDEREAL | swe.FLG_SWIEPH)[0][0]
                lon = (true_node_lon + 180) % 360

            # Record D1 longitudes for Moon & Mercury
            if name == "Moon":
                moon_d1_lon = lon
            if name == "Mercury":
                mercury_d1_lon = lon

            # Classify D1 sign type
            sign_index = int(lon // 30)
            rashi = signs[sign_index]
            d1_sign_num = custom_d1_map[rashi]
            d1_type = classify_sign_type(d1_sign_num)
            d1_types.append(d1_type)

            if name == "Moon":
                moon_d1_type = d1_type
            if name == "Mercury":
                mercury_d1_type = d1_type

            # 3b) Compute D9 (Navamsa) longitude:
            navamsa_size = 30 / 9
            pos_in_sign = lon % 30
            nav_sign_idx = get_d9_sign_index(lon)
            d9_deg_in_sign = (pos_in_sign % navamsa_size) * 9
            d9_lon = nav_sign_idx * 30 + d9_deg_in_sign

            # Record D9 longitudes for Moon & Mercury
            if name == "Moon":
                moon_d9_lon = d9_lon
            if name == "Mercury":
                mercury_d9_lon = d9_lon

            # Classify D9 sign
            d9_sign = signs[nav_sign_idx]
            d9_sign_num = custom_d9_map[d9_sign]
            d9_type = classify_sign_type(d9_sign_num)
            d9_types.append(d9_type)

        # 4) Count Movable/Fixed in D1 & D9
        d1_movable_count = d1_types.count("Movable")
        d1_fixed_count   = d1_types.count("Fixed")
        d9_movable_count = d9_types.count("Movable")
        d9_fixed_count   = d9_types.count("Fixed")

        # 5a) D1 combined classification
        if moon_d1_type == mercury_d1_type:
            mm_d1_status = f"Moon & Mercury: {moon_d1_type}"
        else:
            mm_d1_status = f"Moon: {moon_d1_type}, Mercury: {mercury_d1_type}"

        # 5b) D9 combined classification
        moon_d9_type = classify_sign_type(custom_d9_map[signs[int(moon_d9_lon // 30)]])
        mercury_d9_type = classify_sign_type(custom_d9_map[signs[int(mercury_d9_lon // 30)]])

        if moon_d9_type == mercury_d9_type:
            mm_d9_status = f"Moon & Mercury: {moon_d9_type}"
        else:
            mm_d9_status = f"Moon: {moon_d9_type}, Mercury: {mercury_d9_type}"


        # 6) Compute Moon→Mercury and Mercury→Moon diffs for D1
        #    → raw signed diff (0–360), convert to (−180…+180)
        raw_m2me_d1 = (moon_d1_lon - mercury_d1_lon + 360) % 360
        if raw_m2me_d1 > 180:
            raw_m2me_d1 -= 360
        raw_m2me_d1 = round(raw_m2me_d1, 2)

        raw_me2m_d1 = (mercury_d1_lon - moon_d1_lon + 360) % 360
        if raw_me2m_d1 > 180:
            raw_me2m_d1 -= 360
        raw_me2m_d1 = round(raw_me2m_d1, 2)

        # Minimal absolute difference (0–180)
        min_m2me_d1 = minimal_abs_diff(moon_d1_lon, mercury_d1_lon)
        min_me2m_d1 = minimal_abs_diff(mercury_d1_lon, moon_d1_lon)

        # If within ±1° of {0, 90, 180}, show that key angle; otherwise show actual
        def label_diff(min_diff):
            for target in (0, 90, 180):
                if abs(min_diff - target) <= 1:
                    return f"{target}°"
            return f"{min_diff:.2f}°"

        label_m2me_d1 = label_diff(min_m2me_d1)
        label_me2m_d1 = label_diff(min_me2m_d1)
        mm_d1_label = f"{label_m2me_d1} / {label_me2m_d1}"

        # 7) Compute Moon→Mercury and Mercury→Moon diffs for D9
        raw_m2me_d9 = (moon_d9_lon - mercury_d9_lon + 360) % 360
        if raw_m2me_d9 > 180:
            raw_m2me_d9 -= 360
        raw_m2me_d9 = round(raw_m2me_d9, 2)

        raw_me2m_d9 = (mercury_d9_lon - moon_d9_lon + 360) % 360
        if raw_me2m_d9 > 180:
            raw_me2m_d9 -= 360
        raw_me2m_d9 = round(raw_me2m_d9, 2)

        min_m2me_d9 = minimal_abs_diff(moon_d9_lon, mercury_d9_lon)
        min_me2m_d9 = minimal_abs_diff(mercury_d9_lon, moon_d9_lon)

        label_m2me_d9 = label_diff(min_m2me_d9)
        label_me2m_d9 = label_diff(min_me2m_d9)
        mm_d9_label = f"{label_m2me_d9} / {label_me2m_d9}"

        # 8) Append row
        report_rows.append({
            "Date": current_date.strftime("%Y-%m-%d"),
            "Day Type": day_type,
            "Reason": reason,
            "D1 Movable": d1_movable_count,
            "D9 Movable": d9_movable_count,
            "D1 Fixed": d1_fixed_count,
            "D9 Fixed": d9_fixed_count,
            "Moon & Mercury D1": mm_d1_label,
            "Moon & Mercury D9": mm_d9_label,
            "Moon & Mercury D1 Type": mm_d1_status,
            "Moon & Mercury D9 Type": mm_d9_status
            })

        current_date += datetime.timedelta(days=1)

    # Create DataFrame
    df = pd.DataFrame(report_rows)

    # === Render Highlighted HTML Table ===
    def render_highlighted_report(df):
        rows_html = []
        for _, row in df.iterrows():
            status = row["Moon & Mercury D1 Type"]
            if status.strip() == "Moon & Mercury: Movable":
                style = "background-color: black; color: white;"
            elif status.strip() == "Moon & Mercury: Fixed":
                style = "background-color: #ffcccc;"
            else:
                style = ""


            cells = "".join([f"<td>{row[col]}</td>" for col in df.columns])
            rows_html.append(f"<tr style='{style}'>{cells}</tr>")

        header_html = "".join([f"<th>{col}</th>" for col in df.columns])
        html = f"""
        <div class="scroll-table">
            <table>
                <thead><tr>{header_html}</tr></thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
        """
        return html

    st.markdown("### 📊 Final Daily Report Table")
    st.markdown(render_highlighted_report(df), unsafe_allow_html=True)

elif filter_mode == "Moon–Mercury Aspects":
    st.header("🌕 Mercury–Moon Aspects (D1 & D9 ±1°)")

    import swisseph as swe
    import pandas as pd
    import re
    from datetime import datetime, timedelta
    from collections import defaultdict

    # === Setup ===
    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)

    LAT = 19.076
    LON = 72.8777
    TZ_OFFSET = 5.5  # IST

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2025, 6, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 6, 30))

    # Time Ranges
    D1_START_HOUR, D1_END_HOUR = 0, 23
    D9_START_HOUR, D9_END_HOUR = 8, 16

    # Helpers
    def angular_diff(from_deg, to_deg):
        return round((to_deg - from_deg) % 360, 2)

    def check_aspects(from_deg, to_deg):
        angles = [0, 90, 180]
        matched = []
        diff1 = angular_diff(from_deg, to_deg)
        diff2 = angular_diff(to_deg, from_deg)
        for angle in angles:
            if abs(diff1 - angle) <= 1:
                matched.append(f"Moon→Mercury ≈ {angle}°")
            if abs(diff2 - angle) <= 1:
                matched.append(f"Mercury→Moon ≈ {angle}°")
        return matched if matched else ["nan"]

    def get_d9_longitude(longitude_deg):
        sign_index = int(longitude_deg // 30)
        pos_in_sign = longitude_deg % 30
        navamsa_index = int(pos_in_sign // (30 / 9))
        if sign_index in [0, 3, 6, 9]:
            start = sign_index
        elif sign_index in [1, 4, 7, 10]:
            start = (sign_index + 8) % 12
        else:
            start = (sign_index + 4) % 12
        d9_sign_index = (start + navamsa_index) % 12
        deg_in_navamsa = pos_in_sign % (30 / 9)
        d9_long = d9_sign_index * 30 + deg_in_navamsa * 9
        return d9_long

    def extract_angles(aspect_str):
        if aspect_str == "nan":
            return set()
        return set(re.findall(r"(Moon→Mercury|Mercury→Moon) ≈ (\d+)°", aspect_str))

    # === Process Dates ===
    results_d1 = []
    results_d9 = []

    for day in pd.date_range(start_date, end_date):
        # D1 window
        for hour in range(D1_START_HOUR, D1_END_HOUR + 1):
            dt = datetime(day.year, day.month, day.day, hour, 0)
            utc_dt = dt - timedelta(hours=TZ_OFFSET)
            jd = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour + utc_dt.minute / 60)
            flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH
            moon = swe.calc_ut(jd, swe.MOON, flag)[0][0]
            mercury = swe.calc_ut(jd, swe.MERCURY, flag)[0][0]
            aspects = check_aspects(moon, mercury)
            if aspects != ["nan"]:
                results_d1.append({
                    "Date": dt.date(),
                    "Time (IST)": dt.time(),
                    "D1 Aspects": ", ".join(aspects)
                })

        # D9 window
        for hour in range(D9_START_HOUR, D9_END_HOUR + 1):
            dt = datetime(day.year, day.month, day.day, hour, 0)
            utc_dt = dt - timedelta(hours=TZ_OFFSET)
            jd = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour + utc_dt.minute / 60)
            flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH
            moon = swe.calc_ut(jd, swe.MOON, flag)[0][0]
            mercury = swe.calc_ut(jd, swe.MERCURY, flag)[0][0]
            moon_d9 = get_d9_longitude(moon)
            mercury_d9 = get_d9_longitude(mercury)
            aspects = check_aspects(moon_d9, mercury_d9)
            if aspects != ["nan"]:
                results_d9.append({
                    "Date": dt.date(),
                    "Time (IST)": dt.time(),
                    "D9 Aspects": ", ".join(aspects)
                })

    # === Group By Date ===
    all_dates = pd.date_range(start=start_date, end=end_date).date
    grouped_d1 = defaultdict(set)
    grouped_d1_times = defaultdict(list)
    grouped_d9 = defaultdict(set)
    grouped_d9_times = defaultdict(list)

    for row in results_d1:
        d = row["Date"]
        grouped_d1[d].update(extract_angles(row["D1 Aspects"]))
        grouped_d1_times[d].append(str(row["Time (IST)"]))

    for row in results_d9:
        d = row["Date"]
        grouped_d9[d].update(extract_angles(row["D9 Aspects"]))
        grouped_d9_times[d].append(str(row["Time (IST)"]))

    # === Build Summary ===
    summary = []
    for d in all_dates:
        d1_text = ", ".join([f"{dir} ≈ {deg}°" for dir, deg in sorted(grouped_d1[d])]) if grouped_d1[d] else "0"
        d9_text = ", ".join([f"{dir} ≈ {deg}°" for dir, deg in sorted(grouped_d9[d])]) if grouped_d9[d] else "0"
        d1_times = sorted(set(grouped_d1_times[d]))[0] if grouped_d1_times[d] else "0"
        d9_times = ", ".join(sorted(set(grouped_d9_times[d]))) if grouped_d9_times[d] else "0"

        # === NEW: Get D1 and D9 signs for Moon & Mercury
        utc_dt = datetime(d.year, d.month, d.day, 0, 0) - timedelta(hours=TZ_OFFSET)
        jd = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour)

        moon_d1 = swe.calc_ut(jd, swe.MOON, swe.FLG_SIDEREAL | swe.FLG_SWIEPH)[0][0]
        mercury_d1 = swe.calc_ut(jd, swe.MERCURY, swe.FLG_SIDEREAL | swe.FLG_SWIEPH)[0][0]

        moon_d1_sign = int(moon_d1 // 30)
        mercury_d1_sign = int(mercury_d1 // 30)

        # D9 conversion
        def get_sign_from_deg(deg):
            signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
                    'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
            return signs[int(deg // 30)]

        moon_d9 = get_d9_longitude(moon_d1)
        mercury_d9 = get_d9_longitude(mercury_d1)

        moon_d1_sign_name = get_sign_from_deg(moon_d1)
        mercury_d1_sign_name = get_sign_from_deg(mercury_d1)

        moon_d9_sign_name = get_sign_from_deg(moon_d9)
        mercury_d9_sign_name = get_sign_from_deg(mercury_d9)

        # Check if they're in Gemini, Virgo, or Cancer
        target_signs = ["Gemini", "Virgo", "Cancer"]

        def build_status(moon_sign, mercury_sign):
            parts = []
            if moon_sign in target_signs:
                parts.append(f"Moon: {moon_sign}")
            if mercury_sign in target_signs:
                parts.append(f"Mercury: {mercury_sign}")
            return ", ".join(parts) if parts else "None"

        d1_signs_status = build_status(moon_d1_sign_name, mercury_d1_sign_name)
        d9_signs_status = build_status(moon_d9_sign_name, mercury_d9_sign_name)

        summary.append({
            "Date": d,
            "D1 Aspects": d1_text,
            "D1 Aspect Time(s)": d1_times,
            "Moon–Mercury D1 Signs": d1_signs_status,
            "D9 Aspects": d9_text,
            "D9 Aspect Time(s)": d9_times,
            "Moon–Mercury D9 Signs": d9_signs_status
        })


        df_summary = pd.DataFrame(summary)

    # 🔥 Filter out rows where both D1 and D9 aspects are "0"
    df_summary_filtered = df_summary[
        (df_summary["D1 Aspects"] != "0") | (df_summary["D9 Aspects"] != "0")
    ]

    st.markdown("### 📅 Final Moon–Mercury Aspect Table")
    def render_aspect_table(df):
        rows_html = []
        for _, row in df.iterrows():
            d1 = row["D1 Aspects"]
            d9 = row["D9 Aspects"]
            if d1 != "0" or d9 != "0":
                style = "background-color: black; color: white;"
            else:
                style = ""

            row_cells = "".join([f"<td>{val}</td>" for val in row])
            rows_html.append(f"<tr style='{style}'>{row_cells}</tr>")

        header_html = "".join([f"<th>{col}</th>" for col in df.columns])
        table_html = f"""
        <div class="scroll-table">
            <table>
                <thead><tr>{header_html}</tr></thead>
                <tbody>
                    {''.join(rows_html)}
                </tbody>
            </table>
        </div>
        """
        return table_html

    st.markdown(render_aspect_table(df_summary_filtered), unsafe_allow_html=True)

elif filter_mode == "Planetary Aspects":
    st.header("🔭 Planetary Aspect Report")

    import swisseph as swe
    import pandas as pd
    from datetime import datetime, timedelta
    from collections import defaultdict

    # Setup
    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)

    LAT = 19.076
    LON = 72.8777
    TZ_OFFSET = 5.5

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2025, 6, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 6, 30))

    D1_START_HOUR, D1_END_HOUR = 0, 23
    D9_START_HOUR, D9_END_HOUR = 8, 16

    signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
             'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']

    def get_sign(lon):
        return signs[int(lon // 30)]

    def angular_diff(from_deg, to_deg):
        return round((to_deg - from_deg) % 360, 2)

    def check_aspects(from_deg, to_deg, angles, label):
        matched = []
        for angle in angles:
            if abs(angular_diff(from_deg, to_deg) - angle) <= 0.5:
                matched.append(f"{label} ≈ {angle}°")
        return matched if matched else ["nan"]

    def get_d9_longitude(lon):
        sign_index = int(lon // 30)
        pos_in_sign = lon % 30
        navamsa_index = int(pos_in_sign // (30 / 9))
        if sign_index in [0, 3, 6, 9]:
            start = sign_index
        elif sign_index in [1, 4, 7, 10]:
            start = (sign_index + 8) % 12
        else:
            start = (sign_index + 4) % 12
        d9_sign_index = (start + navamsa_index) % 12
        deg_in_navamsa = pos_in_sign % (30 / 9)
        return d9_sign_index * 30 + deg_in_navamsa * 9

    # === Aspect Config ===
    aspect_config = {
        "Sun→Ketu": {"from": "Sun", "to": "Ketu", "angles": [0, 90, 120]},
        "Venus→Ketu": {"from": "Venus", "to": "Ketu", "angles": [0, 120]},
    }

    planet_map = {
        'Sun': swe.SUN, 'Moon': swe.MOON, 'Mercury': swe.MERCURY, 'Venus': swe.VENUS,
        'Mars': swe.MARS, 'Jupiter': swe.JUPITER, 'Saturn': swe.SATURN,
        'Rahu': swe.MEAN_NODE, 'Ketu': swe.TRUE_NODE  # Ketu = Rahu + 180
    }

    results_d1 = defaultdict(list)
    results_d9 = defaultdict(list)

    for day in pd.date_range(start_date, end_date):
        # D1 loop
        for hour in range(D1_START_HOUR, D1_END_HOUR + 1):
            dt = datetime(day.year, day.month, day.day, hour)
            utc_dt = dt - timedelta(hours=TZ_OFFSET)
            jd = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour)
            flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

            longitudes = {}
            for name, code in planet_map.items():
                lon = swe.calc_ut(jd, code, flag)[0][0]
                if name == "Ketu":
                    lon = (swe.calc_ut(jd, swe.MEAN_NODE, flag)[0][0] + 180) % 360
                longitudes[name] = lon

            for label, config in aspect_config.items():
                aspects = check_aspects(longitudes[config["from"]], longitudes[config["to"]], config["angles"], label)
                if aspects != ["nan"]:
                    results_d1[dt.date()].append((label, aspects[0], dt.time()))

        # D9 loop
        for hour in range(D9_START_HOUR, D9_END_HOUR + 1):
            dt = datetime(day.year, day.month, day.day, hour)
            utc_dt = dt - timedelta(hours=TZ_OFFSET)
            jd = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour)
            flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

            longitudes = {}
            for name, code in planet_map.items():
                lon = swe.calc_ut(jd, code, flag)[0][0]
                if name == "Ketu":
                    lon = (swe.calc_ut(jd, swe.MEAN_NODE, flag)[0][0] + 180) % 360
                longitudes[name] = get_d9_longitude(lon)

            for label, config in aspect_config.items():
                aspects = check_aspects(longitudes[config["from"]], longitudes[config["to"]], config["angles"], label)
                if aspects != ["nan"]:
                    results_d9[dt.date()].append((label, aspects[0], dt.time()))

    # === Build Summary Table ===
    summary = []
    all_dates = pd.date_range(start=start_date, end=end_date).date

    for d in all_dates:
        d1_aspects = [a[1] for a in results_d1[d]]
        d1_times = [str(a[2]) for a in results_d1[d]]
        d9_aspects = [a[1] for a in results_d9[d]]
        d9_times = [str(a[2]) for a in results_d9[d]]

        summary.append({
            "Date": d,
            "D1 Aspects": d1_aspects[0] if d1_aspects else "0",
            "D1 Aspects Time": d1_times[0] if d1_times else "0",
            "D9 Aspects": d9_aspects[0] if d9_aspects else "0",
            "D9 Aspects Time": d9_times[0] if d9_times else "0"
        })

    df = pd.DataFrame(summary)

    # ✅ Only show rows with at least one aspect
    df_filtered = df[(df["D1 Aspects"] != "0") | (df["D9 Aspects"] != "0")]

    # === Render Table ===
    def render_aspect_html(df):
        rows = []
        for _, row in df.iterrows():
            style = "background-color: black; color: white;"
            row_html = "<tr style='{}'>".format(style)
            row_html += "".join(f"<td>{val}</td>" for val in row)
            row_html += "</tr>"
            rows.append(row_html)

        header = "".join([f"<th>{col}</th>" for col in df.columns])
        table = f"""
        <div class="scroll-table">
            <table>
                <thead><tr>{header}</tr></thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """
        return table

    st.markdown("### 🧭 Daily Planetary Aspect Hits")
    st.markdown(render_aspect_html(df_filtered), unsafe_allow_html=True)

elif filter_mode == "Swapt Nadi Chakra":
    st.header("🌀 Swapt Nadi Chakra Report")

    import swisseph as swe
    import pandas as pd
    from datetime import datetime, timedelta

    # === Setup ===
    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)
    LAT, LON = 19.076, 72.8777
    TZ_OFFSET = 5.5

    # === User Input ===
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2025, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 6, 3))

    # === Nadi Classification Setup ===
    nakshatras = [
        "Ashwani", "Bharni", "Krittika", "Rohini", "Mrigsira", "Ardra", "Punarvasu", "Pushya", "Ashlesha",
        "Magha", "P.Phalguni", "U.Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshta",
        "Mula", "P.Ashadha", "U.Ashadha", "Abhijit", "Sharavana", "Dhanishta", "Satabhisha",
        "P.Bhadra Pada", "U.Bhadra Pada", "Revati"
    ]

    nadi_map = {
        'Bharni': 'Prachanda', 'Krittika': 'Prachanda', 'Vishakha': 'Prachanda', 'Anuradha': 'Prachanda',
        'Ashwani': 'Pawan', 'Rohini': 'Pawan', 'Swati': 'Pawan', 'Jyeshta': 'Pawan',
        'Mrigsira': 'Dahan', 'Chitra': 'Dahan', 'Mula': 'Dahan', 'Revati': 'Dahan',
        'Ardra': 'Soumya', 'Hasta': 'Soumya', 'P.Ashadha': 'Soumya', 'U.Bhadra Pada': 'Soumya',
        'Punarvasu': 'Neera', 'U.Phalguni': 'Neera', 'U.Ashadha': 'Neera', 'P.Bhadra Pada': 'Neera',
        'Pushya': 'Jala', 'P.Phalguni': 'Jala', 'Abhijit': 'Jala', 'Satabhisha': 'Jala',
        'Ashlesha': 'Amrit', 'Magha': 'Amrit', 'Sharavana': 'Amrit', 'Dhanishta': 'Amrit'
    }

    planet_list = {
        'Sun': swe.SUN, 'Moon': swe.MOON, 'Mercury': swe.MERCURY, 'Venus': swe.VENUS,
        'Mars': swe.MARS, 'Jupiter': swe.JUPITER, 'Saturn': swe.SATURN,
        'Rahu': swe.MEAN_NODE, 'Ketu': swe.TRUE_NODE
    }

    def get_nakshatra(degree):
        return nakshatras[int(degree // (360 / 27))]

    def get_longitude(jd, planet, flag):
        if planet == 'Ketu':
            rahu_lon = swe.calc_ut(jd, swe.MEAN_NODE, flag)[0][0]
            return (rahu_lon + 180) % 360
        return swe.calc_ut(jd, planet_list[planet], flag)[0][0]

    # === Build Daily Table ===
    daily_data = []
    for day in pd.date_range(start=start_date, end=end_date):
        row = {"Date": day.date(), "Prachanda": [], "Pawan": [], "Dahan": [], "Soumya": [], "Neera": [], "Jala": [], "Amrit": []}
        dt = datetime(day.year, day.month, day.day, 9, 0)
        utc_dt = dt - timedelta(hours=TZ_OFFSET)
        jd = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour + utc_dt.minute / 60)
        flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

        for planet in planet_list:
            lon = get_longitude(jd, planet, flag)
            nak = get_nakshatra(lon)
            nadi = nadi_map.get(nak)
            if nadi:
                row[nadi].append(planet)

        for nadi in ["Prachanda", "Pawan", "Dahan", "Soumya", "Neera", "Jala", "Amrit"]:
            row[nadi] = ", ".join(row[nadi]) if row[nadi] else ""

        daily_data.append(row)

    df_nadi = pd.DataFrame(daily_data)

    # === Display as HTML Table ===
    st.markdown("### 🔮 Daily Nadi Chakra Classification Table")
    html_table = df_nadi.to_html(index=False)
    st.markdown(f'<div class="scroll-table">{html_table}</div>', unsafe_allow_html=True)

elif filter_mode == "Planetary Ingress":
    st.header("🚪 Planetary Ingress Report (Excluding Moon)")

    import swisseph as swe
    import pandas as pd
    from datetime import datetime, timedelta

    # === Setup ===
    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)

    LAT, LON = 19.076, 72.8777
    TZ_OFFSET = 5.5  # IST

    # === User Date Input ===
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2025, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 6, 30))

    # === Planets to Track (excluding Moon) ===
    planet_list = {
        'Sun': swe.SUN,
        'Mercury': swe.MERCURY,
        'Venus': swe.VENUS,
        'Mars': swe.MARS,
        'Jupiter': swe.JUPITER,
        'Saturn': swe.SATURN,
        'Rahu': swe.MEAN_NODE,
        'Ketu': swe.TRUE_NODE  # calculated as opposite of Rahu
    }

    signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
             'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']

    def get_sign_name(degree):
        return signs[int(degree // 30)]

    # === Ingress Detection ===
    results = []
    last_sign = {}

    for day in pd.date_range(start=start_date, end=end_date):
        dt = datetime(day.year, day.month, day.day, 0)
        utc_dt = dt - timedelta(hours=TZ_OFFSET)
        jd = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour)
        flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

        for name, code in planet_list.items():
            lon = swe.calc_ut(jd, code, flag)[0][0]
            if name == "Ketu":
                lon = (swe.calc_ut(jd, swe.MEAN_NODE, flag)[0][0] + 180) % 360

            current_sign = get_sign_name(lon)

            if name not in last_sign:
                last_sign[name] = current_sign
            elif last_sign[name] != current_sign:
                results.append({
                    "Date": dt.date(),
                    "Planet": name,
                    "From Sign": last_sign[name],
                    "To Sign": current_sign
                })
                last_sign[name] = current_sign
 
    df_ingress = pd.DataFrame(results)

    # === Render as HTML Table ===
    st.markdown("### 📆 Planetary Ingress Events")
    if not df_ingress.empty:
        html = df_ingress.to_html(index=False)
        st.markdown(f'<div class="scroll-table">{html}</div>', unsafe_allow_html=True)
    else:
        st.info("No ingress events found in selected date range.")

elif filter_mode == "AOT Monthly Calendar":
    st.header("📅 AOT Monthly Report")

    import swisseph as swe
    import pandas as pd
    from datetime import datetime, timedelta

    # === Config
    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)
    LAT = 19.0760
    LON = 72.8777
    TZ_OFFSET = 5.5  # IST

    # === Input
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=30), key="aot_start")
    with col2:
        end_date = st.date_input("End Date", value=datetime.today(), key="aot_end")

    nadi_map = {
    'Bharni': 'Prachanda', 'Krittika': 'Prachanda', 'Vishakha': 'Prachanda', 'Anuradha': 'Prachanda',
    'Ashwani': 'Pawan', 'Rohini': 'Pawan', 'Swati': 'Pawan', 'Jyeshta': 'Pawan',
    'Mrigsira': 'Dahan', 'Chitra': 'Dahan', 'Mula': 'Dahan', 'Revati': 'Dahan',
    'Ardra': 'Soumya', 'Hasta': 'Soumya', 'P.Ashadha': 'Soumya', 'U.Bhadra Pada': 'Soumya',
    'Punarvasu': 'Neera', 'U.Phalguni': 'Neera', 'U.Ashadha': 'Neera', 'P.Bhadra Pada': 'Neera',
    'Pushya': 'Jala', 'P.Phalguni': 'Jala', 'Abhijit': 'Jala', 'Satabhisha': 'Jala',
    'Ashlesha': 'Amrit', 'Magha': 'Amrit', 'Sharavana': 'Amrit', 'Dhanishta': 'Amrit'
    }

    nakshatras = [
        "Ashwani", "Bharni", "Krittika", "Rohini", "Mrigsira", "Ardra", "Punarvasu", "Pushya", "Ashlesha",
        "Magha", "P.Phalguni", "U.Phalguni", "Hasta", "Chitra", "Swati", "Vishakha", "Anuradha", "Jyeshta",
        "Mula", "P.Ashadha", "U.Ashadha", "Abhijit", "Sharavana", "Dhanishta", "Satabhisha",
        "P.Bhadra Pada", "U.Bhadra Pada", "Revati"
    ]

    def get_nakshatra_name(degree):
        return nakshatras[int(degree // (360 / 27))]

    nadi_types = ["Prachanda", "Pawan", "Dahan", "Soumya", "Neera", "Jala", "Amrit"]

    planet_list = {
        'Sun': swe.SUN, 'Moon': swe.MOON, 'Mercury': swe.MERCURY, 'Venus': swe.VENUS,
        'Mars': swe.MARS, 'Jupiter': swe.JUPITER, 'Saturn': swe.SATURN,
        'Rahu': swe.MEAN_NODE, 'Ketu': swe.TRUE_NODE  # Ketu will be 180° from Rahu
    }

    # === Planet definitions
    planets = {
        "Sun": swe.SUN, "Moon": swe.MOON, "Mercury": swe.MERCURY,
        "Venus": swe.VENUS, "Mars": swe.MARS, "Jupiter": swe.JUPITER,
        "Saturn": swe.SATURN, "Rahu": swe.TRUE_NODE, "Ketu": swe.TRUE_NODE,
        "Uranus": swe.URANUS, "Neptune": swe.NEPTUNE, "Pluto": swe.PLUTO
    }

    signs = ['Aries', 'Taurus', 'Gemini', 'Cancer', 'Leo', 'Virgo',
             'Libra', 'Scorpio', 'Sagittarius', 'Capricorn', 'Aquarius', 'Pisces']
    
    def get_sign_name(degree):
        return signs[int(degree // 30)]
    
    planet_ingress_signs = {}
    planet_list_ingress = {
        'Sun': swe.SUN,
        'Mercury': swe.MERCURY,
        'Venus': swe.VENUS,
        'Mars': swe.MARS,
        'Jupiter': swe.JUPITER,
        'Saturn': swe.SATURN,
        'Rahu': swe.MEAN_NODE,
        'Ketu': swe.TRUE_NODE  # Opposite of Rahu
    }

    def classify_sign_type(num):
        if num in [1, 4, 7, 10]:
            return "Movable"
        elif num in [2, 5, 8, 11]:
            return "Fixed"
        elif num in [3, 6, 9, 12]:
            return "Dual"
        return "Unknown"

    custom_d1_map = {sign: i + 1 for i, sign in enumerate(signs)}
    sign_types = {
        "Movable": [1, 4, 7, 10],
        "Fixed": [2, 5, 8, 11],
        "Dual": [3, 6, 9, 12]
    }

    planet_order = ["Moon", "Mercury", "Venus", "Sun", "Mars",
                    "Jupiter", "Rahu", "Ketu", "Saturn", "Uranus", "Neptune", "Pluto"]
    planet_rank = {p: i for i, p in enumerate(planet_order)}

    def classify_sign_type(sign_number):
        for k, v in sign_types.items():
            if sign_number in v:
                return k
        return "Unknown"

    # === Loop through dates
    rows = []
    current = start_date
    while current <= end_date:
        utc_dt = datetime(current.year, current.month, current.day) - timedelta(hours=TZ_OFFSET)
        jd = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day)
        jd2 = swe.julday(current.year, current.month, current.day, 0)

        planet_data = {}
        for name, pid in planets.items():
            lon, speed = get_planet_data(jd2, name, pid)
            planet_data[name] = {"deg": lon, "speed": speed}

        day_type = "-"
        reason = ""
        for p1, p2 in combinations(planet_data.keys(), 2):
            r1 = planet_rank.get(p1, 999)
            r2 = planet_rank.get(p2, 999)
            fast, slow = (p1, p2) if r1 < r2 else (p2, p1)

            d1 = planet_data[fast]["deg"]
            d2 = planet_data[slow]["deg"]
            diff = signed_diff(d1, d2)

            if abs(diff) <= 1.0:
                day_type = "Red Day" if diff < 0 else "Green Day"
                reason = f"{fast}→{slow}: {diff}°"
                break

        # === Fixed Time: 9:00 AM IST for D1 and D9 Type Classification
        classification_dt = datetime(current.year, current.month, current.day, 9, 0)
        utc_dt_class = classification_dt - timedelta(hours=TZ_OFFSET)
        jd_class = swe.julday(utc_dt_class.year, utc_dt_class.month, utc_dt_class.day, utc_dt_class.hour + utc_dt_class.minute / 60)

        d1_types = {"Movable": [], "Fixed": [], "Dual": []}
        d9_types = {"Movable": [], "Fixed": [], "Dual": []}
        flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

        for name, pid in planets.items():
            lon = swe.calc_ut(jd_class, pid, flag)[0][0]
            if name == "Ketu":
                rahu_lon = swe.calc_ut(jd_class, swe.MEAN_NODE, flag)[0][0]
                lon = (rahu_lon + 180) % 360

            # === D1 classification
            d1_sign_index = int(lon // 30)
            d1_sign_number = d1_sign_index + 1
            d1_type = classify_sign_type(d1_sign_number)
            d1_types[d1_type].append(name)

            # === D9 classification
            def get_d9_longitude(lon):
                sign_index = int(lon // 30)
                pos_in_sign = lon % 30
                navamsa_index = int(pos_in_sign // (30 / 9))
                if sign_index in [0, 3, 6, 9]:
                    start = sign_index
                elif sign_index in [1, 4, 7, 10]:
                    start = (sign_index + 8) % 12
                else:
                    start = (sign_index + 4) % 12
                d9_sign_index = (start + navamsa_index) % 12
                deg_in_navamsa = pos_in_sign % (30 / 9)
                return d9_sign_index * 30 + deg_in_navamsa * 9

            d9_lon = get_d9_longitude(lon)
            d9_sign_index = int(d9_lon // 30)
            d9_sign_number = d9_sign_index + 1
            d9_type = classify_sign_type(d9_sign_number)
            d9_types[d9_type].append(name)

        # Format result strings
        d1_classified_str = " | ".join([f"{k}: {', '.join(v)}" for k, v in d1_types.items() if v])
        d9_classified_str = " | ".join([f"{k}: {', '.join(v)}" for k, v in d9_types.items() if v])

        # === Moon Nakshatra & Pada (Lahiri, 9:00 IST)
        moon_dt = datetime(current.year, current.month, current.day, 9, 0)
        moon_utc = moon_dt - timedelta(hours=TZ_OFFSET)
        moon_jd = swe.julday(moon_utc.year, moon_utc.month, moon_utc.day, moon_utc.hour + moon_utc.minute / 60.0)

        moon_lon = swe.calc_ut(moon_jd, swe.MOON)[0][0]

        nak_index = int(moon_lon // (360 / 27))
        moon_nak = nakshatras[nak_index]
        moon_pada = int(((moon_lon % (360 / 27)) // (360 / 27 / 4)) + 1)


        ingress_changes = []

        flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH
        for name, code in planet_list_ingress.items():
            lon = swe.calc_ut(jd, code, flag)[0][0]
            if name == "Ketu":
                rahu_lon = swe.calc_ut(jd, swe.MEAN_NODE, flag)[0][0]
                lon = (rahu_lon + 180) % 360

            current_sign = get_sign_name(lon)

            if name not in planet_ingress_signs:
                planet_ingress_signs[name] = current_sign
            elif planet_ingress_signs[name] != current_sign:
                ingress_changes.append((name, planet_ingress_signs[name], current_sign))
                planet_ingress_signs[name] = current_sign

        # === D1 Aspect Detection
        d1_aspect_result = "0"
        for hour in range(0, 23):
            dt = datetime(current.year, current.month, current.day, hour, 0, 0)
            utc_dt = dt - timedelta(hours=TZ_OFFSET)
            jd_hour = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour + utc_dt.minute / 60)
            flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

            longitudes = {}
            for name, code in planet_map.items():
                lon = swe.calc_ut(jd_hour, code, flag)[0][0]
                if name == "Ketu":
                    lon = (swe.calc_ut(jd_hour, swe.MEAN_NODE, flag)[0][0] + 180) % 360
                longitudes[name] = lon

            for label, config in aspect_config.items():
                asp = check_aspects(longitudes[config["from"]], longitudes[config["to"]], config["angles"], label)
                if asp:
                    d1_aspect_result = asp
                    break
            if d1_aspect_result != "0":
                break

        # === D9 Aspect Detection
        d9_aspect_result = "0"
        for hour in range(8, 16):  # 8 AM to 4 PM
            dt = datetime(current.year, current.month, current.day, hour, 0, 0)
            utc_dt = dt - timedelta(hours=TZ_OFFSET)
            jd_hour = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour  + utc_dt.minute / 60)
            flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

            longitudes = {}
            for name, code in planet_map.items():
                lon = swe.calc_ut(jd_hour, code, flag)[0][0]
                if name == "Ketu":
                    lon = (swe.calc_ut(jd_hour, swe.MEAN_NODE, flag)[0][0] + 180) % 360
                longitudes[name] = get_d9_longitude(lon)

            for label, config in aspect_config.items():
                asp = check_aspects(longitudes[config["from"]], longitudes[config["to"]], config["angles"], label)
                if asp:
                    d9_aspect_result = asp
                    break
            if d9_aspect_result != "0":
                break

        nadi_result = {n: [] for n in nadi_types}
        flag = swe.FLG_SIDEREAL | swe.FLG_SWIEPH

        for planet in planet_list:
            if planet == 'Ketu':
                rahu_lon = swe.calc_ut(jd, swe.MEAN_NODE, flag)[0][0]
                lon = (rahu_lon + 180) % 360
            else:
                lon = swe.calc_ut(jd, planet_list[planet], flag)[0][0]

            nak = get_nakshatra_name(lon)
            nadi = nadi_map.get(nak)
            if nadi:
                nadi_result[nadi].append(planet)

            # Optional: combine only Prachanda & Pawan for now
            prachanda_str = ", ".join(nadi_result["Prachanda"]) if nadi_result["Prachanda"] else ""
            pawan_str = ", ".join(nadi_result["Pawan"]) if nadi_result["Pawan"] else ""


        # Julian Day
        # === Moon Nakshatra & Pada (Lahiri, 9:00 IST)
        asc_dt = datetime(current.year, current.month, current.day, 9, 0)
        asc_utc = asc_dt - timedelta(hours=TZ_OFFSET)
        asc_jd = swe.julday(asc_utc.year, asc_utc.month, asc_utc.day, asc_utc.hour + asc_utc.minute / 60.0)

        flags = swe.FLG_SIDEREAL
        ascmc, cusp = swe.houses_ex(asc_jd, LAT, LON, b'A', flags)
        asc = ascmc[0]

        nak_index = int(asc // (360 / 27))
        pada = int(((asc % (360 / 27)) // (360 / 27 / 4)) + 1)
        nak = nakshatras[nak_index]

        # === Moon–Mercury D1 Type
        moon_d1 = get_planet_deg(jd, "Moon")
        mercury_d1 = get_planet_deg(jd, "Mercury")
        moon_sign = int(moon_d1 // 30)
        mercury_sign = int(mercury_d1 // 30)
        moon_d1_type = classify_sign_type(custom_d1_map[signs[moon_sign]])
        mercury_d1_type = classify_sign_type(custom_d1_map[signs[mercury_sign]])

        if moon_d1_type == mercury_d1_type:
            mm_d1_status = f"Moon & Mercury: {moon_d1_type}"
        else:
            mm_d1_status = f"Moon: {moon_d1_type}, Mercury: {mercury_d1_type}"

        # === Moon–Mercury D9 Type
        moon_d9 = get_d9_longitude(moon_d1)
        mercury_d9 = get_d9_longitude(mercury_d1)
        moon_d9_sign = int(moon_d9 // 30)
        mercury_d9_sign = int(mercury_d9 // 30)
        moon_d9_type = classify_sign_type(moon_d9_sign + 1)
        mercury_d9_type = classify_sign_type(mercury_d9_sign + 1)

        if moon_d9_type == mercury_d9_type:
            mm_d9_status = f"Moon & Mercury: {moon_d9_type}"
        else:
            mm_d9_status = f"Moon: {moon_d9_type}, Mercury: {mercury_d9_type}"

        # === Moon–Mercury D1 Aspect (0–23 IST)
        d1_aspect = "0"
        for hour in range(0, 23):
            dt = datetime(current.year, current.month, current.day, hour, 0)
            utc_dt = dt - timedelta(hours=TZ_OFFSET)
            jd_hour = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour + utc_dt.minute / 60)
            m_deg = get_planet_deg(jd_hour, "Moon")
            mc_deg = get_planet_deg(jd_hour, "Mercury")
            asp = check_mm_aspects(m_deg, mc_deg)
            if asp != "0":
                d1_aspect = asp
                break

        # === Moon–Mercury D9 Aspect (8–16 IST)
        d9_aspect = "0"
        for hour in range(8, 16):
            dt = datetime(current.year, current.month, current.day, hour, 0)
            utc_dt = dt - timedelta(hours=TZ_OFFSET)
            jd_hour = swe.julday(utc_dt.year, utc_dt.month, utc_dt.day, utc_dt.hour + utc_dt.minute / 60)
            m_deg = get_planet_deg(jd_hour, "Moon")
            mc_deg = get_planet_deg(jd_hour, "Mercury")
            m_d9 = get_d9_longitude(m_deg)
            mc_d9 = get_d9_longitude(mc_deg)
            asp = check_mm_aspects(m_d9, mc_d9)
            if asp != "0":
                d9_aspect = asp
                break

        rows.append({
            "Date": current.strftime("%Y-%m-%d"),
            "Day Type": day_type,
            "Reason": reason,
            "Moon & Mercury D1 Type": mm_d1_status,
            "Moon & Mercury D9 Type": mm_d9_status,
            "Moon & Mercury D1 Aspect": d1_aspect,
            "Moon & Mercury D9 Aspect": d9_aspect,
            "Ascendant Nakshatra": nak,
            "Ascendant Pada": pada,
            "Prachanda": prachanda_str,
            "Pawan": pawan_str,
            "Planetary D1 Aspects": d1_aspect_result,
            "Planetary D9 Aspects": d9_aspect_result,
            "Ingress Planet": ingress_changes[0][0] if ingress_changes else "",
            "From Sign": ingress_changes[0][1] if ingress_changes else "",
            "To Sign": ingress_changes[0][2] if ingress_changes else "",
            "Moon Nakshatra": moon_nak,
            "Moon Pada": moon_pada,
            "Planets M/F/D D1": d1_classified_str,
            "Planets M/F/D D9": d9_classified_str,
            })

        current += timedelta(days=1)

    df = pd.DataFrame(rows)

    st.markdown("### 📊 Daily AOT View (with Moon–Mercury Aspects)")
    html = df.to_html()
    st.markdown(f'<div class="scroll-table">{html}</div>', unsafe_allow_html=True)

elif filter_mode == "Rule 1: Speed of Mercury":
    st.header("📈 Nifty Candlestick + 🔭 Mercury Speed")

    import pandas as pd
    import numpy as np
    import swisseph as swe
    import datetime
    from scipy.signal import argrelextrema
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # === Swiss Ephemeris setup
    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)

    # === User input
    col1, col2 = st.columns(2)
    with col1:
        mercury_start = st.date_input("Mercury Start", value=datetime.date(2023, 1, 1))
    with col2:
        mercury_end = st.date_input("Mercury End", value=datetime.date(2026, 1, 1))

    col3, col4 = st.columns(2)
    with col3:
        price_start = st.date_input("Nifty Start", value=datetime.date(2023, 1, 1))
    with col4:
        price_end = st.date_input("Nifty End", value=datetime.date.today())

    index_choice = st.selectbox("Select Index", ["Nifty", "BankNifty"], index=0)

    # === Mercury speed data
    def get_mercury_speed_df(start_date, end_date):
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        speeds = []
        for d in dates:
            jd = swe.julday(d.year, d.month, d.day)
            pos, _ = swe.calc_ut(jd, swe.MERCURY)
            speeds.append(pos[3])
        return pd.DataFrame({"date": dates, "mercury_speed": speeds})

    def find_tops_bottoms(speed_series):
        arr = np.array(speed_series)
        tops = argrelextrema(arr, np.greater)[0]
        bottoms = argrelextrema(arr, np.less)[0]
        return tops, bottoms

    df_speed = get_mercury_speed_df(mercury_start, mercury_end)
    tops, bottoms = find_tops_bottoms(df_speed['mercury_speed'])
    top_dates = df_speed.iloc[tops]['date']
    bottom_dates = df_speed.iloc[bottoms]['date']

    # === Fetch Nifty/BankNifty OHLC from NeonDB
    @st.cache_data
    def fetch_ohlc(index_name, start, end):
        query = """
            SELECT "Date", "Open", "High", "Low", "Close", "Vol(in M)"
            FROM ohlc_index
            WHERE index_name = %s AND "Date" BETWEEN %s AND %s
            ORDER BY "Date"
        """
        df_ohlc = pd.read_sql(query, engine, params=(index_name, start, end))
        df_ohlc['Date'] = pd.to_datetime(df_ohlc['Date'])
        return df_ohlc

    df_nifty_filtered = fetch_ohlc(index_choice, price_start, price_end)

    # === Plotting
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.65, 0.35],
        subplot_titles=("Nifty Candlestick Chart", "Mercury Speed (°/day)")
    )

    # Row 1: Nifty candlestick chart
    fig.add_trace(go.Candlestick(
        x=df_nifty_filtered['Date'],
        open=df_nifty_filtered['Open'],
        high=df_nifty_filtered['High'],
        low=df_nifty_filtered['Low'],
        close=df_nifty_filtered['Close'],
        name="Nifty",
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_nifty_filtered['Date'],
        y=[None] * len(df_nifty_filtered),
        mode='markers',
        marker=dict(opacity=0),
        name='Date Only',
        hovertemplate="%{x|%b %d, %Y}<extra></extra>",
        showlegend=False
    ), row=1, col=1)

    # Row 2: Mercury speed line
    fig.add_trace(go.Scatter(
        x=df_speed['date'],
        y=df_speed['mercury_speed'],
        mode='lines',
        name="Mercury Speed",
        line=dict(color='orange'),
        hovertemplate="%{x|%b %d, %Y}<br>Mercury Speed: %{y:.6f}<extra></extra>"
    ), row=2, col=1)

    # Tops and bottoms
    fig.add_trace(go.Scatter(
        x=top_dates,
        y=df_speed.iloc[tops]['mercury_speed'],
        mode='markers',
        marker=dict(color='red', size=8),
        name="Speed Tops",
        hovertemplate="%{x|%b %d, %Y}<br>Top: %{y:.6f}<extra></extra>"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=bottom_dates,
        y=df_speed.iloc[bottoms]['mercury_speed'],
        mode='markers',
        marker=dict(color='blue', size=8),
        name="Speed Bottoms",
        hovertemplate="%{x|%b %d, %Y}<br>Bottom: %{y:.6f}<extra></extra>"
    ), row=2, col=1)

    # Vertical lines for tops and bottoms
    for d in top_dates:
        fig.add_vline(x=d, line_color='red', line_dash='dot', opacity=0.5, row=1, col=1)
        fig.add_vline(x=d, line_color='red', line_dash='dot', opacity=0.5, row=2, col=1)

    for d in bottom_dates:
        fig.add_vline(x=d, line_color='blue', line_dash='dot', opacity=0.5, row=1, col=1)
        fig.add_vline(x=d, line_color='blue', line_dash='dot', opacity=0.5, row=2, col=1)

    # Layout
    fig.update_layout(
        height=800,
        hovermode="x unified",
        showlegend=True,
        xaxis_rangeslider_visible=False,
        xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        yaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        xaxis2=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        yaxis2=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot')
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Mercury Speed Table"):
        st.dataframe(df_speed)

    with st.expander("📈 Nifty Price Table"):
        st.dataframe(df_nifty_filtered[['Date', 'Open', 'High', 'Low', 'Close']])

elif filter_mode == "Rule 2: Mars–Mercury 59-Min Rule":
    st.header("📉 Mars–Mercury 59-Minute Speed Differential Rule")

    import pandas as pd
    import numpy as np
    import swisseph as swe
    from datetime import datetime, timedelta
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # === Ephemeris setup
    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)

    # === Inputs
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    index_choice = st.selectbox("Select Index", ["Nifty", "BankNifty"], index=0)

    # === Constants
    TARGET_DIFF = 59 / 60       # 0.9833°
    TOLERANCE = 4 / 60          # ±4 minutes
    EXACT_RANGE = (0.98, 0.99)  # for black line

    # === Compute Mars–Mercury speed difference
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    rows = []

    for date in dates:
        jd = swe.julday(date.year, date.month, date.day)
        mars_pos, _ = swe.calc_ut(jd, swe.MARS)
        merc_pos, _ = swe.calc_ut(jd, swe.MERCURY)
        mars_speed = mars_pos[3]
        merc_speed = merc_pos[3]
        diff = abs(merc_speed - mars_speed)
        is_signal = abs(diff - TARGET_DIFF) <= TOLERANCE
        is_exact = EXACT_RANGE[0] <= diff <= EXACT_RANGE[1]

        # Format as arcminutes/arcseconds
        total_minutes = diff * 60
        deg_min = int(total_minutes)
        deg_sec = round((total_minutes - deg_min) * 60)
        time_format = f"{deg_min}′ {deg_sec}″"

        rows.append({
            "Date": date,
            "Mars Speed": mars_speed,
            "Mercury Speed": merc_speed,
            "Degree Diff": diff,
            "Diff (Time)": time_format,
            "Sell Signal": is_signal,
            "Exact Match": is_exact
        })

    df = pd.DataFrame(rows)
    df['Date'] = pd.to_datetime(df['Date'])
    signal_df = df[df['Sell Signal']]

    st.success(f"📌 Found {len(signal_df)} signal(s)")
    st.dataframe(signal_df[['Date', 'Mars Speed', 'Mercury Speed', 'Degree Diff', 'Diff (Time)', 'Exact Match']])

    # === Load OHLC from NeonDB
    @st.cache_data
    def fetch_ohlc(index_name, start, end):
        query = """
            SELECT "Date", "Open", "High", "Low", "Close", "Vol(in M)"
            FROM ohlc_index
            WHERE index_name = %s AND "Date" BETWEEN %s AND %s
            ORDER BY "Date"
        """
        df_ohlc = pd.read_sql(query, engine, params=(index_name, start, end))
        df_ohlc['Date'] = pd.to_datetime(df_ohlc['Date'])
        return df_ohlc

    df_nifty_filtered = fetch_ohlc(index_choice, start_date, end_date)

    # === Plotting
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.07,
        subplot_titles=["Nifty Candlestick Chart", "Mars–Mercury Speed Degree Difference"]
    )

    # Candlestick chart
    fig.add_trace(go.Candlestick(
        x=df_nifty_filtered['Date'],
        open=df_nifty_filtered['Open'],
        high=df_nifty_filtered['High'],
        low=df_nifty_filtered['Low'],
        close=df_nifty_filtered['Close'],
        name="Nifty",
        showlegend=False,
        increasing=dict(line=dict(color="green", width=1.2), fillcolor="rgba(0,200,0,0.4)"),
        decreasing=dict(line=dict(color="red", width=1.2), fillcolor="rgba(200,0,0,0.4)")
    ), row=1, col=1)

    # Transparent scatter (hover only)
    fig.add_trace(go.Scatter(
        x=df_nifty_filtered['Date'],
        y=[None] * len(df_nifty_filtered),
        mode='markers',
        marker=dict(opacity=0),
        hovertemplate="%{x|%b %d, %Y}<extra></extra>",
        showlegend=False
    ), row=1, col=1)

    # Speed difference line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Degree Diff'],
        mode='lines',
        name="Deg. Diff (Mercury–Mars)",
        line=dict(color="orange", width=2)
    ), row=2, col=1)

    # 59-min threshold
    fig.add_hline(
        y=0.9833,
        line_color="gray",
        line_dash="dash",
        opacity=0.6,
        row=2, col=1
    )

    # Signal annotations
    for _, row_sig in signal_df.iterrows():
        d = row_sig['Date']
        color = 'black' if row_sig['Exact Match'] else 'purple'
        fig.add_vline(x=d, line_color=color, line_dash='dot', opacity=0.8, row=1, col=1)
        fig.add_vline(x=d, line_color=color, line_dash='dot', opacity=0.8, row=2, col=1)
        fig.add_annotation(
            x=d,
            y=df_nifty_filtered['High'].max(),
            text="🔻",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(color=color, size=14),
            row=1, col=1
        )

    # Layout
    fig.update_layout(
        height=800,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        yaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        xaxis2=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        yaxis2=dict(
            title="° Difference",
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikedash='dot'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full Speed Comparison Table"):
        st.dataframe(df)

elif filter_mode == "Rule 3: 161° Mars–Mercury Bullish Signal":
    st.header("📈 Mars–Mercury 161°32′18″ Angular Separation (Bullish Signal)")

    import pandas as pd
    import numpy as np
    import swisseph as swe
    from datetime import datetime, timedelta
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # === Ephemeris config
    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)

    # === User input
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    index_choice = st.selectbox("Select Index", ["Nifty", "BankNifty"], index=0)

    # === Constants
    TARGET_ANGLE = 161 + 32/60 + 18/3600  # 161.5383°
    ANGLE_TOLERANCE = 0.5

    # === Compute Mars–Mercury angular separation
    rows = []
    dates = pd.date_range(start=start_date, end=end_date, freq="D")

    for date in dates:
        jd = swe.julday(date.year, date.month, date.day)
        mars_pos, _ = swe.calc_ut(jd, swe.MARS)
        merc_pos, _ = swe.calc_ut(jd, swe.MERCURY)

        angle_diff = abs((merc_pos[0] - mars_pos[0]) % 360)
        angle_diff = min(angle_diff, 360 - angle_diff)
        is_match = abs(angle_diff - TARGET_ANGLE) <= ANGLE_TOLERANCE
        mars_retro = mars_pos[3] < 0

        deg_whole = int(angle_diff)
        minutes = (angle_diff - deg_whole) * 60
        min_whole = int(minutes)
        seconds = round((minutes - min_whole) * 60)
        diff_time = f"{deg_whole}° {min_whole}′ {seconds:02d}″"

        rows.append({
            "Date": date,
            "Mars°": mars_pos[0],
            "Mercury°": merc_pos[0],
            "Angular Diff": round(angle_diff, 3),
            "Diff (Time)": diff_time,
            "Mars Retrograde": mars_retro,
            "Bullish Signal": "🟢" if is_match else ""
        })

    df = pd.DataFrame(rows)
    df['Date'] = pd.to_datetime(df['Date'])
    signal_df = df[df['Bullish Signal'] == "🟢"]

    st.success(f"📌 Found {len(signal_df)} potential bullish signal(s)")
    st.dataframe(signal_df[['Date', 'Mars°', 'Mercury°', 'Angular Diff', 'Diff (Time)', 'Mars Retrograde', 'Bullish Signal']])

    # === Load OHLC from NeonDB
    @st.cache_data
    def fetch_ohlc(index_name, start, end):
        query = """
            SELECT "Date", "Open", "High", "Low", "Close", "Vol(in M)"
            FROM ohlc_index
            WHERE index_name = %s AND "Date" BETWEEN %s AND %s
            ORDER BY "Date"
        """
        df_ohlc = pd.read_sql(query, engine, params=(index_name, start, end))
        df_ohlc['Date'] = pd.to_datetime(df_ohlc['Date'])
        return df_ohlc

    df_nifty_filtered = fetch_ohlc(index_choice, start_date, end_date)

    # === Plotting
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.65, 0.35],
        subplot_titles=["Nifty Candlestick Chart", "Mars–Mercury Angular Separation"]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df_nifty_filtered['Date'],
        open=df_nifty_filtered['Open'],
        high=df_nifty_filtered['High'],
        low=df_nifty_filtered['Low'],
        close=df_nifty_filtered['Close'],
        name="Nifty",
        increasing=dict(line=dict(color="green", width=1.2), fillcolor="rgba(0,200,0,0.4)"),
        decreasing=dict(line=dict(color="red", width=1.2), fillcolor="rgba(200,0,0,0.4)"),
        showlegend=False
    ), row=1, col=1)

    # Transparent hover layer
    fig.add_trace(go.Scatter(
        x=df_nifty_filtered['Date'],
        y=[None] * len(df_nifty_filtered),
        mode='markers',
        marker=dict(opacity=0),
        hovertemplate="%{x|%b %d, %Y}<extra></extra>",
        showlegend=False
    ), row=1, col=1)

    # Angle diff line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Angular Diff'],
        mode='lines',
        name="Mars–Mercury Angle",
        line=dict(color="orange", width=2)
    ), row=2, col=1)

    # Signal markers
    for _, row_sig in signal_df.iterrows():
        d = row_sig['Date']
        fig.add_vline(x=d, line_color='green', line_dash='dot', opacity=0.8, row=1, col=1)
        fig.add_vline(x=d, line_color='green', line_dash='dot', opacity=0.8, row=2, col=1)
        fig.add_annotation(
            x=d,
            y=df_nifty_filtered['High'].max(),
            text="🟢",
            showarrow=False,
            xanchor="center",
            yanchor="bottom",
            font=dict(color="green", size=14),
            row=1, col=1
        )

    # Layout
    fig.update_layout(
        height=800,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        yaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        xaxis2=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        yaxis2=dict(
            title="Angular Separation (°)",
            showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full Angular Separation Table"):
        st.dataframe(df)

elif filter_mode == "test Mars–Mercury 59-Min Rule":
    st.header("📉 Mars–Mercury 59-Minute Speed Differential Rule (Fast – Using Excel)")

    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import datetime

    # === Inputs
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime.today())

    # === Load precomputed speed difference Excel
    @st.cache_data
    def load_speed_data():
        df = pd.read_excel("mars_mercury_daily_speed_diff.xlsx")
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    df = load_speed_data()
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    signal_df = df[df['Sell Signal']]

    # === Load Nifty OHLC data
    df_nifty = pd.read_excel("nifty.xlsx")
    df_nifty['Date'] = pd.to_datetime(df_nifty['Date'], dayfirst=True)
    df_nifty['date'] = df_nifty['Date']
    df_nifty_filtered = df_nifty[
        (df_nifty['date'] >= pd.to_datetime(start_date)) &
        (df_nifty['date'] <= pd.to_datetime(end_date))
    ]

    # === Show Data
    st.subheader("📋 Daily Degree Difference Between Mercury & Mars")
    st.dataframe(df[['Date', 'Mars Speed', 'Mercury Speed', 'Degree Diff', 'Diff (Time)']])

    st.success(f"📌 Found {len(signal_df)} signal(s)")
    st.dataframe(signal_df[['Date', 'Mars Speed', 'Mercury Speed', 'Degree Diff', 'Diff (Time)', 'Exact Match']])

    # === Plotting ===
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.07,
        subplot_titles=["Nifty Candlestick Chart", "Mars–Mercury Speed Degree Difference"]
    )

    # Candlestick Chart (Row 1)
    fig.add_trace(go.Candlestick(
        x=df_nifty_filtered['date'],
        open=df_nifty_filtered['Open'],
        high=df_nifty_filtered['High'],
        low=df_nifty_filtered['Low'],
        close=df_nifty_filtered['Close'],
        name="Nifty",
        showlegend=False,
        increasing=dict(line=dict(color="green", width=1.2), fillcolor="rgba(0,200,0,0.4)"),
        decreasing=dict(line=dict(color="red", width=1.2), fillcolor="rgba(200,0,0,0.4)")
    ), row=1, col=1)

    # Hover Sync
    fig.add_trace(go.Scatter(
        x=df_nifty_filtered['date'],
        y=[None] * len(df_nifty_filtered),
        mode='markers',
        marker=dict(opacity=0),
        hovertemplate="%{x|%b %d, %Y}<extra></extra>",
        showlegend=False
    ), row=1, col=1)

    # Degree Difference Line (Row 2)
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Degree Diff'],
        mode='lines',
        name="Deg. Diff (Merc–Mars)",
        line=dict(color="orange", width=2)
    ), row=2, col=1)

    # Threshold Line
    fig.add_hline(
        y=0.9833,
        line_color="gray",
        line_dash="dash",
        opacity=0.6,
        row=2, col=1
    )

    # === Add Vertical Lines & Annotations
    for _, row_sig in signal_df.iterrows():
        d = row_sig['Date']
        color = 'black' if row_sig['Exact Match'] else 'purple'

        fig.add_vline(x=d, line_color=color, line_dash='dot', opacity=0.8, row=1, col=1)
        fig.add_vline(x=d, line_color=color, line_dash='dot', opacity=0.8, row=2, col=1)

        try:
            high_val = df_nifty_filtered['High'].max()
            fig.add_annotation(
                x=d,
                y=high_val,
                text="🔻",
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                font=dict(color=color, size=14),
                row=1, col=1
            )
        except:
            pass

    # === Layout
    fig.update_layout(
        height=800,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        xaxis=dict(
            showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'
        ),
        yaxis=dict(
            showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'
        ),
        xaxis2=dict(
            showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'
        ),
        yaxis2=dict(
            title="° Difference",
            showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full Speed Comparison Table"):
        st.dataframe(df)

elif filter_mode == "Rule 4: Mercury Retrograde Echo":
    st.header("🔁 Mercury Retrograde Echo Rule")

    import swisseph as swe
    import pandas as pd
    from datetime import datetime, timedelta
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Swiss Ephemeris
    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)

    # Input: Dates + Index
    col1, col2 = st.columns(2)
    with col1:
        retro_start = st.date_input("Retrograde Scan Start", value=datetime(2010, 1, 1))
    with col2:
        retro_end = st.date_input("Retrograde Scan End", value=datetime(2025, 12, 31))

    index_choice = st.selectbox("Select Index", ["Nifty", "BankNifty"], index=0)

    # === Retrograde detection
    def get_mercury_retrograde_dates(start, end):
        step = timedelta(days=1)
        current = start
        retro_dates = []

        while current <= end:
            jd0 = swe.julday(current.year, current.month, current.day)
            jd1 = swe.julday((current + step).year, (current + step).month, (current + step).day)

            spd0 = swe.calc_ut(jd0, swe.MERCURY)[0][3]
            spd1 = swe.calc_ut(jd1, swe.MERCURY)[0][3]

            if spd0 > 0 and spd1 < 0:
                lon = swe.calc_ut(jd1, swe.MERCURY)[0][0]
                sign = int(lon / 30)
                deg = int(lon % 30)
                min = int(((lon % 30) - deg) * 60)

                zodiac = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
                          "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"][sign]

                retro_dates.append({
                    "Retro Date": current + step,
                    "Longitude": f"{deg}°{min:02d}' {zodiac}",
                    "Echo Date": (current + step + timedelta(days=365))
                })

            current += step

        return pd.DataFrame(retro_dates)

    df_retro = get_mercury_retrograde_dates(retro_start, retro_end)

    st.subheader("🪞 Mercury Retrograde Dates & Echo Points")
    st.dataframe(df_retro)

    # === Load OHLC from DB
    @st.cache_data
    def fetch_ohlc(index_name):
        query = f"""
            SELECT "Date", "Open", "High", "Low", "Close"
            FROM ohlc_index
            WHERE index_name = %s
            ORDER BY "Date"
        """
        df = pd.read_sql(query, engine, params=(index_name,))
        df['Date'] = pd.to_datetime(df['Date'])
        df['date'] = df['Date']
        return df

    df_nifty = fetch_ohlc(index_choice)

    echo_dates = df_retro['Echo Date']
    matched_nifty = df_nifty[df_nifty['date'].isin(echo_dates)].copy()

    # === Return logic
    def compute_trend(df, date, days=3):
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df.sort_index(inplace=True)

        date = pd.to_datetime(date)
        available_dates = df.index[df.index >= date]
        if available_dates.empty:
            return "None"
        date = available_dates[0]

        future_target = date + timedelta(days=days)
        future_dates = df.index[df.index >= future_target]
        if future_dates.empty:
            return "None"
        future_date = future_dates[0]

        try:
            close_today = df.loc[date, 'Close']
            close_future = df.loc[future_date, 'Close']
            pct_change = ((close_future - close_today) / close_today) * 100
            if pct_change > 0.5:
                return "Up 🔺"
            elif pct_change < -0.5:
                return "Down 🔻"
            else:
                return "Neutral"
        except:
            return "None"

    # === Analyze each retro → echo
    result_rows = []
    for _, row in df_retro.iterrows():
        retro_date = row['Retro Date']
        echo_date = row['Echo Date']

        retro_trend = compute_trend(df_nifty, retro_date)
        echo_trend = compute_trend(df_nifty, echo_date)

        if retro_trend == "Up 🔺":
            predicted = "Down 🔻"
        elif retro_trend == "Down 🔻":
            predicted = "Up 🔺"
        else:
            predicted = "Neutral"

        result_rows.append({
            "Retro Date": retro_date,
            "Echo Date": echo_date,
            "Retro Trend": retro_trend,
            "Predicted Echo Reaction": predicted,
            "Actual Echo Trend": echo_trend
        })

    df_analysis = pd.DataFrame(result_rows)

    st.subheader("🔍 Echo Reversal Prediction Table")
    st.dataframe(df_analysis)

    # === Plotting chart
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_nifty['date'],
        open=df_nifty['Open'],
        high=df_nifty['High'],
        low=df_nifty['Low'],
        close=df_nifty['Close'],
        name="Nifty",
        increasing=dict(line=dict(color="green"), fillcolor="rgba(0,200,0,0.4)"),
        decreasing=dict(line=dict(color="red"), fillcolor="rgba(200,0,0,0.4)")
    ))

    for _, row in df_analysis.iterrows():
        echo_date = row['Echo Date']
        trend_symbol = row['Actual Echo Trend']
        symbol = trend_symbol if trend_symbol in ["Up 🔺", "Down 🔻"] else "🔁"

        fig.add_vline(x=echo_date, line_color="blue", line_dash="dot", opacity=0.6)
        fig.add_annotation(
            x=echo_date,
            y=df_nifty['High'].max(),
            text=symbol,
            showarrow=False,
            yanchor="bottom",
            font=dict(color="blue", size=14)
        )

    max_date = max(df_nifty['date'].max(), pd.to_datetime(df_retro['Echo Date'].max()))
    min_date = min(df_nifty['date'].min(), pd.to_datetime(df_retro['Echo Date'].min()))

    fig.update_layout(
        height=600,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        title="Nifty Chart with Mercury Echo Dates",
        xaxis=dict(
            range=[min_date - timedelta(days=5), max_date + timedelta(days=10)],
            showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'
        ),
        yaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot')
    )

    st.plotly_chart(fig, use_container_width=True)

elif filter_mode == "Rule 13: Neptune Log Distance":
    st.header("📘 Neptune Log Distance Differential Rule (Precomputed Excel)")

    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from datetime import datetime

    # Raphael's key delta values
    KEY_NUMBERS = [-2412, -2340, -1800, -900, 0, 900, 1800, 2340, 2412]

    # === Date range input
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2000, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 12, 31))

    index_choice = st.selectbox("Select Index", ["Nifty", "BankNifty"], index=0)

    # === Load Neptune Δ log distance data from Excel
    df = pd.read_excel("neptune_delta.xlsx")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]

    # Signal detection
    df['Signal'] = df['Delta'].apply(lambda x: any(abs(x - key) <= 1 for key in KEY_NUMBERS))
    signal_df = df[df['Signal'] == True]

    # === Load OHLC from NeonDB
    @st.cache_data
    def fetch_ohlc(index_name, start, end):
        query = """
            SELECT "Date", "Open", "High", "Low", "Close", "Vol(in M)"
            FROM ohlc_index
            WHERE index_name = %s AND "Date" BETWEEN %s AND %s
            ORDER BY "Date"
        """
        df_ohlc = pd.read_sql(query, engine, params=(index_name, start, end))
        df_ohlc['Date'] = pd.to_datetime(df_ohlc['Date'])
        return df_ohlc

    df_nifty_filtered = fetch_ohlc(index_choice, start_date, end_date)

    # === Results
    st.success(f"📌 Found {len(signal_df)} signal(s) matching Raphael's key levels.")
    st.subheader("📋 Neptune Daily Log Distance Differential Table")
    st.dataframe(df)

    # === Plotting
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.07,
        subplot_titles=["Nifty Candlestick Chart", "Neptune Δ log(True Distance)"]
    )

    # Nifty chart (from DB)
    fig.add_trace(go.Candlestick(
        x=df_nifty_filtered['Date'],
        open=df_nifty_filtered['Open'],
        high=df_nifty_filtered['High'],
        low=df_nifty_filtered['Low'],
        close=df_nifty_filtered['Close'],
        name="Nifty",
        showlegend=False
    ), row=1, col=1)

    # Neptune delta line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Delta'],
        mode='lines',
        name="Neptune Δ Log Distance",
        line=dict(color='blue')
    ), row=2, col=1)

    # Add signal markers
    for _, row in signal_df.iterrows():
        fig.add_vline(x=row['Date'], line_color='red', line_dash='dot', opacity=0.7, row=1, col=1)
        fig.add_vline(x=row['Date'], line_color='red', line_dash='dot', opacity=0.7, row=2, col=1)

    # Layout
    fig.update_layout(
        height=800,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        xaxis2=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        yaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        yaxis2=dict(
            title="Δ log(True Distance)",
            showspikes=True,
            spikemode='across',
            spikesnap='cursor',
            spikedash='dot'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

elif filter_mode == "Monthly OHLC Viewer":
    st.header("📆 Monthly OHLC & Daily Breakdown (Year-wise View)")

    import pandas as pd
    import plotly.graph_objects as go
    from datetime import timedelta
    from sqlalchemy import create_engine

    # === Load Tithi data from Excel
    df_tithi = pd.read_excel("tithi.xlsx")
    df_tithi['Start_IST'] = pd.to_datetime(df_tithi['Start_IST'])
    df_tithi['End_IST'] = pd.to_datetime(df_tithi['End_IST'])
    sunrise_time = timedelta(hours=6, minutes=0)

    def get_tithi_marker_dates(df_tithi):
        marker_dates = []
        for _, row in df_tithi.iterrows():
            tithi = row['Tithi']
            start = row['Start_IST']
            end = row['End_IST']
            day_cursor = start.normalize()
            while day_cursor <= end.normalize():
                sunrise_dt = day_cursor + sunrise_time
                if start <= sunrise_dt <= end:
                    marker_dates.append({"Date": sunrise_dt.date(), "Tithi": tithi})
                    break
                day_cursor += timedelta(days=1)
        return pd.DataFrame(marker_dates)

    df_tithi_markers = get_tithi_marker_dates(df_tithi)

    # === Month selection
    month_map = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    selected_months = st.multiselect("Select Month(s):", list(month_map.values()), default=["January"])
    selected_month_nums = [k for k, v in month_map.items() if v in selected_months]

    # === Load OHLC from NeonDB (PostgreSQL)
    index_choice = st.selectbox("Select Index", ["Nifty", "BankNifty"], index=0)

    @st.cache_data
    def load_ohlc_from_db(index_name):
        query = f"""
            SELECT "Date", "Open", "High", "Low", "Close", "Vol(in M)"
            FROM ohlc_index
            WHERE index_name = %s
            ORDER BY "Date"
        """
        df = pd.read_sql(query, engine, params=(index_name,))
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['MonthName'] = df['Date'].dt.strftime("%B")
        return df

    df_ohlc = load_ohlc_from_db(index_choice)

    if selected_months:
        df_filtered = df_ohlc[df_ohlc['Month'].isin(selected_month_nums)]

        for month_num in selected_month_nums:
            month_name = month_map[month_num]
            df_month = df_filtered[df_filtered['Month'] == month_num]

            st.subheader(f"📅 {month_name}")

            years = sorted(df_month['Year'].unique())
            for year in years:
                df_year_month = df_month[df_month['Year'] == year]
                if df_year_month.empty:
                    continue

                df_year_month_sorted = df_year_month.sort_values('Date')
                open_price = df_year_month_sorted.iloc[0]['Open']
                close_price = df_year_month_sorted.iloc[-1]['Close']
                high_price = df_year_month_sorted['High'].max()
                low_price = df_year_month_sorted['Low'].min()
                pct_change = ((close_price - open_price) / open_price) * 100

                # Monthly candle
                fig_month = go.Figure()
                fig_month.add_trace(go.Candlestick(
                    x=[f"{month_name} {year}"],
                    open=[open_price],
                    high=[high_price],
                    low=[low_price],
                    close=[close_price],
                    increasing=dict(line=dict(color="green")),
                    decreasing=dict(line=dict(color="red"))
                ))
                fig_month.update_layout(
                    title=f"Monthly Candle: {month_name} {year}",
                    xaxis_title="Month",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    height=700,
                    annotations=[
                        dict(
                            x=0,
                            y=high_price,
                            xref="x",
                            yref="y",
                            text=f"O: {open_price:.2f}, H: {high_price:.2f}, L: {low_price:.2f}, C: {close_price:.2f}, %Chg: {pct_change:.2f}%",
                            showarrow=False,
                            xanchor='left',
                            yanchor='bottom',
                            font=dict(size=17)
                        )
                    ]
                )

                # Daily candles
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df_year_month_sorted['Date'],
                    open=df_year_month_sorted['Open'],
                    high=df_year_month_sorted['High'],
                    low=df_year_month_sorted['Low'],
                    close=df_year_month_sorted['Close'],
                    increasing=dict(line=dict(color="green")),
                    decreasing=dict(line=dict(color="red")),
                    name=f"{month_name} {year}"
                ))

                # Tithi markers
                marker_subset = df_tithi_markers[
                    (df_tithi_markers['Date'] >= df_year_month_sorted['Date'].min().date()) &
                    (df_tithi_markers['Date'] <= df_year_month_sorted['Date'].max().date())
                ]
                for _, mark in marker_subset.iterrows():
                    fig.add_vline(
                        x=pd.to_datetime(mark['Date']),
                        line_color="blue" if mark['Tithi'] == "Poornima" else "purple",
                        line_dash="dot",
                        opacity=0.6
                    )
                    fig.add_annotation(
                        x=pd.to_datetime(mark['Date']),
                        y=df_year_month_sorted['High'].max(),
                        text=mark['Tithi'],
                        showarrow=False,
                        yanchor="bottom",
                        font=dict(size=12, color="blue" if mark['Tithi'] == "Poornima" else "purple")
                    )

                fig.update_layout(
                    title=f"Daily Chart for {month_name} {year}",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    xaxis_rangeslider_visible=False,
                    height=400
                )

                st.markdown(f"## 📊 {month_name} {year}")
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Monthly Candle: {month_name} {year}**")
                    st.plotly_chart(fig_month, use_container_width=True)

                with col2:
                    st.markdown(f"**Daily Chart for {month_name} {year}**")
                    st.plotly_chart(fig, use_container_width=True)

elif filter_mode == "Rule 13: All Planets Log Distance":
    st.header("🪐 Planetary Log Distance")

    from skyfield.api import load
    import pandas as pd
    import numpy as np
    from datetime import datetime
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Raphael's key delta values
    KEY_NUMBERS = [-2412, -2340, -1800, -900, 0, 900, 1800, 2340, 2412]

    # Planet selection
    planet_map = {
        "Mercury": "mercury",
        "Venus": "venus",
        "Earth": "earth",
        "Mars": "mars",
        "Jupiter": "jupiter barycenter",
        "Saturn": "saturn barycenter",
        "Uranus": "uranus barycenter",
        "Neptune": "neptune barycenter",
        "Pluto": "pluto barycenter"
    }

    planet_name = st.selectbox("Select Planet:", list(planet_map.keys()), index=7)

    # Date inputs
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2000, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2025, 12, 31))

    TOLERANCE = st.number_input(
    "Set Tolerance (e.g. 0.01 = ±0.01 arcsec)",
    min_value=0.0,
    max_value=5.0,
    value=0.01,
    step=0.01,
    format="%.4f"
)


    # Load ephemeris and timescale
    ts = load.timescale()
    eph = load('de421.bsp')

    earth = eph["earth"]
    body = eph[planet_map[planet_name]]

    # Date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    log_values = []
    distances = []

    for date in dates:
        t = ts.utc(date.year, date.month, date.day)
        astrometric = earth.at(t).observe(body).apparent()
        distance_au = astrometric.distance().au
        distances.append(distance_au)
        log_scaled = int(round(np.log10(distance_au) * 1e7))  # Raphael-style format
        log_values.append(log_scaled)

    df = pd.DataFrame({
        'Date': dates,
        'Distance_AU': distances,
        'Log_Distance': log_values
    })
    df['Delta'] = df['Log_Distance'].diff().fillna(0).astype(int)
    df['Signal'] = df['Delta'].apply(lambda x: any(abs(x - k) <= TOLERANCE for k in KEY_NUMBERS))

    signal_df = df[df['Signal']]

    import sqlalchemy

    # Optional: dropdown to choose index
    index_choice = st.selectbox("Select Index", ["Nifty", "BankNifty"], index=0)

    @st.cache_data
    def fetch_ohlc_from_db(index_name, start, end):
        query = f"""
            SELECT "Date", "Open", "High", "Low", "Close", "Vol(in M)"
            FROM ohlc_index
            WHERE index_name = %s AND "Date" BETWEEN %s AND %s
            ORDER BY "Date"
        """
        df = pd.read_sql(query, engine, params=(index_name, start, end))
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    df_nifty_filtered = fetch_ohlc_from_db(index_choice, start_date, end_date)


    st.success(f"📌 Found {len(signal_df)} signal(s) based on {planet_name}'s log distance differential.")

    st.subheader("📋 Log Distance Differential Table")
    st.dataframe(df)

    # === Plotting
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.07,
        subplot_titles=["Nifty Candlestick Chart", f"{planet_name} Δ log₁₀(Distance) × 10⁷"]
    )

    # Nifty Candlestick
    fig.add_trace(go.Candlestick(
        x=df_nifty_filtered['Date'],
        open=df_nifty_filtered['Open'],
        high=df_nifty_filtered['High'],
        low=df_nifty_filtered['Low'],
        close=df_nifty_filtered['Close'],
        name="Nifty",
        showlegend=False
    ), row=1, col=1)

    # Planetary Delta Line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Delta'],
        mode='lines',
        name=f"{planet_name} Δ log₁₀(Distance) × 10⁷",
        line=dict(color='blue')
    ), row=2, col=1)

    # Signal markers
    for _, row in signal_df.iterrows():
        fig.add_vline(x=row['Date'], line_color='red', line_dash='dot', opacity=0.7, row=1, col=1)
        fig.add_vline(x=row['Date'], line_color='red', line_dash='dot', opacity=0.7, row=2, col=1)

    fig.update_layout(
        height=800,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        xaxis2=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        yaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        yaxis2=dict(
            title="Δ log₁₀(Distance) × 10⁷",
            showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

elif filter_mode == "Rule 7: Mars–Venus Perihelium Reversals":
    st.header("🪐 Rule 7: Mars–Venus Perihelium Reversal Signals")

    import pandas as pd
    import numpy as np
    import swisseph as swe
    from datetime import datetime, timedelta
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Config
    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)

    VENUS_PERI = 130 + 42/60 + 46/3600  # 130.713°
    MARS_PERI = 334 + 56/60 + 9/3600    # 334.936°
    TOLERANCE = 2.5

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2026, 1, 1))

    index_choice = st.selectbox("Select Index", ["Nifty", "BankNifty"], index=0)

    # Generate planetary position table
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    events = []

    for date in dates:
        jd = swe.julday(date.year, date.month, date.day)

        venus_pos = swe.calc_ut(jd, swe.VENUS)[0][0]
        mars_pos = swe.calc_ut(jd, swe.MARS)[0][0]

        # Check Venus crossing
        if abs((venus_pos - VENUS_PERI + 360) % 360 - 0) <= TOLERANCE:
            events.append({"Date": date, "Planet": "Venus", "Signal": "Low 📉"})

        # Check Mars crossing
        if abs((mars_pos - MARS_PERI + 360) % 360 - 0) <= TOLERANCE:
            events.append({"Date": date, "Planet": "Mars", "Signal": "Reversal 🔁"})

    df_events = pd.DataFrame(events)
    st.success(f"📌 Found {len(df_events)} Perihelium events.")
    st.dataframe(df_events)

    st.write("✅ Insights:Venus signals are best when price is already falling before the event")
    st.write("✅ Insights:Mars signals are more flexible: if price breaks previous highs → go long; otherwise → expect reversal")

    # === Load OHLC from DB
    @st.cache_data
    def fetch_ohlc(index_name, start, end):
        query = """
            SELECT "Date", "Open", "High", "Low", "Close"
            FROM ohlc_index
            WHERE index_name = %s AND "Date" BETWEEN %s AND %s
            ORDER BY "Date"
        """
        df = pd.read_sql(query, engine, params=(index_name, start, end))
        df['Date'] = pd.to_datetime(df['Date'])
        return df

    df_price = fetch_ohlc(index_choice, start_date, end_date)

    # === Plotting
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df_price['Date'],
        open=df_price['Open'],
        high=df_price['High'],
        low=df_price['Low'],
        close=df_price['Close'],
        name="Price",
        increasing=dict(line=dict(color="green")),
        decreasing=dict(line=dict(color="red"))
    ))

    # === Annotate Perihelium Crossings
    for _, row in df_events.iterrows():
        fig.add_vline(x=row['Date'], line_dash="dot", line_color="purple", opacity=0.7)
        fig.add_annotation(
            x=row['Date'],
            y=df_price['High'].max(),
            text=f"{row['Planet']} {row['Signal']}",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=12, color="purple")
        )

    fig.update_layout(
        height=700,
        hovermode="x unified",
        title="Price Chart with Mars/Venus Perihelium Reversals",
        xaxis_rangeslider_visible=False
    )

    st.plotly_chart(fig, use_container_width=True)

elif filter_mode == "Rule 27: Mercury's Speed Triggers (59′ and 1°58′)":
    st.header("📏 Rule 27: Mercury Speed Triggers – 59′ and 1°58′ Reversal Rule")

    import pandas as pd
    import numpy as np
    import swisseph as swe
    from datetime import datetime, timedelta
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Constants
    ARC59 = 59 / 60       # 0.9833°
    ARC118 = 118 / 60     # 1.9667°
    TOLERANCE = 0.05

    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2026, 1, 1))

    index_choice = st.selectbox("Select Index", ["Nifty", "BankNifty"], index=0)

    # Calculate daily Mercury speed
    dates = pd.date_range(start=start_date - timedelta(days=1), end=end_date, freq="D")
    speeds = []
    for i in range(1, len(dates)):
        jd0 = swe.julday(dates[i - 1].year, dates[i - 1].month, dates[i - 1].day)
        jd1 = swe.julday(dates[i].year, dates[i].month, dates[i].day)

        lon0 = swe.calc_ut(jd0, swe.MERCURY)[0][0]
        lon1 = swe.calc_ut(jd1, swe.MERCURY)[0][0]

        # Unwrap angle to account for 360° rollover
        if lon1 < lon0:
            lon1 += 360
        speed = lon1 - lon0

        speeds.append({
            "Date": dates[i],
            "Speed": round(speed, 6),
            "Signal": (
                abs(speed - ARC59) <= TOLERANCE or abs(speed - ARC118) <= TOLERANCE
            )
        })

    df = pd.DataFrame(speeds)
    signal_df = df[df['Signal']]

    st.success(f"📌 Found {len(signal_df)} Mercury speed signal(s) near 59′ or 1°58′")
    st.dataframe(df)

    # === Load OHLC from NeonDB
    @st.cache_data
    def fetch_ohlc(index_name, start, end):
        query = """
            SELECT "Date", "Open", "High", "Low", "Close"
            FROM ohlc_index
            WHERE index_name = %s AND "Date" BETWEEN %s AND %s
            ORDER BY "Date"
        """
        df_ohlc = pd.read_sql(query, engine, params=(index_name, start, end))
        df_ohlc['Date'] = pd.to_datetime(df_ohlc['Date'])
        return df_ohlc

    df_price = fetch_ohlc(index_choice, start_date, end_date)

    # === Plot
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.07,
        subplot_titles=["Price Chart", "Mercury Daily Speed (°/day)"]
    )

    # Price chart
    fig.add_trace(go.Candlestick(
        x=df_price['Date'],
        open=df_price['Open'],
        high=df_price['High'],
        low=df_price['Low'],
        close=df_price['Close'],
        name=index_choice,
        increasing=dict(line=dict(color="green")),
        decreasing=dict(line=dict(color="red"))
    ), row=1, col=1)

    # Speed line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Speed'],
        mode='lines',
        name="Mercury Speed",
        line=dict(color='orange')
    ), row=2, col=1)

    # Vertical signal markers
    for _, row in signal_df.iterrows():
        fig.add_vline(x=row['Date'], line_color='purple', line_dash='dot', opacity=0.6, row=1, col=1)
        fig.add_vline(x=row['Date'], line_color='purple', line_dash='dot', opacity=0.6, row=2, col=1)
        fig.add_annotation(
            x=row['Date'],
            y=df_price['High'].max(),
            text="⚡",
            showarrow=False,
            yanchor="bottom",
            font=dict(color="purple", size=14),
            row=1, col=1
        )

    fig.update_layout(
        height=800,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        xaxis2=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        yaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'),
        yaxis2=dict(
            title="°/day",
            showspikes=True, spikemode='across', spikesnap='cursor', spikedash='dot'
        )
    )

    st.plotly_chart(fig, use_container_width=True)

elif filter_mode == "Rule 23: Saturn Latitude Differential":
    st.header("🪐 Rule 23: Saturn Latitude Differential (Swiss Ephemeris)")

    import swisseph as swe
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # === Ephemeris config
    swe.set_ephe_path("C:/ephe")
    swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)

    # === Input
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input("End Date", value=datetime(2026, 12, 31))

    index_choice = st.selectbox("Select Index", ["Nifty", "BankNifty"], index=0)

    # Raphael values: 2.39, 2.19, ..., 0.00, ..., 2.39 (mirror)
    BASE = 2.39
    STEP = 0.20
    TOLERANCE = st.number_input(
    "Set Tolerance (e.g. 0.01 = ±0.01 arcsec)",
    min_value=0.0,
    max_value=5.0,
    value=0.01,
    step=0.01,
    format="%.4f"
)

    key_levels = [round(BASE - i * STEP, 2) for i in range(13)] + \
                 [round(i * STEP, 2) for i in range(1, 13)]

    # === Compute heliocentric latitude differential using Swiss Ephemeris
    dates = pd.date_range(start=start_date - timedelta(days=1), end=end_date, freq='D')
    results = []

    for date in dates:
        jd = swe.julday(date.year, date.month, date.day)

        # heliocentric position
        result, _ = swe.calc_ut(jd, swe.SATURN, swe.FLG_SWIEPH | swe.FLG_HELCTR)
        lat_deg = result[1]  # heliocentric latitude in degrees

        results.append({
            "Date": date,
            "Latitude": lat_deg
        })

    df = pd.DataFrame(results)
    df['Motion'] = df['Latitude'].diff().abs() * 3600  # arcseconds
    df['Signal'] = df['Motion'].apply(lambda x: any(abs(x - k) <= TOLERANCE for k in key_levels))
    signal_df = df[df['Signal'] == True]

    # === Load OHLC from NeonDB
    @st.cache_data
    def fetch_ohlc(index_name, start, end):
        query = """
            SELECT "Date", "Open", "High", "Low", "Close"
            FROM ohlc_index
            WHERE index_name = %s AND "Date" BETWEEN %s AND %s
            ORDER BY "Date"
        """
        df_ohlc = pd.read_sql(query, engine, params=(index_name, start, end))
        df_ohlc['Date'] = pd.to_datetime(df_ohlc['Date'])
        return df_ohlc

    df_price = fetch_ohlc(index_choice, start_date, end_date)

    # === Backtest duration
    def detect_trend_duration(df_price, signal_date, threshold=0.5, max_days=20):
        df_price = df_price.set_index('Date').sort_index()
        if signal_date not in df_price.index:
            future = df_price[df_price.index > signal_date]
            if future.empty:
                return 0, "No Data"
            signal_date = future.index[0]

        P0 = df_price.loc[signal_date]['Close']
        days = 0
        direction = None

        for i in range(1, max_days + 1):
            future_date = signal_date + timedelta(days=i)
            if future_date not in df_price.index:
                continue
            Pn = df_price.loc[future_date]['Close']
            change = ((Pn - P0) / P0) * 100

            if direction is None:
                if change > threshold:
                    direction = "Up 📈"
                elif change < -threshold:
                    direction = "Down 📉"
                else:
                    continue
            elif direction == "Up 📈" and change < -threshold:
                break
            elif direction == "Down 📉" and change > threshold:
                break

            days += 1

        return days, direction or "Neutral ➖"

    # Apply duration check
    durations = []
    for _, row in signal_df.iterrows():
        duration, trend = detect_trend_duration(df_price.copy(), row['Date'])
        durations.append((duration, trend))

    signal_df['Trend Duration (days)'] = [d[0] for d in durations]
    signal_df['Trend Direction'] = [d[1] for d in durations]

    st.success(f"📌 Found {len(signal_df)} Saturn signal(s).")
    st.dataframe(signal_df[['Date', 'Motion', 'Trend Direction', 'Trend Duration (days)']])

    # === Plotting
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.07,
        subplot_titles=["Market Chart", "Saturn Heliocentric Latitude Motion (arcsec/day)"]
    )

    fig.add_trace(go.Candlestick(
        x=df_price['Date'],
        open=df_price['Open'],
        high=df_price['High'],
        low=df_price['Low'],
        close=df_price['Close'],
        name="Price",
        increasing=dict(line=dict(color="green")),
        decreasing=dict(line=dict(color="red"))
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Motion'],
        mode='lines',
        name="Saturn Δ Latitude (arcsec)",
        line=dict(color='orange')
    ), row=2, col=1)

    for _, row in signal_df.iterrows():
        fig.add_vline(x=row['Date'], line_color='purple', line_dash='dot', opacity=0.6, row=1, col=1)
        fig.add_vline(x=row['Date'], line_color='purple', line_dash='dot', opacity=0.6, row=2, col=1)
        fig.add_annotation(
            x=row['Date'],
            y=df_price['High'].max(),
            text=f"{row['Trend Direction']} ({row['Trend Duration (days)']}d)",
            showarrow=False,
            yanchor="bottom",
            font=dict(size=12, color="purple"),
            row=1, col=1
        )

    fig.update_layout(
        height=800,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        xaxis=dict(showspikes=True),
        xaxis2=dict(showspikes=True),
        yaxis=dict(showspikes=True),
        yaxis2=dict(title="Arcseconds/day", showspikes=True)
    )

    st.plotly_chart(fig, use_container_width=True)
