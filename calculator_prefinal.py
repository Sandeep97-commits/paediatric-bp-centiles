import os

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")

import streamlit as st
import pandas as pd
from datetime import date, datetime

from fpdf import FPDF
import pytz, random, string

def create_pdf(age, gender, height, sbp, dbp, map_val,
               final_class, sbp_result, dbp_result):
    # Patient ID
    patient_id = random.choice(string.ascii_uppercase)

    # Current IST date & time
    ist = pytz.timezone("Asia/Kolkata")
    now = datetime.now(ist)
    date_str = now.strftime("%d-%m-%Y %H:%M IST")

    # Initialize PDF
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(200, 10, "Paediatric BP Centile Report", ln=True, align="C")

    # Date top right
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, date_str, align="R", ln=True)
    pdf.ln(5)

    # Patient ID
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, f"Patient: {patient_id}", ln=True)
    pdf.ln(5)

    # Helper for headings
    def add_field(heading, value):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(50, 10, f"{heading}:", ln=0)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, str(value), ln=1)

    # Core values
    add_field("Age", age)
    add_field("Gender", gender)
    add_field("Height", f"{height:.1f} cm")
    add_field("SBP", f"{sbp:.1f} mmHg")
    add_field("DBP", f"{dbp:.1f} mmHg")
    add_field("MAP", f"{map_val:.1f} mmHg")
    pdf.ln(5)

    # Classification
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "Classification:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, final_class)
    pdf.ln(5)

    # Centile notes (with adjustment note)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"SBP: {sbp_result} (adjusted for height, age and gender).")
    pdf.multi_cell(0, 10, f"DBP: {dbp_result} (adjusted for height, age and gender).")
    pdf.ln(10)

    # References
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, "References", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 8,
        "- AAP Clinical Practice Guideline for Screening and Diagnosis "
        "(Pediatrics. 2017;140(3):e20171904)\n"
        "- PALS 2024 Algorithm (American Heart Association)"
    )

    return pdf.output(dest="S").encode("latin-1")

# ---------------- Data Loader ----------------
@st.cache_data
def load_height_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if "Age (years)" in df.columns:
        df.set_index("Age (years)", inplace=True)
    else:
        df = pd.read_csv(path, header=1, encoding="utf-8-sig")
        df.set_index("Age (years)", inplace=True)
    df.index = pd.to_numeric(df.index, errors="coerce")
    df = df[~df.index.isna()]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df

def load_bp_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.rename(columns={df.columns[0]: "BP_centile"}, inplace=True)
    df["BP_centile"] = df["BP_centile"].astype(str).str.replace("%", "").astype(float)
    new_columns = ["BP_centile"]
    for c in df.columns[1:]:
        new_columns.append(c.replace("th", "").strip())
    df.columns = new_columns
    df.set_index("BP_centile", inplace=True)
    return df

# ---------------- Core Functions ----------------
def get_height_centiles(age, gender, height, boys_df, girls_df):
    df = boys_df if gender.lower().startswith('b') else girls_df
    age_years = round(float(age))
    available_ages = df.index.values.astype(float)
    closest_age = min(available_ages, key=lambda x: abs(x - age_years))
    row = df.loc[closest_age].astype(float)

    diffs = row - height
    abs_diffs = diffs.abs()
    min_diff = abs_diffs.min()

    closest_centiles = abs_diffs[abs_diffs == min_diff].index.tolist()
    closest_heights = [row[c] for c in closest_centiles]
    return list(zip(closest_centiles, closest_heights)), int(closest_age)

def get_bp_centile(bp_value: float, df: pd.DataFrame, height_centile: int) -> tuple:
    """Return centile description and numeric band (for classification)"""
    col = str(height_centile)
    if col not in df.columns:
        return f"Height centile {col} not in table", None, None
    centiles = df.index.astype(float).tolist()
    values = df[col].astype(float).tolist()

    # below 5th centile
    if bp_value < values[0]:
        return f"Below {centiles[0]}th centile (value: {values[0]} mm Hg)", centiles[0], values[0]

    # exact match
    if bp_value in values:
        idx = values.index(bp_value)
        return f"At {centiles[idx]}th centile (value: {values[idx]} mm Hg)", centiles[idx], values[idx]

    # in between
    for i in range(len(values)-1):
        if values[i] < bp_value < values[i+1]:
            return f"Between {centiles[i]}th (value: {values[i]} mm Hg) and {centiles[i+1]}th (value: {values[i+1]} mm Hg)", centiles[i], values[i]

    # above
    return f"Above {centiles[-1]}th centile (value: {values[-1]} mm Hg)", centiles[-1], values[-1]

def classify_bp(sbp, dbp, sbp_info, dbp_info):
    """Classify per AAP 2017 based on SBP and DBP"""
    def classify_single(bp_value, centile, ref_value):
        if centile is None:
            return "Unknown"

        if centile < 90:
            return "Normal BP"

        elif 90 <= centile < 95:
            return "Elevated BP"

        elif 95 <= centile and bp_value <= ref_value + 12:
            return "Stage 1 HTN"

        elif bp_value > ref_value + 12:
            return "Stage 2 HTN"

        return "Unknown"

    sbp_class = classify_single(sbp, sbp_info[1], sbp_info[2])
    dbp_class = classify_single(dbp, dbp_info[1], dbp_info[2])

    # take worst of SBP/DBP
    severity_order = ["Hypotension?", "Normal BP", "Elevated BP", "Stage 1 HTN", "Stage 2 HTN"]
    final_class = max([sbp_class, dbp_class], key=lambda x: severity_order.index(x))
    return final_class

# ---------------- Streamlit UI ----------------
st.title("Paediatric Blood Pressure Centile Calculator 1-17")

# --- New Age Input Block ---
st.subheader("Age")
age_input_method = st.radio(
    "Choose how to enter age:",
    ["Age in years (decimal)", "Pick from calendar", "Enter DD/MM/YYYY"],
    index=0
)

today = date.today()

if age_input_method == "Age in years (decimal)":
    age = st.number_input(
        "Age (years):", min_value=1.0, max_value=17.0, step=0.1
    )

elif age_input_method == "Pick from calendar":
    dob = st.date_input(
        "Select Date of Birth:",
        min_value=date(1900,1,1),
        max_value=today
    )
    delta = today - dob
    age_days = delta.days
    age_decimal = age_days / 365.25
    if age_decimal < 1:
        st.error("Age must be ≥1 year")
    elif age_decimal > 17:
        st.error("Age must be ≤17 years")
    age = min(max(age_decimal,1.0),17.0)
    years = age_days // 365
    months = (age_days % 365) // 30
    days = (age_days % 365) % 30
    st.info(f"Age: {age_decimal:.2f} years (~{years} y {months} m {days} d)")

elif age_input_method == "Enter DD/MM/YYYY":
    day = st.number_input("Day", min_value=1, max_value=31, value=1)
    month = st.number_input("Month", min_value=1, max_value=12, value=1)
    year = st.number_input("Year", min_value=1900, max_value=today.year, value=2010)
    try:
        dob = date(int(year), int(month), int(day))
        delta = today - dob
        age_days = delta.days
        age_decimal = age_days / 365.25
        if age_decimal < 1:
            st.error("Age must be ≥1 year")
        elif age_decimal > 17:
            st.error("Age must be ≤17 years")
        age = min(max(age_decimal,1.0),17.0)
        years = age_days // 365
        months = (age_days % 365) // 30
        days = (age_days % 365) % 30
        st.info(f"Age: {age_decimal:.2f} years (~{years} y {months} m {days} d)")
    except ValueError:
        st.error("Invalid date entered")
        age = 1.0  # fallback

# ---------------- Height Input (ADDED OPTIONS) ----------------
gender = st.radio("Gender:", ["Boy", "Girl"])

st.subheader("Height Input")
height_input_method = st.radio(
    "Choose how to enter height:",
    ["Centimeters (cm)", "Inches", "Feet + Inches"],
    index=0
)

if height_input_method == "Centimeters (cm)":
    height = st.number_input("Height (cm):", min_value=50.0, max_value=200.0, step=0.1)

elif height_input_method == "Inches":
    height_in = st.number_input("Height (inches):", min_value=20.0, max_value=80.0, step=0.1)
    height = height_in * 2.54
    st.info(f"Converted height: {height:.1f} cm")

elif height_input_method == "Feet + Inches":
    feet = st.number_input("Feet:", min_value=1, max_value=7, value=5)
    inches = st.number_input("Inches:", min_value=0, max_value=11, value=6)
    height = (feet * 12 + inches) * 2.54
    st.info(f"Converted height: {height:.1f} cm")

# ---------------- BP Inputs ----------------
sbp = st.number_input("Systolic BP (SBP, mm Hg):", min_value=50.0, max_value=200.0, step=0.1)
dbp = st.number_input("Diastolic BP (DBP, mm Hg):", min_value=30.0, max_value=130.0, step=0.1)

# Load height tables
boys_df = load_height_csv(os.path.join(DATA_DIR, "height_boys.csv"))
girls_df = load_height_csv(os.path.join(DATA_DIR, "Height_girls.csv"))

# Process
height_results, rounded_age = get_height_centiles(age, gender, height, boys_df, girls_df)

if height_results:
    st.subheader("Nearest Height Centile")
    for centile, val in height_results:
        st.write(f"{centile}: {val:.1f} cm")

    # For each height centile, check SBP + DBP
    st.subheader("Blood Pressure Centile (with reference values)")
    for centile, _ in height_results:
        height_centile_num = int(centile.replace("Height_", "").replace("th", ""))

        # pick the right SBP + DBP files
        sbp_path = os.path.join(DATA_DIR, f"{gender.lower()}s_sbp_age{rounded_age}.csv")
        dbp_path = os.path.join(DATA_DIR, f"{gender.lower()}s_dbp_age{rounded_age}.csv")

        sbp_df = load_bp_csv(sbp_path)
        dbp_df = load_bp_csv(dbp_path)

        sbp_result, sbp_centile, sbp_ref = get_bp_centile(sbp, sbp_df, height_centile_num)
        dbp_result, dbp_centile, dbp_ref = get_bp_centile(dbp, dbp_df, height_centile_num)

        st.write(f"**For height centile {height_centile_num}:**")
        st.write(f"- SBP {sbp:.1f} mm Hg → {sbp_result}")
        st.write(f"- DBP {dbp:.1f} mm Hg → {dbp_result}")

        # Classification
        final_class = classify_bp(sbp, dbp, (sbp_result, sbp_centile, sbp_ref), (dbp_result, dbp_centile, dbp_ref))

        # --- Hypotension logic override ---
        age_years = rounded_age  # already rounded from get_height_centiles
        hypotension_text = None

        # PALS 2024 hypotension: SBP
        if 1 <= age_years < 10 and sbp < (2 * age_years + 70):
            hypotension_text = "Hypotension (PALS 2024 Algorithm)"
        elif age_years > 10 and sbp < sbp_ref:  # sbp_ref is 5th centile
            hypotension_text = "Hypotension (PALS 2024 Algorithm)"
         # DBP < 5th centile in isolation
        elif dbp < dbp_ref:
             hypotension_text = "?Hypotension"

        if hypotension_text:
           final_class = hypotension_text


            # Color box display
        if final_class.startswith("Hypotension") or final_class == "?Hypotension":
            color = "orange"
        elif final_class == "Normal BP":
            color = "#90EE90"  # light green
        elif final_class == "Elevated BP":
            color = "#FFFF99"  # yellow
        elif final_class == "Stage 1 HTN":
            color = "#FF9999"  # light red
        else:
            color = "#FF4C4C"  # Stage 2 #Dark Red


        aap_tag = " (AAP 2017)" if not final_class.startswith("Hypotension") and final_class != "?Hypotension" else ""
        st.markdown(
            f"<div style='background-color:{color}; padding:15px; border-radius:10px; text-align:center;'>"
            f"<span style='font-size:20px; font-weight:bold; color:black;'>{final_class} {aap_tag}</span>"
            "</div>",
            unsafe_allow_html=True,
        )
            # --- Calculate MAP ---
        map_value = dbp + (sbp - dbp) / 3

          # --- Display MAP in blue box ---
        st.markdown(
            f"<div style='background-color:#ADD8E6; padding:15px; border-radius:10px; text-align:center;'>"
            f"<span style='font-size:20px; font-weight:bold; color:black;'>Mean Arterial Pressure (MAP): {map_value:.1f} mm Hg</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        # --- PDF Export Button ---
        if st.button("Generate PDF Report"):
            pdf_bytes = create_pdf(
                age, gender, height, sbp, dbp, map_value,
                final_class, sbp_result, dbp_result
            )
            st.download_button(
                label="📄 Download Patient Report",
                data=pdf_bytes,
                file_name="bp_report.pdf",
                mime="application/pdf",
            )


         # ---------------- References (moved outside loop, at bottom) ----------------
        st.markdown("---------------")  # longer separator to push it down visually
        st.subheader("References:")
        st.markdown(
            """
        - [AAP Clinical Practice Guideline for Screening and Diagnosis](https://publications.aap.org/pediatrics/article/140/3/e20171904/38358/Clinical-Practice-Guideline-for-Screening-and?autologincheck=redirected)  
        - [PALS 2024 Algorithm](https://www.acls-pals-bls.com/algorithms/pals/)
        """
        )
