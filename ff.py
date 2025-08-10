import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import io
import textwrap
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from PIL import Image as PILImage

# Set page config and custom CSS
st.set_page_config(page_title="Oil & Gas Field Prioritization", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    .stApp { background-color: #F5F5F5; color: #333333; }
    .stHeader { background-color: #E8ECEF; padding: 10px; border-radius: 5px; border-bottom: 2px solid #4A90E2; }
    .stSlider > div > div > div > div { background-color: #4A90E2; height: 12px; }
    .stButton > button { background-color: #4A90E2; color: #FFFFFF; border-radius: 4px; padding: 5px 15px; font-weight: bold; font-size: 12px; }
    .stButton > button:hover { background-color: #357ABD; }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
current_time = datetime.now().strftime("%I:%M %p CEST, %B %d, %Y")
st.markdown(f'<div class="stHeader"><h1>Oil & Gas Field Prioritization Suite</h1><p>Generated at {current_time}</p></div>', unsafe_allow_html=True)

# Ideal profile and default fields
ideal_field = {
    "Volume of Recoverable Reserves (MMbbl)": 150.0,
    "CAPEX ($M)": 0.0,
    "OPEX ($M)": 0.0,
    "IRR (%)": 100.0,
    "Distance from Pipelines (km)": 0.0,
    "Distance from Roads (km)": 0.0,
    "Distance from Processing Facilities (km)": 0.0,
    "Porosity (%)": 30.0,
    "Permeability (mD)": 100.0,
    "Viscosity of Reservoir Fluid (cP)": 0.0,
    "Technical Risk (0-1)": 0.0,
    "Operational Risk (0-1)": 0.0,
    "Commercial Risk (0-1)": 0.0,
    "Economic Risk (0-1)": 0.0,
    "Political Risk (0-1)": 0.0
}

default_fields = {
    "Field A": {
        "Volume of Recoverable Reserves (MMbbl)": 100.0,
        "CAPEX ($M)": 50.0,
        "OPEX ($M)": 10.0,
        "IRR (%)": 15.0,
        "Distance from Pipelines (km)": 10.0,
        "Distance from Roads (km)": 5.0,
        "Distance from Processing Facilities (km)": 15.0,
        "Porosity (%)": 20.0,
        "Permeability (mD)": 50.0,
        "Viscosity of Reservoir Fluid (cP)": 5.0,
        "Technical Risk (0-1)": 0.2,
        "Operational Risk (0-1)": 0.3,
        "Commercial Risk (0-1)": 0.1,
        "Economic Risk (0-1)": 0.4,
        "Political Risk (0-1)": 0.1
    },
    "Field B": {
        "Volume of Recoverable Reserves (MMbbl)": 80.0,
        "CAPEX ($M)": 40.0,
        "OPEX ($M)": 8.0,
        "IRR (%)": 12.0,
        "Distance from Pipelines (km)": 5.0,
        "Distance from Roads (km)": 3.0,
        "Distance from Processing Facilities (km)": 10.0,
        "Porosity (%)": 18.0,
        "Permeability (mD)": 40.0,
        "Viscosity of Reservoir Fluid (cP)": 3.0,
        "Technical Risk (0-1)": 0.4,
        "Operational Risk (0-1)": 0.2,
        "Commercial Risk (0-1)": 0.3,
        "Economic Risk (0-1)": 0.2,
        "Political Risk (0-1)": 0.3
    }
}

if 'fields' not in st.session_state:
    st.session_state.fields = default_fields.copy()
if 'weights' not in st.session_state:
    st.session_state.weights = {param: 50.0 for param in ideal_field.keys()}

# Define sections for categorization
sections = {
    "General Specifications": ["Volume of Recoverable Reserves (MMbbl)", "CAPEX ($M)", "OPEX ($M)", "IRR (%)"],
    "Distance from Infrastructure": ["Distance from Pipelines (km)", "Distance from Roads (km)", "Distance from Processing Facilities (km)"],
    "Rock and Fluid Quality": ["Porosity (%)", "Permeability (mD)", "Viscosity of Reservoir Fluid (cP)"],
    "Risk Levels": ["Technical Risk (0-1)", "Operational Risk (0-1)", "Commercial Risk (0-1)", "Economic Risk (0-1)", "Political Risk (0-1)"]
}

# Sidebar
with st.sidebar:
    st.header("Configuration")
    selected_params = st.multiselect(
        "Select Parameters to Compare",
        options=list(ideal_field.keys()),
        default=list(ideal_field.keys())
    )

# Inputs
with st.expander("Field Data Input", expanded=True):
    for section, params in sections.items():
        with st.container():
            st.subheader(section)
            # Add column headers for the section
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                st.markdown("**Field A**")
            with col2:
                st.markdown("**Field B**")
            with col3:
                st.markdown("**Weight**")
            
            # Input fields for parameters
            for param in params:
                if param in selected_params:
                    col1, col2, col3 = st.columns([1, 1, 1])
                    key_a = f"a_{param.replace(' ', '_')}"
                    key_b = f"b_{param.replace(' ', '_')}"
                    key_w = f"w_{param.replace(' ', '_')}"

                    with col1:
                        st.session_state.fields["Field A"][param] = st.number_input(
                            param,
                            value=st.session_state.fields["Field A"].get(param, 0.0),
                            step=1.0,
                            key=key_a,
                            format="%.1f"
                        )
                    with col2:
                        st.session_state.fields["Field B"][param] = st.number_input(
                            param,
                            value=st.session_state.fields["Field B"].get(param, 0.0),
                            step=1.0,
                            key=key_b,
                            format="%.1f"
                        )
                    with col3:
                        st.session_state.weights[param] = st.slider(
                            "",
                            min_value=0.0,
                            max_value=100.0,
                            value=st.session_state.weights.get(param, 50.0),
                            step=1.0,
                            key=key_w
                        )

# Scoring Function
def calculate_scores(ideal, fields, weights, selected_params):
    scores = {}
    normalized_fields = {}
    
    # Define lower-is-better parameters
    lower_is_better = [
        "CAPEX ($M)", "OPEX ($M)", "Distance from Pipelines (km)",
        "Distance from Roads (km)", "Distance from Processing Facilities (km)",
        "Viscosity of Reservoir Fluid (cP)", "Technical Risk (0-1)",
        "Operational Risk (0-1)", "Commercial Risk (0-1)", "Economic Risk (0-1)",
        "Political Risk (0-1)"
    ]

    # Calculate maximum values dynamically for normalization
    max_values = {}
    for param in selected_params:
        if param in lower_is_better:
            max_val = max(field_data[param] for field_data in fields.values())
            max_values[param] = max_val if max_val > 0 else 1.0  # Avoid division by zero
        else:
            max_values[param] = ideal[param] if ideal[param] > 0 else 1.0

    for field_name, field_data in fields.items():
        norm_values = []
        for param in selected_params:
            ideal_val = ideal[param]
            field_val = field_data[param]
            
            if param in lower_is_better:
                if field_val == 0.0:
                    score = 1.0
                else:
                    score = max(0.0, 1.0 / (1.0 + field_val / max_values[param]))
            else:  # Higher-is-better parameters
                score = min(1.0, field_val / max_values[param] if max_values[param] > 0 else 1.0)
            
            norm_values.append(score)
        normalized_fields[field_name] = norm_values

    # Convert normalized fields to similarity scores
    for field, values in normalized_fields.items():
        weighted_sum = sum(val * (weights[param] / 100) for val, param in zip(values, selected_params))
        total_weight = sum(weights[param] / 100 for param in selected_params)
        scores[field] = (weighted_sum / total_weight * 100) if total_weight > 0 else 0.0

    return scores, normalized_fields

# Function to calculate category-wise scores
def calculate_category_scores(norm_data, weights, selected_params, sections):
    category_scores = {"Field A": {}, "Field B": {}}
    for category, cat_params in sections.items():
        cat_selected = [p for p in cat_params if p in selected_params]
        if cat_selected:
            idx = [selected_params.index(p) for p in cat_selected]
            cat_weights = [weights[p] / 100 for p in cat_selected]
            total_weight = sum(cat_weights)
            if total_weight > 0:
                for field in ["Field A", "Field B"]:
                    cat_norm = [norm_data[field][i] for i in idx]
                    weighted_sum = sum(n * w for n, w in zip(cat_norm, cat_weights))
                    category_scores[field][category] = (weighted_sum / total_weight) * 100
    return category_scores

# Improved Radar Chart (Smaller Size)
def plot_radar_chart(data_dict, selected_params):
    labels = selected_params
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    labels += labels[:1]
    angles += angles[:1]

    # Smaller figure size
    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw=dict(polar=True))

    # Adjust position to fit smaller size
    ax.set_position([0.15, 0.15, 0.7, 0.7])  # [left, bottom, width, height]

    for label, values in data_dict.items():
        values += values[:1]
        ax.plot(angles, values, label=label, linewidth=0.5)
        ax.fill(angles, values, alpha=0.2)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    # Customize grid and background
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=3)
    ax.grid(True, color='gray', linestyle='--', alpha=0.5)
    ax.set_facecolor('#F8F9FA')
    # Tick labels (axes labels)
    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        labels[:-1],
        fontsize=3,
        ha='center'
    )

    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
        label.set_verticalalignment('center')
        label.set_fontsize(3)
        label.set_rotation_mode('anchor')
        label.set_y(label.get_position()[1] + 0)

    ax.set_title("Radar Chart - Field Similarity to Ideal", fontsize=6, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=3)

    return fig

# Function to plot category bar chart (Smaller Size)
def plot_category_bar_chart(category_scores, categories):
    x = np.arange(len(categories))
    width = 0.35

    # Smaller figure size
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, [category_scores['Field A'].get(cat, 0) for cat in categories], width, label='Field A', color='#4A90E2')
    ax.bar(x + width/2, [category_scores['Field B'].get(cat, 0) for cat in categories], width, label='Field B', color='#50C878')

    ax.set_ylabel('Similarity to Ideal (%)', fontsize=8)
    ax.set_title('Category-wise Similarity to Ideal Field', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig

# Function to save figure as image and return bytes
def fig_to_image_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf

# Function to generate PDF report
def generate_pdf_report(current_time, ideal_field, fields, weights, selected_params, scores, category_scores, radar_fig, bar_fig, sections):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("Oil & Gas Field Development Prioritization Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Date: {current_time}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Ideal Field Profile
    story.append(Paragraph("Ideal Field Profile:", styles['Heading2']))
    for param, value in ideal_field.items():
        if param in selected_params:
            story.append(Paragraph(f"{param}: {value}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Field Data
    story.append(Paragraph("Field Data:", styles['Heading2']))
    for field, data in fields.items():
        story.append(Paragraph(f"{field}:", styles['Heading3']))
        for param, value in data.items():
            if param in selected_params:
                story.append(Paragraph(f"  {param}: {value}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Parameter Weights
    story.append(Paragraph("Parameter Weights:", styles['Heading2']))
    for param, weight in weights.items():
        if param in selected_params:
            story.append(Paragraph(f"{param}: {weight}%", styles['Normal']))
    story.append(Spacer(1, 12))

    # Overall Prioritization Results
    story.append(Paragraph("Overall Prioritization Results:", styles['Heading2']))
    for field, score in scores.items():
        story.append(Paragraph(f"{field}: {score:.1f}% similarity to Ideal Field", styles['Normal']))
    story.append(Spacer(1, 12))

    # Category-wise Prioritization Results
    story.append(Paragraph("Category-wise Prioritization Results:", styles['Heading2']))
    for category in sections.keys():
        story.append(Paragraph(f"{category}:", styles['Heading3']))
        for field in ["Field A", "Field B"]:
            score = category_scores[field].get(category, 0.0)
            story.append(Paragraph(f"  {field}: {score:.1f}% similarity to Ideal", styles['Normal']))
    story.append(Spacer(1, 12))

    # Charts
    story.append(Paragraph("Radar Chart - Field Similarity to Ideal:", styles['Heading2']))
    radar_buf = fig_to_image_bytes(radar_fig)
    radar_img = PILImage.open(radar_buf)
    radar_img = radar_img.resize((300, 300))  # Resize for PDF
    story.append(Image(radar_buf, width=300, height=300))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Category-wise Similarity to Ideal Field:", styles['Heading2']))
    bar_buf = fig_to_image_bytes(bar_fig)
    bar_img = PILImage.open(bar_buf)
    bar_img = bar_img.resize((400, 300))  # Resize for PDF
    story.append(Image(bar_buf, width=400, height=300))
    story.append(Spacer(1, 12))

    # Category-wise Table
    story.append(Paragraph("Category-wise Similarity Table:", styles['Heading2']))
    data = [["Category", "Field A (%)", "Field B (%)"]]
    for category in sections.keys():
        data.append([
            category,
            f"{category_scores['Field A'].get(category, 0.0):.1f}",
            f"{category_scores['Field B'].get(category, 0.0):.1f}"
        ])
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)

    doc.build(story)
    buffer.seek(0)
    return buffer

# Prioritization
if st.button("Calculate Prioritization"):
    scores, norm_data = calculate_scores(ideal_field, st.session_state.fields, st.session_state.weights, selected_params)
    
    # Add Ideal Field
    norm_data["Ideal Field"] = [1.0] * len(selected_params)

    fig_radar = plot_radar_chart(norm_data, selected_params)
    st.pyplot(fig_radar)

    st.subheader("Overall Results")
    for field, score in scores.items():
        st.write(f"**{field}:** {score:.1f}% similarity to Ideal Field")

    # Category-wise scores
    category_scores = calculate_category_scores(norm_data, st.session_state.weights, selected_params, sections)
    categories = list(sections.keys())

    st.subheader("Category-wise Similarity Scores")
    fig_bar = plot_category_bar_chart(category_scores, categories)
    st.pyplot(fig_bar)

    # Display table for technical details
    data = {
        'Category': categories,
        'Field A (%)': [category_scores['Field A'].get(cat, 0.0) for cat in categories],
        'Field B (%)': [category_scores['Field B'].get(cat, 0.0) for cat in categories]
    }
    df = pd.DataFrame(data)
    st.table(df)

# Report
if st.button("Generate Decision Report"):
    scores, norm_data = calculate_scores(ideal_field, st.session_state.fields, st.session_state.weights, selected_params)
    category_scores = calculate_category_scores(norm_data, st.session_state.weights, selected_params, sections)
    
    # Generate charts for the report
    norm_data["Ideal Field"] = [1.0] * len(selected_params)
    radar_fig = plot_radar_chart(norm_data, selected_params)
    bar_fig = plot_category_bar_chart(category_scores, list(sections.keys()))

    # Generate PDF
    pdf_buffer = generate_pdf_report(
        current_time, ideal_field, st.session_state.fields, st.session_state.weights,
        selected_params, scores, category_scores, radar_fig, bar_fig, sections
    )
    st.download_button(
        "📄 Download Report (PDF)",
        data=pdf_buffer,
        file_name="Field_Prioritization_Report.pdf",
        mime="application/pdf"
    )
