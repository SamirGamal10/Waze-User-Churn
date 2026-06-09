"""
Waze User Churn Prediction — Streamlit App

Sidebar radio navigation between:
  - Analysis       → EDA with Plotly histograms & box plots
  - ML Prediction  → single-user inference via trained pipeline
"""

import os
import joblib  # pyright: ignore[reportMissingImports]
import pandas as pd
import plotly.express as px # pyright: ignore[reportMissingImports]
import streamlit as st  # pyright: ignore[reportMissingImports]

# ---------------------------------------------------------------------------
# Paths — anchored to this script, independent of CWD
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'src', 'models', 'waze_churn_model.pkl')
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'cleaned', 'waze_dataset_cleaned.csv')

# Feature order must match the trained pipeline exactly
FEATURE_NAMES = [
    'sessions', 'drives', 'total_sessions', 'n_days_after_onboarding',
    'total_navigations_fav1', 'total_navigations_fav2', 'driven_km_drives',
    'duration_minutes_drives', 'activity_days', 'driving_days', 'device',
]
DEVICE_MAP = {0: 'Android', 1: 'iPhone'}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title='Waze Churn', page_icon='🚗', layout='wide')


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Read cleaned CSV; encode label & device to numeric, add churn_status."""
    if not os.path.isfile(DATA_PATH):
        return None
    df = pd.read_csv(DATA_PATH)
    df['label'] = df['label'].map({'retained': 0, 'churned': 1})
    df['device'] = df['device'].map({'Android': 0, 'iPhone': 1})
    df['churn_status'] = df['label'].map({0: 'Retained', 1: 'Churned'})
    return df


@st.cache_resource
def load_model():
    """Load trained pipeline (StandardScaler → SMOTE → Random Forest)."""
    if not os.path.isfile(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


# ---------------------------------------------------------------------------
# Sidebar — page navigation
# ---------------------------------------------------------------------------
st.sidebar.title('🚗 Waze Churn')
page = st.sidebar.radio('Navigate', ['Analysis', 'ML Prediction'])
st.sidebar.markdown('---')


# ===================================================================
# Page 1 — Analysis
# ===================================================================
def page_analysis():
    st.header('📊 Exploratory Data Analysis')
    df = load_data()

    if df is None:
        st.warning(f'Dataset not found at {DATA_PATH}.')
        return

    # KPI summary row
    total = len(df)
    churn_rate = df['label'].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric('Total Users', f'{total:,}')
    c2.metric('Churn Rate', f'{churn_rate:.1%}')
    c3.metric('Retained Rate', f'{1 - churn_rate:.1%}')
    st.markdown('---')

    # Row 1 — churn pie + sessions histogram
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader('Churn Distribution')
        counts = df['churn_status'].value_counts()
        fig = px.pie(
            names=counts.index,
            values=counts.values,
            color=counts.index,
            color_discrete_map={'Retained': '#00CC96', 'Churned': '#EF553B'},
            hole=0.4,
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=340, margin=dict(t=10, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader('Sessions Distribution')
        fig = px.histogram(
            df,
            x='sessions',
            nbins=40,
            color='churn_status',
            color_discrete_map={'Retained': '#00CC96', 'Churned': '#EF553B'},
            barmode='overlay',
            opacity=0.65,
            labels={'sessions': 'Sessions', 'count': 'Users'},
        )
        fig.update_layout(height=340, margin=dict(t=10, b=0, l=0, r=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('---')

    # Row 2 — box plots for key numeric features
    st.subheader('Distribution of Key Features by Churn Status')
    num_cols = ['drives', 'driven_km_drives', 'activity_days', 'driving_days']
    melted = df.melt(
        id_vars=['churn_status'],
        value_vars=num_cols,
        var_name='feature',
        value_name='value',
    )
    fig = px.box(
        melted,
        x='feature',
        y='value',
        color='churn_status',
        color_discrete_map={'Retained': '#00CC96', 'Churned': '#EF553B'},
        labels={'value': 'Value', 'feature': ''},
    )
    fig.update_layout(height=420, margin=dict(t=10, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        'Churned users consistently show **lower** values across all metrics. '
        'Declining engagement is a strong churn signal.'
    )

    # Row 3 — correlation heatmap
    st.subheader('Correlation Heatmap')
    corr = df[FEATURE_NAMES + ['label']].corr()
    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        zmin=-1,
        zmax=1,
        aspect='auto',
    )
    fig.update_layout(height=520, margin=dict(t=10, b=0, l=0, r=0))
    st.plotly_chart(fig, use_container_width=True)

    # Raw data preview
    with st.expander('📋 View raw data sample'):
        st.dataframe(df.head(100), use_container_width=True)


# ===================================================================
# Page 2 — ML Prediction
# ===================================================================
def page_ml_prediction():
    st.header('🔮 ML Prediction')
    pipeline = load_model()

    if pipeline is None:
        st.warning(
            f'Model not found at `{MODEL_PATH}`. '
            'Run the training notebook first.'
        )
    else:
        st.success('Model loaded and ready for inference.')

    st.markdown('Enter the user profile below and click **Predict**.')

    with st.form('predict_form'):
        col_a, col_b = st.columns(2)

        with col_a:
            sessions = st.number_input('Sessions', 0, 800, 80)
            drives = st.number_input('Drives', 0, 600, 67)
            total_sessions = st.number_input('Total Sessions', 0, 1300, 190)
            n_days_after_onboarding = st.number_input(
                'Days Since Onboarding', 0, 3500, 1750,
            )
            total_navigations_fav1 = st.number_input(
                'Navigations to Fav 1', 0, 1300, 120,
            )
            total_navigations_fav2 = st.number_input(
                'Navigations to Fav 2', 0, 450, 30,
            )

        with col_b:
            driven_km_drives = st.number_input('Driven KM', 0.0, 22000.0, 4000.0)
            duration_minutes_drives = st.number_input(
                'Duration (min)', 0.0, 16000.0, 1860.0,
            )
            activity_days = st.number_input('Activity Days', 0, 31, 16)
            driving_days = st.number_input('Driving Days', 0, 31, 12)
            device = st.selectbox(
                'Device',
                options=[0, 1],
                format_func=lambda x: DEVICE_MAP[x],
            )

        submitted = st.form_submit_button(
            '🔮 Predict', type='primary', use_container_width=True,
        )

    if submitted:
        if pipeline is None:
            st.error('Cannot predict — model is unavailable.')
            return

        input_df = pd.DataFrame(
            [[
                sessions, drives, total_sessions, n_days_after_onboarding,
                total_navigations_fav1, total_navigations_fav2,
                driven_km_drives, duration_minutes_drives, activity_days,
                driving_days, device,
            ]],
            columns=FEATURE_NAMES,
        )

        pred = pipeline.predict(input_df)[0]
        proba = pipeline.predict_proba(input_df)[0][1]

        if pred == 1:
            st.error(
                f'### ❌ User is likely to **Churn**\n\n'
                f'Probability: **{proba:.1%}**',
            )
        else:
            st.success(
                f'### ✅ User is likely to **Stay**\n\n'
                f'Probability: **{proba:.1%}**',
            )

        with st.expander('📄 View input data'):
            st.dataframe(input_df, use_container_width=True)


# ===================================================================
# Router
# ===================================================================
if page == 'Analysis':
    page_analysis()
else:
    page_ml_prediction()

st.markdown('---')
st.caption('Built with Streamlit · Model: Random Forest + SMOTE + StandardScaler')
