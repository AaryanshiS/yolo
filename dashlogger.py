import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ----------------------------
# CONFIG
# ----------------------------
CSV_FILE = "live130925.csv"  # Your KPI CSV file path

# ----------------------------
# Dash setup
# ----------------------------
dashboard = dash.Dash(__name__)
server = dashboard.server  # expose server for deployment

# ----------------------------
# Layout
# ----------------------------
dashboard.layout = html.Div([
    html.H1("📊 KPI & Anomaly Dashboard"),
    
    html.H2("📊 KPI Graphs"),
    html.Div(id='kpi-summary'),
    dcc.Graph(id='throughput-graph'),
    dcc.Graph(id='latency-graph'),
    dcc.Graph(id='pdv-graph'),

    html.H3("🔎 Select Anomaly"),
    dcc.Dropdown(id='anomaly-selector', placeholder="Select an anomaly"),

    dcc.Interval(id='interval-component', interval=5000, n_intervals=0)  # refresh every 5s
])

# ----------------------------
# Callback
# ----------------------------
@dashboard.callback(
    [
        Output("kpi-summary", "children"),
        Output("throughput-graph", "figure"),
        Output("latency-graph", "figure"),
        Output("pdv-graph", "figure"),
        Output("anomaly-selector", "options"),
        Output("anomaly-selector", "value")
    ],
    [Input("interval-component", "n_intervals"),
     Input("anomaly-selector", "value")]
)
def update_dashboard(n, selected_anomaly):
    empty_fig = go.Figure()

    if not os.path.exists(CSV_FILE):
        return html.Div("⚠️ Waiting for data..."), empty_fig, empty_fig, empty_fig, [], None

    df = pd.read_csv(CSV_FILE)
    if df.empty or "Time" not in df or "Length" not in df:
        return html.Div("⚠️ CSV missing required columns."), empty_fig, empty_fig, empty_fig, [], None

    # --- KPI Calculations ---
    df["Time"] = df["Time"] - df["Time"].min()
    df["RTT"] = df["Time"].diff().fillna(0) * 1000
    df["Jitter"] = df["RTT"].diff().fillna(0).abs()
    df["Throughput"] = df["Length"] * 8 / 1e3
    df["PDV"] = df["RTT"].rolling(5).std().fillna(0)
    df["Time_Window"] = df["Time"].astype(int)

    agg = df.groupby("Time_Window").agg({
        "Length": "sum",
        "RTT": "mean",
        "Jitter": "mean",
        "Throughput": "mean",
        "PDV": "mean"
    })

    # --- Anomaly Detection ---
    features = ["RTT", "Jitter", "Throughput"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(agg[features].fillna(0))
    model = IsolationForest(contamination=0.05, random_state=42)
    agg["Anomaly"] = model.fit_predict(X_scaled)
    anomalies = agg[agg["Anomaly"] == -1]

    # --- KPI Summary ---
    summary = html.Table([
        html.Tr([html.Th("Metric"), html.Th("Value")]),
        html.Tr([html.Td("Latency (Avg)"), html.Td(f"{agg['RTT'].mean():.2f} ms")]),
        html.Tr([html.Td("Jitter (Avg)"), html.Td(f"{agg['Jitter'].mean():.2f} ms")]),
        html.Tr([html.Td("Throughput (Avg)"), html.Td(f"{agg['Throughput'].mean():.2f} Kbps")]),
        html.Tr([html.Td("PDV (Avg)"), html.Td(f"{agg['PDV'].mean():.2f} ms")]),
        html.Tr([html.Td("Anomalies Detected"), html.Td(f"{(agg['Anomaly'] == -1).sum()}")])
    ])

    # --- Graphs ---
    throughput_fig = px.line(agg, x=agg.index, y="Throughput", title="Throughput Over Time")
    throughput_fig.add_scatter(x=anomalies.index, y=anomalies["Throughput"], mode="markers",
                               marker=dict(color="red", size=10), name="Anomaly")

    latency_fig = px.line(agg, x=agg.index, y="RTT", title="Latency Over Time")
    latency_fig.add_scatter(x=anomalies.index, y=anomalies["RTT"], mode="markers",
                            marker=dict(color="red", size=10), name="Anomaly")

    pdv_fig = px.line(agg, x=agg.index, y="PDV", title="Packet Delay Variation (PDV)")

    # --- Anomaly Dropdown ---
    anomaly_indices = list(anomalies.index)
    dropdown_options = [{'label': f'Anomaly at Time {idx}', 'value': idx} for idx in anomaly_indices]
    if not selected_anomaly and anomaly_indices:
        selected_anomaly = anomaly_indices[0]

    return summary, throughput_fig, latency_fig, pdv_fig, dropdown_options, selected_anomaly

# ----------------------------
# Run Dash App
# ----------------------------
if __name__ == "__main__":
    dashboard.run(debug=True, port=8050)
