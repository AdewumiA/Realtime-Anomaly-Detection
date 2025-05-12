import os
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib,pickle
import uvicorn
from datetime import datetime
import logging
import dash
import pandas as pd
from typing import Dict,List,Tuple
from multiprocessing import Manager


import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output,State

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf


# Get logger for this module
logger = logging.getLogger(__name__)

def get_shared_data():
    manager = Manager()
    shared_data = manager.dict()
    shared_data["ohlc_df"] = pd.DataFrame()
    shared_data["outliers"] = []
    return shared_data

class AnomalyModel():

    """
    Args:
        isolation_forest_path: Path to trained Isolation Forest model
        autoencoder_path: Path to trained Autoencoder model
        scaler_path: Path to fitted scaler
    """

    def __init__(self, isolation_forest_path, scaler_path):
        self.feature_cols = ['close','ATR', 'BB_Width', 'RSI', 'Returns','Hour', 'DayOfWeek','Month','IsWeekend','Trend','Residual']
        self.iso_features = ["close","ATR","BB_Width","RSI","Returns","Hour","DayOfWeek"]
        self.threshold_percentile = 99
        self.isolation_forest_path = isolation_forest_path
        self.scaler_path = scaler_path
        #self.autoencoder_path = autoencoder_path

        self.outliers: Dict[str, List[Tuple[pd.Timestamp, float]]] = {
            'isolation_forest': [],
            'autoencoder': [],
            'distance-threshold': []
        }
        

        try:
            self.isolation_forest = joblib.load(self.isolation_forest_path)
            #self.autoencoder = tf.keras.models.load_model(self.autoencoder_path)
            with open(self.scaler_path, "rb") as rbs:
                self.scaler = pickle.load(rbs)


        except FileNotFoundError as e:
            logger.error(f"FileNotFoundError: Could not load model or scaler. Details: {e}")
            raise
        except Exception as e:
            logger.error(f"Anomaly_model as an error: {e}", exc_info=True)
            raise

    def detect_distance_threshold(self, window, sample,ohlc_df): 
        try:      
            if (~np.isnan(window)).all():
                    min_distances = np.abs(np.array(window)[:, None] - sample).min(axis=1)          
                    threshold = np.percentile(min_distances, self.threshold_percentile)
                    if min_distances[-1] > threshold:
                        latest_value= window[-1]
                        time = ohlc_df.iloc[-1]['Timestamp']
                        self._record_outlier(time, latest_value,'distance-threshold')
                        return f"Outlier detected: {latest_value}"
                    else:
                        return "No anomaly"
        except Exception as e:
            logger.error(f"Distance_threshold as an error {e}", )
            raise

    def detect_outliers_isolation_forest(self,ohlc_df):
        
        try:
            self.scaled_data = self.scaler.transform(ohlc_df[self.iso_features])
            self.scaled_df = pd.DataFrame(self.scaled_data, columns=self.iso_features)

            pred =  self.isolation_forest.predict(self.scaled_df[self.iso_features][-1:]) 

            if pred == -1:
                latest_value= ohlc_df.iloc[-1]['close']
                time = ohlc_df.iloc[-1]['Timestamp']
                self._record_outlier(time, latest_value,'isolation_forest')
                return {'status': 'anomaly_detected', 'method': 'isolation_forest'}
            else:
                return "No anomaly"
        except Exception as e:
            logger.error(f"Isolation Forest predictor error: {e}", exc_info=True)
            raise

    """
    def _detect_autoencoder(self, threshold=0.05) -> Dict:
        pred = self.models['autoencoder'].predict(self.scaled_df[self.iso_features][-1:])
        mse = np.mean(np.square(self.scaled_df[self.iso_features][-1:] - pred))
        if mse > threshold:
            self._record_outlier('autoencoder')
            return {'status': 'anomaly', 'method': 'autoencoder'}
        return {'status': 'normal'}
    """

    def _record_outlier(self, time,latest_value,method: str) -> None:
        self.outliers[method].append((
            time,
            float(latest_value)
        ))

    def get_outliers(self, method: str = None) -> Dict:
        """
        Outliers key can be one of the following:

        1. isolation_forest
        2. autoencoder
        3. distance-threshold
        """
        return self.outliers[method] if method else self.outliers

class dashboard():
    def __init__(self, detector, shared_data):
        self.app_dash = dash.Dash(__name__)
        self.detector = detector
        self.shared_data = shared_data
        self.full_history = pd.DataFrame(columns=['Timestamp', 'close'])
        self.last_update = None
           
        self.last_xaxis_range = None   # Store the last known x-axis range
        # Initialize layout
        self.app_dash.layout = html.Div([
            html.H1(f"Live Anomalies Detection Using - {self.detector.detection_method} Method",
                   style={'textAlign': 'center'}),
            dcc.Graph(
                id="live-graph",
                config={'displayModeBar': True},
                style={'height': '80vh'}
            ),
            dcc.Interval(
                id="interval-component",
                interval=1000,  # 1 second updates
                n_intervals=0
            ),
            html.Div(
                id="outlier-info",
                style={
                    'margin-top': '20px',
                    'font-size': '1.2em',
                    'font-weight': 'bold',
                    'textAlign': 'center'
                }
            ),
            html.Div(
                id="data-stats",
                style={
                    'margin-top': '10px',
                    'color': '#666',
                    'textAlign': 'center'
                }
            )
        ])
     
        self.app_dash.callback(
            [Output("live-graph", "figure"),
            Output("outlier-info", "children"),
            Output("data-stats", "children")],
            [Input("interval-component", "n_intervals")],
            [State("live-graph", "relayoutData")]
        )(self.update_graph)
    
    def update_graph(self, n_intervals, relayout_data):
        try:
            ohlc_df = self.shared_data.get("ohlc_df", pd.DataFrame())

            all_outliers = self.shared_data.get("outliers")
            
            print(ohlc_df.tail(3))

            if ohlc_df.empty:
                return go.Figure(), "Waiting for initial data...", ""

            ohlc_df["Timestamp"] = pd.to_datetime(ohlc_df["Timestamp"])
            self.full_history = pd.concat([self.full_history, ohlc_df[['Timestamp', 'close']]]).drop_duplicates('Timestamp',keep="first")

            # Preserve user-selected range
            if relayout_data:
                self.last_xaxis_range = relayout_data.get("xaxis.range") or [
                    relayout_data.get("xaxis.range[0]"),
                    relayout_data.get("xaxis.range[1]")
                ]

            fig = go.Figure()
            fig.add_trace(go.Scattergl(
                x=self.full_history['Timestamp'],
                y=self.full_history['close'],
                mode='lines',
                name='Close Price',
                line={'color': '#1f77b4', 'width': 1.5},
                hovertemplate='%{x|}<br>%{y:.2f}<extra></extra>'
            ))

            # Apply stored x-axis range
            if self.last_xaxis_range:
                fig.update_xaxes(range=self.last_xaxis_range)

            # Remove duplicates efficiently while maintaining order
            unique_outliers = list({ts: price for ts, price in all_outliers}.items()) if all_outliers else []


            # Ensure anomalies align with close price timestamps
            close_timestamps = set(self.full_history["Timestamp"])  # Convert close price timestamps to a set
            filtered_outliers = [(ts, price) for ts, price in unique_outliers if ts in close_timestamps]

            # Plot anomalies
            if filtered_outliers:
                fig.add_trace(go.Scattergl(
                    x=[ts for ts, _ in filtered_outliers],
                    y=[price for _, price in filtered_outliers],
                    mode='markers',
                    name='Anomalies',
                    marker={'color': 'Red', 'size': 8, 'line': {'width': 1, 'color': 'DarkSlateGrey'}},
                    hovertemplate='Anomaly at %{x|%H:%M:%S}<br>Price: %{y:.4f}<extra></extra>'
                ))


            # Apply layout configuration with range selector
            fig.update_layout(
                title_text=f"{self.detector.detection_method} Anomalies",
                title_x=0.5,
                xaxis={
                    'title': 'Time',
                    'rangeselector': {
                        'buttons': [
                            dict(count=1, label="1m", step="minute", stepmode="backward"),
                            dict(count=5, label="5m", step="minute", stepmode="backward"),
                            dict(count=15, label="15m", step="minute", stepmode="backward"),
                            dict(count=1, label="1h", step="hour", stepmode="backward"),
                            dict(step="all", label="All")
                        ],
                        'bgcolor': 'rgba(150,200,250,0.4)',
                        'activecolor': 'Green'
                    },
                    'rangeslider': {
                        'visible': True,
                        'thickness': 0.1,
                        'bgcolor': 'rgba(150,200,250,0.2)'
                    },
                    'type': 'date',
                },
                yaxis_title="Price",
                yaxis_tickformat=".4f",
                hovermode="x unified",
                template="plotly_white",
                margin=dict(l=50, r=30, t=60, b=40),
                legend=dict(orientation="h", y=1.02, x=1)
            )

            return fig, "", ""

        except Exception as e:
            logger.error(f"Graph update error: {str(e)}")
            return go.Figure(), f"Error: {str(e)}", ""
  

    def run(self):
        try:
            logger.info("\n[INFO] Starting dashboard server...")
            if "ohlc_df" not in self.shared_data:
                self.shared_data["ohlc_df"] = pd.DataFrame()

            if "outliers" not in self.shared_data:
                self.shared_data["outliers"] = []
            
            self.app_dash.run(
                debug=True,
                port=8500,
                use_reloader=False
            )
        except Exception as e:
            logger.error(f"Dashboard server failed: {str(e)}", exc_info=True)
            raise