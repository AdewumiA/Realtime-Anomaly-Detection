import os
import numpy as np
import joblib
import logging
import pandas as pd
from typing import Dict,List,Tuple
from multiprocessing import Manager

import dash
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# Get logger for this module
logger = logging.getLogger(__name__)

def get_shared_data():
    manager = Manager()
    shared_data = manager.dict()
    shared_data["ohlc_df"] = pd.DataFrame()
    shared_data["outliers"] = []
    return shared_data


def rolling_zscore_df(df, window=30):

    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns  # Select only numeric columns
    
    if window is None:  # If window is not specified, use the entire dataset
        mean = df[numeric_cols].mean()
        std = df[numeric_cols].std()
    else:
        mean = df[numeric_cols].rolling(window=window, min_periods=1).mean()
        std = df[numeric_cols].rolling(window=window, min_periods=1).std()

    return df.assign(**{col: (df[col] - mean[col]) / (std[col] + 1e-8) for col in numeric_cols})

class AnomalyModel():

    """
    Args:
        isolation_forest_path: Path to trained Isolation Forest model
    """

    def __init__(self, isolation_forest_path):
        self.feature_cols = ['close','ATR', 'BB_Width', 'RSI', 'Returns','Hour', 'DayOfWeek','Month','IsWeekend','Trend','Residual']
        self.iso_features = ["close","ATR","BB_Width","RSI","Returns","Hour","DayOfWeek"]
        self.threshold_percentile = 95
        self.isolation_forest_path = isolation_forest_path

        self.outliers: Dict[str, List[Tuple[pd.Timestamp, float]]] = {
            'isolation_forest': [],
            'distance-threshold': []
        }
        

        try:
            self.isolation_forest = joblib.load(self.isolation_forest_path)
            

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
                        return "anomaly_detected"
                    else:
                        return "No anomaly"
        except Exception as e:
            logger.error(f"Distance_threshold as an error {e}", )
            raise

    def detect_outliers_isolation_forest(self,ohlc_df):
        
        try:
            self.scaled_df = rolling_zscore_df(ohlc_df[self.iso_features])

            pred =  self.isolation_forest.predict(self.scaled_df.iloc[[-1]]) 

            if pred == -1:
                latest_value= ohlc_df.iloc[-1]['close']
                time = ohlc_df.iloc[-1]['Timestamp']
                self._record_outlier(time, latest_value,'isolation_forest')
                return ('anomaly_detected')
            else:
                return "No anomaly"
        except Exception as e:
            logger.error(f"Isolation Forest predictor error: {e}", exc_info=True)
            raise


    def _record_outlier(self, time,latest_value,method: str) -> None:
        self.outliers[method].append((
            time,
            float(latest_value)
        ))

    def get_outliers(self, method: str = None) -> Dict:
        """
        Outliers key can be one of the following:

        1. distance-threshold
        2. isolation_forest
        """
        return self.outliers[method] if method else self.outliers


class dashboard():
    def __init__(self, detector, shared_data):
        self.app_dash = dash.Dash(__name__)
        self.detector = detector
        self.shared_data = shared_data
        self.full_history = pd.DataFrame(columns=['Timestamp', 'close'])
        self.max_points = 300  
        self.last_update = None
        
        # Initialize layout
        self.app_dash.layout = html.Div([
            html.H1(f"Live Trading - {self.detector.detection_method} Anomalies",
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
        
        # Register callback
        self.app_dash.callback(
            [Output("live-graph", "figure"),
             Output("outlier-info", "children"),
             Output("data-stats", "children")],
            [Input("interval-component", "n_intervals")]
        )(self.update_graph)
        
    def update_graph(self):
        try:
            # Get current data with thread safety
            ohlc_df = self.shared_data.get("ohlc_df", pd.DataFrame())
            all_outliers = self.shared_data.get("outliers", [])

            if ohlc_df.empty:
                return go.Figure(), "Waiting for initial data...", ""

            # Ensure Timestamp is properly formatted
            ohlc_df['Timestamp'] = pd.to_datetime(ohlc_df['Timestamp'])

            # Update full history while maintaining rolling window
            current_time = pd.Timestamp.now()
            if self.last_update is None or (current_time - self.last_update).seconds >= 1:
                self.last_update = current_time
                self.full_history = pd.concat([self.full_history, ohlc_df[['Timestamp', 'close']]]).drop_duplicates('Timestamp',keep="last")

                if len(self.full_history) > self.max_points:
                    self.full_history = self.full_history.iloc[-self.max_points:]


            # Create figure
            fig = go.Figure()

            # Plot FULL price history
            fig.add_trace(go.Scattergl(
                x=self.full_history['Timestamp'],
                y=self.full_history['close'],
                mode='lines',
                name='Close Price',
                line={'color': '#1f77b4', 'width': 1.5},
                hovertemplate='%{x|%H:%M:%S}<br>%{y:.4f}<extra></extra>'
            ))

            # Remove duplicates efficiently while maintaining order
            unique_outliers = list({ts: price for ts, price in all_outliers}.items())

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
            logger.error(f"Graph update error: {str(e)}", exc_info=True)
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
