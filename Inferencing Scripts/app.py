import numpy as np
import threading
import re
import MetaTrader5 as mt5
import gradio as gr
import plotly.express as px

from logger import setup_logger
from preprocessing import process
from Predictors import AnomalyModel, dashboard, get_shared_data
from AlertSystem import Email, Telegram, encryptor, decryptor

# Configuration 
CONFIG = {
    "window_size": 300,
    "sample_size": 40,
    "timer": 1,
    "symbol": "BTCUSD",
    "timeframe": mt5.TIMEFRAME_M1,
    "isolation_forest_path": "Isolation_forest.joblib",
}

logger = setup_logger('app')

logger.info("\n" + "=" * 50 + "\nApplication Started\n" + "=" * 50 + "\n")

class AnomalyDetector:
    def __init__(self, shared_data, detection_method='distance-threshold'):
        self.shared_data = shared_data
        self.detection_method = detection_method
        self.window_size = CONFIG['window_size']
        self.sample_size = CONFIG['sample_size']
        self.detector = None
        self.running = False
        self.data_processor = process(
            self.window_size,
            self.sample_size,
            CONFIG['symbol'],
            CONFIG['timeframe'],
            CONFIG['timer']
        )
    def update_data(self, new_outliers):
        try:
            
            self.shared_data["outliers"] = new_outliers
        except Exception as e:
            logger.error(f"Error updating shared data: {str(e)}", exc_info=True)
        
    def start_detection(self):
        self.running = True
        logger.info(f"Starting detection: {self.detection_method}")
        try:
            for ohlc, window, sample in self.data_processor.start_mt5_stream(
            ):
                if not self.running:
                    break
                # Update OHLC data
                self.shared_data['ohlc_df'] = ohlc

                # Initialize model
                if self.detector is None:
                    self.detector = AnomalyModel(
                        isolation_forest_path=CONFIG['isolation_forest_path']
                    )

                # Run detection
                if self.detection_method == 'distance-threshold':
                    result = self.detector.detect_distance_threshold(window, sample, ohlc)
                else:
                    result = self.detector.detect_outliers_isolation_forest(ohlc)
                print("Detection result:", result)
                # Extract outliers
                outliers = self.detector.get_outliers(self.detection_method)
                if outliers:
                    self.update_data(outliers)
                    send_alert(result, self.shared_data)
        except Exception as e:
            logger.error(f"Detection error: {e}", exc_info=True)
        finally:
            mt5.shutdown()
            self.running = False
            logger.info("MT5 shutdown complete")

    def stop_detection(self):
        self.running = False
        logger.info("Stopping detection")
        # No need for SIGINT kill here; thread will exit

def validate_email(email_str):
    pattern = r"(^[\w\.-]+@[\w\.-]+\.\w{2,}$)"
    return bool(re.match(pattern, email_str))



def send_alert(result, shared_data):
    if result == 'anomaly_detected':
        ts = shared_data['outliers'][-1][0]
        msg = f"Anomaly detected at {ts} UTC. Check your dashboard."
        method = shared_data.get('alert_method')
        if method == 'Telegram':
            tg.send_telegram(msg)
        elif method == 'Email':
            email_client.send_email(decryptor(shared_data['Email']), msg)


def start_and_update(method, shared_data, detector, dashboard_runner):
    if detector is None or not detector.running:
        detector = AnomalyDetector(shared_data, detection_method=method)
        threading.Thread(target=detector.start_detection, daemon=True).start()
        dashboard_runner = dashboard(detector, shared_data)
        threading.Thread(target=dashboard_runner.update_graph, daemon=True).start()
    status = f"Detection started with: {method}"
    return shared_data, detector, dashboard_runner, status


def stop_and_update(detector):
    if detector and detector.running:
        detector.stop_detection()
        return None, "Detection stopped."
    return detector, "Detection not running."


def update_dashboard_plot(detector, dashboard_runner):
    if detector is None or not detector.running:
        return px.line(title="Detection Not Started")
    fig, _, _ = dashboard_runner.update_graph()
    return fig

def on_select_telegram(alert_method):
    if alert_method == "Telegram":
        try:
            internal_id = np.random.randint(100000, 999999)
            
        except KeyError:
            return gr.update(value="❌ Invalid phone number.", visible=True)
        
        return gr.update(value=tg.get_link(internal_id), visible=True)
    else:
        return gr.update(visible=False)
    
if __name__ == "__main__":
    tg = Telegram()
    email_client = Email()
    tg.start_bot()



    custom_css = """

        body, .gradio-container {
            background-color: #000000 !important;
            color: #ffffff !important;
        }


        select, textarea {
            padding: 8px 10px;
            background-color: rgb(35, 34, 39) !important;
            border: 2px solid rgb(255, 255, 255) !important;
            color: white !important;
        }


        input.svelte-1hfxrpf.svelte-1hfxrpf {
            margin: var(--spacing-sm);                 /* Uses a small spacing variable for margin */
            padding: 8px 10px;
            border: 2px solid rgb(255, 255, 255) !important;
            background-color:#666666 !important; /* Transparent background */
            color: white !important;
            font-size: 12px       
            height: 100%;                              
        }

        /* Dark theme style override for toggle/checkbox/radio button labels */
        label.svelte-k79vs1.svelte-k79vs1.svelte-k79vs1 {
            background-color: #232227 !important;
            color: #ffffff !important;
            font-size: 14px !important;
            font-weight: 500 !important;
            cursor: pointer;
        }

        /* Override Gradio block title styling for dark theme */
        span.svelte-1gfkn6j {
            background-color: #232227 !important;
            color: #ffffff !important; 
            font-weight: bold !important;
            font-size: 16px !important;
        }


        /* Override white wrapper around input elements */
        .wrap-inner.svelte-1hfxrpf.svelte-1hfxrpf {
            background-color: #232227 !important;
        }

        h1, h2, h3 {
            color: #FFFFFF !important;
        }

        .primary-heading {
            text-align: center;
            color: gold !important;
            font-family: 'Arial Black', Gadget, sans-serif;
            font-size: 40px;
            font-weight: 500;
            margin-bottom: 5px;
        }
            .centred-heading {
                text-align: center;
            }


        #start-btn {
            display: inline-block !important;
            width: auto !important;             
            max-width: fit-content !important;  
            min-width: 0 !important; 
            align-self: center !important;
            font-size: 16px;
            background-color: #28a745;
            color: white;
            margin:0 auto !important;
            border: none;
            border-radius: 8px;
            transition: all 0.3s ease-in-out;
            cursor: pointer;
        }

        #start-btn:hover:enabled {
            background-color: #218838;
            transform: scale(1.05);
            
        }

        #start-btn:disabled {
            background-color: #ccc;
            color: #666;
            cursor: not-allowed;

        }

        @media (max-width: 600px) {
            #start-btn {
                width: 100%;
                font-size: 18px;
            }
        }


    """


    class PureDark(gr.themes.Base):
        def __init__(self):
            super().__init__()
            self.set(block_background_fill="#232227")

    # Build Gradio app with per-session state
    with gr.Blocks(css=custom_css, title="Anomaly_detection", theme=PureDark()) as app:
        shared_data_state       = gr.State(get_shared_data())
        detector_state          = gr.State(None)
        dashboard_state         = gr.State(None)

        gr.HTML("<h1 class='primary-heading'>Real-Time Anomalies Detection</h1>")  

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    method_dropdown = gr.Dropdown(
                        choices=["distance-threshold", "isolation_forest"],
                        label="Detection Method"
                    )
                with gr.Row():
                    alert_toggle = gr.Radio(
                    choices=["No", "Yes"], label="Receive Alerts?"
                    )
                    with gr.Column():
                        alert_method = gr.Dropdown(
                            choices=["Email", "Telegram"],
                            label="Alert Method", visible=False
                        )
                with gr.Row():
                    with gr.Column():

                        email_input = gr.Textbox(
                            placeholder="Enter email", label="Email", visible=False
                        )
                    with gr.Column():  
                       telegram = gr.Markdown(visible=False)

                start_btn = gr.Button("Start Detection", elem_id="start-btn", interactive=False)
                
            with gr.Column(scale=2):
                with gr.Row():
                    status = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    gr.Markdown("## Real-Time Anomaly Detection Plot", elem_classes= ["centred-heading"])
                with gr.Row():   
                    plot_out = gr.Plot()
                with gr.Row():
                    stop_btn = gr.Button("Stop Detection",elem_id=["start-btn"])
                

        # Toggle visibility of alert inputs
        def toggle_alert(receive, method):
            show_method = receive == "Yes"
            show_email = show_method and method == "Email"
            show_phone = show_method and method == "Telegram"
            return (
                gr.update(visible=show_method),  # alert_method
                gr.update(visible=show_email),   # email_input
                gr.update(visible=show_phone)    
            )

        alert_toggle.change(
            toggle_alert,
            inputs=[alert_toggle, alert_method],
            outputs=[alert_method, email_input, telegram]
        )
        alert_method.change(
            toggle_alert,
            inputs=[alert_toggle, alert_method],
            outputs=[alert_method, email_input, telegram]
        )

        alert_method.change(
                on_select_telegram,
                inputs=[alert_method],
                outputs=[telegram]
            )


        def check_inputs(method, receive, alert_m, email_addr, shared_data):
            """
            Validate inputs AND write the user’s alert prefs into shared_data.
            Returns (shared_data, button_update).
            """
            if not method:
                return shared_data, gr.update(interactive=False)

            shared_data['detection_method'] = method

            if receive == "Yes":
                shared_data['receive_alerts'] = True
                shared_data['alert_method'] = alert_m

                if alert_m == "Email":
                    # only enable once email is valid
                    if validate_email(email_addr):
                        shared_data['Email'] = encryptor(email_addr)
                        return shared_data, gr.update(interactive=True)
                    else:
                        return shared_data, gr.update(interactive=False)

                elif alert_m == "Telegram":
                        shared_data["alert_method"] = "Telegram"

                        return shared_data, gr.update(interactive=True)

            else:
                # No alerts: clear any old prefs
                shared_data['receive_alerts'] = False
                shared_data.pop('alert_method', None)
                shared_data.pop('Email', None)
                return shared_data, gr.update(interactive=True)

        for comp in [method_dropdown, alert_toggle, alert_method, email_input]:

            comp.change(
                check_inputs,
                inputs=[method_dropdown, alert_toggle, alert_method, email_input,
                    shared_data_state
                ],
                outputs=[
                    shared_data_state,  
                    start_btn          
                ]
            )

        # Main callbacks
        start_btn.click(
            start_and_update,
            inputs=[
                method_dropdown,
                shared_data_state,
                detector_state,
                dashboard_state,

            ],
            outputs=[
                shared_data_state,
                detector_state,
                dashboard_state,
                status
            ],
            concurrency_limit=5
        )

        stop_btn.click(
            stop_and_update,
            inputs=[detector_state],
            outputs=[detector_state, status],
            concurrency_limit=10
        )

        timer = gr.Timer()
        timer.tick(
            fn=update_dashboard_plot,
            inputs=[detector_state, dashboard_state],
            outputs=[plot_out],
            concurrency_limit=None
        )

    app.queue(default_concurrency_limit=None)
    app.launch()
