import json
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

load_dotenv()


from futureexpert import (
    DataDefinition,
    ExpertClient,
    FileSpecification,
    ForecastingConfig,
    PreprocessingConfig,
    ReportConfig,
    TsCreationConfig,
    MethodSelectionConfig,
)
import futureexpert.checkin as checkin
from futureexpert.plot import plot_forecast




def run_futureexpert_forecast_mcp(
    cleaned_csv_path: str,
    config_json_path: str,
    output_csv_path: str = "birte_cleaned/futureexpert_forecast.csv",
    output_plot_path: str = "birte_cleaned/futureexpert_plot.png"
):
    """
    Thin wrapper for MCP tools.
    It calls run_futureexpert_forecast() but returns only structured output
    and does not print anything to stdout.
    """
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    result = run_futureexpert_forecast(
        cleaned_csv_path=cleaned_csv_path,
        config_json_path=config_json_path,
        output_csv_path=output_csv_path,
        output_plot_path=output_plot_path,
    )
    return result




# Map short frequency codes → full FutureExpert literals
FREQ_MAP = {
    "Y": "yearly",
    "Q": "quarterly",
    "M": "monthly",
    "W": "weekly",
    "D": "daily",
    "H": "hourly",
    "30min": "halfhourly",
}


def run_futureexpert_forecast(
    cleaned_csv_path: str,
    config_json_path: str,
    output_csv_path="birte_cleaned/futureexpert_forecast.csv",
    output_plot_path="birte_cleaned/futureexpert_plot.png",
):
    print("\n=== Running FutureExpert Forecast ===\n")

    # ----------------------------------------------------------
    # Load cleaned data
    # ----------------------------------------------------------
    print(f"Loading cleaned CSV: {cleaned_csv_path}")
    df = pd.read_csv(cleaned_csv_path)

    

    # ----------------------------------------------------------
    # Load user config
    # ----------------------------------------------------------
    print(f"Loading FutureExpert config: {config_json_path}")
    with open(config_json_path, "r") as f:
        config = json.load(f)

    fx_cfg = config["futureexpert_config"]

    time_col = fx_cfg["time_column"]
    value_col = fx_cfg["value_column"]
    raw_freq = fx_cfg["frequency"]
    covariates = fx_cfg.get("covariate_columns", [])

    df= df[[time_col, value_col]].dropna()

    df[df.columns[0]] = df[df.columns[0]].astype(str)

    # Convert "M" → "monthly"
    frequency = FREQ_MAP.get(raw_freq, raw_freq)

    print(f"Using time granularity: {frequency}")

    # ----------------------------------------------------------
    # Initialize FutureExpert Client
    # (username+password should come from .env in production)
    # ----------------------------------------------------------
    print("Preparing FutureExpert configuration...")
    

    user = os.getenv("FUTUREEXPERT_USER")
    password = os.getenv("FUTUREEXPERT_PASSWORD")


    future_client = ExpertClient(
        user=user,
        password=password,
    )



    # ----------------------------------------------------------
    # Data Definition
    # ----------------------------------------------------------
    data_definition = DataDefinition(
    date_column=checkin.DateColumn(name=time_col, format="%Y-%m-%d"),
    value_columns=[checkin.ValueColumn(name=value_col)],
    #group_columns=[checkin.GroupColumn(name=c) for c in covariates],
)


    # ----------------------------------------------------------
    # Time Series Creation
    # ----------------------------------------------------------
    ts_creation_config = TsCreationConfig(
        time_granularity=frequency,
        value_columns_to_save=[value_col],
        #grouping_level=covariates,
        missing_value_handler="setToZero",
    )

    # ----------------------------------------------------------
    # Report Config
    # ----------------------------------------------------------
    fc_report_config = ReportConfig(
        title=f"Forecast_{datetime.datetime.now()}".replace(" ", "_"),
        forecasting=ForecastingConfig(fc_horizon=14),
        preprocessing=PreprocessingConfig(),
        method_selection=MethodSelectionConfig(number_iterations=1),
    )

    file_spec = FileSpecification(delimiter=",", decimal=".")

    # ----------------------------------------------------------
    # Start forecast job
    # ----------------------------------------------------------
    print("Submitting data to FutureExpert...")

    forecast_id = future_client.start_forecast_from_raw_data(
        raw_data_source=df,
        data_definition=data_definition,
        config_ts_creation=ts_creation_config,
        config_fc=fc_report_config,
        file_specification=file_spec,
    )

    print(f"Forecast job ID: {forecast_id}")

    # ----------------------------------------------------------
    # Polling
    # ----------------------------------------------------------
    print("\nWaiting for FutureExpert to finish...\n")

    wait_seconds = 20
    while True:
        status = future_client.get_report_status(id=forecast_id)

        if status.is_finished:
            print("\nForecast finished!\n")
            break

        print(f"[{datetime.datetime.now()}] Not finished yet → waiting {wait_seconds}s")
        time.sleep(wait_seconds)

    # ----------------------------------------------------------
    # Retrieve results
    # ----------------------------------------------------------
    print("Fetching forecast results...")
    results = future_client.get_fc_results(id=forecast_id)

    # results = client.get_fc_results(...)

    # Extract forecast table
    df_result = results.export_forecasts_to_pandas()

    print(df_result.head())


    

    df_result.to_csv(output_csv_path, index=False)

    print(f"Forecast saved to: {output_csv_path}")

    # ----------------------------------------------------------
    # Plot results
    # ----------------------------------------------------------
    print("Saving forecast plot...")
    plot_forecast(result=results[0])     # this draws the plot but returns None
    fig = plt.gcf()                      # get the current active figure
    fig.savefig(output_plot_path, dpi=300)
    plt.close(fig)
    

    print(f"Plot saved to: {output_plot_path}")
    print("\n=== FutureExpert Forecast Completed Successfully ===\n")

    return {
        "forecast_csv": output_csv_path,
        "plot_path": output_plot_path,
        "forecast_id": forecast_id,
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run FutureExpert Forecast")
    parser.add_argument("--cleaned_csv", required=True, help="Path to cleaned CSV file")
    parser.add_argument("--config", required=True, help="Path to FutureExpert config JSON")
    parser.add_argument("--out_csv", default="birte_cleaned/futureexpert_forecast.csv")
    parser.add_argument("--out_plot", default="birte_cleaned/futureexpert_plot.png")

    args = parser.parse_args()

    run_futureexpert_forecast(
        cleaned_csv_path=args.cleaned_csv,
        config_json_path=args.config,
        output_csv_path=args.out_csv,
        output_plot_path=args.out_plot,
    )
