"""
birte_preprocessing.py

Script for Birte's Agent-Based Data Preprocessing (Work Package 1).
Uses ProgAI chat endpoint to propose transformation plans, interactively
clarify if needed, then applies transformations with pandas and writes:
 - cleaned CSV(s)
 - a JSON config describing the dataset for futureEXPERT check-in.

Environment:
  PROGAI_TOKEN must be set in your environment (ProgAI API token).
"""

from email.mime import base
import os
import sys
import json
import argparse
import textwrap
from typing import List, Optional, Dict, Any
import pandas as pd
import requests
from dateutil.parser import parse as date_parse
import re
import csv

# ----------------------
# Configuration
# ----------------------

PROGAI_BASE = "https://ai.prognostica.de/models/chat/v1/chat/completions"
PROGAI_TOKEN = os.getenv("PROGAI_TOKEN")
if not PROGAI_TOKEN:
    print("ERROR: PROGAI_TOKEN environment variable not found. Set it and re-run.", file=sys.stderr)
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {PROGAI_TOKEN}",
    "Content-Type": "application/json",
}


# ----------------------
# Utilities
# ----------------------

def summarize_df(df: pd.DataFrame, n_sample: int = 20) -> Dict[str, Any]:
    """Return a compact summary of the dataframe for the LLM prompt."""
    samples = df.head(n_sample).fillna("").to_dict(orient="records")
    cols = []
    for col in df.columns:
        col_data = df[col].dropna()
        inferred = "unknown"
        try:
            pd.to_numeric(col_data.astype(str), errors='raise')
            inferred = "numeric"
        except Exception:
            try:
                if len(col_data) > 0:
                    date_parse(str(col_data.iloc[0]))
                    inferred = "date_like"
            except Exception:
                inferred = "string"
        n_unique = df[col].nunique(dropna=True)
        n_null = df[col].isna().sum()
        cols.append({
            "name": col,
            "inferred_type": inferred,
            "n_unique": int(n_unique),
            "n_null": int(n_null),
            "example_values": list(df[col].dropna().astype(str).head(3).values)
        })
    return {"columns": cols, "n_rows": int(len(df)), "samples": samples}


def call_progai_chat(messages: List[Dict[str, str]], timeout: int = 120) -> Dict[str, Any]:
    """Call ProgAI chat endpoint (OpenAI-compatible REST)."""
    payload = {"messages": messages, "temperature": 0.0}
    resp = requests.post(PROGAI_BASE, headers=HEADERS, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def extract_json_from_model(content: str) -> Any:
    """Safely extract JSON from model response."""
    content = content.strip()
    if content.startswith("{") or content.startswith("["):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    match2 = re.search(r"(\{[\s\S]*\})", content)
    if match2:
        try:
            return json.loads(match2.group(1))
        except json.JSONDecodeError:
            pass
    raise ValueError("Could not parse JSON from model response.")


# ----------------------
# Prompt Template
# ----------------------

PLAN_PROMPT_TEMPLATE = textwrap.dedent("""
You are an expert data-preprocessing assistant for time-series forecasting.
You will receive a JSON summary of one or more datasets.
Your task: analyze the schema and propose a transformation plan.

Return your response strictly as JSON, following this schema:

{{
  "status": "ok" | "ask_user",
  "explanation": "Brief description of reasoning and key cleaning steps.",
  "transformations": [
     {{
       "type": "rename_column" | "parse_dates" | "split_column" | "set_index" |
                "resample" | "fill_na" | "drop_columns" | "cast_type" |
                "standardize_headers" | "deduplicate_rows" | "strip_whitespace",
       "details": {{ ... }}
     }}
  ],
  "questions": [
     {{
       "id": "q1",
       "question": "Ask if any ambiguity exists",
       "choices": null | ["option1", "option2"],
       "example_expected_answer": "e.g. The date column is 'Timestamp'"
     }}
  ],
  "futureexpert_config": {{
     "time_column": "<column name or null>",
     "value_column": "<main numeric column or null>",
     "frequency": "<e.g. 'yearly', 'quarterly', 'monthly', 'weekly', 'daily', 'hourly' or 'halfhourly' 
     "covariates": ["colA", "colB"]
  }}
}}

Guidelines:
- Always return syntactically valid JSON.
- Include robust transformations such as trimming spaces, deduplication, parsing dates, splitting multi-value columns, etc.
- If uncertain, use "ask_user" and include specific questions.
- Please be smart, only ask questions if it is really necessary. 
- Avoid plain text output — only JSON.
Input summary:
{summary_json}
""").strip()


# ----------------------
# Transformation Executor
# ----------------------

def remove_prefixes(df):
    """
    Removes prefixes from all string columns.
    Works dynamically based on column names.
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace(f"{col}:", "", regex=False)
    return df


def apply_transformations(df_map: Dict[str, pd.DataFrame], plan: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Apply each transformation in the LLM-generated plan with improved date handling and resampling."""
    dfs = {k: v.copy() for k, v in df_map.items()}
    transformations = plan.get("transformations", [])
    print("Transformation plan:")
    print(json.dumps(plan.get("transformations", []), indent=2))

    for tr in transformations:
        ttype = tr.get("type")
        details = tr.get("details", {})

        try:
            if ttype == "rename_column":
                fname, mapping = details.get("file") or list(dfs.keys())[0], details.get("mapping", {})
                if fname in dfs:
                    dfs[fname].rename(columns=mapping, inplace=True)
                    print(f"[rename_column] {fname}: {mapping}")

            elif ttype == "parse_dates":
                fname = details.get("file")or list(dfs.keys())[0]
                # Support both "column" and "columns" in plan
                col = details.get("column")
                cols = details.get("columns")
                if col is None and cols:
                    col = cols[0]  # Take the first column if multiple given

                if fname in dfs and col in dfs[fname].columns:
                    series = dfs[fname][col].astype(str)
                    # Use the provided format or infer
                    fmt = details.get("format")
                    if fmt:
                        # Convert common format strings like DD.MM.YYYY to Python's %d.%m.%Y
                        fmt = fmt.replace("DD", "%d").replace("MM", "%m").replace("YYYY", "%Y")
                        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
                    else:
                        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)

                    dfs[fname][col] = parsed
                    print(f"[parse_dates] {fname}.{col} parsed to datetime")
            
        

            elif ttype == "split_column":
                fname = details.get("file") or list(dfs.keys())[0]
                col = details.get("column")
                user_delim = details.get("delimiter") or details.get("sep")
                new_cols = details.get("new_columns") or details.get("names", [])
                keep_original = details.get("keep_original", False)

                if fname in dfs and col in dfs[fname].columns:
                    series = dfs[fname][col].astype(str)

                    # Try user-defined delimiter, else detect, else fallback to regex
                    if user_delim:
                        delimiter = user_delim
                    else:
                        try:
                            sample = series.dropna().iloc[0]
                            delimiter = csv.Sniffer().sniff(sample).delimiter
                            print(f"[split_column] Auto-detected delimiter for {col!r}: '{delimiter}'")
                        except Exception:
                            delimiter = r'[;,\|\t:]'  # fallback regex pattern

                    # Split using regex-compatible expression
                    split_df = series.str.split(delimiter, expand=True)

                    # Assign column names
                    if new_cols:
                        split_df.columns = new_cols[:split_df.shape[1]]
                    else:
                        split_df.columns = [f"{col}_{i}" for i in range(split_df.shape[1])]

                    # Join back into main DataFrame
                    if keep_original:
                        dfs[fname] = dfs[fname].join(split_df)
                    else:
                        dfs[fname] = dfs[fname].drop(columns=[col]).join(split_df)

                    dfs[fname] = remove_prefixes(dfs[fname])

                    print(f"[split_column] {fname}.{col} -> {list(split_df.columns)} (delimiter={delimiter})")


            elif ttype == "fill_na":
                fname, cols = details.get("file") or list(dfs.keys())[0], details.get("columns", [])
                strategy = details.get("strategy", "ffill")
                if fname in dfs:
                    for c in cols:
                        if c in dfs[fname].columns:
                            if strategy == "ffill":
                                dfs[fname][c] = dfs[fname][c].fillna(method="ffill")
                            elif strategy == "bfill":
                                dfs[fname][c] = dfs[fname][c].fillna(method="bfill")
                            elif strategy == "zero":
                                dfs[fname][c] = dfs[fname][c].fillna(0)
                            else:
                                dfs[fname][c] = dfs[fname][c].fillna(strategy)
                            print(f"[fill_na] {fname}.{c} using {strategy}")

            elif ttype == "standardize_headers":
                fname = details.get("file") or list(dfs.keys())[0]
                if fname in dfs:
                    dfs[fname].columns = [c.strip().upper().replace(" ", "_") for c in dfs[fname].columns]
                    print(f"[standardize_headers] {fname}")

            elif ttype == "strip_whitespace":
                fname = details.get("file") or list(dfs.keys())[0]
                if fname in dfs:
                    dfs[fname] = dfs[fname].applymap(lambda x: x.strip() if isinstance(x, str) else x)
                    print(f"[strip_whitespace] {fname}")

            elif ttype == "deduplicate_rows":
                fname = details.get("file") or list(dfs.keys())[0]
                if fname in dfs:
                    before = len(dfs[fname])
                    dfs[fname] = dfs[fname].drop_duplicates()
                    print(f"[deduplicate_rows] {fname}: {before - len(dfs[fname])} removed")

            elif ttype == "drop_columns":
                fname, cols = details.get("file") or list(dfs.keys())[0], details.get("columns", [])
                if fname in dfs:
                    dfs[fname].drop(columns=[c for c in cols if c in dfs[fname].columns], inplace=True)
                    print(f"[drop_columns] {fname}: {cols}")

            elif ttype == "cast_type":
                fname = details.get("file") or list(dfs.keys())[0]
                columns = details.get("columns", {})
                for col, dtype in columns.items():
                    if col in dfs[fname].columns:
                        dfs[fname][col] = dfs[fname][col].astype(dtype, errors="ignore")
                        print(f"[cast_type] {fname}.{col} -> {dtype}")


            elif ttype == "set_index":
                fname = details.get("file") or list(dfs.keys())[0]
                col = details.get("column")
                if fname in dfs and col in dfs[fname].columns:
                    # Ensure index is datetime if it's DATE
                    if col.upper() == "DATE":
                        dfs[fname][col] = pd.to_datetime(dfs[fname][col], errors="coerce", dayfirst=True)
                    dfs[fname] = dfs[fname].set_index(col).sort_index()
                    print(f"[set_index] {fname}.{col} set as index")


            elif ttype == "resample":
                fname = details.get("file") or list(dfs.keys())[0]
                rule = details.get("rule") or details.get("frequency", "MS")
                agg = details.get("agg", {})

                if fname in dfs:
                    df = dfs[fname]
                    if not isinstance(df.index, pd.DatetimeIndex):
                        if "DATE" in df.columns:
                            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
                            df = df.set_index("DATE").sort_index()
                    dfs[fname] = df.resample(rule).agg(agg) if agg else df.resample(rule).mean()
                    print(f"[resample] {fname} with rule={rule}")

                    
            else:
                print(f"[skip] Unsupported transformation type: {ttype}")

        except Exception as e:
            print(f"[ERROR] Failed {ttype} on {details}: {e}")

    return dfs


# ----------------------
# Main Orchestration 
# ----------------------

def prepare_data(files: List[str], out_dir: str, answers: Optional[List[Dict[str, str]]] = None):
    """
    
    - If 'answers' are given, uses them directly.
    - If 'ask_user' is returned but no answers are given, returns the questions instead of calling input().
    """
    df_map = {}
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in [".xlsx", ".xls", ".xlsm", ".ods"]:
            sheets = pd.read_excel(f, sheet_name=None, dtype=str)
            for sheet_name, sheet_df in sheets.items():
                key = f"{os.path.basename(f)}::{sheet_name}"
                df_map[key] = sheet_df
                print(f"Loaded {f} [{sheet_name}]: {sheet_df.shape[0]} rows, {sheet_df.shape[1]} cols")
        elif ext == ".csv":
            df = pd.read_csv(f, sep=None, engine="python", dtype=str)
            df_map[f] = df
            print(f"Loaded {f}: {df.shape[0]} rows, {df.shape[1]} cols")
        else:
            print(f"Skipping unsupported file type: {f}")

    summary = {f: summarize_df(df) for f, df in df_map.items()}
    summary_json = json.dumps(summary, indent=2)

    system_msg = {"role": "system", "content": "You are Birte’s AI data-preprocessing assistant."}
    user_msg = {"role": "user", "content": PLAN_PROMPT_TEMPLATE.format(summary_json=summary_json)}

    print("Requesting transformation plan from ProgAI...")
    resp = call_progai_chat([system_msg, user_msg])
    content = resp["choices"][0]["message"].get("content", "")
    plan = extract_json_from_model(content)

    
    if plan.get("status") == "ask_user":
        if answers is None:
            # Instead of using input(), return questions to the caller
            print("Model requires clarification — returning questions instead of blocking.")
            return {"status": "ask_user", "questions": plan.get("questions", [])}
        else:
            # Use provided answers
            followup = {"role": "user", "content": json.dumps({"answers": answers})}
            print("\nSending clarifications to ProgAI...")
            resp2 = call_progai_chat([system_msg, user_msg, followup])
            plan = extract_json_from_model(resp2["choices"][0]["message"]["content"])

    print("\nApplying transformations...")
    updated = apply_transformations(df_map, plan)


    saved_files = []
    os.makedirs(out_dir, exist_ok=True) 
    for fname, df in updated.items():
        base, _ = os.path.splitext(os.path.basename(fname))
        safe_name = base.replace("::", "__").replace(":", "_")
        outp = os.path.join(out_dir, f"cleaned_{safe_name}.csv")
        df.reset_index().to_csv(outp, index=False)
        saved_files.append(outp)
        print(f"Saved cleaned: {outp}")


    PANDAS_FREQ_TO_HUMAN = {
    "MS": "MONTHLY",
    "M": "MONTHLY",
    "QS": "QUARTERLY",
    "Q": "QUARTERLY",
    "YS": "YEARLY",
    "Y": "YEARLY",
    "D": "DAILY",
    "H": "HOURLY",
    "30T": "HALFHOURLY",
    }

    fx_cfg = plan.get("futureexpert_config", {}).copy()
    freq = fx_cfg.get("frequency")
    if freq in PANDAS_FREQ_TO_HUMAN:
        fx_cfg["frequency"] = PANDAS_FREQ_TO_HUMAN[freq]

    cfg = {
    "files": list(updated.keys()),
    "futureexpert_config": fx_cfg,
    "explanation": plan.get("explanation", "")
    }

    cfg_path = os.path.join(out_dir, "futureexpert_checkin_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config saved: {cfg_path}")

    return {"status": "success", "outputs": saved_files, "config": cfg}


# ----------------------
# CLI 
# ----------------------

def main():
    parser = argparse.ArgumentParser(description="Birte preprocessing agent using ProgAI (MCP-compatible)")
    parser.add_argument("-i", "--input", nargs="+", required=True, help="Input CSV(s)")
    parser.add_argument("-o", "--out-dir", default="birte_cleaned", help="Output directory")
    parser.add_argument("--answers", help="Optional JSON string of answers to clarification questions")
    args = parser.parse_args()

    answers = json.loads(args.answers) if args.answers else None
    result = prepare_data(args.input, args.out_dir, answers=answers)

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()