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

def summarize_df(df: pd.DataFrame, n_sample: int = 30) -> Dict[str, Any]:
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
     "frequency": "<e.g., D, W, M or null>",
     "covariates": ["colA", "colB"]
  }}
}}

Guidelines:
- Propose transformations that are likely to improve forecasting performance.
-specicy details for each transformation (e.g. column names, date formats, resampling rules).
-Please don't miss important transformations, but also avoid unnecessary ones. Be concise.
-Please don't miss column splitting if you see multi-value columns!.
- Always return syntactically valid JSON.
- Include robust transformations such as trimming spaces, deduplication, parsing dates, splitting multi-value columns, etc.
- If uncertain, use "ask_user" and include specific questions.
- Please be smart, only ask questions if it is really necessary. 
- Avoid plain text output — only JSON.
-Suggest "fill_na" instead of resampling and always provide filling strategies.
-Please be very smart with the filling strategy, don't suggest constant if ffill or bfill would work well, Use ffill for grouped empty rows!. 
-Only drop columns when it is Absolutely necessary.                                       

Each transformation type MUST follow this exact structure:

1. rename_column
{{
  "type": "rename_column",
  "details": {{
    "file": "<filename or null>",
    "mapping": {{
      "<old_name>": "<new_name>"
    }}
  }}
}}

2. parse_dates
{{
  "type": "parse_dates",
  "details": {{
    "file": "<filename or null>",
    "column": "<column_name>",
    "format": "<optional date format string or null>"
  }}
}}

3. split_column
{{
  "type": "split_column",
  "details": {{
    "file": "<filename or null>",
    "column": "<column_name>",
    "delimiter": "<delimiter or null>",
    "new_columns": ["col1", "col2", "..."],
    "keep_original": false
  }}
}}

4. fill_na
{{
  "type": "fill_na",
  "details": {{
    "file": "<filename or null>",
    "columns": ["col1", "col2"],
    "strategy": "ffill" | "bfill" | "zero" | "<constant value>"
  }}
}}

5. drop_columns
{{
  "type": "drop_columns",
  "details": {{
    "file": "<filename or null>",
    "columns": ["col1", "col2"]
  }}
}}

6. cast_type
{{
  "type": "cast_type",
  "details": {{
    "file": "<filename or null>",
    "columns": {{
      "col1": "float64",
      "col2": "int64"
    }}
  }}
}}

7. set_index
{{
  "type": "set_index",
  "details": {{
    "file": "<filename or null>",
    "column": "<column_name>"
  }}
}}

8. standardize_headers
{{
  "type": "standardize_headers",
  "details": {{
    "file": "<filename or null>"
  }}
}}

9. deduplicate_rows
{{
  "type": "deduplicate_rows",
  "details": {{
    "file": "<filename or null>"
  }}
}}

10. strip_whitespace
{{
  "type": "strip_whitespace",
  "details": {{
    "file": "<filename or null>"
  }}
}}

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

#-----------------------
#File Loader
#-----------------------




def robust_read_delimited_file(path):
    """
    Robust CSV/TSV/pipe reader with:
    - delimiter detection
    - decimal detection
    - encoding fallback
    """

    # --- Read sample ---
    with open(path, "rb") as f:
        raw = f.read(10000)

    sample = raw.decode("utf-8", errors="ignore")

    # --- Detect delimiter using Sniffer ---
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "|", "\t"])
        delimiter = dialect.delimiter
    except Exception:
        # fallback frequency detection
        delimiters = [",", ";", "|", "\t"]
        counts = {d: sample.count(d) for d in delimiters}
        delimiter = max(counts, key=counts.get)

    # --- Detect decimal ---
    if delimiter != "," and re.search(r"\d+,\d+", sample):
        decimal = ","
    else:
        decimal = "."

    print(f"[robust_read] delimiter='{delimiter}' decimal='{decimal}'")

    # --- Encoding fallback ---
    encodings = ["utf-8", "latin1", "cp1252"]

    for enc in encodings:
        try:
            df = pd.read_csv(
                path,
                sep=delimiter,
                decimal=decimal,
                encoding=enc,
                engine="python"
            )
            print(f"[robust_read] encoding='{enc}'")
            return df
        except Exception:
            continue

    raise ValueError("Failed to read delimited file.")





def robust_load_file(path):
    """
    Universal file loader for:
    - Excel (.xlsx, .xls, .xlsm, .ods)
    - CSV
    - TSV
    - Pipe separated
    - TXT delimited files

    Returns:
        dict {file_key: DataFrame}
    """

    ext = os.path.splitext(path)[1].lower()
    basename = os.path.basename(path)
    result = {}

    # =========================
    # EXCEL FILES
    # =========================
    if ext in [".xlsx", ".xls", ".xlsm", ".ods"]:

        # Try safe engine fallback
        excel_engines = [None, "openpyxl", "xlrd"]

        for engine in excel_engines:
            try:
                sheets = pd.read_excel(
                    path,
                    sheet_name=None,
                    engine=engine,
                    dtype=str
                )
                break
            except Exception:
                continue
        else:
            raise ValueError("Could not read Excel file with available engines.")

        for sheet_name, sheet_df in sheets.items():

            # Drop completely empty rows/columns
            sheet_df = sheet_df.dropna(how="all").dropna(axis=1, how="all")

            key = f"{basename}::{sheet_name}"
            result[key] = sheet_df

        return result

    # =========================
    # DELIMITED FILES
    # =========================
    elif ext in [".csv", ".tsv", ".txt"]:

        df = robust_read_delimited_file(path)

        # Drop fully empty rows/cols
        df = df.dropna(how="all").dropna(axis=1, how="all")

        result[path] = df
        return result

    else:
        raise ValueError(f"Unsupported file type: {ext}")



def auto_detect_and_clean_value_column(df: pd.DataFrame, threshold: float = 0.7):
    """
    Detects columns that contain values like 'DMD:167.0'
    or similar prefix:number patterns, strips prefixes,
    converts to float, and returns the best candidate column.
    
    Returns:
        (df, detected_value_column or None)
    """

    numeric_pattern = re.compile(r":\s*([-+]?\d*\.?\d+)")
    candidates = []

    for col in df.columns:

        if df[col].dtype != "object":
            continue

        # Try extract numeric part
        extracted = df[col].astype(str).str.extract(numeric_pattern)[0]

        match_ratio = extracted.notna().mean()

        if match_ratio >= threshold:
            converted = pd.to_numeric(extracted, errors="coerce")

            # Only accept if conversion meaningful
            if converted.notna().mean() >= threshold:
                df[col] = converted
                candidates.append(col)
                print(f"[value_detect] Converted column '{col}' (match_ratio={match_ratio:.2f})")

    # If multiple candidates, pick best
    if candidates:
        # Heuristic: highest variance usually = value column
        variances = {col: df[col].var() for col in candidates}
        best_col = max(variances, key=variances.get)
        print(f"[value_detect] Selected value column: {best_col}")
        return df, best_col

    return df, None



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
        try:
            loaded = robust_load_file(f)

            for key, df in loaded.items():
                df_map[key] = df
                print(f"Loaded {key}: {df.shape[0]} rows, {df.shape[1]} cols")

        except Exception as e:
            print(f"[ERROR] Failed loading {f}: {e}")


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


    detected_values = {}

    for fname, df in updated.items():
        df, value_col = auto_detect_and_clean_value_column(df)
        updated[fname] = df
        if value_col:
            detected_values[fname] = value_col



    os.makedirs(out_dir, exist_ok=True)
    for fname, df in updated.items():
        safe_name = os.path.basename(fname).replace("::", "__").replace(":", "_")
        outp = os.path.join(out_dir, f"cleaned_{safe_name}.csv")
        df.reset_index().to_csv(outp, index=False)
        print(f"Saved cleaned: {outp}")




    future_cfg = plan.get("futureexpert_config", {})

    # If LLM did not specify value column → auto inject
    if not future_cfg.get("value_column"):
        if detected_values:
            #   Take first detected (single dataset case)
            future_cfg["value_column"] = list(detected_values.values())[0]
            print(f"[futureexpert] Auto-assigned value_column: {future_cfg['value_column']}")


    cfg = {
    "files": list(updated.keys()),
    "futureexpert_config": future_cfg,
    "explanation": plan.get("explanation", "")
}






    cfg_path = os.path.join(out_dir, "futureexpert_checkin_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config saved: {cfg_path}")

    return {"status": "success", "outputs": list(updated.keys()), "config": cfg}



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
