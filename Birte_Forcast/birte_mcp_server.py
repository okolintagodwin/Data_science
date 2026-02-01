# birte_mcp_server.py

from typing import List, Optional, Dict
import asyncio

import google.generativeai as genai
import os
from fastmcp import FastMCP
from birte_preprocessing import prepare_data
from forcasting import run_futureexpert_forecast_mcp
from event import run_wp2_pipeline
import json
import re



# Create MCP server instance
mcp = FastMCP("birte_mcp_server")


@mcp.tool()
async def preprocess_files(
    files: List[str],
    out_dir: str = "birte_cleaned",
    answers: Optional[List[Dict[str, str]]] = None
) -> Dict:
    """
    Cleans user time-series files using WP1 preprocessing.
    Runs prepare_data() in a background thread to avoid blocking the MCP event loop.
    """
    return await asyncio.to_thread(
        prepare_data,
        files,
        out_dir,
        answers
    )


@mcp.tool()
async def clean(
    file_paths: List[str],
    out_dir: str = "birte_cleaned"
) -> str:
    """
    Simple slash command (Gemini CLI compatible):
      /clean file_paths="data.csv"
    """
    await preprocess_files(file_paths, out_dir=out_dir)
    return f"Cleaning completed. Output written to: {out_dir}"

@mcp.tool()
async def forecast_futureexpert(
    cleaned_csv: str,
    config_json: str,
    output_csv: str = "birte_cleaned/futureexpert_forecast.csv",
    output_plot: str = "birte_cleaned/futureexpert_plot.png",
) -> dict:
    # Runs the forecast in a background thread so the event loop is not blocked
    import asyncio
    return await asyncio.to_thread(
        run_futureexpert_forecast_mcp,
        cleaned_csv,
        config_json,
        output_csv,
        output_plot,
    )


@mcp.tool()
async def search_events(user_prompt: str) -> dict:
    """
    WP2 – News Intelligence / Event Discovery.
    Uses Gemini + NewsAPI to extract structured global events.
    """
    try:
        return await asyncio.to_thread(run_wp2_pipeline, user_prompt)
    except Exception as e:
        return {
            "status": "error",
            "message": f"WP2 failed: {str(e)}"
        }





@mcp.tool()
async def ask_birte(user_prompt: str, context: Optional[Dict] = None) -> dict:
    """
    Conversational agent: interprets user requests and routes to appropriate MCP tools.
    """

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        return {"status": "error", "message": "GEMINI_API_KEY not set"}

    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Prompt Gemini to map user request to tool + args
    prompt = f"""
You are an intelligent assistant managing the following tools:

1. preprocess_files(files, out_dir, answers)
2. forecast_futureexpert(cleaned_csv, config_json, output_csv, output_plot)
3. search_events(user_prompt)

Analyze the user request:
\"\"\"
{user_prompt}
\"\"\"

Return a JSON ONLY with:
{{
  "tool": "<name of tool: preprocess_files | forecast_futureexpert | search_events>",
  "args": {{
      "<arg_name>": "<value>",
      ...
  }},
  "missing_args": ["<list of missing arguments if any>"]
}}
"""
    raw_resp = model.generate_content(prompt).text
    try:
        import json, re
        cleaned = re.sub(r"```json|```", "", raw_resp).strip()
        instruction = json.loads(cleaned)
    except Exception as e:
        return {"status": "error", "message": f"Failed to parse Gemini response: {e}", "raw": raw_resp}

    tool_name = instruction.get("tool")
    args = instruction.get("args", {})
    missing_args = instruction.get("missing_args", [])

    # If required arguments are missing, ask the user
    if missing_args:
        return {"status": "need_info", "missing_args": missing_args, "tool": tool_name}

    # Map tool name to MCP tool
    tool_mapping = {
        "preprocess_files": preprocess_files,
        "forecast_futureexpert": forecast_futureexpert,
        "search_events": search_events
    }
    selected_tool = tool_mapping.get(tool_name)
    if not selected_tool:
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}

    # Execute selected tool in background thread
    import asyncio
    result = await asyncio.to_thread(selected_tool, **args)
    return {"status": "success", "tool_used": tool_name, "result": result}



if __name__ == "__main__":
    # Use stdio for Gemini CLI (no web server)
    mcp.run(transport="stdio")
