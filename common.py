import re
import sys
import subprocess
import runpy
import tempfile
import os


def install_and_execute(code: str):
    # 查找所有import和from-import语句
    # imports = re.findall(r"^\s*(?:import|from)\s+([a-zA-Z_][a-zA-Z0-9_]*)", code, re.MULTILINE)
    # packages = set(imports)

    # # 安装缺失的库
    # for package in packages:
    #     try:
    #         __import__(package)
    #     except ImportError:
    #         print(f"{package} is not installed. Installing...")
    #         subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w", encoding="utf-8") as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        # Execute the code using runpy.run_path
        try:
            result_vars = runpy.run_path(temp_file_path)
            result = result_vars.get("result")
            if result is None:
                return None, "No 'result' variable found in the executed code."
            else:
                return result, None
        except Exception as e:
            return None, f"An error occurred during execution: {e}"

    finally:
        # Clean up by removing the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def round_values(data):

    if isinstance(data, float):
        return round(data, 2)
    elif isinstance(data, bool):
        return "TRUE" if data else "FALSE"
    elif isinstance(data, list):
        return [round_values(item) for item in data]
    elif isinstance(data, dict):
        return {key: round_values(value) for key, value in data.items()}
    else:
        return data
