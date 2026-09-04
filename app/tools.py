# ==========================================
# Calculator Tool
# ==========================================

def calculator(expression):
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)

    except Exception:
        return "Invalid calculation."