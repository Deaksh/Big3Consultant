# Function to classify response and determine visualization type
def classify_response(response_text):
    """
    Uses basic keyword matching and structured extraction.
    Ideally, function calling would return a JSON like: {"chart_type": "line_chart", "data": {...}}
    """
    response_text = response_text.lower()
    
    if "trend" in response_text or "growth" in response_text:
        return "line_chart"
    elif "comparison" in response_text or "market share" in response_text:
        return "bar_chart"
    elif "decision" in response_text or "strategy" in response_text:
        return "decision_tree"
    elif "distribution" in response_text or "proportion" in response_text:
        return "pie_chart"
    else:
        return None  # No visualization needed
