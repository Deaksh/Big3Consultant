def classify_response1(evaluation_text):
    """
    Maps evaluation response to appropriate consulting framework, visualization, and method.
    """
    evaluation_text = evaluation_text.lower()
    
    # Define mappings for consulting frameworks and visual aids
    framework_chart_mapping = {
        "market entry": ("3C Framework", "market_size_graph"),
        "pricing strategy": ("Value-Based Pricing", "pricing_waterfall_chart"),
        "cost reduction": ("Zero-Based Budgeting (ZBB)", "cost_breakdown_bar"),
        "competitive analysis": ("Porter's Five Forces", "porters_five_forces"),  # Updated visual aid
        "growth strategy": ("Ansoff Matrix", "growth_trends_chart"),
        "process optimization": ("Lean Six Sigma", "process_flowchart"),
        "organizational restructuring": ("McKinsey 7S Framework", "org_structure_chart"),
        "decision making": ("Decision Tree Analysis", "decision_tree_visualization"),

        # Additional frameworks for better visual representation
        "swot": ("SWOT Analysis", "swot_matrix"),
        "porter": ("Porter's Five Forces", "porters_five_forces"),
        "mece": ("MECE Framework", "mece_structure"),
        "bcg matrix": ("BCG Growth-Share Matrix", "bcg_matrix"),
        "business model canvas": ("Business Model Canvas", "business_model_canvas"),
    }

    # Find best match in text
    for key in framework_chart_mapping:
        if key in evaluation_text:
            return framework_chart_mapping[key]  # (Framework, Chart Type)

    return ("General Consulting Approach", "none")  # Default case
