from graph import app 

def save_langgraph_png(output_path="state_machine_langgraph.png"):
    """
    Save the LangGraph state machine as a PNG image.
    Works in both Jupyter and normal Python scripts.
    """
    try:
        # Generate PNG bytes using LangGraph's Mermaid rendering
        png_bytes = app.get_graph().draw_mermaid_png()

        # Write bytes to file
        with open(output_path, "wb") as f:
            f.write(png_bytes)

        print(f"✅ LangGraph state machine diagram saved to {output_path}")

    except Exception as e:
        print(f"⚠️ Error generating graph image: {e}")
