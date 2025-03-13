import streamlit as st
import os
import groq
from dotenv import load_dotenv
import requests
import queue
import threading
import av
from deepgram import Deepgram
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from pydub import AudioSegment
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import networkx as nx
import random
import json
import pandas as pd
import matplotlib.pyplot as plt
from llmformatter import format_llm_response
from classifyresponse import classify_response
from classifylikeconsultant import classify_response1
import re

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

groq_client = groq.Client(api_key=GROQ_API_KEY)
deepgram = Deepgram(DEEPGRAM_API_KEY)

SYSTEM_PROMPT = """
You are a highly experienced McKinsey consultant specializing in Strategy & Business Transformation.
Provide structured, data-driven, and actionable recommendations.
Use right kind of frameworks that a big3 consultant use wherever applicable.
Your responses should be concise, strategic, and impactful.
"""

def query_llama(prompt):
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Function to transcribe audio using Deepgram
def transcribe_audio(audio_path):
    url = "https://api.deepgram.com/v1/listen"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"
    }
    
    with open(audio_path, "rb") as audio_file:
        response = requests.post(url, headers=headers, data=audio_file)

    if response.status_code == 200:
        return response.json().get("results", {}).get("channels", [{}])[0].get("alternatives", [{}])[0].get("transcript", "")
    else:
        st.error(f"Deepgram STT Error: {response.json()}")
        return None

# Function to generate speech using Deepgram TTS
def generate_speech(text):
    url = "https://api.deepgram.com/v1/speak"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"text": text}  # Ensure only "text" or "url" is sent

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.content
    else:
        st.error(f"Deepgram TTS Error: {response.json()}")
        return None

# Function to generate a line chart (Trend Analysis)
def generate_line_chart():
    data = pd.DataFrame({
        "Year": list(range(2018, 2025)),
        "Revenue": [random.randint(50, 100) for _ in range(7)]
    })
    
    fig = px.line(data, x="Year", y="Revenue", title="Market Trend Over Time")
    return fig

# Function to generate a bar chart (Comparison)
def generate_bar_chart():
    data = pd.DataFrame({
        "Category": ["Product A", "Product B", "Product C"],
        "Market Share": [random.randint(10, 50) for _ in range(3)]
    })
    
    fig = px.bar(data, x="Category", y="Market Share", title="Market Share Comparison", text_auto=True)
    return fig

# Function to generate a pie chart (Distribution)
def generate_pie_chart():
    data = pd.DataFrame({
        "Category": ["Segment A", "Segment B", "Segment C"],
        "Percentage": [30, 45, 25]
    })
    
    fig = px.pie(data, names="Category", values="Percentage", title="Market Segment Distribution")
    return fig

def generate_decision_tree():
    G = nx.DiGraph()
    
    # Define decision paths
    G.add_edges_from([
        ("Start", "Expand Market"),
        ("Expand Market", "Enter New Region"),
        ("Expand Market", "Target New Customer Segments"),
        ("Enter New Region", "Partner with Local Firms"),
        ("Enter New Region", "Build In-House Distribution"),
        ("Target New Customer Segments", "Develop New Products"),
        ("Target New Customer Segments", "Adjust Pricing Strategy")
    ])
    
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, edge_color="gray")
    plt.title("Decision Tree for Market Expansion")
    return plt

def generate_growth_trends_chart():
    data = pd.DataFrame({
        "Year": [2020, 2021, 2022, 2023, 2024],
        "Revenue ($M)": [10, 15, 22, 30, 45]
    })
    fig = px.line(data, x="Year", y="Revenue ($M)", title="Revenue Growth Over Time")
    return fig

# Function to generate Porter's Five Forces diagram
def generate_porters_five_forces():
    st.image("porters_five_forces.png", caption="Porter's Five Forces Framework")

# Function to generate SWOT Matrix
def generate_swot_matrix():
    st.image("swot_matrix_example.png", caption="SWOT Analysis")

# Function to generate MECE structure
def generate_mece_structure():
    fig = px.treemap(
        names=["Market", "Customer Segments", "Pricing Strategies", "Product Lines"],
        parents=["", "Market", "Market", "Market"],
        title="MECE Framework Breakdown"
    )
    return fig



# Initialize session state variables
if "solution_approaches" not in st.session_state:
    st.session_state.solution_approaches = []
if "selected_bucket" not in st.session_state:
    st.session_state.selected_bucket = None
if "evaluations" not in st.session_state:
    st.session_state.evaluations = {}
# Initialize session state variables if they don't exist
if "approach_titles" not in st.session_state:
    st.session_state.approach_titles = []

# Streamlit UI
st.title("AI Strategy Consultant")
problem = st.text_area("Describe your business problem:", "How can we improve market penetration in Europe?")

# **Upload Audio for STT**
uploaded_audio = st.file_uploader("Upload an audio file (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])
if uploaded_audio:
    file_path = f"temp_{uploaded_audio.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_audio.read())

    st.write("Transcribing...")
    transcribed_text = transcribe_audio(file_path)
    st.write(f"**Transcribed Text:** {transcribed_text}")
    if transcribed_text:
     problem = transcribed_text
     

    # **Send to AI Model**
    if st.button("Ask AI"):
        ai_response = query_llama(transcribed_text)
        st.write("### AI Response:")
        st.write(ai_response)

        # **Convert AI response to Speech**
        st.write("Generating Speech...")
        tts_file = generate_speech(ai_response)
        if isinstance(tts_file, str) and tts_file.endswith(".mp3"):
            st.audio(tts_file)
        else:
            st.error(f"Failed to generate speech: {tts_file}")

# Text-to-Speech Button
#if st.button("🔊 Listen to Problem"):
 #   speech_file = text_to_speech(problem)
  #  if isinstance(speech_file, str) and speech_file.endswith(".mp3"):
   #     st.audio(speech_file)
    #else:
     #   st.error(f"Failed to generate speech: {speech_file}")

# Step-by-step structured process
if st.button("Interpret Problem"):
    interpret_prompt = f"Interpret and refine the following business problem: '{problem}'. Provide a structured breakdown."
    interpretation = query_llama(interpret_prompt)
    
    if interpretation:
        st.write("### Problem Interpretation:")
        st.write(interpretation)
    else:
        st.error("Could not interpret problem. Please refine your input.")


# ✅ Reset solutions when a new problem is entered
if "previous_problem" not in st.session_state or st.session_state.previous_problem != problem:
    st.session_state.previous_problem = problem
    st.session_state.solution_approaches = []
    st.session_state.approach_titles = []

# Step 2: Solution Buckets - Modularized Approach
if st.button("Suggest Solutions"):
    if st.session_state.get("previous_problem") != problem:
        st.session_state.previous_problem = problem
        st.session_state.solution_approaches = []
        st.session_state.approach_titles = []

    bucket_prompt = f"Given the problem: '{problem}', suggest 3-5 high-level solution approaches."
    solution_buckets = query_llama(bucket_prompt)
    
    if solution_buckets:
        approaches = [opt.strip() for opt in solution_buckets.split("\n") if opt.strip()]

        st.session_state.solution_approaches = []
        st.session_state.approach_titles = []

        for line in approaches:
            if re.match(r'^\d+\.|^-|\*', line):  # More flexible regex
                st.session_state.solution_approaches.append(f"### {line}")
                st.session_state.approach_titles.append(line)  # For dropdown
            else:
                st.session_state.solution_approaches.append(f"- {line}")  # Bullet points
        
        st.rerun()  # Force UI refresh

    else:
        st.error("No solutions found. Please try again.")

# ✅ Display structured solutions
if st.session_state.get("solution_approaches"):
    st.write("### Suggested Solution Approaches:")
    for approach in st.session_state.solution_approaches:
        if approach.startswith("### "):  # Approach titles
            st.markdown(f'<p style="font-size:16px; margin-top:10px;">{approach.replace("### ", "")}</p>', unsafe_allow_html=True)
        else:  # Bullet points
            st.markdown(f'<p style="font-size:16px;">{approach}</p>', unsafe_allow_html=True)


# ✅ Ensure dropdown appears consistently
if st.session_state.get("approach_titles"):
    selected_option = st.selectbox("Select a solution approach:", st.session_state.approach_titles, key="approach_selector")
    
    if selected_option:
        st.session_state.selected_bucket = selected_option
        st.write(f"**You selected:** {selected_option}")

# Step 3: Solution Evaluation
if st.session_state.selected_bucket:
    if st.button("Evaluate Solution"):
        eval_prompt = f"Evaluate the solution approach: '{st.session_state.selected_bucket}' using qualitative and quantitative metrics."
        evaluation = query_llama(eval_prompt)

        # Save evaluation in session state
        st.session_state["evaluation"] = evaluation  # Store it persistently
        
        if evaluation:
            st.session_state.evaluations[st.session_state.selected_bucket] = evaluation
            st.write("### Solution Evaluation:")
            
            # Classify response
            framework, vis_type1 = classify_response1(st.session_state["evaluation"])
            vis_type = classify_response(st.session_state["evaluation"])


            # Display selected consulting framework
            st.subheader("Consulting Framework Used:")
            st.write(f"**{framework}**")

            # Generate and display the correct visualization
            if vis_type1 == "market_size_graph":
               st.plotly_chart(generate_growth_trends_chart())
            elif vis_type1 == "decision_tree_visualization":
               st.pyplot(generate_decision_tree())
            elif vis_type1 == "cost_breakdown_bar":
               st.plotly_chart(generate_bar_chart())
            elif vis_type1 == "swot_matrix":
               generate_swot_matrix()  # Displays SWOT Analysis
            elif vis_type1 == "porters_five_forces":
               generate_porters_five_forces()  # Porter's Five Forces Diagram
            elif vis_type1 == "mece_structure":
               st.plotly_chart(generate_mece_structure())  # MECE Breakdown
            elif vis_type1 == "bcg_matrix":
               generate_bcg_matrix()  # BCG Growth-Share Matrix
            elif vis_type1 == "business_model_canvas":
               st.image("business_model_canvas.png", caption="Business Model Canvas")
            else:
               st.write("No visual aid required for this framework.")

                
             # Render appropriate visualization
            if vis_type == "line_chart":
               st.plotly_chart(generate_line_chart())
            elif vis_type == "bar_chart":
               st.plotly_chart(generate_bar_chart())
            elif vis_type == "pie_chart":
               st.plotly_chart(generate_pie_chart())
            elif vis_type == "decision_tree":
               st.pyplot(generate_decision_tree())
            else:
               st.write("No visual aid required for this response.")
            st.write(evaluation)
        else:
            st.error("Could not evaluate the solution.")



# **Inline Feedback for Refinement**
feedback_eval = st.text_area("Provide feedback on the evaluation (optional):")
if st.button("Refine Evaluation"):
    refined_eval_prompt = f"Refine the evaluation of '{st.session_state.selected_bucket}' considering this feedback: '{feedback_eval}'"
    refined_evaluation = query_llama(refined_eval_prompt)
        
    if refined_evaluation:
        st.session_state.evaluations[st.session_state.selected_bucket] = refined_evaluation
        st.write("### Refined Solution Evaluation:")
        st.write(refined_evaluation)
    else:
        st.error("Could not refine evaluation.")

# Step 4: Implementation Plan
if st.session_state.selected_bucket:
    if st.button("Generate Execution Plan"):
        exec_prompt = f"Create a step-by-step execution plan for '{st.session_state.selected_bucket}'."
        execution_plan = query_llama(exec_prompt)

        # Save evaluation in session state
        st.session_state["execution_plan"] = execution_plan  # Store it persistently
        
        
        if execution_plan:
            st.write("### Execution Plan:")
            framework, vis_type = classify_response1(st.session_state["execution_plan"])

            # Display selected consulting framework
            st.subheader("Consulting Framework Used:")
            st.write(f"**{framework}**")

            # Generate and display the correct visualization
            if vis_type == "market_size_graph":
               st.plotly_chart(generate_growth_trends_chart())
            elif vis_type == "decision_tree_visualization":
               st.pyplot(generate_decision_tree())
            elif vis_type == "cost_breakdown_bar":
               st.plotly_chart(generate_bar_chart())  # Assume this function exists
            elif vis_type == "swot_matrix":
               st.image("swot_matrix_example.png")  # Predefined SWOT image
            else:
               st.write("No visual aid required for this response.")
            st.write(execution_plan)
        else:
            st.error("Execution plan could not be generated.")
# **Inline Feedback for Execution Plan**
feedback_exec = st.text_area("Provide feedback on the execution plan (optional):")
if st.button("Refine Execution Plan"):
    refined_exec_prompt = f"Refine the execution plan for '{st.session_state.selected_bucket}' considering this feedback: '{feedback_exec}'"
    refined_execution_plan = query_llama(refined_exec_prompt)
        
    if refined_execution_plan:
        st.write("### Refined Execution Plan:")
        st.write(refined_execution_plan)
    else:
        st.error("Could not refine execution plan.")
