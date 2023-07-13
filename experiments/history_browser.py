"""
Simple streamlit app to browse the history of an adversarial prompting game.
"""
import os

import streamlit as st
import pandas as pd
import tkinter as tk
from tkinter import filedialog


st.set_page_config(layout="wide")
st.button("Refresh")

# Set up tkinter
root = tk.Tk()
root.withdraw()

# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)

# Folder picker button
st.title('Folder Picker')
st.write('Please select a folder:')
clicked = st.button('Folder Picker')
if clicked:
    dirname = st.text_input('Selected folder:', filedialog.askdirectory(master=root))
    os.chdir(dirname)


def color_rows(row):
    color = {
        "player1": "pink",
        "player1_reflection": "pink",
        "player2": "lime",
        "judge": "cyan",
    }
    return [f"background-color: {color[row.type]}"]*len(row)

# def color_survived(val):
#     color = 'green' if val else 'red'
#     return f'background-color: {color}'

# st.dataframe(df.style.apply(highlight_survived, axis=1))
# st.dataframe(df.style.applymap(color_survived, subset=['Survived']))
history = pd.read_json("history.jsonl", lines=True)
st.markdown("## History")
st.dataframe(history[["round", "time", "type", "response", "prompt"]].style.apply(color_rows, axis=1))

st.markdown("## Insights")
insights = pd.read_json("insights.jsonl", lines=True)
st.dataframe(insights[["round", "time", "type", "response", "prompt"]].style.apply(color_rows, axis=1))

