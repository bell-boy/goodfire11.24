from IPython.display import HTML, IFrame, clear_output, display

def get_gemma_2b_dashboard(
  layer: int,
  feature_idx: int,
):
  url = f"https://neuronpedia.org/gemma-2-2b/{layer}-gemmascope-res-16k/{feature_idx}?embed=true&embedexplanation=true&embedplots=true&embedtest=true"
  display(IFrame(url, width=800, height=100))