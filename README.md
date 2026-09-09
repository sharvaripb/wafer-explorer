# Wafer Explorer

Interactive Streamlit portfolio reproduction of a semiconductor wafer-image exploration tool developed during an industrial data science internship.

The public version preserves the image-feature clustering and wafer exploration workflow while replacing confidential manufacturing data with 284 synthetic wafer maps and synthetic feature embeddings.

### Features
- DINOv3-S / DINOv3-B-style feature selection
- K-Means and Agglomerative clustering
- Adjustable cluster count
- t-SNE wafer-map visualisation
- Click-to-preview wafer inspection
- Cluster counts and optional cluster outlines

Clustering is performed on standardized feature embeddings; t-SNE is used only for 2D visualisation.

**Portfolio disclosure:** No proprietary wafer images, process data, labels, or confidential manufacturing information are included.

### Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
