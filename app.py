import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from kneed import KneeLocator

st.set_page_config(page_title="Spotify Mood Map", layout="wide")

# 1. Sidebar for Upload
st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload Spotify CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    available_cols = df.columns.tolist()

    hover_cols = [col for col in ['track_name', 'artist_name'] if col in available_cols]
    
    # Selecting the features
    if 'valence' in df.columns and 'energy' in df.columns:
        X = df[['valence', 'energy']]
    
    # 2. Internal Optimal K Calculation
    distortions = []
    K_range = range(1, 11)
    for k in K_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        distortions.append(model.inertia_)
    
    # Auto-detect the 'elbow'
    kn = KneeLocator(K_range, distortions, curve='convex', direction='decreasing')
    optimal_k = kn.elbow if kn.elbow else 4 # Default to 4 if no clear elbow
    
    # 3. Final Clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['mood_cluster'] = kmeans.fit_predict(X).astype(str)
    
    # 4. Interactive Plotly Graph
    st.title("🎧 Your Personal Spotify Mood Map")
    st.markdown(f"**AI detected {optimal_k} distinct moods in your library.**")
    
    fig = px.scatter(
        df, x='valence', y='energy', color='mood_cluster',
        hover_data=hover_cols,
        title="Hover over points to see song details",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Prism
    )
    
    # Visualizing quadrants
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey")
    fig.add_vline(x=0.5, line_dash="dash", line_color="grey")
    
    st.plotly_chart(fig, use_container_width=True)

    # 5. Dataset Details & Points Toggle
    if st.button("Describe Dataset Details"):
        st.write("### Cluster Centers (Mood Anchors)")
        centers = pd.DataFrame(kmeans.cluster_centers_, columns=['Valence', 'Energy'])
        st.table(centers)
        
        st.write("### Raw Data Samples")
        st.dataframe(df[['track_name', 'artist_name', 'valence', 'energy', 'mood_cluster']])

else:
    st.info("Please upload your Spotify Tracks CSV in the sidebar to begin.")