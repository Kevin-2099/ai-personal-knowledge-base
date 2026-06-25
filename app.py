import streamlit as st
import numpy as np
import pandas as pd
import faiss
import networkx as nx
import plotly.graph_objects as go

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Personal Knowledge Base",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# MODEL
# =====================================================

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# =====================================================
# SESSION STATE
# =====================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = []

# =====================================================
# HELPERS
# =====================================================

def rebuild_faiss():
    if len(st.session_state.embeddings) == 0:
        return None

    vectors = np.array(
        st.session_state.embeddings,
        dtype=np.float32
    )

    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)

    faiss.normalize_L2(vectors)
    index.add(vectors)

    return index


def add_document(title, text):

    embedding = model.encode(text)

    st.session_state.documents.append({
        "title": title,
        "text": text
    })

    st.session_state.embeddings.append(
        embedding.astype(np.float32)
    )


def semantic_search(query, k=5):

    index = rebuild_faiss()

    if index is None:
        return []

    q = model.encode(query).astype(np.float32)
    q = np.expand_dims(q, axis=0)

    faiss.normalize_L2(q)

    scores, ids = index.search(
        q,
        min(k, len(st.session_state.documents))
    )

    results = []

    for score, idx in zip(scores[0], ids[0]):

        results.append({
            "score": float(score),
            "doc": st.session_state.documents[idx]
        })

    return results


def build_graph():

    if len(st.session_state.documents) < 2:
        return None

    embeddings = np.array(
        st.session_state.embeddings
    )

    sims = cosine_similarity(
        embeddings
    )

    G = nx.Graph()

    for i, doc in enumerate(st.session_state.documents):
        G.add_node(
            i,
            label=doc["title"]
        )

    threshold = 0.45

    for i in range(len(sims)):
        for j in range(i + 1, len(sims)):

            if sims[i][j] > threshold:

                G.add_edge(
                    i,
                    j,
                    weight=float(sims[i][j])
                )

    return G, sims


# =====================================================
# HEADER
# =====================================================

st.title("🧠 AI Personal Knowledge Base")

st.markdown("""
Guarda conocimiento, encuentra relaciones entre ideas
y crea automáticamente un mapa de conceptos.
""")

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("➕ Nuevo conocimiento")

    title = st.text_input(
        "Título"
    )

    text = st.text_area(
        "Contenido",
        height=250
    )

    if st.button("Guardar"):

        if text.strip():

            add_document(
                title if title else f"Documento {len(st.session_state.documents)+1}",
                text
            )

            st.success(
                "Documento guardado"
            )

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📚 Base",
    "🔍 Buscar",
    "🕸 Relaciones",
    "📊 Insights"
])

# =====================================================
# TAB 1
# =====================================================

with tab1:

    st.subheader("Base de conocimiento")

    if len(st.session_state.documents) == 0:

        st.info(
            "Todavía no hay documentos."
        )

    for doc in st.session_state.documents:

        with st.expander(doc["title"]):

            st.write(doc["text"])

# =====================================================
# TAB 2
# =====================================================

with tab2:

    st.subheader("Consulta semántica")

    query = st.text_input(
        "Haz una pregunta"
    )

    if st.button(
        "🔍 Enviar",
        key="search_button"
    ):

        if not query.strip():

            st.warning(
                "Escribe una pregunta."
            )

        else:

            results = semantic_search(query)

            if results:

                st.success(
                    f"Se encontraron {len(results)} resultados."
                )

                for r in results:

                    st.markdown(
                        f"### {r['doc']['title']}"
                    )

                    st.caption(
                        f"Relevancia: {r['score']:.3f}"
                    )

                    st.write(
                        r["doc"]["text"][:800]
                    )

                    st.divider()

            else:

                st.warning(
                    "No se encontraron resultados."
                )

# =====================================================
# TAB 3
# =====================================================

with tab3:

    st.subheader(
        "Mapa de conocimiento"
    )

    result = build_graph()

    if result is None:

        st.info(
            "Necesitas al menos 2 documentos."
        )

    else:

        G, sims = result

        pos = nx.spring_layout(
            G,
            seed=42
        )

        edge_x = []
        edge_y = []

        for edge in G.edges():

            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]

            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            hoverinfo="none"
        )

        node_x = []
        node_y = []
        labels = []

        for node in G.nodes():

            x, y = pos[node]

            node_x.append(x)
            node_y.append(y)

            labels.append(
                G.nodes[node]["label"]
            )

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(
                size=20
            )
        )

        fig = go.Figure(
            data=[
                edge_trace,
                node_trace
            ]
        )

        fig.update_layout(
            height=700,
            showlegend=False
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

# =====================================================
# TAB 4
# =====================================================

with tab4:

    st.subheader(
        "Insights automáticos"
    )

    docs_count = len(
        st.session_state.documents
    )

    st.metric(
        "Documentos",
        docs_count
    )

    if docs_count > 1:

        _, sims = build_graph()

        avg = (
            sims.sum() - docs_count
        ) / (
            docs_count * (docs_count - 1)
        )

        st.metric(
            "Conexión promedio",
            round(avg, 3)
        )

        upper = np.triu(
            sims,
            1
        )

        i, j = np.unravel_index(
            np.argmax(upper),
            upper.shape
        )

        doc1 = st.session_state.documents[i]["title"]
        doc2 = st.session_state.documents[j]["title"]

        st.success(
            f"Mayor relación encontrada: {doc1} ↔ {doc2}"
        )

        st.info(
            "Las relaciones se calculan por similitud semántica, no por palabras clave."
        )
