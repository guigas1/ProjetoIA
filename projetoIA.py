import streamlit as st
import pandas as pd
import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SomBR · Recomendações Musicais",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* Reset & base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0ece3;
}

.stApp {
    background: #0d0d0d;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 3.5rem 1rem 1.5rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f0ece3;
    margin: 0;
    line-height: 1.05;
}
.hero-title span {
    color: #c9a84c;
    font-style: italic;
}
.hero-sub {
    font-size: 1rem;
    color: #888;
    margin-top: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 300;
}
.hero-divider {
    width: 60px;
    height: 2px;
    background: #c9a84c;
    margin: 1.5rem auto 0;
}

/* ── Section titles ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: #f0ece3;
    margin-bottom: 0.3rem;
    letter-spacing: -0.01em;
}
.section-sub {
    font-size: 0.82rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 1.4rem;
}

/* ── Cards ── */
.card {
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s, transform 0.2s;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: #c9a84c;
    opacity: 0;
    transition: opacity 0.2s;
}
.card:hover {
    border-color: #3a3a3a;
    transform: translateY(-2px);
}
.card:hover::before { opacity: 1; }

.card-rank {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    color: #2a2a2a;
    position: absolute;
    right: 1.2rem;
    top: 0.8rem;
    line-height: 1;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    color: #f0ece3;
    margin-bottom: 0.2rem;
}
.card-artist {
    font-size: 0.85rem;
    color: #c9a84c;
    margin-bottom: 0.6rem;
    font-weight: 500;
}
.card-tags {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
}
.tag {
    background: #1e1e1e;
    border: 1px solid #2e2e2e;
    border-radius: 20px;
    padding: 0.18rem 0.65rem;
    font-size: 0.72rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.tag.genre { border-color: #3a2a00; color: #c9a84c; }

.score-bar-wrap {
    margin-top: 0.7rem;
}
.score-label {
    font-size: 0.72rem;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}
.score-bar {
    height: 3px;
    background: #1e1e1e;
    border-radius: 2px;
    overflow: hidden;
}
.score-fill {
    height: 100%;
    background: linear-gradient(90deg, #c9a84c, #e8c76a);
    border-radius: 2px;
}

/* ── Form area ── */
.form-card {
    background: #111;
    border: 1px solid #222;
    border-radius: 16px;
    padding: 2rem;
}

/* ── Stickers / badge ── */
.badge {
    display: inline-block;
    background: #1a1400;
    border: 1px solid #c9a84c44;
    color: #c9a84c;
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    margin-bottom: 0.5rem;
}

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div {
    background: #161616 !important;
    border-color: #2a2a2a !important;
    color: #f0ece3 !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input {
    background: #161616 !important;
    border-color: #2a2a2a !important;
    color: #f0ece3 !important;
    border-radius: 8px !important;
}
.stButton > button {
    background: #c9a84c !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.8rem !important;
    transition: opacity 0.15s !important;
    font-size: 0.9rem !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    border-bottom: 1px solid #1e1e1e;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #555 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    padding: 0.5rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    color: #c9a84c !important;
    border-bottom-color: #c9a84c !important;
}

/* Success / info messages */
.stSuccess, .stInfo {
    background: #111 !important;
    border-color: #c9a84c44 !important;
    color: #f0ece3 !important;
}

/* Divider */
.sep { border: none; border-top: 1px solid #1e1e1e; margin: 2rem 0; }

label, .stSelectbox label, .stTextInput label {
    color: #999 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# BACKEND — idêntico ao código original
# ─────────────────────────────────────────────
@st.cache_data
def carregar_dados():
    generos = ['MPB', 'Samba', 'Rock Nacional', 'Bossa Nova', 'Rap']
    contexto = ['Romântico', 'Alegre', 'Melancólico', 'Político', 'Festa', 'Calmo', 'Ambiente']
    decada = ['Anos 60', 'Anos 70', 'Anos 80', 'Anos 90', 'Anos 2000', 'Anos 2010']

    musicas_reais = [
        ("Águas de Março", "Tom Jobim", "MPB", "Anos 70"),
        ("Construção", "Chico Buarque", "MPB", "Anos 70"),
        ("O Leãozinho", "Caetano Veloso", "MPB", "Anos 70"),
        ("Como Nossos Pais", "Elis Regina", "MPB", "Anos 70"),
        ("Velha Infância", "Tribalistas", "MPB", "Anos 2000"),
        ("Anunciação", "Alceu Valença", "MPB", "Anos 80"),
        ("Meu Abrigo", "Melim", "MPB", "Anos 2010"),
        ("As Rosas Não Falam", "Cartola", "Samba", "Anos 70"),
        ("O Mundo é um Moinho", "Cartola", "Samba", "Anos 70"),
        ("Não Deixe o Samba Morrer", "Alcione", "Samba", "Anos 70"),
        ("Vou Festejar", "Beth Carvalho", "Samba", "Anos 70"),
        ("Coração em Desalinho", "Zeca Pagodinho", "Samba", "Anos 2000"),
        ("Tá Escrito", "Grupo Revelação", "Samba", "Anos 2000"),
        ("Deixa Acontecer", "Grupo Revelação", "Samba", "Anos 2000"),
        ("Tempo Perdido", "Legião Urbana", "Rock Nacional", "Anos 80"),
        ("Pais e Filhos", "Legião Urbana", "Rock Nacional", "Anos 80"),
        ("Exagerado", "Cazuza", "Rock Nacional", "Anos 80"),
        ("Pro Dia Nascer Feliz", "Barão Vermelho", "Rock Nacional", "Anos 80"),
        ("Só os Loucos Sabem", "Charlie Brown Jr.", "Rock Nacional", "Anos 2000"),
        ("Dias de Luta, Dias de Glória", "Charlie Brown Jr.", "Rock Nacional", "Anos 2000"),
        ("Me Adora", "Pitty", "Rock Nacional", "Anos 2000"),
        ("Renata", "Tihuana", "Rock Nacional", "Anos 2000"),
        ("Garota de Ipanema", "Tom Jobim", "Bossa Nova", "Anos 60"),
        ("Chega de Saudade", "João Gilberto", "Bossa Nova", "Anos 60"),
        ("Desafinado", "João Gilberto", "Bossa Nova", "Anos 60"),
        ("Wave", "Tom Jobim", "Bossa Nova", "Anos 70"),
        ("Samba de Uma Nota Só", "Tom Jobim", "Bossa Nova", "Anos 60"),
        ("Corcovado", "Tom Jobim", "Bossa Nova", "Anos 60"),
        ("Insensatez", "Tom Jobim", "Bossa Nova", "Anos 60"),
        ("Para Machucar meu coração", "João Gilberto", "Bossa Nova", "Anos 60"),
        ("Negro Drama", "Racionais MC's", "Rap", "Anos 2000"),
        ("Vida Loka Parte II", "Racionais MC's", "Rap", "Anos 2000"),
        ("Diário de um Detento", "Racionais MC's", "Rap", "Anos 90"),
        ("AmarElo", "Emicida", "Rap", "Anos 2010"),
        ("Principia", "Emicida", "Rap", "Anos 2010"),
        ("Sulicídio", "Baco Exu do Blues", "Rap", "Anos 2010"),
        ("Te Amo Disgraça", "Baco Exu do Blues", "Rap", "Anos 2010"),
        ("Trem das Onze", "Adoniran Barbosa", "Samba", "Anos 60"),
        ("Mas Que Nada", "Jorge Ben Jor", "MPB", "Anos 60"),
        ("País Tropical", "Jorge Ben Jor", "MPB", "Anos 70"),
        ("Anna Júlia", "Los Hermanos", "Rock Nacional", "Anos 90"),
        ("Do Seu Lado", "Jota Quest", "Rock Nacional", "Anos 2000"),
        ("Na Sua Estante", "Pitty", "Rock Nacional", "Anos 2000"),
        ("Boa Sorte / Good Luck", "Vanessa da Mata", "MPB", "Anos 2000"),
        ("Ai, Ai, Ai", "Vanessa da Mata", "MPB", "Anos 2000"),
        ("Pra Você Guardei o Amor", "Nando Reis", "MPB", "Anos 2000"),
        ("Por Onde Andei", "Nando Reis", "MPB", "Anos 2000"),
        ("Aquarela", "Toquinho", "MPB", "Anos 80"),
        ("Epitáfio", "Titãs", "Rock Nacional", "Anos 2000"),
        ("Flores", "Titãs", "Rock Nacional", "Anos 80"),
        ("O Sol", "Vitor Kley", "MPB", "Anos 2010"),
    ]

    np.random.seed(42)
    random.seed(42)

    df_musicas = pd.DataFrame(musicas_reais, columns=["titulo", "artista", "genero", "decada"])
    df_musicas["musica_id"] = range(1, len(df_musicas) + 1)
    num_musicas = len(df_musicas)
    df_musicas["contexto"] = np.random.choice(contexto, size=len(df_musicas))
    df_musicas = df_musicas[["musica_id", "titulo", "artista", "genero", "contexto", "decada"]]

    num_usuarios = 1000
    usuarios = []
    for usuario_id in range(1, num_usuarios + 1):
        usuarios.append({
            "usuario_id": usuario_id,
            "genero_favorito": random.choice(generos),
            "contexto_favorito": random.choice(contexto),
            "decada_favorita": random.choice(decada)
        })
    df_usuarios = pd.DataFrame(usuarios)

    def gerar_nota(usuario, musica):
        score = 0
        if usuario["genero_favorito"] == musica["genero"]: score += 3
        if usuario["contexto_favorito"] == musica["contexto"]: score += 2
        if usuario["decada_favorita"] == musica["decada"]: score += 1
        ruido = np.random.normal(0, 0.8)
        nota_continua = 2.5 + score * 0.45 + ruido
        return min(max(round(nota_continua), 1), 5)

    num_avaliacoes = 30000
    pares = set()
    while len(pares) < num_avaliacoes:
        uid = random.randint(1, num_usuarios)
        mid = random.randint(1, num_musicas)
        pares.add((uid, mid))

    avaliacoes = []
    for uid, mid in pares:
        u = df_usuarios.loc[df_usuarios["usuario_id"] == uid].iloc[0]
        m = df_musicas.loc[df_musicas["musica_id"] == mid].iloc[0]
        avaliacoes.append({"usuario_id": uid, "musica_id": mid, "nota": gerar_nota(u, m)})
    df_avaliacoes = pd.DataFrame(avaliacoes)

    matriz_utilidade = df_avaliacoes.pivot_table(index="usuario_id", columns="musica_id", values="nota")
    matriz_item_based = matriz_utilidade.T.fillna(0)

    modelo_knn = NearestNeighbors(metric="cosine", algorithm="brute")
    modelo_knn.fit(matriz_item_based)

    return df_musicas, df_usuarios, df_avaliacoes, matriz_item_based, modelo_knn, generos, contexto, decada

# ─────────────────────────────────────────────
# FUNÇÕES DE RECOMENDAÇÃO — idênticas ao original
# ─────────────────────────────────────────────
def recomendar_para_novo_usuario(df_musicas, matriz_item_based, modelo_knn,
                                  genero_alvo, contexto_alvo, decada_alvo, n=5, n_sementes=3):
    df_novo = df_musicas.copy()
    df_novo["score_perfil"] = 0
    df_novo.loc[df_novo["genero"] == genero_alvo, "score_perfil"] += 3
    df_novo.loc[df_novo["contexto"] == contexto_alvo, "score_perfil"] += 2
    df_novo.loc[df_novo["decada"] == decada_alvo, "score_perfil"] += 1

    sementes = (
        df_novo.sort_values("score_perfil", ascending=False)
        .head(n_sementes)["musica_id"].tolist()
    )

    recomendacoes = {}
    for musica_id in sementes:
        musica_idx = musica_id - 1
        distancias, indices = modelo_knn.kneighbors(
            matriz_item_based.iloc[musica_idx].values.reshape(1, -1),
            n_neighbors=n + n_sementes + 1
        )
        for i in range(1, len(indices[0])):
            mid = matriz_item_based.index[indices[0][i]]
            if mid in sementes: continue
            sim = 1 - distancias[0][i]
            recomendacoes[mid] = recomendacoes.get(mid, 0) + sim

    ordenadas = sorted(recomendacoes.items(), key=lambda x: x[1], reverse=True)[:n]
    resultado = []
    for mid, score in ordenadas:
        m = df_musicas[df_musicas["musica_id"] == mid].iloc[0]
        resultado.append({
            "musica_id": mid,
            "titulo": m["titulo"], "artista": m["artista"],
            "genero": m["genero"], "contexto": m["contexto"],
            "decada": m["decada"], "score_knn": round(score, 4)
        })
    return pd.DataFrame(resultado), sementes


def recomendar_musicas_parecidas(df_musicas, matriz_item_based, modelo_knn, musica_id, n=5):
    musica_idx = musica_id - 1
    distancias, indices = modelo_knn.kneighbors(
        matriz_item_based.iloc[musica_idx].values.reshape(1, -1),
        n_neighbors=n + 1
    )
    resultado = []
    for i in range(1, len(indices[0])):
        mid = matriz_item_based.index[indices[0][i]]
        m = df_musicas[df_musicas["musica_id"] == mid].iloc[0]
        resultado.append({
            "musica_id": mid,
            "titulo": m["titulo"], "artista": m["artista"],
            "genero": m["genero"], "contexto": m["contexto"],
            "decada": m["decada"],
            "similaridade": round(1 - distancias[0][i], 4)
        })
    return pd.DataFrame(resultado)

# ─────────────────────────────────────────────
# HELPER — card HTML
# ─────────────────────────────────────────────
def render_card(rank, titulo, artista, genero, contexto, decada, score_val, score_max, score_label):
    pct = min(score_val / score_max * 100, 100) if score_max > 0 else 0
    st.markdown(f"""
    <div class="card">
        <div class="card-rank">#{rank}</div>
        <div class="card-title">{titulo}</div>
        <div class="card-artist">{artista}</div>
        <div class="card-tags">
            <span class="tag genre">{genero}</span>
            <span class="tag">{contexto}</span>
            <span class="tag">{decada}</span>
        </div>
        <div class="score-bar-wrap">
            <div class="score-label">{score_label}: {score_val:.4f}</div>
            <div class="score-bar">
                <div class="score-fill" style="width:{pct:.1f}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
with st.spinner("Carregando base musical e treinando modelos KNN..."):
    df_musicas, df_usuarios, df_avaliacoes, matriz_item_based, modelo_knn, \
        generos, contexto_list, decadas = carregar_dados()

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <p class="hero-sub">Sistema de Recomendação · KNN Colaborativo</p>
    <h1 class="hero-title">Som<span>BR</span></h1>
    <p class="hero-sub">Músicas nacionais escolhidas para o seu gosto</p>
    <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🎧  Novo Usuário · Cold Start", "🔍  Músicas Similares · Item-based"])

# ══════════════════════════════════════════════
# TAB 1 — CADASTRO + COLD START
# ══════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    col_form, col_gap, col_result = st.columns([1.1, 0.15, 1.75])

    with col_form:
        st.markdown('<div class="badge">Passo 1 · Cadastro</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Conte seu gosto musical</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Perfil para cold start via KNN</p>', unsafe_allow_html=True)

        nome = st.text_input("Seu nome", placeholder="ex: Ana Lima")
        genero_sel = st.selectbox("Gênero favorito", generos)
        contexto_sel = st.selectbox("Contexto preferido", contexto_list)
        decada_sel = st.selectbox("Década favorita", decadas)
        n_rec = st.slider("Quantidade de recomendações", 3, 10, 5)

        gerar = st.button("Gerar recomendações →")

    with col_result:
        if gerar:
            if not nome.strip():
                st.warning("Por favor, insira seu nome para continuar.")
            else:
                st.markdown(f'<div class="badge">Olá, {nome.strip()}!</div>', unsafe_allow_html=True)
                st.markdown('<p class="section-title">Suas recomendações</p>', unsafe_allow_html=True)
                st.markdown('<p class="section-sub">Cold start · KNN item-based com músicas-semente</p>', unsafe_allow_html=True)

                with st.spinner("Calculando vizinhos mais próximos..."):
                    df_rec, sementes = recomendar_para_novo_usuario(
                        df_musicas, matriz_item_based, modelo_knn,
                        genero_sel, contexto_sel, decada_sel, n=n_rec
                    )

                # Mostrar músicas-semente
                with st.expander("🌱 Músicas-semente usadas pelo KNN"):
                    for sid in sementes:
                        s = df_musicas[df_musicas["musica_id"] == sid].iloc[0]
                        st.markdown(f"**{s['titulo']}** — {s['artista']} · _{s['genero']}_ · {s['decada']}")

                st.markdown("<hr class='sep'>", unsafe_allow_html=True)

                if df_rec.empty:
                    st.info("Nenhuma recomendação encontrada para esse perfil.")
                else:
                    score_max = df_rec["score_knn"].max()
                    for i, row in df_rec.iterrows():
                        rank = list(df_rec.index).index(i) + 1
                        render_card(rank, row["titulo"], row["artista"], row["genero"],
                                    row["contexto"], row["decada"],
                                    row["score_knn"], score_max, "Score KNN acumulado")

                # Salvar na session_state para usar na tab 2
                st.session_state["usuario_cadastrado"] = {
                    "nome": nome.strip(),
                    "genero": genero_sel,
                    "contexto": contexto_sel,
                    "decada": decada_sel,
                }
        else:
            st.markdown("""
            <div style="height:100%;display:flex;flex-direction:column;justify-content:center;
                        align-items:center;padding:4rem 1rem;text-align:center;gap:1rem;">
                <div style="font-size:3.5rem">🎶</div>
                <p style="color:#444;font-size:0.9rem;line-height:1.7;max-width:280px;">
                    Preencha seu perfil ao lado e clique em<br>
                    <strong style="color:#c9a84c;">Gerar recomendações</strong><br>
                    para ver a mágica do KNN acontecer.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 2 — MÚSICAS SIMILARES
# ══════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_gap2, col_b = st.columns([1.1, 0.15, 1.75])

    with col_a:
        st.markdown('<div class="badge">KNN · Item-based</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">Encontre músicas similares</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Similaridade do cosseno sobre a matriz de utilidade</p>', unsafe_allow_html=True)

        # Seleção da música de referência
        opcoes = df_musicas.apply(lambda r: f"{r['titulo']} — {r['artista']}", axis=1).tolist()
        musica_sel_label = st.selectbox("Escolha uma música de referência", opcoes)

        musica_idx_sel = opcoes.index(musica_sel_label)
        musica_ref = df_musicas.iloc[musica_idx_sel]

        n_sim = st.slider("Número de similares", 3, 10, 5, key="sim_n")

        buscar = st.button("Buscar similares →")

        # Info da música selecionada
        st.markdown("<hr class='sep'>", unsafe_allow_html=True)
        st.markdown('<p class="section-sub">Música de referência</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card" style="border-color:#2a1a00;">
            <div class="card-title">{musica_ref['titulo']}</div>
            <div class="card-artist">{musica_ref['artista']}</div>
            <div class="card-tags">
                <span class="tag genre">{musica_ref['genero']}</span>
                <span class="tag">{musica_ref['contexto']}</span>
                <span class="tag">{musica_ref['decada']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        if buscar:
            st.markdown('<p class="section-title">Músicas similares</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="section-sub">Vizinhos KNN de "{musica_ref["titulo"]}"</p>', unsafe_allow_html=True)

            with st.spinner("Calculando similaridade por cosseno..."):
                df_sim = recomendar_musicas_parecidas(
                    df_musicas, matriz_item_based, modelo_knn,
                    musica_ref["musica_id"], n=n_sim
                )

            if df_sim.empty:
                st.info("Nenhuma música similar encontrada.")
            else:
                score_max_sim = df_sim["similaridade"].max()
                for i, row in df_sim.iterrows():
                    rank = list(df_sim.index).index(i) + 1
                    render_card(rank, row["titulo"], row["artista"], row["genero"],
                                row["contexto"], row["decada"],
                                row["similaridade"], score_max_sim, "Similaridade do cosseno")
        else:
            st.markdown("""
            <div style="height:100%;display:flex;flex-direction:column;justify-content:center;
                        align-items:center;padding:4rem 1rem;text-align:center;gap:1rem;">
                <div style="font-size:3.5rem">🔍</div>
                <p style="color:#444;font-size:0.9rem;line-height:1.7;max-width:280px;">
                    Selecione qualquer música ao lado e clique em<br>
                    <strong style="color:#c9a84c;">Buscar similares</strong><br>
                    para ver o KNN item-based em ação.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:1.5rem;border-top:1px solid #1a1a1a;">
    <p style="color:#333;font-size:0.75rem;letter-spacing:0.12em;text-transform:uppercase;">
        SomBR · KNN Colaborativo Item-based · Filtragem por similaridade do cosseno
    </p>
    <p style="color:#222;font-size:0.7rem;margin-top:0.3rem;">
        Guilherme Nascimento · Letícia Brito · Manuela Mattos
    </p>
</div>
""", unsafe_allow_html=True)