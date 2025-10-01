## VFINAL - CORRIGIDO: GRÁFICOS NO FLUXO DO CHAT ###
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# =====================================
# CONFIGURAÇÃO STREAMLIT
# =====================================
st.set_page_config(
    page_title="🔍 Agente EDA Inteligente", 
    page_icon="🔍", 
    layout="wide"
)

st.title("🔍 Agente para Análise Exploratória de Dados (EDA)")
st.markdown("*Upload um arquivo CSV e converse com seus dados usando IA!*")

# Verificar se a API key foi carregada
if not os.getenv("GROQ_API_KEY"):
    st.error("⚠️ **API Key do Groq não encontrada!**")
    st.info("📝 **Como configurar:**\n\n1. Crie um arquivo `.env` na pasta do projeto\n2. Adicione: `GROQ_API_KEY=sua_chave_aqui`\n3. Obtenha sua chave em: https://console.groq.com  ")
    st.stop()

# =====================================
# CONFIGURAÇÃO LLM - GROQ
# =====================================
@st.cache_resource
def get_llm():
    """Inicializa e cacheia a conexão com a LLM Groq"""
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        return llm
    except Exception as e:
        st.error(f"❌ Erro ao conectar com a API Groq: {e}")
        return None

llm = get_llm()
if llm is None:
    st.stop()

# =====================================
# SESSION STATE PARA MEMÓRIA
# =====================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        (AIMessage(content="Olá! Sou seu agente de análise de dados. Upload um arquivo CSV e faça suas perguntas!"), None)
    ]

if "df" not in st.session_state:
    st.session_state.df = None

if "conclusions" not in st.session_state:
    st.session_state.conclusions = []

if "data_context" not in st.session_state:
    st.session_state.data_context = ""

# =====================================
# FUNÇÃO PARA CRIAR GRÁFICOS (SEM RENDERIZAR)
# =====================================
def create_automatic_plot(column, plot_type="auto"):
    """Cria gráfico automaticamente baseado no tipo de dados. Retorna (mensagem, fig)"""
    if st.session_state.df is None or column not in st.session_state.df.columns:
        return f"❌ Coluna '{column}' não encontrada", None
    
    df = st.session_state.df
    
    try:
        is_numeric = df[column].dtype in ['int64', 'float64']
        
        if plot_type in ["pizza", "pie"]:
            value_counts = df[column].value_counts().head(8)
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=f"🍕 Pizza - {column}")
            
        elif plot_type in ["boxplot", "box"]:
            if not is_numeric:
                return f"❌ Boxplot requer coluna numérica. '{column}' é categórica.", None
            fig = px.box(df, y=column, title=f"📦 Boxplot - {column}")
            
        elif plot_type in ["linha", "line"]:
            if is_numeric:
                fig = px.line(df.reset_index(), x='index', y=column, title=f"📈 Linha - {column}")
            else:
                value_counts = df[column].value_counts().head(10)
                fig = px.line(x=range(len(value_counts)), y=value_counts.values, title=f"📈 Linha - {column}")
                
        elif plot_type in ["scatter", "dispersão"]:
            if not is_numeric:
                return f"❌ Scatter plot requer coluna numérica. '{column}' é categórica.", None
            fig = px.scatter(df, x=df.index, y=column, title=f"🎯 Scatter - {column}")
            
        elif plot_type in ["histograma", "histogram"]:
            if not is_numeric:
                return f"❌ Histograma requer coluna numérica. '{column}' é categórica.", None
            fig = px.histogram(df, x=column, title=f"📊 Histograma - {column}", color_discrete_sequence=['#636EFA'])
            
        elif plot_type in ["barras", "bar"]:
            value_counts = df[column].value_counts().head(10)
            fig = px.bar(
                x=value_counts.index, 
                y=value_counts.values, 
                title=f"📊 Barras - {column}",
                color_discrete_sequence=['#00CC96']
            )
            
        elif plot_type == "auto":
            if is_numeric:
                fig = px.histogram(df, x=column, title=f"📊 Distribuição - {column}", color_discrete_sequence=['#636EFA'])
            else:
                value_counts = df[column].value_counts().head(10)
                fig = px.bar(
                    x=value_counts.index, 
                    y=value_counts.values, 
                    title=f"📊 Top 10 - {column}",
                    color_discrete_sequence=['#00CC96']
                )
        else:
            return f"❌ Tipo de gráfico '{plot_type}' não suportado", None
        
        # Salvar no session_state para seção fixa
        plot_key = f"plot_{column}_{plot_type}"
        st.session_state[plot_key] = fig
        
        return f"✅ Gráfico de {plot_type} da coluna '{column}' criado", fig
    
    except Exception as e:
        return f"❌ Erro ao criar gráfico: {str(e)}", None

# =====================================
# FUNÇÃO DE ANÁLISE DOS DADOS
# =====================================
def analyze_data_with_question(question):
    """Analisa dados baseado na pergunta do usuário. Retorna (mensagem, fig)"""
    if st.session_state.df is None:
        return "❌ Nenhum dataset carregado", None
    
    df = st.session_state.df
    analysis_result = []
    question_lower = question.lower()
    
    # Verifica se é pedido de gráfico
    if any(word in question_lower for word in ['gráfico', 'plot', 'visualização', 'histograma', 'boxplot', 'crie', 'criar', 'gerar', 'pizza', 'linha', 'scatter', 'dispersão', 'barras']):
        graph_created = False
        generated_fig = None
        
        # Detecção do tipo de gráfico
        plot_type = "auto"
        if any(w in question_lower for w in ['pizza', 'pie', 'torta']):
            plot_type = "pizza"
        elif any(w in question_lower for w in ['boxplot', 'box', 'caixa']):
            plot_type = "boxplot"
        elif any(w in question_lower for w in ['linha', 'line', 'linear']):
            plot_type = "linha"
        elif any(w in question_lower for w in ['scatter', 'dispersão', 'pontos']):
            plot_type = "scatter"
        elif any(w in question_lower for w in ['histograma', 'histogram']):
            plot_type = "histograma"
        elif any(w in question_lower for w in ['barras', 'bar', 'colunas']):
            plot_type = "barras"
        
        # Busca por colunas
        for col in df.columns:
            col_variations = [str(col).lower(), str(col).replace('_', ' ').lower(), str(col).replace('_', '').lower()]
            if any(variation in question_lower for variation in col_variations):
                result, fig = create_automatic_plot(col, plot_type)
                analysis_result.append(f"🎯 GRÁFICO CRIADO:")
                analysis_result.append(result)
                generated_fig = fig
                graph_created = True
                break
        
        # Busca por palavras-chave
        if not graph_created:
            words = question_lower.replace('gráfico', '').replace('pizza', '').replace('linha', '').replace('barras', '').replace('boxplot', '').replace('da', '').replace('de', '').replace('do', '').replace('por', '').replace('crie', '').replace('um', '').strip().split()
            for word in words:
                if len(word) > 2:
                    for col in df.columns:
                        if word in str(col).lower() or str(col).lower() in word:
                            result, fig = create_automatic_plot(col, plot_type)
                            analysis_result.append(f"🎯 GRÁFICO CRIADO ('{word}' → '{col}'):")
                            analysis_result.append(result)
                            generated_fig = fig
                            graph_created = True
                            break
                    if graph_created:
                        break
        
        if not graph_created:
            analysis_result.append("❌ NÃO FOI POSSÍVEL IDENTIFICAR A COLUNA")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            text_cols = df.select_dtypes(include=['object']).columns.tolist()
            if numeric_cols:
                analysis_result.append(f"🔢 Numéricas: {numeric_cols}")
            if text_cols:
                analysis_result.append(f"📝 Categóricas: {text_cols}")
            analysis_result.append(f"\n🎨 TIPOS DE GRÁFICO DISPONÍVEIS:")
            analysis_result.append("• **Pizza/Torta:** 'gráfico de pizza da coluna X'")
            analysis_result.append("• **Barras:** 'gráfico de barras da coluna X'")
            analysis_result.append("• **Histograma:** 'histograma da coluna X' (só numéricas)")
            analysis_result.append("• **Boxplot:** 'boxplot da coluna X' (só numéricas)")
            analysis_result.append("• **Linha:** 'gráfico de linha da coluna X'")
            analysis_result.append("• **Scatter:** 'scatter plot da coluna X' (só numéricas)")
            if text_cols:
                analysis_result.append(f"\n💡 EXEMPLO: 'Crie um gráfico de pizza da coluna {text_cols[0]}'")
            if numeric_cols:
                analysis_result.append(f"• 'Crie um boxplot da coluna {numeric_cols[0]}'")
        
        return "\n".join(analysis_result), generated_fig
    
    # Outras análises (sem gráfico)
    if any(word in question_lower for word in ['informação', 'básico', 'estrutura', 'colunas', 'linhas']):
        analysis_result.append("📊 INFORMAÇÕES BÁSICAS:")
        analysis_result.append(f"- Dataset com {len(df)} linhas e {len(df.columns)} colunas")
        analysis_result.append(f"- Colunas: {list(df.columns)}")
        analysis_result.append(f"- Tipos: {dict(df.dtypes)}")
    
    if any(word in question_lower for word in ['estatística', 'média', 'mediana', 'resumo', 'distribuição']):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis_result.append("\n📈 ESTATÍSTICAS DESCRITIVAS:")
            analysis_result.append(df[numeric_cols].describe().to_string())
        else:
            analysis_result.append("\n❌ Não há colunas numéricas para estatísticas")
    
    if any(word in question_lower for word in ['ausente', 'nulo', 'missing', 'faltante']):
        missing = df.isnull().sum()
        analysis_result.append("\n🔍 VALORES AUSENTES:")
        if missing.sum() == 0:
            analysis_result.append("✅ Não há valores ausentes!")
        else:
            for col, count in missing.items():
                if count > 0:
                    pct = (count / len(df)) * 100
                    analysis_result.append(f"- {col}: {count} ausentes ({pct:.1f}%)")
    
    if any(word in question_lower for word in ['correlação', 'relacionamento']):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            analysis_result.append("\n📊 MATRIZ DE CORRELAÇÃO:")
            analysis_result.append(corr_matrix.round(3).to_string())
        else:
            analysis_result.append("\n❌ Precisa de pelo menos 2 colunas numéricas")
    
    if any(word in question_lower for word in ['dados', 'tabela', 'preview', 'mostrar', 'ver']) and 'gráfico' not in question_lower:
        analysis_result.append("\n📊 VISUALIZAÇÃO DOS DADOS:")
        analysis_result.append(f"\nPrimeiras 10 linhas:")
        analysis_result.append(df.head(10).to_string())
        if len(df) > 10:
            analysis_result.append(f"\n... e mais {len(df) - 10} linhas no total")
    
    return "\n".join(analysis_result) if analysis_result else "🤔 Tente ser mais específico na sua pergunta.", None

# =====================================
# FUNÇÃO PRINCIPAL DO CHAT
# =====================================
def chat_with_eda_agent(user_input):
    """Função principal de chat com memória"""
    
    if any(word in user_input.lower() for word in ['conclus', 'resumo', 'descobriu', 'insights']):
        if st.session_state.conclusions:
            conclusion_text = "\n\n".join([
                f"🔍 **Pergunta:** {c['question']}\n💡 **Conclusão:** {c['conclusion']}"
                for c in st.session_state.conclusions[-3:]
            ])
            response_content = f"📋 **RESUMO DAS CONCLUSÕES:**\n\n{conclusion_text}"
        else:
            response_content = "🤔 Ainda não tenho conclusões. Faça perguntas sobre seus dados primeiro!"
        ai_response = AIMessage(content=response_content)
        generated_fig = None
    else:
        if st.session_state.df is not None:
            try:
                specific_analysis, generated_fig = analyze_data_with_question(user_input)
                
                system_prompt = """Você é um analista de dados especializado em EDA.

IMPORTANTE: 
- NUNCA retorne código Python ou instruções técnicas genéricas
- Foque apenas na interpretação dos resultados fornecidos
- Se um gráfico não pôde ser criado, explique o motivo de forma clara
- Seja direto e objetivo nas respostas
- Comente sobre padrões observados nos dados reais

REGRAS:
1. Se a análise mostra que um gráfico foi criado, comente sobre o gráfico
2. Se mostra erro de coluna não encontrada, explique quais colunas existem
3. NUNCA dê instruções sobre "como fazer" - apenas interprete os dados
4. Seja específico sobre os dados analisados

Com base na análise fornecida:

Análise realizada:
{specific_analysis}

Responda de forma clara, explicando o que aconteceu e dando insights sobre os dados disponíveis.
NÃO INCLUA INSTRUÇÕES TÉCNICAS OU CÓDIGO."""

                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{question}")
                ])
                
                chain = prompt_template | llm | StrOutputParser()
                response = chain.invoke({
                    "question": user_input,
                    "specific_analysis": specific_analysis
                })
                response = response.strip()
                if "娓" in response and "哨" in response:
                    response = response.split("哨")[-1].strip()
                response = response.replace("娓", "").replace("哨", "").strip()
                
                conclusion = {
                    "question": user_input,
                    "conclusion": response,
                    "timestamp": pd.Timestamp.now()
                }
                st.session_state.conclusions.append(conclusion)
                ai_response = AIMessage(content=response)
                
            except Exception as e:
                ai_response = AIMessage(content=f"❌ Erro: {str(e)}")
                generated_fig = None
        else:
            ai_response = AIMessage(content="📤 Faça upload de um CSV primeiro!")
            generated_fig = None
    
    return ai_response, generated_fig

# =====================================
# INTERFACE STREAMLIT
# =====================================

with st.sidebar:
    st.header("📤 Upload dos Dados")
    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            sample = uploaded_file.read(1024).decode('utf-8')
            uploaded_file.seek(0)
            separators = {',': sample.count(','), ';': sample.count(';'), '\t': sample.count('\t')}
            best_sep = max(separators, key=separators.get) if max(separators.values()) > 0 else ','
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, sep=best_sep, encoding=encoding)
                    if len(df.columns) > 1 and len(df) > 0:
                        break
                except:
                    continue
            st.session_state.df = df
            st.session_state.data_context = ""
            st.success(f"✅ Arquivo carregado!")
            st.info(f"📊 **{len(df)} linhas** e **{len(df.columns)} colunas**")
            with st.expander("👀 Preview dos Dados", expanded=True):
                st.dataframe(df.head(3), use_container_width=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Tipos únicos", df.dtypes.nunique())
                with col2:
                    missing_count = df.isnull().sum().sum()
                    st.metric("Valores nulos", missing_count)
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
            st.info("💡 Verifique se é um arquivo CSV válido")
    
    st.markdown("---")
    st.markdown("**💡 Recursos:**")
    st.markdown("""
    - Análise automática de CSV
    - Estatísticas descritivas
    - Gráficos automáticos
    - Memória de conclusões
    - Interface conversacional
    """)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat com o Agente")
    
    # Seção fixa de gráficos (opcional)
    plot_keys = [key for key in st.session_state.keys() if key.startswith("plot_") and st.session_state[key] is not None]
    if plot_keys:
        st.subheader("📊 Gráficos Gerados")
        for key in plot_keys:
            if st.session_state[key] is not None:
                st.plotly_chart(st.session_state[key], use_container_width=True, key=f"display_{key}")
        st.divider()
    
    # Exibir histórico do chat com gráficos integrados
    for message, fig in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, key=f"chat_fig_{id(fig)}")
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)

# Input do usuário
user_input = st.chat_input("Faça uma pergunta sobre seus dados...")

if user_input:
    st.session_state.chat_history.append((HumanMessage(content=user_input), None))
    
    with st.spinner("🧠 Analisando..."):
        response, generated_fig = chat_with_eda_agent(user_input)
    
    st.session_state.chat_history.append((response, generated_fig))
    
    # Forçar atualização para mostrar o novo gráfico imediatamente
    st.rerun()

with col2:
    st.subheader("📊 Status")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.metric("📄 Linhas", len(df))
        st.metric("📋 Colunas", len(df.columns))
        st.metric("🎯 Conclusões", len(st.session_state.conclusions))
        
        if st.button("👁️ Visualizar Dados Completos", use_container_width=True):
            st.session_state.show_full_data = not st.session_state.get('show_full_data', False)
        
        if st.session_state.get('show_full_data', False):
            st.subheader("📋 Dados Completos")
            col_a, col_b = st.columns(2)
            with col_a:
                n_rows = st.selectbox("Linhas:", [10, 25, 50, 100, "Todas"], index=0)
            with col_b:
                show_info = st.checkbox("Info técnica", value=False)
            if n_rows == "Todas":
                st.dataframe(df, use_container_width=True, height=400)
            else:
                st.dataframe(df.head(n_rows), use_container_width=True, height=400)
            if show_info:
                st.subheader("ℹ️ Informações Técnicas")
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write("**Tipos de Dados:**")
                    st.write(df.dtypes)
                with col_info2:
                    st.write("**Valores Ausentes:**")
                    missing = df.isnull().sum()
                    st.write(missing[missing > 0] if missing.sum() > 0 else "Nenhum")
        
        st.subheader("💡 Sugestões")
        st.markdown("""
        • *Informações básicas dos dados*
        • *Estatísticas principais*
        • *Há valores ausentes?*
        • *Correlações entre variáveis*
        • *Visualizar os dados*
        • *Gráfico da coluna X*
        • *Suas conclusões?*
        """)
        
        st.subheader("📋 Colunas")
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                st.write(f"🔢 {col}")
            else:
                st.write(f"📝 {col}")
    else:
        st.info("📤 Upload um CSV para começar!")
        st.markdown("""
        **Recursos disponíveis:**
        - Análise automática de dados
        - Gráficos interativos
        - Estatísticas detalhadas
        - Memória conversacional
        """)

if 'show_full_data' not in st.session_state:
    st.session_state.show_full_data = False

st.markdown("---")
st.markdown("*🤖 Agente EDA com Groq API - LangChain + Streamlit*")