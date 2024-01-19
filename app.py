import pickle

import folium
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from streamlit_folium import folium_static


def base_dados():
    """Carrega a base de dados de alugueis."""
    dados = pd.read_csv('./dados/house_to_rent_tratado.csv')
    return dados


def modelo_regressao():
    """Carrega o modelo de regressão linear."""
    modelo = pickle.load(open('./modelo/modelo_regressao.sav', 'rb'))
    return modelo


def pre_processamento(
    cidade,
    area,
    quartos,
    banheiros,
    vagas,
    andar,
    animais,
    mobilia,
    codominio,
    iptu,
    seguro,
):
    """Pré processamento dos dados para aplicação do regressor."""

    # Nova linha
    nova_linha = {
        'cidade': cidade,
        'area': area,
        'quartos': quartos,
        'banheiros': banheiros,
        'vagas': vagas,
        'andar': andar,
        'animais': animais,
        'mobilia': mobilia,
        'condominio': codominio,
        'iptu': iptu,
        'seguro': seguro,
    }
    nova_linha = pd.DataFrame(nova_linha, index=[0])
    dados = base_dados()
    dados = dados.drop(['aluguel', 'total'], axis=True)
    dados = pd.concat([dados, nova_linha], ignore_index=True)

    # Definindo variáveis com features numéricas e categóricas
    variavel_num = [
        'area',
        'quartos',
        'banheiros',
        'vagas',
        'andar',
        'animais',
        'mobilia',
        'condominio',
        'iptu',
        'seguro',
    ]
    variavel_cat = ['cidade']

    # Definido transformações com estimador final
    num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

    # Criando o pré-processador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, variavel_num),
            ('cat', cat_transformer, variavel_cat),
        ]
    )

    # Aplicando o pré-processamento
    processed = preprocessor.fit_transform(dados)
    return processed


def regressor(dados):
    """Aplicando o modelo de regressao linear."""
    modelo = modelo_regressao()
    return modelo.predict(dados)[len(dados) - 1]


def mapa_cidades(dados):
    """Cria um mapa com as cidades."""
    coordenadas_centro = [-23.5489, -46.6388]

    mapa = folium.Map(
        location=coordenadas_centro, zoom_start=5, prefer_canvas=True
    )

    localizacao = dados

    for loc in localizacao:
        popup_html = f"""
            <h4 style="margin:0;padding:0;">{loc['cidade']}</h4>
            <p style="margin:0;padding:0;">Total de Imóveis: {loc['qtd_imoveis']}</p>
            <p style="margin:0;padding:0;">Média de Alugueis: R$ {loc['media_aluguel']}</p>
            <p style="margin:0;padding:0;">Valor do M²: R$ {loc['metro_quadrado']}</p>
        """
        folium.Marker(
            location=loc['location'],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color='red', icon='ok-sign'),
        ).add_to(mapa)

    folium_static(mapa)


def main():
    st.set_page_config(page_title='Brazil Rent', page_icon='🏘')
    st.markdown(
        "<h3 style='text-align:center; font-family:Verdana'>Consulta de Alugueis</h3>",
        unsafe_allow_html=True,
    )
    st.header('', divider='red')
    st.sidebar.image('./img/logos/logo_app.png', use_column_width='always')
    st.sidebar.markdown(
        "<h4 style='text-align:center; font-family:Verdana'>Selecione as Configurações do Imóvel Usando os Filtros:</h4>",
        unsafe_allow_html=True,
    )
    # st.sidebar.header('', divider='red')

    # Filtros
    cidade = st.sidebar.selectbox(
        'Selecione a Cidade:',
        [
            'Belo Horizonte',
            'Campinas',
            'Porto Alegre',
            'Rio de Janeiro',
            'São Paulo',
        ],
    )
    area = st.sidebar.slider('Área do Imóvel m²:', 11, 500, 78)
    quartos = st.sidebar.slider('Quantidade de Quartos:', 1, 5, 2)
    banheiros = st.sidebar.slider('Quantidade de Banheiros:', 1, 6, 2)
    vagas = st.sidebar.slider('Vagas de Estacionamento:', 0, 4, 1)
    andar = st.sidebar.slider('Andar:', 0, 20, 3)
    coluna1, colona2 = st.sidebar.columns(2)
    animais_filtro = coluna1.radio('Aceita animais?', ['Sim', 'Não'], index=1)
    mobilia_filtro = colona2.radio('é Mobiliado?', ['Sim', 'Não'], index=1)
    condominio = st.sidebar.slider('Valor do Condomínio:', 0, 2800, 530)
    iptu = st.sidebar.slider('Valor do IPTU:', 0, 876, 94)
    seguro = st.sidebar.slider('Valor do Seguro Contra Incendio:', 3, 136, 30)

    animais = 0
    if animais_filtro == 'Sim':
        animais = 1

    mobilia = 0
    if mobilia_filtro == 'Sim':
        mobilia = 1

    if st.sidebar.button('CONSULTAR'):
        processed = pre_processamento(
            cidade,
            area,
            quartos,
            banheiros,
            vagas,
            andar,
            animais,
            mobilia,
            condominio,
            iptu,
            seguro,
        )
        previsao = round(regressor(processed), 2)
        aluguel = f'R$ {previsao:.2f}'
        texto = f'Com base nas informações selecionadas, o valor do aluguel previsto para cidade de {cidade}:'
        st.markdown(texto)
        cl1, cl2, cl3 = st.columns(3)
        cl2.header(aluguel)

        fixa = {
            'Cidade': cidade,
            'Área m²': area,
            'Quantidade de Quartos': quartos,
            'Quantidade de Banheiros': banheiros,
            'Quantidade de Vagas': vagas,
            'Andar': andar,
            'Aceita Animais': animais_filtro,
            'Mobilia': mobilia_filtro,
            'Valor do Condomínio': condominio,
            'Valor do IPTU': iptu,
            'Valor do Seguro Incendio': seguro,
        }
        c1, c2, c3 = st.columns([0.6, 2.3, 0.2])
        c2.write('Configuração do Imóvel:')
        c2.dataframe(fixa, width=400, height=422)
        st.header('', divider='red')

    dados = base_dados().copy()
    dados['metro_quadrado'] = round(dados['total'] / dados['area'], 2)
    metro_quadrado = round(dados.groupby('cidade')['metro_quadrado'].mean(), 2)
    media_aluguel = round(dados.groupby('cidade')['aluguel'].mean(), 2)
    total_imoveis = dados['cidade'].value_counts()
    total_imoveis = dados['cidade'].value_counts().reset_index()
    total_imoveis.columns = ['cidade', 'total']
    total_imoveis = total_imoveis.sort_values('cidade')

    localizacao = [
        {
            'cidade': 'Belo Horizonte',
            'location': [-19.9167, -43.9345],
            'qtd_imoveis': total_imoveis.loc[3][1],
            'media_aluguel': media_aluguel[0],
            'metro_quadrado': metro_quadrado[0],
        },
        {
            'cidade': 'Campinas',
            'location': [-22.9064, -47.0616],
            'qtd_imoveis': total_imoveis.loc[4][1],
            'media_aluguel': media_aluguel[1],
            'metro_quadrado': metro_quadrado[1],
        },
        {
            'cidade': 'Porto Alegre',
            'location': [-30.0277, -51.2287],
            'qtd_imoveis': total_imoveis.loc[2][1],
            'media_aluguel': media_aluguel[2],
            'metro_quadrado': metro_quadrado[2],
        },
        {
            'cidade': 'Rio de Janeiro',
            'location': [-22.9035, -43.2096],
            'qtd_imoveis': total_imoveis.loc[1][1],
            'media_aluguel': media_aluguel[3],
            'metro_quadrado': metro_quadrado[3],
        },
        {
            'cidade': 'São Paulo',
            'location': [-23.5489, -46.6388],
            'qtd_imoveis': total_imoveis.loc[0][1],
            'media_aluguel': media_aluguel[4],
            'metro_quadrado': metro_quadrado[4],
        },
    ]

    st.markdown(
        "<h5 style='text-align:center; font-family:Verdana'>Dados Observados Por Cidade</h5>",
        unsafe_allow_html=True,
    )
    mapa_cidades(localizacao)
    st.header('', divider='red')
    st.caption('versão 1.0')


if __name__ == '__main__':
    main()
