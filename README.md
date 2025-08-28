## Predição de Nota do IMDB — EDA + Modelagem (Ridge)
### Repositório do desafio de Ciência de Dados com:
- EDA completa do dataset de filmes;
- Análises sobre a coluna Overview (texto) e termos por gênero;
### Modelagem para prever a nota do IMDB em dois cenários:
- A (pré-lançamento) e B (pós-lançamento);
- Pipeline final salvo em .pkl pronto para inferência.
------------------------------------------
### Sumário
- Requisitos e instalação
- Como executar (EDA + Modelagem)
- Treinar e salvar o modelo .pkl
- Como usar o modelo salvo para prever
- Resultados (métricas)
- Checklist de entrega exigida
- Boas práticas adotadas
  
-------------------------------------------

###  Requisitos e instalação
- Testado com Python 3.12, scikit-learn 1.6.1.
#### Clone o repositório:
- git clone https://github.com/<seu-usuario>/<seu-repo>.git
- cd <seu-repo>
#### (Opcional) Crie e ative um ambiente virtual:
- Windows (PowerShell)
- python -m venv .venv
- .venv\Scripts\Activate.ps1
#### macOS / Linux
- python3 -m venv .venv
- source .venv/bin/activate
#### Instale as dependências:
- pip install -r requirements.txt
- O requirements.txt fixa as versões usadas e garante reprodutibilidade.
  
-------------------------------------------------------------

### Como executar (EDA + Modelagem)
#### Abra o notebook principal e execute as células na ordem:
- jupyter notebook notebooks/LH_CD_VICTOR_TINTEL_MARTINS.ipynb

#### No notebook você encontrará:

##### Passo 1 — EDA
- Estatísticas, distribuições, correlações (Spearman), análise por Gênero e Classificação etária, outliers plausíveis vs. erros (ex.: duração 321 → corrigido para 240), além de boxplots/histogramas.
##### Passo 2 — Overview (texto)
- Tamanho das sinopses em palavras/caracteres;
- TF-IDF (1–2-grams) + χ² para levantar termos característicos por gênero;
- Conclusão: é possível extrair sinais de gênero do Overview, mas ele não é perfeito sozinho (ambiguidade, classes raras).
##### Passo 3 — Modelagem da Nota do IMDB
- Cenário A (pré-lançamento): só variáveis disponíveis antes do filme “existir” publicamente.
- Features: Duracao_min, idade_filme (= 2025 − Ano), ovw_len_palavras, ovw_len_chars, genero_prim, class_std, overview_txt (TF-IDF).
- Modelo: Ridge Regression (linear com L2).
- Cenário B (pós-lançamento): adiciona log_votos, log_fat e Media_Ponderada_Criticas.
- Modelo: Ridge Regression.
- Validação: KFold(5) com métricas RMSE, MAE e R².
##### Passo 4 — Previsão exemplo
- “The Shawshank Redemption”: nota prevista ~ 8,31 (intervalo ~95%: [7,85; 8,77]).
------------------------------------------------
### Treinar e salvar o modelo .pkl
- No final do notebook há as células para treinar o Cenário B com 100% dos dados e salvar o pipeline com joblib:
- Arquivo gerado: models/imdb_model_ridgeB.pkl
- O pacote inclui metadados (versões, data/hora, features esperadas) para auditoria.
- Se preferir recriar: execute as células “Treinar o pipeline final (Cenário B)…” e “Salvar com joblib”.
-------------------------------------------------
### Como usar o modelo salvo para prever
- O pipeline salvo espera as mesmas features derivadas do treino:
- Numéricas: Duracao_min, idade_filme, ovw_len_palavras, ovw_len_chars,
log_votos, log_fat, Media_Ponderada_Criticas
- Categóricas: genero_prim, class_std
- Texto: overview_txt (TF-IDF aplicado dentro do pipeline)
------------------------------------------------------
### Resultados (métricas)
- Validação cruzada KFold(5):
- Cenário A (pré-lançamento)
RMSE 0.263 ± 0.018 | MAE 0.209 | R² 0.043
- Cenário B (pós-lançamento)
RMSE 0.231 ± 0.010 | MAE 0.183 | R² 0.265
- Modelo: Ridge Regression (linear com regularização L2).
- Por que Ridge? Escala bem com TF-IDF (muitas colunas), é robusto a multicolinearidade e rápido para validar por KFold.
------------------------------------------------------
### Checklist de entrega exigida
- README explicando como instalar e executar (este arquivo)
- requirements.txt com pacotes e versões
- Relatórios EDA/estatística — notebooks/LH_CD_VICTOR_TINTEL_MARTINS.ipynb
(opcional: exportar em PDF/HTML via nbconvert)
- Códigos de modelagem (no mesmo notebook)
- Arquivo .pkl — models/imdb_model_ridgeB.pkl
----------------------------------------------------------
Boas práticas adotadas
- Pipeline + ColumnTransformer para pré-processamento + modelo de ponta a ponta;
- TF-IDF (1–2-grams) com min_df=3, max_features=40000 para Overview;
- Padronização de categorias (class_std) e extrações de texto (overview_txt) ao longo do fluxo;
- Versões travadas no requirements.txt e metadados junto ao .pkl;
- Reprodutibilidade: random_state=42, KFold(5), comentários e células separadas por cenários.
----------------------------------------------------------
