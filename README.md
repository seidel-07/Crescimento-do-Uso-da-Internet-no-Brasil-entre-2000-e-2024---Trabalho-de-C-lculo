
# Correlação e Regressão Linear: uso da internet no Brasil

Projeto acadêmico do curso de Sistemas de Informação do Instituto Federal do Espírito Santo (IFES), que analisa a evolução do uso da internet no Brasil entre 2000 e 2024 utilizando Python.

## Objetivo

Analisar a relação entre o tempo e a porcentagem da população brasileira que utiliza internet por meio de técnicas de estatística descritiva, correlação de Pearson e regressão linear.

O projeto também calcula resíduos, realiza previsões e testa o modelo utilizando dados de 2000 a 2019 para estimar os valores de 2020 a 2024.

## Tecnologias utilizadas

- Python
- Pandas
- Matplotlib
- SciPy
- Scikit-learn

## Principais resultados

- Correlação de Pearson: 0,9961.
- Coeficiente de determinação: 0,9922.
- Equação da reta: ŷ = 4,1898 + 3,5996x.
- Aumento médio estimado: 3,60 pontos percentuais por ano.
- Erro absoluto médio no teste de 2020 a 2024: 4,61 pontos percentuais.

Os resultados indicam uma correlação positiva muito forte entre o tempo e o uso de internet. Entretanto, a correlação não comprova causalidade, e a regressão linear apresenta limitações para previsões futuras devido à desaceleração do crescimento.

## Estrutura do projeto

```text
correlacao-regressao-internet-brasil/
├── analise_internet_brasil.py
├── README.md
├── requirements.txt
├── dados/
│   └── uso_internet_brasil.csv
└── graficos/
    ├── grafico_dispersao.png
    ├── grafico_reta.png
    └── grafico_residuos.png
```

Os gráficos são gerados automaticamente pelo programa.

## Como executar

Instale o Python e as dependências necessárias:

```bash
python -m pip install -r requirements.txt
```

Execute o programa:

```bash
python analise_internet_brasil.py
```

Os resultados serão exibidos no terminal, e os três gráficos serão salvos na pasta `graficos`.

## Fonte dos dados

União Internacional de Telecomunicações (ITU) e Banco Mundial.

Indicador: IT.NET.USER.ZS.

https://data.worldbank.org/indicator/IT.NET.USER.ZS?locations=BR
