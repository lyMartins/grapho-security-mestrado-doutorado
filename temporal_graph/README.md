# `temporal_graph`: grafo temporal para previsao de ameacas

Este modulo constroi e treina um grafo temporal heterogeneo para prever, a partir de mensagens de grupos de Telegram, o volume e os tipos de incidentes ciberneticos reportados pelo Hackmageddon no dia seguinte.

Apesar dos nomes internos usarem `weekly`, a tarefa atual nao e uma previsao semanal agregada. O que existe hoje e uma janela movel diaria de 7 dias: para cada dia de snapshot `D`, o modelo recebe as mensagens observadas de `D-6` ate `D` e tenta prever os eventos Hackmageddon reportados em `D+1`.

## Como reproduzir

Os dois comandos principais sao:

```bash
uv run build_weekly_graph.py
uv run train_weekly.py
```

O primeiro comando gera:

- `temporal_graph/output/weekly_dataset.pt`: objeto `HeteroData` do PyTorch Geometric.
- `temporal_graph/output/weekly_metadata.json`: estatisticas do grafo, distribuicoes e parametros de construcao.

O segundo comando gera:

- `temporal_graph/output/weekly_metrics.json`: metricas de treino, validacao e teste.
- `temporal_graph/output/weekly_training_history.png`
- `temporal_graph/output/weekly_count_bucket_metrics.png`
- `temporal_graph/output/weekly_type_multilabel_metrics.png`
- `temporal_graph/output/weekly_type_count_regression_metrics.png`
- `temporal_graph/output/weekly_val_count_confusion_matrix.png`
- `temporal_graph/output/weekly_test_count_confusion_matrix.png`
- `temporal_graph/output/weekly_test_type_metrics.png`

Por padrao, `train_weekly.py` usa `--split-policy stratified`: treino,
validacao e teste sao amostrados de forma reprodutivel, estratificados pelo
bucket de contagem futura (`y_future_count_bucket`). As politicas cronologicas
antigas continuam disponiveis via `--split-policy chronological` e
`--split-policy balanced_chronological` para comparacao.

Tambem existe uma visualizacao HTML:

```bash
uv run visualize_weekly.py
```

Ela le `output/weekly_dataset.pt` e escreve `output/graph_viz.html`.
