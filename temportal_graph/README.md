# `temportal_graph`: grafo temporal para previsao de ameacas

Este modulo constroi e treina um grafo temporal heterogeneo para prever, a partir de mensagens de grupos de Telegram, o volume e os tipos de incidentes ciberneticos reportados pelo Hackmageddon no dia seguinte.

Apesar dos nomes internos usarem `weekly`, a tarefa atual nao e uma previsao semanal agregada. O que existe hoje e uma janela movel diaria de 7 dias: para cada dia de snapshot `D`, o modelo recebe as mensagens observadas de `D-6` ate `D` e tenta prever os eventos Hackmageddon reportados em `D+1`.

## Como reproduzir

Os dois comandos principais sao:

```bash
uv run build_weekly_graph.py
uv run train_weekly.py
```

O primeiro comando gera:

- `temportal_graph/output/weekly_dataset.pt`: objeto `HeteroData` do PyTorch Geometric.
- `temportal_graph/output/weekly_metadata.json`: estatisticas do grafo, distribuicoes e parametros de construcao.

O segundo comando gera:

- `temportal_graph/output/weekly_metrics.json`: metricas de treino, validacao e teste.
- `temportal_graph/output/weekly_training_history.png`
- `temportal_graph/output/weekly_val_count_confusion_matrix.png`
- `temportal_graph/output/weekly_test_count_confusion_matrix.png`
- `temportal_graph/output/weekly_test_type_metrics.png`

Tambem existe uma visualizacao HTML:

```bash
uv run visualize_weekly.py
```

Ela le `output/weekly_dataset.pt` e escreve `output/graph_viz.html`.