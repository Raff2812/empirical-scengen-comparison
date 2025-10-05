## Pipeline di analisi su dati nuovi

### Rigenerazione dei dataset su dati nuovi
Per analizzare **nuovi log arricchiti** (da `data_gathering/enriching/orchestrator.py`) è necessario:

1. **Modificare `pipeline/run_comparison.py`**
   Decommentare la parte relativa a `run_clustering` per riattivare l’esecuzione completa del clustering.

2. **Modificare `pipeline/clustering.py`**
   Nella funzione `run_clustering()`, decommentare il blocco di codice iniziale che carica e salva i dataset:

   ```python
   def run_clustering():
       # Load and prepare data
       #critical_df, full_df = load_dataset(Path("../logs/enriched")) #punta alla cartella in cui sono salvati i nuovi logs

       # Save datasets
       #full_df.to_csv("../datasets/full_dataset.csv", index=False)
       #critical_df.to_csv("../datasets/critical_dataset.csv", index=False)

       # Load critical dataset
       critical_df = pd.read_csv("../datasets/critical_dataset.csv")
       ...
   ```

   Questo permetterà di rigenerare i dataset a partire dai log nuovi.
La rigenerazione sovrascriverà i file nella cartella `datasets/`:

* `full_dataset.csv` → tutti gli scenari (critici e non)
* `critical_dataset.csv` → solo scenari con collisioni/violazioni
* `full_dataset_with_clusters.csv` → tutti gli scenari con assegnazione cluster

---

## Avvio effettivo dell'analisi

Dopo aver seguito le istruzioni di sopra, basta semplicemente spostarsi nella cartella **pipeline** ed eseguire il file **run_comparison.py**.

---
## Output generati

L’esecuzione della pipeline produce diversi output suddivisi in due cartelle principali:

1. **`clustering_results/`**

   * `clustering_metrics_comparison.svg` → rappresentazione grafica della scelta del numero ottimale di cluster.
   * `clustering_results.json` → valori di Silhouette e WCSS per ciascun numero di cluster considerato.
   * `plotting/` → visualizzazioni dei cluster trovati con K-Means.

2. **`results/`**

   * `latent_risk_summary.csv` → statistiche descrittive sulle metriche di near-miss per ciascun tool.
   * `summary_table.csv` → sintesi dei risultati relativi alla criticità.
   * `tool_scores.json` → risultati aggregati delle tre dimensioni: criticità, diversità ed efficacia.
   * `graphs/` → grafici di supporto per l’interpretazione dei dati numerici (criticità, diversità, efficacia, outcome, rischio latente).
