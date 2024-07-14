## Processo di Addestramento della Main Network in DQN

### Setup Iniziale

- **Definizione dell'Ambiente e della Rete**:
  - L'ambiente fornisce gli stati e le ricompense basati sulle interazioni dell'agente.
  - La rete principale stima i Q-values basati sugli stati osservati.

### Loop di Addestramento per ogni Episodio

Per ogni episodio di addestramento:

1. **Reset dell'Ambiente**:
   - Inizia un nuovo episodio e ottieni lo stato iniziale dell'ambiente.
   - **Codice**:
     ```python
     stato = env.reset()
     ```

2. **Iterazione su ogni Step dell'Episodio**:
   - Per ogni step del episodio, esegui le seguenti operazioni:

   #### Selezione dell'Azione
   - **Decisione dell'Azione** (Politica ?-greedy):
     - Seleziona un'azione in modo casuale con probabilità ? (esplorazione) o l'azione che massimizza i Q-values stimati con probabilità 1-? (sfruttamento).
     - **Codice**:
       ```python
       azione = epsilon_greedy_policy(stato, epsilon, num_actions)
       ```

   #### Interazione con l'Ambiente
   - **Esecuzione dell'Azione**:
     - Applica l'azione selezionata all'ambiente.
     - Ricevi il nuovo stato, la ricompensa e l'informazione se l'episodio è terminato.
     - **Codice**:
       ```python
       nuovo_stato, ricompensa, terminato, info = env.step(azione)
       ```

   #### Aggiornamento del Buffer di Replay
   - **Archiviazione dell'Esperienza**:
     - Salva l'esperienza (stato, azione, ricompensa, nuovo stato, terminato) nel buffer di replay.
     - **Codice**:
       ```python
       buffer_di_replay.push(stato, azione, ricompensa, nuovo_stato, terminato)
       ```

   #### Campionamento dal Buffer di Replay
   - **Estrazione di un Batch di Esperienze**:
     - Campiona un batch di esperienze dal buffer di replay per l'addestramento della rete.
     - **Codice**:
       ```python
       batch = buffer_di_replay.sample(batch_size)
       ```

   #### Aggiornamento della Rete Principale
   - **Calcolo dei Target Q-values**:
     - Utilizza la rete target per calcolare i Q-values target per le azioni nel batch.
     - **Codice**:
       ```python
       Q_target = compute_targets(batch, network_target)
       ```
   - **Ottimizzazione della Rete Principale**:
     - Aggiorna i pesi della rete principale per minimizzare la differenza tra i Q-values stimati e i Q-values target.
     - **Codice**:
       ```python
       network_main.train_on_batch(batch_states, Q_target)
       ```

   #### Aggiornamento Condizionale della Rete Target
   - **Sincronizzazione delle Reti**:
     - Ogni N step, aggiorna i pesi della rete target per allinearli con quelli della rete principale.
     - **Codice**:
       ```python
       if step % update_target_frequency == 0:
           update_target_network(network_main, network_target)
       ```

   #### Check di Terminazione
   - **Verifica se l'Episodio è Terminato**:
     - Se `terminato` è vero, interrompi il loop e inizia un nuovo episodio.
     - **Codice**:
       ```python
       if terminato:
           break
       ```

### Conclusione del Loop Episodico

- Al termine di ogni episodio, aggiorna eventuali metriche di performance, logga i risultati, e preparati per il prossimo episodio.
- Adegua il valore di ? (epsilon) per ridurre gradualmente l'esplor





