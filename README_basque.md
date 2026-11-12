# basque_experiment.py: guía de ejecución

Experimento PQK en `ibm_basquecountry`: los mismos circuitos en la mejor y en la peor cadena de 6 qubits del chip, con 100 puntos de entrenamiento y 50 de prueba de Sonar, para medir cuánto recupera el kernel la selección de observables basada en la calibración. Es la versión con presupuesto del experimento de octubre en Marrakesh y Kingston.

**Regla de oro:** el script no gasta un segundo de QPU sin imprimir la estimación y pedir que escriban `YES`. Si la estimación supera lo que queda de los 50 minutos, se detiene solo.

## 1. Instalación (una vez)

Descomprimir `pqk_basque.zip` (o clonar el repositorio) y, dentro de la carpeta:

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements_basque.txt
```

La cuenta de IBM Quantum debe estar guardada localmente con el nombre `unai` (`QiskitRuntimeService(name="unai")`). Si usan otro nombre, cambien `"service_name"` en `basque_config.json`. El script no pide ni guarda tokens.

## 2. Comprobaciones sin QPU (antes de la primera sesión)

```bash
python basque_experiment.py verify
```
Reanaliza los conteos crudos de octubre con este código y compara número a número. Debe terminar en `RESULT: PASS`. Si no, hay una diferencia de versiones que conviene mirar antes de gastar.

```bash
python basque_experiment.py dry-run --set n_train=24 n_test=12
```
Todo el flujo (cadenas, circuitos, chunks, análisis) en el simulador ruidoso FakeMarrakesh. Unos dos minutos en una laptop, cero QPU. Escribe `results/basque_basque_dry.json` y dos figuras; ese archivo nunca se confunde con uno real.

## 3. La configuración: `basque_config.json`

| clave | valor por defecto | qué es |
|---|---|---|
| `n_qubits` | 6 | 6 → 153 observables y 15 bases; 12 → 630 y 19; 20 → 1770 y 25 |
| `n_train`, `n_test` | 100, 50 | puntos de Sonar medidos en hardware |
| `shots` | 1000 | como en octubre (etapas PQK) |
| `fid_shots` | `"matched"` | shots por circuito de fidelidad: `"matched"` = el mismo total de shots que el PQK repartido entre los n² circuitos (226 con 150 puntos); un entero; o `null` = igual que `shots` |
| `stages` | `["pqk_good","pqk_bad"]` | añadir `"fid_good","fid_bad"` para el kernel de fidelidad: mismos puntos, misma cadena, mismo total de shots que el PQK |
| `max_circuits_per_job` | 1000 | el tope actual de la máquina |
| `budget_minutes` | 50 | cuota mensual |
| `qpu_safety_factor` | 1.5 | toda estimación se multiplica por esto antes de comparar con la cuota |
| `chains` | `null` | el script elige la mejor y la peor cadena de hoy; `{"good":[...],"bad":[...]}` para fijarlas |

Cualquier clave se puede cambiar sin editar el archivo: `--set n_qubits=12 n_train=60`.

## 4. Los comandos, en el orden de uso

```bash
python basque_experiment.py estimate                   # circuitos, jobs y minutos por etapa contra la cuota restante. Sin QPU.
python basque_experiment.py plan                       # lee la calibración de hoy y muestra la mejor y la peor cadena. Sin QPU.
python basque_experiment.py submit --stages pqk_good   # envía UNA etapa en Batch mode (3 jobs de ≤1000 circuitos). Pide YES.
python basque_experiment.py fetch                      # recoge los conteos de los jobs terminados y los segundos cobrados.
python basque_experiment.py submit --stages pqk_bad    # la otra cadena; puede ser otro día.
python basque_experiment.py fetch
python basque_experiment.py analyze                    # opcional aquí; lo corremos nosotros con el archivo que nos envíen.
```

Costos estimados con el factor ×1.5 (reales ≈ dividir entre 1.5): `pqk_good` 15.7 min, `pqk_bad` 15.7 min; los dos, 31.4 de los 50. Cada etapa de fidelidad a presupuesto igual (`fid_shots: matched`) cuesta 16.1 min más: las cuatro etapas con 100 + 50 puntos son 64 min (dos meses); con 60 + 30 puntos son 38 min y caben en uno. `estimate` lo muestra para cualquier combinación.

## 5. Qué hace `submit` para proteger la cuota

- Congela el diseño (puntos, cadenas, backend, shots) en `results/basque_basque.json` la primera vez. Las etapas siguientes, aunque sean otro día, miden los mismos puntos en los mismos qubits; cambiar `n_train` o `shots` con un archivo existente se rechaza.
- Cada job lleva al frente 2 circuitos de readout check (|0…0⟩ y |1…1⟩) y etiquetas con el experimento, una huella del diseño, la etapa y el chunk.
- El ID de cada job se guarda antes de enviar el siguiente. Si se corta la conexión, volver a ejecutar el mismo `submit` reenvía solo los chunks que faltan; nada se envía dos veces. Un ID que el servidor tiene y el archivo perdió se recupera por las etiquetas.
- `fetch` marca un chunk en `ERROR` como fallido y lo dice en pantalla; `submit --stages <etapa>` reenvía solo ese chunk.
- `fetch` guarda los segundos de QPU cobrados por cada job (`usage_by_job`). Tras la primera etapa comparamos ese número con la estimación y ajustamos el modelo si hace falta.

## 6. Lo que nos envían

Un solo archivo: **`results/basque_basque.json`**. Contiene configuración, puntos y etiquetas, cadenas, calibración y readout de cada chunk, IDs y etiquetas de los jobs, segundos cobrados y los conteos crudos de cada circuito. Con él rehacemos el análisis completo y es el material suplementario del artículo. Si corren `analyze`, las figuras quedan en `figures/basque_basque_*.png` y el resumen en `results/basque_basque_analysis.json`; no hace falta enviarlos.

Si algo falla, envíennos la salida de pantalla del comando: dice qué etapa, qué chunk y qué job.

Contacto: David (d.vicho@hotmail.com).
