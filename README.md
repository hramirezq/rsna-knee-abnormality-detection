# Detección multietiqueta de anomalías de rodilla en RM

Clasificador CNN que predice **12 hallazgos** a partir de los estudios de resonancia magnética (DICOM) de la
competencia de Kaggle [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

Solo 58 estudios traen etiqueta experta (*gold*). El resto trae únicamente el reporte radiológico en texto libre,
escrito en unos 12 idiomas. Por eso el proyecto tiene dos partes:

1. **Etiquetas débiles:** un extractor de reglas multilingües lee cada reporte y produce las 12 etiquetas, con una confianza para cada una.
2. **Clasificador de imágenes:** una CNN 2.5D (desde cero o ResNet preentrenada en ImageNet) se entrena con esas etiquetas y se evalúa sobre los 58 gold.

Hallazgos: `ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`.

## El desafío de Kaggle

| | |
|---|---|
| Competencia | https://www.kaggle.com/competitions/rsna-knee-abnormality-detection |
| Datos | pestaña [Data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data): ≈570 GB, 819 640 DICOM, `studies → series → instances` |
| Reglas | pestaña [Rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules): hay que aceptarlas para descargar los datos |
| Métrica | AUC-ROC macro: promedio simple del AUC de los 12 hallazgos |
| Formato | *Code Competition*: el notebook de envío corre en Kaggle con internet apagado y escribe `submission.csv` |

Los datos de la competencia **no están en este repositorio**, por su tamaño y por las reglas de uso.

## Estructura del repositorio

```
.
├── rsna_knee_kaggle.ipynb   # todo el pipeline en un solo notebook (5 modos, ver "Ejecución")
├── requirements.txt         # versiones probadas (Python 3.13)
├── output-data/             # corrida 1 en Kaggle (GPU T4, 600 estudios, semilla 42)
│   ├── resumen_experimentos.csv    # AUC macro por experimento, con IC 95 %
│   ├── auc_por_hallazgo.csv        # AUC de cada hallazgo, por experimento
│   ├── metricas_por_hallazgo.csv   # umbral, precisión, recall y F1, por experimento y hallazgo
│   ├── calidad_etiquetas.csv       # extractor de reportes vs los 58 gold, por hallazgo
│   ├── calidad_por_idioma.csv      # extractor de reportes, por idioma
│   ├── prevalencia.csv             # % de positivos por hallazgo en train, holdout y gold
│   ├── experimentos.png            # comparación gráfica de los experimentos
│   ├── gradcam.png                 # mapas Grad-CAM de 4 hallazgos
│   └── knee-train.log              # log completo de la corrida en Kaggle
├── output-data-v2/          # corrida 2: los 7 experimentos con las semillas 42, 1 y 2
│   ├── semillas.csv                # AUC macro y por hallazgo, por experimento y semilla
│   ├── resumen_experimentos.csv    # igual al de la corrida 1 salvo los minutos (la semilla 42 se reproduce exacta)
│   ├── knee-train-v2.ipynb         # notebook ejecutado en Kaggle
│   └── knee-train-v2.log           # log completo de la corrida 2
├── informe/                 # paper en formato IEEE (plantilla del curso)
│   ├── main.tex                    # paper completo; compilar con `tectonic main.tex` o en Overleaf
│   ├── referencias.bib
│   └── figs/                       # copias de experimentos.png y gradcam.png
└── work/                    # (ignorado por git) datos sintéticos y salidas de la corrida local
```

Algunos archivos se generan pero no se versionan (ver `.gitignore`): los checkpoints `output-data*/*.pt`
(≈840 MB entre las dos corridas), `output-data*/weak_labels.csv`, los *wheels* de `pylibjpeg` y toda la carpeta `work/`.
Los demás CSV y PNG de la corrida 2 son copias exactas de los de `output-data/`, porque la semilla 42 es determinista.

## Ejecución

### 1. Local, con datos sintéticos (un comando)

No necesita los datos de Kaggle. El notebook genera DICOM y reportes falsos con el mismo esquema que los reales
y corre el pipeline completo: etiquetas, caché, 7 experimentos, métricas y Grad-CAM. Sirve para comprobar que todo
funciona y para estudiar el código. Los números que da **no tienen valor clínico**: las imágenes son ruido con discos brillantes.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/jupyter nbconvert --to notebook --execute rsna_knee_kaggle.ipynb --output executed.ipynb
```

- Tarda ≈10–15 min en CPU. La primera vez descarga los pesos de ResNet-18/50 (≈150 MB), así que necesita internet.
- Las salidas quedan en `work/`. Si la última celda imprime `OK: pipeline sintético completo`, todo funcionó.
- Para recorrerlo celda por celda: `.venv/bin/jupyter notebook rsna_knee_kaggle.ipynb`.

### 2. En Kaggle, con los datos reales

Se usa el mismo notebook, importado en Kaggle, cambiando solo `MODE` en la primera celda. La salida de cada
notebook es el *Input* del siguiente:

| Notebook en Kaggle | `MODE` | Máquina | Inputs | Salida |
|---|---|---|---|---|
| `knee-preprocess` | `preprocess` | CPU, Internet ON | competencia | caché `train_vols.npy` (≈10.6 GB), `train_masks.npy`, `train_ids.npy` |
| `knee-labels` (opcional) | `labels` | CPU | competencia | calidad del extractor; sirve para iterar el léxico |
| `knee-train` | `train` | GPU T4, Internet ON | competencia + `knee-preprocess` | todo lo de `output-data/` + `semillas.csv` + checkpoints |
| `knee-submit` (opcional) | `infer` | GPU, Internet OFF | competencia + `knee-train` | `submission.csv` |

Pasos en Kaggle:

1. Verificar el teléfono en la cuenta (sin eso no se habilitan la GPU ni el internet) y aceptar las [reglas](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules) de la competencia.
2. **+ Create → New Notebook → File → Import Notebook** y subir `rsna_knee_kaggle.ipynb`.
3. Agregar los Inputs de la tabla con **+ Add Input** y configurar la máquina en **Session options**.
4. Cambiar `MODE` y hacer una prueba corta: `LIMIT = 50` en `preprocess`, o `RUN = ['A_scratch', 'B_resnet18']` con 1 época en `train`.
5. Para la corrida completa, **Save Version → Save & Run All (Commit)**. Corre en segundo plano hasta 12 h, y las salidas quedan en la pestaña **Output**.

### Parámetros de la primera celda

| Variable | Valor por defecto | Para qué sirve |
|---|---|---|
| `MODE` | `'synthetic'` | `synthetic`, `preprocess`, `labels`, `train` o `infer` |
| `LIMIT` | `0` | en `preprocess`, con un valor >0 procesa solo N estudios (prueba rápida) |
| `SUBSET` | `600` | estudios de entrenamiento + holdout, muestreados de forma estratificada |
| `RUN` | `None` | `None` corre los 7 experimentos; o una lista, p. ej. `['A_scratch', 'B_resnet18']` |
| `SEEDS` | `[42, 1, 2]` | cada experimento se repite con estas semillas; la primera alimenta tablas y figuras |
| `CFG` | — | tamaño de imagen, cortes por serie, épocas, *batch size*, *learning rates* |

## Cómo funciona

```
reporte (≈12 idiomas) ─► reglas: anatomía + patología, negación/duda (estilo NegEx) ─► 12 etiquetas blandas + confianza
DICOM ─► 1 serie por plano (sag/cor/ax) ─► cortes ordenados por geometría ─► 16 cortes de 224 px ─► caché uint8
ventanas 2.5D (3 cortes vecinos = RGB) ─► CNN (desde cero | ResNet ImageNet) ─► rasgos por ventana
   + embedding de plano + codificación posicional ─► 2 bloques de self-attention
   ─► 12 consultas de atención (una por hallazgo) ─► 12 logits ─► 12 sigmoides
```

- **Datos:** subconjunto estratificado de 600 estudios. El 10 % es *holdout* con etiquetas débiles y se usa para elegir la época y los umbrales. Los 58 gold se usan **solo** en la evaluación final.
- **Pérdida:** BCE (o focal) sobre objetivos blandos, ponderada por la confianza de cada etiqueta (1 afirmado/negado, 0.5 duda, 0.3 silencio), con `pos_weight` para los hallazgos raros.
- **Sin flips horizontales:** medial/lateral es parte del diagnóstico.

### Mapa del notebook

| § | Sección | Qué hace |
|---|---|---|
| 0 | Datos sintéticos | genera la competencia falsa (solo en `synthetic`) |
| 1 | Exploración | cuenta estudios, gold y series por plano/secuencia |
| 2 | Preprocesamiento | DICOM → volumen `(3 planos, 16 cortes, 224, 224)` en caché |
| 3 | Etiquetas débiles | extractor multilingüe, pruebas de regresión, calidad vs gold y por idioma |
| 4 | Subconjunto y prevalencia | muestreo estratificado, holdout, prevalencia por hallazgo |
| 5 | Arquitecturas | ventanas 2.5D, aumentos, CNN desde cero, ResNet, cabeza de atención |
| 6 | Experimentos | entrena los 7 experimentos (un factor por vez) |
| 7 | Evaluación | AUC macro con IC 95 % bootstrap, métricas por hallazgo, umbrales |
| 8 | Grad-CAM | mapas de activación para ACL, menisco medial, derrame y quiste de Baker |
| 9 | Inferencia | `submission.csv` (en `infer`) |
| 10 | Verificaciones | `assert` sobre el formato de las salidas |

### Experimentos

Todos cambian **un solo factor** respecto de la referencia `B_resnet18`:

| Experimento | Cambio respecto de `B` |
|---|---|
| `A_scratch` | CNN estilo VGG entrenada desde cero |
| `B_resnet18` | referencia: ResNet-18 ImageNet, BCE + `pos_weight`, etiquetas ponderadas, 3 planos |
| `C_focal` | pérdida focal (γ = 2) |
| `D_confident` | solo etiquetas confiables (descarta silencios y dudas) |
| `E_sagittal` | solo el plano sagital |
| `F_sin_posweight` | sin `pos_weight` |
| `G_resnet50` | backbone más profundo (ResNet-50) |

## Cómo leer los resultados

Los archivos de la corrida 1 están en `output-data/`; la corrida 2 agrega `output-data-v2/semillas.csv`. Para correr el notebook de nuevo en local, los mismos archivos aparecen en `work/`.

**`resumen_experimentos.csv`:** la tabla principal. Columnas:
- `AUC_gold`: AUC macro sobre los 58 gold (la métrica de la competencia). 0.5 equivale a azar y 1.0 es perfecto.
- `IC95`: intervalo de confianza bootstrap (1 000 remuestreos de estudios).
- `delta_vs_ref` / `IC95_delta`: diferencia frente a `B_resnet18`, con su IC pareado. **Si el IC cruza 0, la diferencia no es concluyente**: con 58 estudios es lo habitual.
- `AUC_holdout`: AUC sobre el holdout de etiquetas débiles, usado para elegir la `mejor_epoca`.
- `minutos`: tiempo de entrenamiento.

Resultados de la corrida incluida:

| Experimento | AUC gold | IC 95 % | Δ vs B |
|---|---|---|---|
| `A_scratch` | 0.589 | [0.525, 0.647] | −0.140 (significativo) |
| `B_resnet18` | 0.729 | [0.673, 0.782] | — |
| `C_focal` | 0.740 | [0.686, 0.789] | +0.011 (no concluyente) |
| `D_confident` | 0.707 | [0.648, 0.761] | −0.022 (no concluyente) |
| `E_sagittal` | 0.734 | [0.681, 0.784] | +0.005 (no concluyente) |
| `F_sin_posweight` | 0.737 | [0.684, 0.787] | +0.008 (no concluyente) |
| `G_resnet50` | 0.741 | [0.690, 0.793] | +0.012 (no concluyente) |

Lo único que la evidencia sostiene con claridad es que el preentrenamiento en ImageNet aporta mucho: `A` contra `B`.
Las demás variantes quedan dentro del ruido estadístico (pero ver `semillas.csv`: `C` y `D` mantienen el signo en las tres semillas).

**`semillas.csv`** (corrida 2): una fila por experimento y semilla, con el AUC macro, los 12 AUC por hallazgo y `delta_vs_ref`
(diferencia contra `B_resnet18` entrenado con la misma semilla). Mide cuánto cambia el resultado solo por el azar del entrenamiento:

| Experimento | AUC macro (media ± DE) | Δ vs B por semilla (42 / 1 / 2) | Lectura |
|---|---|---|---|
| `A_scratch` | 0.595 ± 0.006 | −0.140 / −0.145 / −0.126 | peor en las tres |
| `B_resnet18` | 0.732 ± 0.010 | — | — |
| `C_focal` | 0.742 ± 0.007 | +0.011 / +0.006 / +0.012 | mejor en las tres, pero poco |
| `D_confident` | 0.699 ± 0.008 | −0.022 / −0.052 / −0.025 | peor en las tres |
| `E_sagittal` | 0.737 ± 0.002 | +0.005 / −0.006 / +0.014 | cambia de signo: ruido |
| `F_sin_posweight` | 0.735 ± 0.009 | +0.008 / −0.019 / +0.018 | cambia de signo: ruido |
| `G_resnet50` | 0.736 ± 0.004 | +0.012 / −0.010 / +0.008 | cambia de signo: ruido |

La desviación entre semillas (≤ 0.010) es mucho menor que el ancho del IC bootstrap (≈ ±0.05): la incertidumbre principal
viene de tener 58 gold. Por hallazgo la variación es mayor (hasta 0.08), así que las comparaciones por hallazgo de una sola semilla son frágiles.

**`auc_por_hallazgo.csv`:** una fila por experimento y una columna por hallazgo. Muestra qué hallazgos aprende bien el modelo y cuáles no.

**`metricas_por_hallazgo.csv`:** por experimento y hallazgo: `prevalencia_gold`, `AUC` y el `umbral` que maximiza F1
en el holdout, aplicado sin cambios al gold, con la `precision`, el `recall` y el `F1` resultantes. El umbral nunca se elige en gold, para no inflar las métricas.

**`calidad_etiquetas.csv`:** compara el extractor de reportes con los 58 gold:
- `mencion`: fracción de reportes gold en que el hallazgo aparece (afirmado o negado).
- `acuerdo`, `sensibilidad`, `especificidad`, `AUC`: calidad de la etiqueta débil frente a la experta.

Un hallazgo con sensibilidad baja tiene muchos positivos no detectados en el texto, y eso limita lo que el clasificador puede aprender.

**`calidad_por_idioma.csv`:** por idioma del reporte:
- `silencio_medio`: fracción de hallazgos que el extractor no encuentra.
- `positivos_medios`: positivos detectados por reporte.
- `acuerdo_gold`: acuerdo con los gold de ese idioma.

Un silencio alto indica un léxico pobre para ese idioma, es decir, más falsos negativos en las etiquetas de esos centros.

**`prevalencia.csv`:** % de positivos por hallazgo en train y holdout (etiquetas débiles) y en gold (etiquetas expertas).
Las diferencias entre columnas reflejan los sesgos del extractor. Por ejemplo, `Fracture` tiene 13 % de positivos en train y 31 % en gold.

**`experimentos.png`:** versión gráfica del resumen. **`gradcam.png`:** para cada hallazgo se elige el estudio gold positivo
con mayor probabilidad y se muestra la ventana a la que más atendió el modelo. Al leerlo, la pregunta es si el mapa cae
en la anatomía correcta (meniscos, receso suprapatelar, fosa poplítea) o en el fondo.

**`knee-train.log`:** log de Kaggle en formato JSON por línea. Contiene el avance de cada época y las tablas impresas por el notebook.

## Referencias

- RSNA. *RSNA Knee Abnormality Detection*. Kaggle, 2026. https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
- Bien, N. et al. (2018). *Deep-learning-assisted diagnosis for knee MRI: development and retrospective validation of MRNet.* PLOS Medicine.
- He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Simonyan, K., Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR.
- Lin, T.-Y. et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.
- Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- Chapman, W. W. et al. (2001). *A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries* (NegEx). J. Biomed. Inform.
- Ratner, A. et al. (2017). *Snorkel: Rapid Training Data Creation with Weak Supervision.* VLDB.
- Ilse, M., Tomczak, J., Welling, M. (2018). *Attention-based Deep Multiple Instance Learning.* ICML.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.

Notebooks públicos de la competencia que se estudiaron como trabajo relacionado (sin copiar su código):
[pilkwang, *RSNA Knee baseline v1*](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1) y
[dreaddevelopment, *training the twelve-finding model*](https://www.kaggle.com/code/dreaddevelopment/knee-mri-training-the-twelve-finding-model).
