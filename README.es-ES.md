# CONQUER: REPRESENTACIÓN CONSCiente DEL CONTEXTO CON MEJORA DE CONSULTAS PARA BÚSQUEDA DE PERSONAS POR TEXTO
[![IEEE](https://img.shields.io/badge/IEEE-11461248-0078C8.svg)](https://ieeexplore.ieee.org/document/11461248)
[![Project Page](https://img.shields.io/badge/GitHub-CONQUER-181717.svg)](https://github.com/zqxie77/CONQUER)

**Artículo**: https://ieeexplore.ieee.org/document/11461248

## Introducción

Este repositorio contiene la implementación en PyTorch para el artículo [CONQUER: CONTEXT-AWARE REPRESENTATION WITH QUERY ENHANCEMENT FOR TEXT-BASED PERSON SEARCH]. Nuestro trabajo introduce un marco de dos etapas diseñado para abordar los desafíos de discrepancias cruzado-modales y consultas ambiguas de usuarios en la Búsqueda de Personas por Texto.

**Código fuente oficial**: [https://github.com/zqxie77/CONQUER](https://github.com/zqxie77/CONQUER)

### ¡Novedad!

* **[2026-01-25]** 🎉 **CONQUER** ha sido aceptado por **ICASSP 2026**!
* **[2025-09-20]** Se han publicado el código y los modelos preentrenados.
 
### Marco CONQUER

A diferencia de los métodos existentes que realizan una búsqueda directa utilizando la consulta de texto original, el marco CONQUER mejora la consulta durante la inferencia sin necesidad de reentrenar el modelo base. El proceso comienza identificando una imagen ancla relevante. Un Modelo de Lenguaje Grande Multimodal (MLLM) luego aprende los detalles visuales clave de esta imagen a través de un proceso de preguntas y respuestas. Finalmente, estos detalles se fusionan con el texto original para crear una consulta mejorada que se utiliza para reordenar los resultados de búsqueda. Todo esto es apoyado por la fase de entrenamiento, donde el módulo de Mejora de Representación Consciente del Contexto (CARE) aprende incrustaciones cruzado-modales robustas.

## Requisitos y Conjuntos de Datos

* PyTorch
* OpenAI CLIP ViT-B/16 (Codificador de Imágenes)
* CLIP Transformer (Codificador de Texto) 
* Qwen2.5-VL-7B (para el módulo IQE) 

### Conjuntos de datos

Evaluamos nuestro modelo en tres referencias comunes para TBPS:

**CUHK-PEDES**.
**ICFG-PEDES**.
**RSTPReid**.

## Entrenamiento y Evaluación

### Etapa 1: Entrenamiento del Módulo CARE

Para entrenar un nuevo modelo CONQUER desde cero, ejecute el siguiente script. Esta etapa entrena el módulo de Mejora de Representación Consciente del Contexto (CARE) para aprender incrustaciones cruzado-modales robustas.

```bash
sh run_CONQUER.sh
```
### Etapa 2: Inferencia con el Módulo IQE

Para realizar inferencias y evaluar un modelo entrenado, ejecute el siguiente script. Esta etapa utiliza el módulo plug-and-play de Mejora Interactiva de Consultas (IQE) para refinar las consultas y mejorar los resultados de recuperación.
```bash
sh run_IQE.sh
```

### Cita

Si encuentra este trabajo útil en su investigación, considere citar:

**BibTeX:**
```bash
@INPROCEEDINGS{11461248,
  author={Zeng, Chenxi and Duan, Yipeng and Xie, Zequn and Han, Xiaosong},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={CONQUER: Context-Aware Representation with Query Enhancement for Text-Based Person Search}, 
  year={2026},
  volume={},
  number={},
  pages={12867-12871},
  keywords={Protocols;HTTP;LoRa;Local area networks;Videos;Communication systems;Video equipment;Data communication;Plugs;Fuses;Text-Based Person Search;Cross-modal Learning;Optimal Transport;Query Enhancement},
  doi={10.1109/ICASSP55912.2026.11461248}}
```

**RIS:**
```bash
TY  - CONF
TI  - CONQUER: Context-Aware Representation with Query Enhancement for Text-Based Person Search
T2  - ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
SP  - 12867
EP  - 12871
AU  - C. Zeng
AU  - Y. Duan
AU  - Z. Xie
AU  - X. Han
PY  - 2026
DO  - 10.1109/ICASSP55912.2026.11461248
JO  - ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
IS  - 
SN  - 2379-190X
VO  - 
VL  - 
JA  - ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)
Y1  - 3-8 May 2026
ER  - 
```

**Estilo IEEE:**
```
C. Zeng, Y. Duan, Z. Xie and X. Han, "CONQUER: Context-Aware Representation with Query Enhancement for Text-Based Person Search," ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Barcelona, España, 2026, pp. 12867-12871, doi: 10.1109/ICASSP55912.2026.11461248.
```
