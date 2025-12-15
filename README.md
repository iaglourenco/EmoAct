# EmoAct

Sistema de análise de vídeo para detecção de emoções, atividades e identificação de pessoas, com geração automática de sumário por LLM.

## Visão Geral

O EmoAct processa vídeos através de um pipeline modular que extrai múltiplas camadas de informação: detecção facial, reconhecimento de emoções, estimativa de pose corporal, detecção de objetos, transcrição de áudio e classificação de atividades. Ao final, um modelo de linguagem (LLM) gera um resumo narrativo do conteúdo analisado.

## Arquitetura

O sistema utiliza um grafo de estados (StateGraph do LangGraph) para orquestrar o fluxo de processamento:

![](graph.png)

## Módulos e Funcionalidades

### 🎥 Entrada/Saída de Vídeo (`video_io.py`)
Responsável pelo carregamento e salvamento de vídeos usando OpenCV. Extrai frames individuais mantendo a taxa de frames original (FPS) para preservar a temporalidade do vídeo na exportação.

### 👤 Detecção Facial (`face.py`)
Utiliza **InsightFace** (modelo Buffalo_L) para:
- **Localização de faces**: Detecta bounding boxes de rostos no frame
- **Embedding facial**: Gera vetores de características (embeddings) para identificação única
- **Estimativa de gênero e idade**: Infere atributos demográficos a partir da face

O threshold de confiança filtra detecções de baixa qualidade.

### 😊 Reconhecimento de Emoções (`emotions.py`)
Classifica emoções faciais usando um modelo transformer (HuggingFace). A partir da região da face recortada, infere estados emocionais como alegria, tristeza, raiva, surpresa, entre outros.

### 🏃 Estimativa de Pose (`pose.py`)
Emprega **YOLOv11-pose** para detectar keypoints corporais (nariz, ombros, cotovelos, pulsos, quadris, joelhos, tornozelos). As poses detectadas são associadas às faces correspondentes através de cálculo de IoU (Intersection over Union) entre bounding boxes.

### 🔍 Detecção de Objetos (`objects.py`)
Utiliza **YOLOv11** para identificar objetos na cena (exceto pessoas). Cada objeto detectado inclui rótulo, localização e nível de confiança. O threshold de confiança pode ser ajustado para filtrar detecções.

### 🎯 Rastreamento de Pessoas (`tracker.py`)
Mantém identificadores consistentes (IDs) para pessoas ao longo do vídeo combinando:
- **Similaridade de embedding**: Compara vetores faciais usando similaridade de cosseno
- **Proximidade espacial**: Considera a distância entre detecções em frames consecutivos
- **Tolerância a oclusão**: Permite que uma pessoa "desapareça" por alguns frames sem perder o ID

O algoritmo pondera essas métricas para decidir se uma detecção corresponde a uma pessoa já conhecida ou é uma nova pessoa.

### 📐 Classificação de Atividades (`classifier.py`)
Coleta dados brutos para inferência de atividade:
- **Ângulos articulares**: Calcula ângulos entre landmarks (cotovelos, joelhos, quadris, ombros) usando geometria vetorial
- **Classificação de imagem**: Usa **YOLOv11-cls** para obter predições sobre a atividade no frame

Os dados são enviados para o LLM interpretar, sem regras hardcoded de classificação.

### 🎤 Transcrição de Áudio (`audio.py`)
Emprega o **Whisper** (OpenAI) para transcrever o áudio do vídeo em texto. Suporta aceleração por GPU quando disponível.

### 🤖 Sumarização por LLM (`llm.py`)
Integra com um servidor LLM local (API compatível com OpenAI) para:
- Agregar dados de frames amostrados (pessoas, emoções, objetos, poses)
- Combinar transcrição de áudio com dados visuais
- Gerar um resumo narrativo do conteúdo do vídeo

O módulo gerencia chunking de contexto para não exceder limites de tokens.

### 🎨 Visualização (`utils.py`)
Funções utilitárias para desenhar:
- Bounding boxes coloridos para faces e objetos
- Esqueleto de pose com conexões entre landmarks
- Rótulos com informações (ID, emoção, idade/gênero)

### 📊 Tipos de Dados (`types.py`)
Define as estruturas de dados TypedDict para tipagem estática:
- `PersonInfo`: Dados completos de uma pessoa (face, pose, emoções, atividade)
- `FrameInfo`: Informações de um frame (imagem, pessoas, objetos)
- `PipelineState`: Estado global do pipeline durante processamento

## Pipeline de Execução

1. **Carregamento**: Vídeo é decomposto em frames
2. **Detecção**: Faces, poses e objetos são detectados em cada frame
3. **Associação**: Poses são vinculadas às faces correspondentes por proximidade espacial
4. **Rastreamento**: IDs consistentes são atribuídos às pessoas ao longo do tempo
5. **Classificação**: Dados de pose e imagem são coletados para análise de atividade
6. **Transcrição**: Áudio é convertido em texto
7. **Visualização**: Anotações são desenhadas nos frames
8. **Sumarização**: LLM gera resumo combinando todas as informações
9. **Exportação**: Vídeo anotado e arquivo de sumário são salvos

## Modelos Utilizados

| Componente | Modelo | Descrição |
|------------|--------|-----------|
| Face | [InsightFace](https://github.com/deepinsight/insightface) Buffalo_L | Detecção, embedding, gênero/idade |
| Emoções | [dima806/facial_emotions](https://huggingface.co/dima806/facial_emotions_image_detection) | Classificação de emoções faciais |
| Pose | [YOLOv11n-pose](https://docs.ultralytics.com/tasks/pose/) | Estimativa de keypoints corporais |
| Objetos | [YOLOv11n](https://docs.ultralytics.com/pt/tasks/detect/) | Detecção de objetos COCO |
| Atividade | [YOLOv11n-cls](https://docs.ultralytics.com/pt/tasks/classify/) | Classificação de imagem |
| Áudio | [Whisper](https://github.com/openai/whisper) Base | Transcrição speech-to-text |
| Sumário | LLM Local via [LM Studio](https://lmstudio.ai/) | Geração de texto (servidor local) |

## Configuração

### Dependências
```bash
pip install -r requirements.txt
```

## Uso

```python
python -m emoact.pipeline
```

O vídeo de entrada deve estar em `input/input_video.mp4`. A saída inclui:
- Vídeo anotado: `output.mp4`
- Sumário textual: `output_summary.txt`

## Estrutura do Projeto

```
emoact/
├── pipeline.py      # Orquestração do grafo de estados
├── face.py          # Detecção e análise facial
├── emotions.py      # Classificação de emoções
├── pose.py          # Estimativa de pose corporal
├── objects.py       # Detecção de objetos
├── tracker.py       # Rastreamento de pessoas
├── classifier.py    # Coleta de dados de atividade
├── audio.py         # Transcrição de áudio
├── llm.py           # Integração com LLM
├── video_io.py      # I/O de vídeo
├── utils.py         # Utilitários de visualização
└── types.py         # Definições de tipos
```
