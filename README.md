# POC — Reconhecimento Facial

Prova de conceito desenvolvida durante o estágio curricular na Asseco PST, no departamento de Redes e Segurança. O objetivo foi explorar e comparar duas abordagens de reconhecimento facial: uma local com a biblioteca DeepFace, e outra na cloud com o AWS Rekognition.

---

## O que faz

O projeto cobre o ciclo completo de reconhecimento facial — registo, verificação 1:1, identificação 1:N, extração de foto de documento de identidade, análise de qualidade de imagem, avaliação comparativa de modelos e deteção de vivacidade (liveness).

Está dividido em duas "engines" paralelas que partilham o mesmo frontend:

- **DeepFace** — corre localmente, usa o modelo ArcFace com o detetor YuNet e distância cosseno
- **AWS Rekognition** — chama a API da Amazon, inclui liveness via componente React oficial da AWS

---

## Estrutura

```
.
├── backend/
│   ├── main.py                  # FastAPI — servidor principal (porta 8000)
│   ├── config.py                # Thresholds, modelos, diretórios
│   ├── deepface/
│   │   ├── routes/              # verify, identify, register, quality, models,
│   │   │                        # document, live, video, multi, attributes, config
│   │   └── services/            # deepface_service, quality_service,
│   │                            # document_service, model_comparison_service
│   └── aws/
│       ├── routes/              # verify, identify, register, liveness, config
│       └── services/            # rekognition_service, metrics_service,
│                                # anti_spoofing_service
├── liveness/
│   ├── server.py                # Flask — servidor liveness AWS (porta 5001)
│   └── app/                     # App React com @aws-amplify/ui-react-liveness
├── frontend/                    # HTML/CSS/JS — interface web
├── database/
│   ├── faces/                   # Embeddings por pessoa (excluído do git)
│   └── documents/               # Sessões de documento temporárias
├── run.py                       # Arranca o servidor FastAPI
├── run_deepface.py              # Arranca o DeepFace Webservice (porta 8001)
├── config.json                  # Thresholds editáveis em runtime
└── requirements.txt
```

---

## Requisitos

- Python 3.10+
- Conta AWS com acesso ao Rekognition e Face Liveness (para as funcionalidades AWS)
- Node.js 18+ (só necessário para recompilar o app React de liveness)

---

## Instalação

```bash
# Criar e ativar ambiente virtual
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/Mac

# Instalar dependências Python
pip install -r requirements.txt
pip install flask flask-cors python-dotenv requests boto3
```

Copiar `.env.example` para `.env` e preencher as credenciais:

```bash
cp .env.example .env
```

Conteúdo do `.env`:

```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=eu-west-1
AWS_COLLECTION_ID=poc-asseco-faces
```

---

## Executar

```bash
# Ativar o ambiente virtual (se ainda não estiver ativo)
.\venv\Scripts\activate

# Servidor principal — FastAPI + liveness Flask em background
python run.py

# (Opcional) DeepFace Webservice numa segunda janela
python run_deepface.py
```

O servidor principal inicia automaticamente o servidor de liveness na porta 5001 em background. Não é necessário iniciá-lo manualmente.

---

## Endereços

| Serviço | URL |
|---|---|
| Dashboard | http://localhost:8000 |
| Verificação 1:1 (DeepFace) | http://localhost:8000/verify |
| Identificação 1:N (DeepFace) | http://localhost:8000/identify |
| Registo (DeepFace) | http://localhost:8000/register |
| Avaliação de modelos | http://localhost:8000/models |
| Extração documento ID | http://localhost:8000/document |
| AWS — Verificação | http://localhost:8000/aws/verify |
| AWS — Identificação | http://localhost:8000/aws/identify |
| AWS — Registo | http://localhost:8000/aws/register |
| AWS — Liveness | http://localhost:5001 |
| API Docs (Swagger) | http://localhost:8000/docs |
| DeepFace Webservice Docs | http://localhost:8001/docs |

---

## Funcionalidades principais

**DeepFace**
- Verificação 1:1 por upload de ficheiro ou câmara em tempo real
- Identificação 1:N contra a base de dados local (embeddings em JSON)
- Registo de novas pessoas com ou sem anti-spoofing
- Análise de qualidade da imagem: blur (variância Laplacian), brilho, contraste, tamanho da face
- Extração automática de foto de documento de identidade com CLAHE para normalizar a iluminação
- Avaliação comparativa de modelos: ArcFace, Facenet512, VGG-Face, SFace
- Deteção de atributos (idade estimada, género, emoção dominante)
- Pesquisa em vídeo com resultados em stream (SSE)
- Anti-spoofing dinâmico: ativado para câmara, desativado para ficheiros

**AWS Rekognition**
- Verificação 1:1 e identificação 1:N via API da Amazon
- Coleção de faces gerida no Rekognition
- Métricas de latência com percentis p50, p95, p99
- Liveness detection com o componente oficial `@aws-amplify/ui-react-liveness`

---

## Configuração de thresholds

Os thresholds podem ser alterados em runtime pela interface web (sem reiniciar o servidor). Os valores ficam guardados em `config.json`.

Valores por defeito:

| Parâmetro | Valor | Notas |
|---|---|---|
| `df_verify_threshold` | 0.54 | Distância cosseno — verificação 1:1 |
| `df_identify_threshold` | 0.54 | Distância cosseno — identificação 1:N |
| `df_document_threshold` | 0.65 | Mais permissivo — doc vs câmara |
| `df_quality_min_blur` | 30.0 | Variância Laplacian mínima |
| `liveness_threshold` | 75.0 | Score mínimo AWS liveness (0–100) |
| `AWS_VERIFY_THRESHOLD` | 90.0 | Similaridade mínima AWS verify |
| `AWS_IDENTIFY_THRESHOLD` | 80.0 | Similaridade mínima AWS identify |

---

## Notas

- A pasta `database/faces/` contém embeddings faciais e está excluída do repositório por questões de privacidade
- O ficheiro `.env` com credenciais AWS nunca deve ser partilhado nem versionado
- Para produção, as credenciais AWS devem ser geridas com IAM roles ou Cognito Identity Pools em vez de chaves estáticas
- O detetor YuNet é mais rápido que o RetinaFace e adequado para uso em tempo real; o RetinaFace tem melhor precisão em condições difíceis
- A GPU é usada automaticamente via ONNX Runtime DirectML (Windows) se disponível
