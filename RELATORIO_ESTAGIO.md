# Relatório de Estágio

## Prova de Conceito de Reconhecimento Facial
### AWS Rekognition vs DeepFace

**Aluno:** João Freitas
**Empresa:** Asseco PST
**Departamento:** Redes e Segurança
**Duração do estágio:** 210 horas
**Ano letivo:** 2025/2026

---

## Agradecimentos

Quero agradecer à Asseco PST por me ter recebido e pela confiança que depositaram em mim para desenvolver um tema tão atual como o reconhecimento facial. Um agradecimento em particular aos meus orientadores na empresa, que ao longo do estágio me foram dando direções sempre que estava encravado, e que me ensinaram a pensar também na parte de segurança e privacidade, que no início eu não estava habituado a considerar. Agradeço também à minha orientadora na universidade pelo acompanhamento e pelas correções ao longo do processo.

---

## Resumo

Este relatório descreve o estágio que realizei na empresa Asseco PST, no departamento de Redes e Segurança, durante o qual desenvolvi uma Prova de Conceito (POC) para avaliar duas abordagens distintas de reconhecimento facial: uma baseada em serviço cloud (AWS Rekognition) e outra baseada numa biblioteca Python open-source (DeepFace). O objetivo principal foi comparar as duas soluções em termos de precisão, latência, custos, complexidade operacional e segurança, de forma a apresentar à empresa uma recomendação fundamentada sobre qual o caminho a seguir em projetos futuros que envolvam validação biométrica.

Ao longo do estágio, passei por várias fases — começando por uma fase de investigação, onde fiz um levantamento exaustivo das soluções disponíveis no mercado, analisei os modelos de licenciamento e a arquitetura de cada uma, e apresentei as minhas conclusões aos orientadores numa apresentação intermédia. A partir dessa apresentação, os orientadores decidiram quais eram as duas soluções a explorar mais a fundo — DeepFace e AWS Rekognition — e entrei então na fase de implementação propriamente dita.

Na implementação, construí uma aplicação web completa, com backend em Python (FastAPI e Flask), frontend em HTML/CSS/JavaScript e um mini-app em React para o liveness da AWS. A aplicação cobre todos os casos de uso típicos de reconhecimento facial: registo biométrico, verificação 1:1, identificação 1:N, deteção de múltiplas pessoas, pesquisa em vídeo, análise de atributos faciais, extração de fotografia a partir de documento de identificação, câmara ao vivo com identificação em tempo real, avaliação comparativa de modelos de reconhecimento, e prova de vida (liveness) tanto pela AWS como localmente via DeepFace.

Este relatório descreve todo o caminho que percorri, as escolhas técnicas que fiz, as dificuldades que encontrei, e termina com uma comparação final entre as duas soluções e um guia de execução que permite correr o projeto em qualquer computador.

---

## Índice

1. Introdução
2. A empresa
3. Objetivos do estágio
4. Fase 1 — Investigação de soluções
   - 4.1 Levantamento de soluções
   - 4.2 Análise de licenciamento e custos
   - 4.3 Análise da arquitetura
   - 4.4 Apresentação intermédia
5. Fase 2 — Implementação da POC
   - 5.1 Escolha das soluções
   - 5.2 Arquitetura geral do projeto
   - 5.3 Tecnologias utilizadas
   - 5.4 Estrutura do repositório
6. Implementação DeepFace
   - 6.1 Setup do ambiente Python
   - 6.2 Pipeline de registo biométrico
   - 6.3 Verificação 1:1
   - 6.4 Identificação 1:N
   - 6.5 Avaliação de qualidade da imagem
   - 6.6 Avaliação dos melhores modelos
   - 6.7 Extração da fotografia do documento
   - 6.8 Análise de atributos faciais
   - 6.9 Câmara ao vivo
   - 6.10 Deteção multi-pessoa
   - 6.11 Pesquisa em vídeo
   - 6.12 Webservice DeepFace dedicado
7. Implementação AWS Rekognition
   - 7.1 Configuração e preparação do ambiente
   - 7.2 Registo na coleção (IndexFaces)
   - 7.3 Verificação 1:1 (CompareFaces)
   - 7.4 Identificação 1:N (SearchFacesByImage)
   - 7.5 Métricas de qualidade e performance
   - 7.6 Anti-spoofing local complementar
   - 7.7 Face Liveness da AWS
8. Frontend
9. Comparação final: DeepFace vs AWS
10. Documentação da POC
11. Apresentação final
12. Reuniões semanais com os orientadores
13. Dificuldades encontradas
14. Conclusões e próximos passos
15. Guia de execução do projeto
16. Tabela de horas
17. Referências

---

## 1. Introdução

O reconhecimento facial é uma das áreas da biometria que mais se tem desenvolvido nos últimos anos, em grande parte devido ao avanço das redes neuronais profundas e à disponibilidade de serviços cloud que permitem integrar estas funcionalidades sem que as empresas tenham de treinar modelos do zero. Apesar disso, quando uma empresa quer integrar reconhecimento facial num produto, continua a existir uma decisão difícil: usar um serviço cloud pago, que é mais rápido de integrar mas que implica custos contínuos e dependência de terceiros, ou usar bibliotecas open-source, que dão mais controlo mas obrigam a gerir toda a infraestrutura e a manter os modelos atualizados.

Foi exatamente este problema que o meu estágio procurou explorar. A ideia não era construir um produto final, mas sim uma Prova de Conceito que permitisse à empresa ter dados concretos sobre as duas abordagens para poder tomar uma decisão fundamentada em projetos futuros. No fundo, o meu trabalho foi o de fazer o "trabalho de casa" que normalmente não há tempo para fazer quando se está num projeto real com prazos apertados.

O estágio teve a duração total de 210 horas, distribuídas ao longo do semestre, e dividiu-se em duas fases principais: uma primeira fase de investigação e análise de soluções, que resultou numa apresentação intermédia aos orientadores, e uma segunda fase de implementação técnica, em que construí a POC propriamente dita. No final, apresentei os resultados numa apresentação formal na empresa.

---

## 2. A empresa

A Asseco PST faz parte do grupo Asseco, um dos maiores grupos de software da Europa, com forte presença em várias áreas, desde banca e finanças até saúde, administração pública e telecomunicações. Em Portugal, a empresa desenvolve soluções de software para diferentes clientes institucionais e tem um foco importante em áreas de segurança e integração de sistemas, o que faz com que o tema do reconhecimento facial se enquadre naturalmente no contexto dos projetos que a empresa costuma abordar.

Durante o estágio estive integrado no departamento de Redes e Segurança, onde tive o apoio direto dos meus orientadores. O ambiente de trabalho foi bastante positivo e, apesar de ser um estágio de universidade, senti-me tratado como se fosse mais um elemento da equipa, o que me deu bastante à vontade para fazer perguntas e pedir ajuda quando precisei.

---

## 3. Objetivos do estágio

Os objetivos do estágio foram definidos logo na primeira semana, em conjunto com os orientadores, e podem ser resumidos nos seguintes pontos:

- **Investigar** o mercado de soluções de reconhecimento facial existentes, identificando vantagens e desvantagens de cada uma, tanto do ponto de vista técnico como de licenciamento.
- **Comparar** soluções cloud com soluções open-source, percebendo em que casos cada uma faz mais sentido.
- **Implementar** uma POC funcional que cubra todos os casos de uso típicos: registo, 1:1, 1:N, prova de vida, análise de qualidade, extração de fotografia de documento.
- **Avaliar** modelos de reconhecimento facial (ArcFace, Facenet512, VGG-Face, SFace) para perceber qual tem melhor relação precisão/velocidade.
- **Documentar** todo o processo de forma a que no futuro qualquer pessoa da empresa consiga pegar no trabalho e continuar a partir daí.
- **Apresentar** os resultados em duas ocasiões: uma apresentação intermédia após a fase de investigação, e uma apresentação final com toda a POC pronta.

Além disto, havia também um objetivo mais pessoal, que era o de aprender como funciona um ambiente empresarial real, com reuniões semanais, gestão de tarefas em issues no GitHub e apresentações formais.

---

## 4. Fase 1 — Investigação de soluções

### 4.1 Levantamento de soluções de reconhecimento facial

A primeira tarefa que me foi atribuída foi a de fazer um levantamento das soluções de reconhecimento facial disponíveis no mercado que pudessem vir a ser utilizadas num contexto empresarial. Esta fase pareceu-me inicialmente simples — afinal, "basta pesquisar no Google e listar" — mas acabou por ser mais exigente do que eu estava à espera, porque há realmente muitas soluções, e muitas delas apresentam-se de forma confusa (por exemplo, algumas bibliotecas são na verdade wrappers de outras, e outras são apenas serviços pagos sem qualquer camada open-source).

Para organizar o trabalho, defini uma lista de critérios que cada solução tinha de cumprir ou que eu tinha de conseguir avaliar:

- **Validação 1:1** — ser capaz de comparar duas imagens e dizer se são da mesma pessoa.
- **Validação 1:N** — ser capaz de pesquisar uma pessoa numa base de dados de faces.
- **Registo biométrico** — ter uma forma de adicionar pessoas à base de dados.
- **Liveness / prova de vida** — detetar se o que está à frente da câmara é uma pessoa real ou uma fotografia/vídeo.
- **Suporte a Python** — por decisão dos orientadores, a linguagem principal do projeto era Python.

Com estes critérios em mente, fiz o levantamento das seguintes soluções:

- **AWS Rekognition** (Amazon) — Serviço cloud da AWS. Tem funcionalidades completas: `CompareFaces` para 1:1, `SearchFacesByImage` para 1:N, `IndexFaces` para registo na coleção, e `CreateFaceLivenessSession` para prova de vida ativa. Tem SDK oficial para Python (boto3).
- **Google Cloud Vision / Face Detection** — Serviço cloud da Google. Faz deteção de faces e análise de atributos, mas não faz comparação facial propriamente dita (não tem equivalente a CompareFaces nem liveness ativo da mesma forma que a AWS).
- **Azure Face API** (Microsoft) — Serviço cloud da Microsoft. Tem funcionalidades semelhantes à AWS, com verificação, identificação e liveness. A Microsoft tem vindo a restringir o acesso a estas APIs, o que torna a sua adoção mais complicada em contexto empresarial.
- **DeepFace** — Biblioteca open-source em Python, mantida ativamente, que serve como wrapper para vários modelos de reconhecimento (ArcFace, Facenet, VGG-Face, SFace, OpenFace, DeepID, Dlib, ArcFace, etc.). Faz deteção, comparação e análise de atributos. Tem anti-spoofing integrado desde uma versão mais recente.
- **face_recognition (Dlib)** — Biblioteca em Python baseada em Dlib. Simples de usar mas com modelos mais antigos e precisão inferior às soluções modernas.
- **InsightFace** — Conjunto de modelos e utilitários em Python, com foco em modelos de última geração (ArcFace, RetinaFace, SCRFD). Mais "low-level" que o DeepFace mas mais próximo do estado da arte.
- **OpenCV DNN** — O próprio OpenCV permite correr modelos de deteção e reconhecimento facial, mas é mais uma ferramenta do que uma solução completa.
- **Luxand FaceSDK, Neurotechnology VeriLook, Cognitec FaceVACS** — Soluções comerciais, com SDKs pagos.

Para cada solução, analisei se tinha suporte para cada uma das funcionalidades exigidas. O resultado foi uma tabela comparativa que ficou mais ou menos assim (em versão resumida):

| Solução           | 1:1 | 1:N | Registo | Liveness | Python |
|-------------------|-----|-----|---------|----------|--------|
| AWS Rekognition   | Sim | Sim | Sim     | Sim (ativo) | Sim (boto3) |
| Azure Face API    | Sim | Sim | Sim     | Sim      | Sim    |
| Google Cloud      | Não | Não | Não     | Não      | Sim    |
| DeepFace          | Sim | Sim | Sim     | Sim (passivo) | Sim (nativo) |
| face_recognition  | Sim | Sim | Sim     | Não      | Sim    |
| InsightFace       | Sim | Sim | Sim     | Parcial  | Sim    |
| Luxand FaceSDK    | Sim | Sim | Sim     | Sim      | Sim    |

Foi logo evidente que o Google Cloud ficava de fora, porque não cobre os requisitos mínimos. Nas restantes, a decisão não dependia apenas de ter ou não ter a funcionalidade, mas sim da forma como cada uma as implementava e do custo associado. Por isso, entreguei a tabela e a lista de soluções identificadas e passei à tarefa seguinte.

### 4.2 Análise de licenciamento e custos

A segunda tarefa foi analisar os modelos de licenciamento das soluções identificadas. Aqui a divisão mais óbvia é entre soluções gratuitas e pagas, mas dentro de cada categoria há diferenças importantes.

**Soluções gratuitas / open-source:**

- **DeepFace** — Licença MIT. É a licença mais permissiva possível, permite uso comercial sem restrições. O projeto está ativo no GitHub, com commits frequentes, e é mantido por um autor principal (Sefik Ilkin Serengil). A grande vantagem é que, uma vez instalado, não há custos por uso — paga-se apenas o hardware onde corre.
- **face_recognition** — Licença MIT. Também gratuito e open-source, mas com manutenção menos ativa nos últimos anos.
- **InsightFace** — Também com licença permissiva, mas os modelos treinados são distribuídos separadamente e alguns têm restrições próprias.
- **OpenCV** — Licença Apache 2.0. Sem restrições significativas.

**Soluções pagas:**

- **AWS Rekognition** — Pagamento por uso. Os preços são calculados por número de imagens processadas e por operações específicas. À data da análise, os preços eram aproximadamente (em euros, com as tarifas da região `eu-west-1`):
  - `DetectFaces`, `CompareFaces`, `SearchFacesByImage`: cerca de 0,001 $ por imagem processada, com descontos para volumes elevados.
  - `IndexFaces`: semelhante.
  - `CreateFaceLivenessSession`: cerca de 0,0025 $ por sessão.
  - Armazenamento de faces na coleção: 0,00001 $ por face/mês.
  - A AWS oferece um **free tier** de 5.000 imagens processadas por mês durante os primeiros 12 meses, o que permite fazer uma POC sem custos.
- **Azure Face API** — Pagamento por uso, com preços semelhantes à AWS. A Microsoft tem vindo a restringir o acesso a estas APIs para clientes empresariais selecionados.
- **Luxand, Neurotechnology, Cognitec** — Licenciamento por dispositivo ou por chave comercial, com preços que facilmente ultrapassam os milhares de euros. Muitas destas soluções não têm sequer uma página de preços pública — é preciso pedir orçamento.

Da análise dos custos, duas coisas ficaram claras:

1. Para uma POC, qualquer das soluções pagas era viável por causa dos trials e dos free tiers. A AWS era a que tinha o trial mais generoso.
2. Para um sistema em produção com milhares ou milhões de operações por mês, os custos da AWS (e da Azure) crescem rapidamente e podem justificar o esforço de instalar e manter uma solução on-premises com DeepFace.

Entreguei aos orientadores uma tabela comparativa de custos e condições de utilização que resumia tudo isto, e voltou a ficar claro que a escolha final ia depender muito do caso de uso específico.

### 4.3 Análise da arquitetura das soluções

A terceira tarefa foi avaliar a arquitetura técnica das soluções. Aqui o foco era perceber, para cada solução, se era fácil ou difícil de integrar, o que é que era preciso ter instalado, e que tipo de dependências trazia.

**AWS Rekognition (arquitetura cloud):**

- Instalação local: apenas o SDK `boto3` (instalável via pip).
- Requer conta AWS e chaves de acesso (`AWS_ACCESS_KEY_ID` e `AWS_SECRET_ACCESS_KEY`).
- Não requer GPU nem modelos locais — tudo corre nos servidores da AWS.
- As imagens são enviadas para a AWS em cada pedido, o que levanta questões de privacidade (as imagens não ficam armazenadas pela AWS, mas passam pelos servidores deles).
- Dependência forte: se a AWS estiver em baixo, o sistema também está.

**DeepFace (arquitetura on-premises):**

- Instalação local via pip. Requer Python 3.8 ou superior.
- Instala bibliotecas pesadas como TensorFlow, Keras, OpenCV, Pillow, NumPy.
- Os modelos são descarregados automaticamente na primeira execução e guardados em `~/.deepface/weights/`. Alguns modelos são grandes (VGG-Face tem cerca de 500 MB).
- Funciona tanto em CPU como em GPU. Numa máquina sem GPU, as primeiras inferências são lentas (cold start de 5-10 segundos), mas depois estabilizam.
- Nenhum dado sai da máquina, o que é uma vantagem grande do ponto de vista de privacidade.

Com base nisto, escrevi um documento resumo com a arquitetura de cada solução, incluindo diagramas simples a explicar o fluxo de dados em cada caso, e os requisitos técnicos para integração.

### 4.4 Preparação da apresentação intermédia

Com as três análises feitas, o passo seguinte foi preparar uma apresentação intermédia em PowerPoint para apresentar os resultados aos orientadores. O PPT tinha as seguintes secções:

- **Enquadramento** — explicação do contexto e dos objetivos da POC.
- **Soluções analisadas** — lista das soluções que encontrei, com um slide de resumo para cada uma.
- **Comparação** — a tabela comparativa de funcionalidades e a tabela de custos.
- **Critérios de decisão** — lista dos critérios que considerava mais importantes (precisão, custo, facilidade de integração, privacidade, manutenção futura).
- **Conclusões** — a minha recomendação inicial, que já apontava para um caminho misto: usar uma solução cloud para um cenário rápido de integração, mas ter uma alternativa on-premises para cenários em que o custo ou a privacidade sejam críticos.

A apresentação correu bem e foi nesta reunião que os orientadores tomaram a decisão final: avançar com **DeepFace** e **AWS Rekognition** como as duas soluções a implementar na POC. O DeepFace foi escolhido por ser a opção open-source mais completa e ativa, e a AWS foi escolhida por ter a melhor oferta cloud, com trial generoso e com uma função de liveness (Face Liveness) que é considerada uma das mais robustas do mercado.

---

## 5. Fase 2 — Implementação da POC

### 5.1 Escolha das soluções

Como referi na secção anterior, a escolha final foi DeepFace + AWS Rekognition. A partir daqui, o meu trabalho foi o de implementar todas as funcionalidades pedidas nas duas soluções, de forma a que pudessem ser comparadas lado a lado, e construir uma aplicação web que servisse como interface para testar cada funcionalidade.

### 5.2 Arquitetura geral do projeto

Desde cedo decidi que a aplicação tinha de ter uma arquitetura modular, em que cada funcionalidade estivesse isolada numa rota própria, e em que as duas "engines" (DeepFace e AWS) coexistissem dentro do mesmo projeto. A arquitetura geral ficou assim:

```
+----------------------+
|   Browser (cliente)  |
+----------+-----------+
           |
           | HTTP/HTTPS
           |
+----------v-----------+
|   FastAPI (port 8000)|   <-- Servidor principal (backend/main.py)
|   - DeepFace routes  |
|   - AWS routes       |
|   - Serve frontend   |
+------+--------+------+
       |        |
       |        |
       |        +----------> AWS Rekognition (cloud)
       |                     via boto3
       |
       v
+------+--------------+
|  DeepFace (local)   |
|  - ArcFace          |
|  - YuNet            |
|  - Anti-spoofing    |
+---------------------+

+----------------------+
|   Flask (port 5001)  |   <-- Servidor de liveness (liveness/server.py)
|   - API credentials  |
|   - API result       |
|   - Serve React app  |
+----------------------+

+----------------------+
| FastAPI (port 8001)  |   <-- Webservice DeepFace isolado
|   - Só API DeepFace  |       (backend/deepface_webservice.py)
+----------------------+
```

O servidor principal é um **FastAPI** que corre na porta **8000** e serve tanto o frontend como todas as rotas de API. Dentro deste servidor, há dois grandes grupos de rotas: as rotas DeepFace (`/api/verify`, `/api/identify`, `/api/register`, `/api/quality`, `/api/models`, `/api/document`, `/api/multi`, `/api/video`, `/api/live`, `/api/attributes`) e as rotas AWS (`/api/aws/verify`, `/api/aws/identify`, `/api/aws/register`, `/api/aws/liveness`, `/api/aws/config`).

Em paralelo, existe um segundo servidor — uma aplicação **Flask** na porta **5001** — que é dedicada exclusivamente ao liveness da AWS. Esta separação foi uma decisão técnica: o componente React oficial da AWS (`FaceLivenessDetector`, da biblioteca `@aws-amplify/ui-react-liveness`) precisa de credenciais AWS e tem uma forma própria de comunicar com o backend, e achei mais limpo isolar isso num servidor à parte do que misturar com o FastAPI principal. O servidor Flask arranca automaticamente em background quando o FastAPI arranca — isto acontece no `@app.on_event("startup")` do [backend/main.py](backend/main.py), que verifica se o Flask já está a correr e, se não estiver, arranca-o numa thread daemon.

Por fim, existe ainda um **terceiro servidor** (também em FastAPI) na porta **8001**, que expõe apenas as rotas DeepFace sem frontend e sem rotas AWS. Este é o `deepface_webservice.py`, pensado para o caso em que alguém queira usar apenas a API DeepFace sem levantar o resto da aplicação. Serviu-me sobretudo para simular como seria expor o DeepFace como um microserviço isolado, tema que falámos nas reuniões com os orientadores.

### 5.3 Tecnologias utilizadas

Segue a lista das principais tecnologias usadas no projeto, com uma breve justificação para cada uma:

- **Python 3** — Linguagem principal, imposta como requisito no início do estágio.
- **FastAPI** — Framework web escolhido para o servidor principal. Escolhi-o em vez do Flask (para o servidor principal) porque tem suporte nativo para async, documentação automática via OpenAPI (`/docs`), e é muito rápido a desenvolver.
- **Flask** — Usado apenas no servidor de liveness. Não precisava de async ali e queria manter a parte de liveness o mais simples possível.
- **Uvicorn** — Servidor ASGI que corre o FastAPI.
- **DeepFace** — Biblioteca principal para reconhecimento facial local.
- **OpenCV (cv2)** — Usado para manipulação de imagens, cálculos de desfocagem (Laplaciano), conversão de espaços de cor, rotação de imagens, etc.
- **NumPy** — Manipulação numérica dos arrays de imagens e cálculos de distância entre embeddings.
- **Pillow (PIL)** — Manipulação de imagens de alto nível, conversão de formatos, redimensionamento.
- **boto3** — SDK oficial da AWS para Python.
- **React + AWS Amplify UI** — Usados apenas no mini-app de liveness (no diretório `liveness/app/`). O bundle já vem compilado no `dist/`.
- **HTML + CSS + JavaScript puro** — Frontend principal. Escolhi não usar nenhum framework JS para simplificar a manutenção e para poder servir tudo como ficheiros estáticos a partir do próprio FastAPI.
- **face-api.js** — Biblioteca JavaScript usada no frontend para fazer deteção básica de faces no browser antes de enviar para o servidor, o que evita pedidos desnecessários.
- **psutil** — Usado para medir o consumo de memória dos modelos durante a avaliação comparativa.
- **python-dotenv** — Para carregar as variáveis de ambiente do ficheiro `.env`.

### 5.4 Estrutura do repositório

A estrutura do repositório ficou assim:

```
POC - Facial Recognition/
│
├── backend/
│   ├── main.py                          # Ponto de entrada do FastAPI principal
│   ├── deepface_webservice.py           # Webservice DeepFace dedicado
│   ├── config.py                        # Configurações globais
│   │
│   ├── deepface/
│   │   ├── routes/                      # Rotas API DeepFace
│   │   │   ├── verify.py
│   │   │   ├── identify.py
│   │   │   ├── register.py
│   │   │   ├── quality.py
│   │   │   ├── models.py
│   │   │   ├── document.py
│   │   │   ├── multi.py
│   │   │   ├── video.py
│   │   │   ├── live.py
│   │   │   ├── attributes.py
│   │   │   └── config.py
│   │   └── services/                    # Lógica de negócio
│   │       ├── deepface_service.py
│   │       ├── document_service.py
│   │       ├── model_comparison_service.py
│   │       └── quality_service.py
│   │
│   └── aws/
│       ├── routes/                      # Rotas API AWS
│       │   ├── verify.py
│       │   ├── identify.py
│       │   ├── register.py
│       │   ├── liveness.py
│       │   └── config.py
│       └── services/
│           ├── rekognition_service.py
│           ├── metrics_service.py
│           └── anti_spoofing_service.py
│
├── frontend/
│   ├── index.html                       # Dashboard DeepFace
│   ├── verify.html, identify.html, register.html, multi.html
│   ├── video.html, live.html, document.html, compare.html, models.html
│   ├── deepface-config.html
│   ├── aws-index.html, aws-verify.html, aws-identify.html,
│   ├── aws-register.html, aws-multi.html, aws-stats.html, aws-config.html
│   ├── css/style.css
│   ├── js/nav.js, js/smart-camera.js
│   └── models/                          # Modelos face-api.js (tiny_face_detector)
│
├── liveness/
│   ├── server.py                        # Flask (port 5001)
│   └── app/dist/                        # React app compilado (AWS Amplify)
│
├── database/
│   ├── faces/                           # Embeddings DeepFace por pessoa
│   └── documents/                       # Fotos extraídas de documentos
│
├── config.json                          # Thresholds persistentes
├── .env                                 # Credenciais AWS (não versionado)
├── .env.example                         # Template do .env
├── requirements.txt
├── run.py                               # Arranca servidor principal (8000)
└── run_deepface.py                      # Arranca webservice DeepFace (8001)
```

Esta organização foi algo que fui refinando ao longo do estágio. No início, tinha tudo num único ficheiro `main.py` de 800 linhas, mas rapidamente percebi que isso ia ficar impossível de manter. Decidi separar por "engine" (DeepFace vs AWS) e depois, dentro de cada engine, por "rotas" e "serviços", seguindo mais ou menos o padrão que se vê em projetos FastAPI de maior dimensão.

---

## 6. Implementação DeepFace

### 6.1 Setup do ambiente Python

A primeira tarefa da fase de implementação foi preparar um ambiente Python isolado, para garantir reprodutibilidade. Criei um `venv` dentro da pasta do projeto e escrevi um `requirements.txt` com as dependências principais. As dependências são:

```
fastapi>=0.100.0
uvicorn>=0.23.0
python-multipart>=0.0.6
deepface>=0.0.89
opencv-python>=4.8.0
numpy>=1.24.0
Pillow>=10.0.0
psutil>=5.9.0
easyocr>=1.7.0
insightface>=0.7.3
onnxruntime-directml>=1.19.0
```

A primeira coisa que aprendi (da pior forma possível) é que o DeepFace traz consigo uma enorme cadeia de dependências pesadas — TensorFlow, Keras, várias versões de OpenCV, dlib, entre outras — e nem sempre instala de forma limpa em máquinas Windows. A primeira vez que tentei instalar, a instalação demorou quase 20 minutos e deu três erros pelo meio, todos relacionados com versões do NumPy e do protobuf. Tive de fazer bastante pesquisa para perceber qual a combinação de versões que funcionava, e no fim a solução foi fixar as versões de algumas bibliotecas e deixar o DeepFace instalar o resto.

Outra dificuldade no setup foi o facto de a primeira execução do DeepFace descarregar automaticamente os pesos dos modelos (ArcFace, YuNet, modelos de anti-spoofing, modelos de idade/género/emoção) para o diretório `~/.deepface/weights/`. Este download é transparente mas lento na primeira vez, e se não houver internet falha. Para contornar isto, implementei um mecanismo de "warmup" que corre no arranque do servidor e que pré-carrega os modelos principais, para eliminar o cold start na primeira chamada de API. O warmup está em [backend/main.py:187-207](backend/main.py#L187) e faz uma primeira chamada dummy ao `DeepFace.represent` com ArcFace e outra ao `DeepFace.extract_faces` com YuNet, usando uma imagem em branco de 112x112.

### 6.2 Pipeline de registo biométrico

O registo biométrico é o fluxo mais básico de qualquer sistema de reconhecimento facial: pega-se numa imagem de uma pessoa, extrai-se a face, calcula-se o embedding (o vetor numérico que representa a face) e guarda-se esse vetor na base de dados, associado ao nome da pessoa.

Na minha implementação, o pipeline está no ficheiro [backend/deepface/services/deepface_service.py](backend/deepface/services/deepface_service.py), na função `register_face`. O fluxo é:

1. **Normalização do nome** — O nome da pessoa é convertido para ASCII puro, sem acentos nem caracteres especiais, para evitar problemas de codificação em nomes de ficheiros no Windows. Isto é feito pela função `_ascii_safe_name`, que usa o módulo `unicodedata` para normalizar a string em forma NFKD e depois remove os caracteres não-ASCII.
2. **Guardar a imagem num ficheiro temporário** — A imagem recebida é guardada no diretório `temp/` com um nome aleatório (UUID). Antes de guardar, é redimensionada para no máximo 640 pixels no lado maior, porque o DeepFace (com ArcFace + YuNet) não precisa de mais do que isso, e resoluções maiores só atrasam a inferência sem ganhos de precisão.
3. **Auto-orientação** — A imagem é passada pela função `_auto_orient_image_for_face`, que tenta detetar a face em 4 rotações diferentes (0°, 90°, 180°, 270°) e escolhe a rotação em que a deteção tem maior confiança. Esta parte surgiu porque reparei que muitas vezes as pessoas tiravam fotos com o telemóvel virado para o lado, e o DeepFace falhava a deteção apesar de a face estar claramente na imagem.
4. **Extração da face** — Usa `DeepFace.extract_faces` com o detector `yunet` (que é rápido e preciso) e opcionalmente com anti-spoofing ativado se a imagem vier da câmara.
5. **Cálculo do embedding** — Chama `DeepFace.represent` com o modelo `ArcFace`, que devolve um vetor de 512 dimensões. Este vetor é o que fica guardado.
6. **Persistência** — O embedding é adicionado a um ficheiro JSON por pessoa, em `database/faces/<nome>/embeddings.json`. A cada registo, o ficheiro é lido, o novo embedding é adicionado à lista e o ficheiro é gravado novamente. Esta abordagem permite ter várias imagens por pessoa, o que melhora a precisão da identificação 1:N quando há variação de iluminação ou ângulo.

Uma decisão importante que tomei neste pipeline foi a de **não guardar a imagem original** em disco, apenas o embedding. Isto tem duas vantagens:

- **Espaço em disco** — os embeddings ocupam apenas alguns KB cada, enquanto as imagens podem ter vários MB.
- **Privacidade** — se a base de dados for comprometida, o atacante fica apenas com os vetores, que não são invertíveis na prática para reconstruir a imagem original.

Tenho também uma função chamada `get_registered_people` que lê o diretório `database/faces/` e devolve uma lista das pessoas registadas, e uma função `delete_person` que remove uma pessoa da base de dados. Esta última tem uma pequena mas importante verificação de segurança: usa `os.path.realpath` para garantir que o nome recebido não aponta para fora do diretório `DATABASE_DIR` — isto evita ataques do tipo "path traversal" em que alguém tentasse passar um nome como `../../etc/passwd`. Foi uma das preocupações que os orientadores me fizeram ter logo no início, e que eu acabei por replicar em todos os sítios onde há manipulação de nomes de ficheiros.

### 6.3 Verificação 1:1

A verificação 1:1 é o caso em que se pergunta "estas duas imagens são da mesma pessoa?". A rota está em [backend/deepface/routes/verify.py](backend/deepface/routes/verify.py) e aceita duas imagens via upload multipart, mais um parâmetro `mode` que pode ser `"file"` ou `"camera"`.

- Em modo `file`, o anti-spoofing está desativado (porque uma imagem de ficheiro é sempre tecnicamente uma "fotografia" e ia ser marcada como spoof).
- Em modo `camera`, o anti-spoofing está ativado na imagem ao vivo (não na outra, porque isso bloquearia a comparação contra uma foto de referência).

A lógica está na função `verify_1to1` do `deepface_service.py`. O fluxo:

1. Guarda as duas imagens temporariamente.
2. Aplica auto-orientação a ambas (só à segunda se estivermos em modo câmara, porque a câmara já vem à direita).
3. Se for modo câmara, chama `_check_live_anti_spoofing` na imagem da câmara. Se detetar spoof, devolve imediatamente um erro.
4. Chama `DeepFace.verify` com os parâmetros: `model_name=ArcFace`, `detector_backend=yunet`, `distance_metric=cosine`, `threshold=get_df_runtime_config()["df_verify_threshold"]`. O threshold é lido do `config.json` e pode ser alterado em runtime na página de configuração.
5. Devolve o resultado, que inclui `verified` (True/False), `distance`, `threshold`, `model`, `similarity_metric` e `message`.

Uma coisa curiosa que aconteceu aqui: nos primeiros testes, o DeepFace rejeitava bastantes imagens com o erro "Face could not be detected", mesmo quando a face estava visível. Descobri que o YuNet era mais exigente do que o OpenCV básico em termos de qualidade da imagem. Para contornar isto, implementei um **fallback** — se a deteção com YuNet falhar, tenta-se com o detector `opencv`, que é menos preciso mas mais tolerante. Isto evita ter de rejeitar imagens que, na prática, têm uma face detetável.

O threshold usado por defeito é **0.54** (distância cosseno). Abaixo disto, é considerado match; acima, é considerado não-match. Este valor foi afinado empiricamente depois de testar com várias pessoas, e ficou um pouco mais restritivo do que o valor por defeito do ArcFace (0.68), porque com o default tinha alguns falsos positivos (pessoas diferentes a serem consideradas iguais).

### 6.4 Identificação 1:N

A identificação 1:N é o caso em que se pergunta "esta pessoa está na minha base de dados?". A diferença em relação à verificação 1:1 é que em vez de compararmos a imagem com uma só outra imagem, comparamos com **todas** as imagens registadas.

A minha implementação está na função `identify_1toN` e tem uma otimização importante: em vez de chamar `DeepFace.verify` em loop contra cada pessoa (que recalcula o embedding da imagem a cada iteração), eu calculo o embedding da imagem de entrada **uma só vez** com `DeepFace.represent`, e depois comparo esse embedding com todos os embeddings pré-calculados que estão guardados em disco.

O fluxo:

1. Lê a lista de pessoas registadas.
2. Calcula o embedding da imagem de entrada (uma chamada ao `DeepFace.represent`).
3. Para cada pessoa, lê o ficheiro `embeddings.json` e calcula a distância cosseno entre o embedding de entrada e cada embedding guardado.
4. Ordena todos os candidatos por distância crescente.
5. Filtra os que estão abaixo do threshold (0.54 por defeito).
6. Devolve os top-5.

O cálculo da distância cosseno está na função `_compute_distance` e é feito manualmente com NumPy, sem depender do DeepFace:

```python
norm_a = np.linalg.norm(a_arr)
norm_b = np.linalg.norm(b_arr)
return float(1.0 - np.dot(a_arr, b_arr) / (norm_a * norm_b))
```

Esta otimização faz uma diferença gigante em termos de performance. Com esta abordagem, identificar uma pessoa numa base de dados com 50 embeddings demora praticamente o mesmo que com 5 embeddings, porque o custo dominante é a extração do embedding de entrada (cerca de 150-200 ms), não as comparações em si (que são essencialmente operações vetoriais com NumPy, na ordem dos microssegundos).

Uma coisa que também acrescentei foi uma mensagem informativa quando **não** é encontrado match. Em vez de só dizer "Nenhuma correspondência encontrada", a resposta inclui o **candidato mais próximo** e a sua distância, para que o utilizador perceba "por pouco" ou se não havia mesmo nada parecido. Isto foi uma sugestão direta dos orientadores na reunião em que apresentei a primeira versão desta função.

### 6.5 Avaliação de qualidade da imagem

A avaliação de qualidade da imagem foi uma das tarefas mais interessantes da implementação, porque obrigou-me a pensar "fora da caixa do reconhecimento facial" — o problema aqui não é comparar caras, é avaliar se a imagem serve para ser usada em reconhecimento facial.

O serviço está em [backend/deepface/services/quality_service.py](backend/deepface/services/quality_service.py) e avalia vários atributos em sequência:

- **Resolução** — verifica se a imagem tem pelo menos 160×120 pixels. Abaixo disto não vale a pena sequer tentar.
- **Brilho** — calcula o brilho médio da imagem em escala de cinzentos. Se for menor que 20 (muito escuro) ou maior que 240 (muito claro), rejeita. O intervalo aceitável é 20-240.
- **Contraste** — calcula o desvio-padrão dos níveis de cinzento. Abaixo de 15 é considerado baixo contraste.
- **Desfocagem** — usa a **variância do Laplaciano**, uma técnica clássica em visão por computador. Aplica-se o filtro Laplaciano à imagem em tons de cinzento e calcula-se a variância do resultado. Valores baixos indicam imagem desfocada (porque o Laplaciano deteta bordas, e numa imagem desfocada há poucas bordas nítidas). O threshold mínimo é 30, mas este valor é configurável pelo utilizador na página de configuração.
- **Tamanho da face** — depois de detetar a face, verifica se ela tem pelo menos 50 pixels em largura e altura, e se ocupa pelo menos 2% da imagem. Se for demasiado pequena, rejeita (porque o ArcFace não extrai embeddings fiáveis de faces pequenas).
- **Confiança da deteção** — verifica se o YuNet (ou fallback OpenCV) detetou a face com confiança superior a 0.7.

Uma decisão técnica importante foi **calcular o blur na região da face**, não na imagem inteira. A razão é simples: se alguém tira uma foto com um fundo uniforme (uma parede branca, por exemplo), a variância do Laplaciano da imagem toda vai ser baixa só por causa do fundo, mesmo que a face esteja perfeitamente nítida. Ao recortar apenas a região da face antes de calcular o blur, evito estes falsos positivos. Este recorte está em `full_quality_assessment` e usa as coordenadas devolvidas pelo `DeepFace.extract_faces`.

Outra dificuldade que encontrei aqui foi a deteção de faces em imagens mal orientadas. A minha solução foi replicar a lógica de auto-rotação que já usava noutras partes, tentando quatro rotações e escolhendo a que deu melhor confiança.

No final, a função devolve uma lista de "checks", cada um com o seu resultado individual (passou / não passou) e uma mensagem explicativa, mais um flag geral `overall_passed` que é `True` apenas se todos os checks passarem.

### 6.6 Avaliação dos melhores modelos

Uma das tarefas pedidas foi comparar os modelos de reconhecimento disponíveis no DeepFace e decidir qual o mais adequado para o caso de uso. Os modelos avaliados foram:

- **ArcFace** — ResNet-100, embeddings de 512 dimensões, precisão LFW de 99.83%. É considerado estado da arte.
- **Facenet512** — Inception-ResNet, embeddings de 512 dimensões, precisão LFW de 99.65%. Foi desenvolvido pela Google.
- **VGG-Face** — VGG-16, embeddings de 4096 dimensões, precisão LFW de 98.78%. É um modelo mais antigo mas ainda muito usado.
- **SFace** — MobileNet, embeddings de 128 dimensões, precisão LFW de 99.60%. Pensado para dispositivos com poucos recursos.

A lista está em `MODELS_TO_EVALUATE` no `backend/config.py` e a lógica de avaliação está em [backend/deepface/services/model_comparison_service.py](backend/deepface/services/model_comparison_service.py).

A função principal é `evaluate_models_verify`, que recebe duas imagens e corre `DeepFace.verify` com cada modelo, medindo:

- **Tempo de processamento** — medido com `time.time()` antes e depois.
- **Consumo de memória** — medido com `psutil`, comparando a memória residente antes e depois da chamada.
- **Distância** — a distância entre os embeddings gerados pelo modelo.
- **Verificado** — se o resultado é considerado match pelo threshold próprio do modelo.

No final, devolve uma matriz comparativa em JSON que é renderizada no frontend em [frontend/models.html](frontend/models.html) como uma tabela.

Os resultados que obtive (em termos médios, numa máquina com CPU Intel i7, sem GPU) foram mais ou menos os seguintes:

- **ArcFace**: 350-500 ms por verificação, embeddings 512, consumo de memória ~200 MB uma vez carregado.
- **Facenet512**: 400-550 ms, 512 dimensões, consumo semelhante ao ArcFace.
- **VGG-Face**: 700-900 ms, embeddings gigantes de 4096 dimensões, consumo de memória mais de 500 MB. É lento.
- **SFace**: 150-250 ms, embeddings pequenos de 128 dimensões, consumo baixo. O mais rápido de todos.

A conclusão que cheguei foi que, para um sistema em produção com restrições de precisão, o **ArcFace** continua a ser a melhor escolha (e é por isso que está definido como `RECOGNITION_MODEL = "ArcFace"` no `config.py`). Mas para um cenário em que a velocidade é mais importante do que a precisão máxima (por exemplo, numa aplicação móvel ou num sistema embarcado), o **SFace** é uma alternativa muito competitiva. O Facenet512 não me pareceu trazer nenhuma vantagem clara sobre o ArcFace para este caso de uso. O VGG-Face só usaria se tivesse um requisito específico que obrigasse a isso, porque é claramente o mais lento e o mais pesado.

### 6.7 Extração da fotografia do documento de identificação

Esta foi, de longe, a tarefa mais complicada de toda a POC. A ideia, tal como foi pedida, é replicar o fluxo típico de um banco ou de um serviço de onboarding digital: o utilizador tira uma foto ao documento de identificação, o sistema extrai a fotografia do titular do documento, depois o utilizador tira uma selfie, e o sistema verifica se a pessoa na selfie é a mesma pessoa que está no documento.

O serviço está em [backend/deepface/services/document_service.py](backend/deepface/services/document_service.py) e tem cerca de 850 linhas. As rotas estão em [backend/deepface/routes/document.py](backend/deepface/routes/document.py) e expõem quatro endpoints principais:

- `POST /api/document/validate` — valida rapidamente se a imagem é um documento com face detetável.
- `POST /api/document/extract` — extrai a face do documento e cria uma sessão.
- `POST /api/document/verify-live` — recebe a selfie e compara com a face do documento dessa sessão.
- `DELETE /api/document/session/{session_id}` — limpa a sessão.

O fluxo completo é:

1. **Validação rápida** — antes de processar o documento a sério, é feita uma validação rápida que confirma que existe de facto um documento (retângulo detetável) e uma face na imagem. Isto evita que o utilizador carregue uma foto da gaveta ou um screenshot e só descubra o erro depois de todo o processamento.
2. **Deteção do retângulo do documento** — a função `_detect_document_rectangle` usa várias técnicas de OpenCV (Otsu em duas polaridades, Canny com dilatação) para tentar detetar um retângulo proeminente na imagem, que corresponde ao documento. Esta parte foi surpreendentemente difícil, porque a iluminação varia muito e os documentos têm cantos arredondados que confundem as heurísticas.
3. **Extração da face** — depois de validar que há um documento, extrai-se a face com `DeepFace.extract_faces`. Aqui também se tenta várias rotações, porque muitas vezes as pessoas tiram a foto do documento na horizontal.
4. **Avaliação de qualidade** — a face extraída é passada pelo `full_quality_assessment` para verificar se tem qualidade suficiente para ser usada na comparação.
5. **Criação da sessão** — é gerado um `session_id` (UUID) e a face extraída é guardada temporariamente em `database/documents/<session_id>.jpg`. A sessão fica válida por 30 minutos.
6. **Selfie da câmara** — quando o utilizador tira a selfie, a imagem é enviada para `verify-live` juntamente com o `session_id`. O sistema lê a face do documento dessa sessão e faz a comparação 1:1.
7. **Pré-processamento CLAHE** — esta foi uma das grandes descobertas desta parte do projeto. Os resultados iniciais eram muito maus: a distância entre a face do documento e a face da câmara era sempre muito alta, muitas vezes acima do threshold, mesmo para pessoas iguais. O problema é que as fotos dos documentos são muitas vezes a preto e branco (ou com cores muito mortas), tiradas com iluminação de estúdio, e comprimidas. A selfie é a cores, com iluminação ambiente. Estas duas imagens "parecem" muito diferentes para o modelo. A minha solução foi aplicar **CLAHE** (Contrast Limited Adaptive Histogram Equalization) a ambas as imagens antes de as comparar, para normalizar iluminação e contraste. Adicionalmente, verifico se a foto do documento é efetivamente a preto e branco (comparando a diferença média entre os canais R, G e B) e, se for, aplico o CLAHE no canal cinzento e reconverto para BGR para o DeepFace processar.
8. **Threshold mais permissivo** — além do pré-processamento, usei um threshold mais permissivo (`DOCUMENT_THRESHOLD = 0.65`) nesta comparação, porque mesmo com o CLAHE a comparação continua a ser mais "difícil" do que a comparação entre duas selfies.

O threshold e o pré-processamento juntos fizeram com que os resultados passassem de cerca de 30% de taxa de sucesso (no início) para mais de 85% (no final) em testes com as fotos dos meus colegas.

Uma nota sobre segurança: as imagens dos documentos são sempre guardadas numa sessão temporária com TTL de 30 minutos, e há uma tarefa de cleanup que corre no arranque do servidor (`cleanup_old_sessions`) e outra que pode ser chamada via endpoint para apagar sessões antigas. As imagens nunca são registadas em logs.

### 6.8 Análise de atributos faciais

A análise de atributos faciais é uma funcionalidade que o DeepFace oferece "de graça" e que decidi expor porque achei interessante ter essa comparação também com a AWS (que oferece o mesmo tipo de análise em `DetectFaces`).

A função `analyze_face_attributes` chama `DeepFace.analyze` com as ações `["age", "gender", "emotion", "race"]` e devolve:

- **Idade estimada** (com uma margem de erro de ±4 anos, que é a margem típica do modelo de idade do DeepFace).
- **Género dominante** com nível de confiança.
- **Emoção dominante** (feliz, triste, zangado, medo, surpresa, nojo, neutro) com emoji correspondente.
- **Etnia dominante** com distribuição das outras etnias.

Esta análise é acessível pelo endpoint `POST /api/attributes/` e é usada também na câmara ao vivo.

### 6.9 Câmara ao vivo

A câmara ao vivo foi a funcionalidade que mais trabalho me deu do lado do frontend, mas que ficou também a mais vistosa. A ideia é ter uma câmara aberta no browser, com um canvas por cima onde são desenhadas as bounding boxes das faces detetadas, e que em tempo real identifique quem está à frente da câmara.

A rota está em [backend/deepface/routes/live.py](backend/deepface/routes/live.py) e aceita um frame JPEG de cada vez em `POST /api/live/analyze`. Para cada frame:

1. Calcula a nitidez (variância do Laplaciano). Se estiver abaixo de 30, ignora o frame (está demasiado desfocado para valer a pena processar).
2. Deteta todas as faces com YuNet.
3. Para cada face, verifica se tem pelo menos 40 pixels.
4. Recorta a região da face com uma pequena margem.
5. Se o torch estiver disponível, corre o anti-spoofing nessa região.
6. Se a face for considerada real, chama `identify_1toN` apenas com o crop da face (não a imagem inteira).
7. Devolve uma lista de faces com a sua bounding box, o seu estado (`identified`, `not_found`, `spoof`, `blurry_face`), e se aplicável, o nome da pessoa identificada e a similaridade.

Do lado do frontend, a página [frontend/live.html](frontend/live.html) tem a câmara do navegador aberta, captura um frame a cada 400 ms (configurável), envia-o para o endpoint, e desenha as bounding boxes no canvas por cima do vídeo. As bounding boxes são suavizadas com interpolação linear (LERP) para não saltarem bruscamente entre frames, o que dá uma aparência muito mais polida. Cada face tem uma cor diferente consoante o estado: verde se identificada, amarelo se desconhecida, vermelho se spoof detetado, roxo se desfocada.

A parte de suavização foi algo que fui descobrindo ao longo do desenvolvimento. Inicialmente, as caixas saltavam de um lado para o outro a cada frame, o que dava uma sensação muito má. Com o LERP de 0.25 para a posição e 0.15 para a opacidade (valores que encontrei por tentativa e erro), o resultado ficou agradável de ver.

### 6.10 Deteção multi-pessoa

Esta funcionalidade permite carregar uma imagem com várias pessoas e receber de volta todas as faces detetadas, com as suas bounding boxes. Está implementada em [backend/deepface/routes/multi.py](backend/deepface/routes/multi.py) e suporta dois motores: AWS (`DetectFaces`) e DeepFace (`extract_faces` com `yunet`).

As faces são devolvidas já ordenadas da esquerda para a direita (depois, em caso de empate, de cima para baixo), o que facilita a interação no frontend — o utilizador pode selecionar a pessoa "número 3" ou a pessoa "número 7" sem ter de adivinhar a que face é que o sistema se está a referir.

### 6.11 Pesquisa em vídeo

A pesquisa em vídeo é uma funcionalidade mais avançada, em que o utilizador carrega um vídeo e uma fotografia de referência, e o sistema procura ocorrências dessa pessoa no vídeo. A rota está em [backend/deepface/routes/video.py](backend/deepface/routes/video.py) e usa **Server-Sent Events (SSE)** para enviar resultados em streaming ao frontend à medida que vai processando os frames.

O fluxo:

1. Recebe o vídeo e a imagem de referência.
2. Abre o vídeo com OpenCV e determina a duração e o frame rate.
3. Extrai um frame a cada N segundos (configurável, tipicamente 1-2 segundos).
4. Para cada frame, faz uma "janela" de 4 frames adjacentes e escolhe o mais nítido (maior variância do Laplaciano).
5. Aplica um **unsharp mask** para realçar bordas ligeiramente desfocadas.
6. Compara esse frame com a imagem de referência usando `verify_1to1`.
7. Se houver match, envia um evento SSE ao frontend com o timestamp, a distância e um thumbnail do frame.
8. Faz **deduplicação** de matches próximos: se dois matches estão a menos de 2 segundos de distância, considera que é o mesmo encontro e só reporta uma vez.

A limitação que pus foi de 5 minutos de duração máxima por vídeo, para evitar que alguém carregue um filme inteiro e rebente com o servidor.

### 6.12 Webservice DeepFace dedicado

Por fim, há o `deepface_webservice.py`, que é uma versão "slim" do FastAPI que expõe apenas as rotas DeepFace, sem frontend e sem AWS. Corre na porta **8001** e é arrancado com `python run_deepface.py`. Tem a sua própria documentação OpenAPI em `http://localhost:8001/docs`.

A ideia é simular como seria expor o DeepFace como um microserviço autónomo, que poderia correr numa máquina dedicada com GPU e ser consumido por outros serviços da empresa. Falámos sobre isto numa das reuniões semanais e acabei por incluir esta variante precisamente para mostrar essa possibilidade.

---

## 7. Implementação AWS Rekognition

### 7.1 Configuração e preparação do ambiente AWS

Antes de poder usar o Rekognition, tive de criar uma conta AWS e configurar as credenciais. Criei um utilizador IAM dedicado para a POC, com uma política de permissões mínimas que incluía apenas os serviços do Rekognition que ia usar:

- `rekognition:CreateCollection`
- `rekognition:ListCollections`
- `rekognition:IndexFaces`
- `rekognition:ListFaces`
- `rekognition:DeleteFaces`
- `rekognition:SearchFacesByImage`
- `rekognition:CompareFaces`
- `rekognition:DetectFaces`
- `rekognition:CreateFaceLivenessSession`
- `rekognition:GetFaceLivenessSessionResults`

Gerei as chaves de acesso (`AWS_ACCESS_KEY_ID` e `AWS_SECRET_ACCESS_KEY`) e coloquei-as num ficheiro `.env` na raiz do projeto, que é carregado automaticamente pelo `python-dotenv`. O ficheiro `.env` está no `.gitignore` e não vai para o repositório — em vez disso, há um `.env.example` com os nomes das variáveis sem os valores, que serve de template.

A região escolhida foi **eu-west-1** (Irlanda), porque é a mais próxima de Portugal com todos os serviços disponíveis.

O cliente AWS está encapsulado na classe `RekognitionClient` em [backend/aws/services/rekognition_service.py](backend/aws/services/rekognition_service.py). Tem um método `_timed_call` que envolve todas as chamadas boto3 e regista o tempo decorrido em milissegundos, o que depois é usado pelas métricas. Também tem um helper `_to_jpeg` que converte qualquer imagem para JPEG e a redimensiona se for maior que 5 MB, porque o Rekognition tem esse limite.

### 7.2 Registo na coleção (IndexFaces)

A AWS tem o conceito de **coleção** — uma espécie de base de dados remota onde se guardam as faces indexadas. Cada face tem um `FaceId` (gerado pela AWS) e um `ExternalImageId` (gerido por nós, tipicamente o nome da pessoa).

A coleção que criei para a POC chama-se `poc-asseco-faces` (definido em `AWS_COLLECTION_ID`). A rota `POST /api/aws/register` faz os seguintes passos:

1. Garante que a coleção existe (`create_collection`, que é idempotente).
2. Normaliza o nome recebido para caracteres ASCII (a AWS só aceita certos caracteres no `ExternalImageId`).
3. Chama `detect_faces` primeiro para verificar que existe uma face na imagem (e rejeitar cedo se não houver).
4. Chama `index_faces` com `QualityFilter=AUTO`, que diz à AWS para rejeitar automaticamente imagens de má qualidade (faces pequenas, mal orientadas, etc.).
5. Regista a operação no collector de métricas.

O `index_faces` devolve um `FaceId` único e o `confidence` da deteção, que fica também guardado para referência futura.

### 7.3 Verificação 1:1 (CompareFaces)

A verificação 1:1 na AWS usa a operação `CompareFaces`, que recebe duas imagens e devolve o grau de similaridade entre a face da imagem origem e as faces da imagem destino. A rota é `POST /api/aws/verify`.

Uma subtileza importante que apanhei foi a seguinte: a operação `CompareFaces` tem um parâmetro `SimilarityThreshold` que, se for maior que zero, faz com que a AWS **só devolva** matches acima desse valor. O problema é que, se eu usar o threshold da POC (95), perdia a informação da similaridade das faces que estavam abaixo desse valor. A minha solução foi passar `SimilarityThreshold=0` ao boto3 e aplicar **eu** o threshold a seguir, em Python. Assim, consigo saber sempre o valor real da similaridade, mesmo quando não há match — por exemplo, para mostrar ao utilizador "similaridade foi de 82%, ficou abaixo do threshold de 95%".

Além da rota `/verify`, criei também uma variante `/verify/by-name` que recebe um nome e uma imagem, e verifica se a imagem corresponde à pessoa com aquele nome na coleção. Em vez de fazer `CompareFaces`, esta rota faz `SearchFacesByImage` com um threshold baixo (0), recebe todos os matches, e filtra pelo `ExternalImageId` que corresponde ao nome.

### 7.4 Identificação 1:N (SearchFacesByImage)

A identificação 1:N na AWS usa `SearchFacesByImage`. É a operação mais direta: passa-se uma imagem, um `CollectionId` e um `FaceMatchThreshold`, e a AWS devolve a lista de faces na coleção que têm similaridade acima do threshold, ordenadas por similaridade.

A rota está em [backend/aws/routes/identify.py](backend/aws/routes/identify.py) e é praticamente um wrapper fino em volta da chamada AWS. O threshold por defeito é 80 (configurável na página `/aws/config`), e a resposta inclui o top-K de matches, a confiança da face pesquisada, o threshold usado e a latência.

### 7.5 Métricas de qualidade e performance

Uma das tarefas explicitamente pedidas na POC foi medir a latência e a qualidade do Rekognition. Para isto, criei um `MetricsCollector` (classe em [backend/aws/services/metrics_service.py](backend/aws/services/metrics_service.py)) que regista, para cada operação chamada:

- Nome da operação (ex: `CompareFaces`, `SearchFacesByImage`, `IndexFaces`).
- Latência em milissegundos.
- Se foi bem-sucedida ou não.
- Timestamp.
- Metadados extra específicos à operação (ex: similaridade, confiança, número de matches).

O collector calcula automaticamente estatísticas agregadas: contagens, média, mínimo, máximo, **percentil 50 (mediana)**, **percentil 95** e **percentil 99**. Estes percentis são importantes porque a média sozinha não é suficiente — num sistema em produção, interessa perceber como é o pior caso (o p99) e não apenas o caso médio. O p95 em particular é o valor mais usado como SLA em serviços cloud.

As estatísticas são expostas no endpoint `GET /api/aws/stats` e são visualizadas no frontend em [frontend/aws-stats.html](frontend/aws-stats.html), com gráficos simples em JavaScript.

Pelos testes que fiz, as latências típicas do Rekognition na região `eu-west-1` foram (ordem de grandeza):

- `DetectFaces`: 300-500 ms
- `CompareFaces`: 400-700 ms
- `SearchFacesByImage`: 400-800 ms (depende do tamanho da coleção)
- `IndexFaces`: 500-1000 ms

O p95 fica tipicamente 30-40% acima da média, o que é normal para uma API cloud. Grande parte da latência é dominada pelo tempo de rede entre a minha máquina e a região `eu-west-1`.

### 7.6 Anti-spoofing local complementar

Além do Face Liveness oficial da AWS (que descrevo a seguir), implementei também uma camada de anti-spoofing local, em Python, que analisa uma imagem e dá-lhe um score de "vivacidade". O serviço está em [backend/aws/services/anti_spoofing_service.py](backend/aws/services/anti_spoofing_service.py) e combina quatro métricas:

- **Análise de textura** (35% do peso) — variância do Laplaciano. Fotos impressas e ecrãs têm texturas mais "lisas" do que a pele humana.
- **Análise de frequência** (20% do peso) — transformada de Fourier, olhando para a razão entre as frequências altas e baixas. Fotos têm tipicamente menos componentes de alta frequência do que imagens reais.
- **Análise de cor** (15% do peso) — diversidade cromática em espaços HSV e YCrCb. Fotos impressas têm distribuições de cor diferentes das capturadas ao vivo.
- **Qualidade AWS** (30% do peso) — usa os atributos devolvidos pelo `DetectFaces` da AWS (sharpness, brightness, confidence, pose, eyes_open, sunglasses) para compor um score.

O score final é uma média ponderada dos quatro, e se for maior que 50 a imagem é considerada "real". Este threshold é configurável.

Este anti-spoofing local é útil como **camada complementar** — se a imagem já tiver sido validada pelo Face Liveness da AWS, este serviço permite fazer uma validação extra numa única foto, sem precisar de uma nova sessão. É mais rápido (não precisa de ir à AWS) mas também menos preciso que o liveness ativo.

### 7.7 Face Liveness da AWS

O Face Liveness é a funcionalidade mais sofisticada desta parte do projeto e também a que mais me ensinou sobre integração com serviços cloud. A AWS disponibiliza um componente React oficial — `FaceLivenessDetector` da biblioteca `@aws-amplify/ui-react-liveness` — que faz um desafio interativo ao utilizador (posicionar o rosto num oval, depois uma série de flashes de luz aleatórios) e envia o vídeo diretamente para os servidores da AWS através de WebSocket para análise.

A dificuldade é que este componente não se integra facilmente com HTML puro — tem de correr dentro de uma aplicação React. A minha solução foi criar um mini-app React separado, dentro de `liveness/app/`, e compilar esse app com Vite. O bundle compilado está em `liveness/app/dist/`, que é um conjunto de ficheiros estáticos que podem ser servidos por qualquer servidor web.

O fluxo completo é:

1. O utilizador clica no botão "Iniciar liveness" no frontend principal (porta 8000).
2. O frontend faz `POST /api/aws/liveness/create`, que está na rota [backend/aws/routes/liveness.py](backend/aws/routes/liveness.py). Esta rota cria uma sessão na AWS via `create_face_liveness_session()` e devolve um `session_id`.
3. O frontend abre o mini-app React na porta 5001, passando o `session_id` como parâmetro.
4. O mini-app React pede credenciais AWS ao servidor Flask (endpoint `GET /api/credentials`) e inicializa o `FaceLivenessDetector` com o `sessionId` e as credenciais.
5. O utilizador faz o desafio: posiciona o rosto no oval, e o SDK da AWS faz flashes de luz no ecrã. O vídeo é enviado **diretamente** para a AWS via WebSocket (não passa pelo meu servidor).
6. Quando o desafio termina, o SDK chama um callback que o mini-app usa para redirecionar de volta para o FastAPI, com um parâmetro `session_id` na URL.
7. O frontend principal faz então `GET /api/aws/liveness/result/{session_id}`, que faz proxy para o Flask (`GET /api/result/{session_id}`).
8. O Flask chama `get_face_liveness_session_results()` e devolve:
   - `status`: `SUCCEEDED` ou `FAILED`.
   - `confidence`: 0-100 (nível de confiança do liveness).
   - `isLive`: `True` se `status == SUCCEEDED` **e** `confidence >= liveness_threshold` (75 por defeito).
   - `reference_image`: imagem capturada durante o desafio, codificada em base64.

O servidor Flask (na porta 5001) tem ainda algumas medidas de segurança adicionais:

- **Rate limiting** — máximo de 30 pedidos por IP por janela de 60 segundos, para evitar abuso.
- **CORS restrito** — só aceita pedidos da origem `localhost:8000` e da própria porta 5001.
- **Validação do session_id** — só aceita caracteres alfanuméricos e hífen, para evitar injeção.

O Flask arranca automaticamente em background quando o FastAPI arranca, graças a uma thread daemon que é iniciada no `startup_event` do `backend/main.py`.

Uma das dificuldades que tive aqui foi perceber como passar as credenciais AWS ao componente React. A recomendação oficial da AWS é usar **Cognito Identity Pool** com credenciais temporárias STS, mas isso implicava configurar o Cognito e trazia mais complexidade do que o necessário para uma POC. A solução que adotei foi passar as credenciais do `.env` diretamente via o endpoint `/api/credentials`, que só é acessível a partir de localhost (CORS restrito). Deixei comentado no código que, em produção, isto tem de ser substituído pelo Cognito.

---

## 8. Frontend

Apesar de o foco do estágio ser no backend e na integração com o Rekognition / DeepFace, acabei por ter de investir bastante tempo no frontend para conseguir demonstrar as funcionalidades. Decidi fazer tudo em HTML + CSS + JavaScript puro, sem recorrer a frameworks como React ou Vue, pelas seguintes razões:

1. **Simplicidade** — o frontend é essencialmente uma série de páginas isoladas, cada uma com o seu caso de uso. Não justifica a overhead de um framework.
2. **Servir estaticamente** — posso servir tudo a partir do próprio FastAPI, sem precisar de um build step.
3. **Aprendizagem** — quis também mostrar que sei fazer frontend sem dependências pesadas.

O frontend tem duas "secções" principais, separadas no menu de navegação superior: **DeepFace POC** (páginas que usam o motor DeepFace local) e **AWS POC** (páginas que usam o Rekognition). Cada secção tem o seu próprio conjunto de páginas:

- **DeepFace**: dashboard, registo, verificação 1:1, identificação 1:N, multi-pessoa, pesquisa em vídeo, câmara ao vivo, comparar motores, modelos, documento de ID, configuração.
- **AWS**: dashboard, registo, verificação 1:1, identificação 1:N, multi-pessoa, estatísticas, configuração.

Os estilos estão em [frontend/css/style.css](frontend/css/style.css) e há uma pequena lib de navegação em [frontend/js/nav.js](frontend/js/nav.js) que deteta a página ativa e destaca o link correspondente.

Há também uma biblioteca auxiliar, [frontend/js/smart-camera.js](frontend/js/smart-camera.js), que encapsula a lógica de abrir a câmara do browser, capturar frames, e enviá-los para o backend. É usada em várias páginas para evitar duplicação de código.

Um detalhe técnico que vale a pena mencionar é o uso de **face-api.js** no frontend. Esta biblioteca carrega um modelo pequeno (TinyFaceDetector) diretamente no browser, o que permite fazer deteção básica de faces em tempo real antes de enviar para o servidor. Isto é usado, por exemplo, na página de registo para mostrar ao utilizador um overlay verde quando há uma face detetada, antes de ele carregar no botão. Os modelos estão em [frontend/models/](frontend/models/) e são servidos como estáticos em `/faceapi-models/`.

---

## 9. Comparação final: DeepFace vs AWS

A página [frontend/compare.html](frontend/compare.html) foi pensada para permitir comparar os dois motores lado a lado, carregando as mesmas imagens e vendo o que cada um devolve. Mas para efeitos do relatório final, aqui está a tabela de comparação que apresentei aos orientadores:

| Critério                     | AWS Rekognition                | DeepFace                        |
|------------------------------|--------------------------------|---------------------------------|
| Precisão 1:1                 | Muito alta (~99%)              | Muito alta (~99% com ArcFace)   |
| Precisão 1:N                 | Muito alta                     | Muito alta                      |
| Latência 1:1                 | 400-700 ms (inclui rede)       | 350-500 ms (local)              |
| Latência 1:N                 | 400-800 ms                     | 150-300 ms (depende do tamanho) |
| Escalabilidade               | Excelente (gerida pela AWS)    | Limitada ao hardware local      |
| Custos                       | Por uso, mas pode ser caro     | Só hardware (grátis ao nível de SDK) |
| Complexidade de instalação   | Muito baixa                    | Alta (dependências pesadas)     |
| Segurança / Privacidade      | Dados vão para a cloud         | Totalmente local                |
| Liveness                     | Ativo (estado da arte)         | Passivo (menos preciso)         |
| Manutenção                   | Nenhuma                        | Atualizações manuais            |
| Dependência tecnológica      | Forte (AWS)                    | Fraca (open-source)             |
| Análise de atributos         | Muito completa (idade, género, emoção, óculos, barba, etc.) | Completa (idade, género, emoção, etnia) |

**Recomendação fundamentada:**

A minha recomendação final — que apresentei na apresentação final — foi que a escolha entre AWS e DeepFace depende muito do contexto do projeto:

- **Para projetos com requisitos de privacidade fortes** (banca, saúde, administração pública), ou para cenários em que não se quer depender de um fornecedor externo, **o DeepFace é a escolha preferível**. A precisão está ao mesmo nível da AWS, o custo é só o hardware e, muito importante, os dados biométricos nunca saem da infraestrutura da empresa.
- **Para projetos em que a velocidade de integração é crítica e o volume é moderado**, **a AWS é a escolha mais pragmática**. A integração é rápida (um SDK, algumas chamadas), a manutenção é nula, e o Face Liveness oficial é considerado um dos melhores do mercado.
- **Para projetos de alto volume em produção**, vale a pena calcular o ponto de equilíbrio entre os dois modelos — a partir de um certo volume, o custo da AWS torna-se superior ao custo de manter uma infraestrutura DeepFace, e o DeepFace começa a ser mais económico.

Os orientadores concordaram com esta análise e acharam útil ter os dois caminhos explorados em detalhe.

---

## 10. Documentação da POC

Paralelamente ao desenvolvimento, fui mantendo documentação interna da POC. A documentação cobre:

- **Arquitetura** — descrição do que é cada componente e como comunicam entre si.
- **Passos de instalação** — o que está descrito na secção 15 deste relatório, mas mais técnico.
- **Limitações identificadas** — o que o sistema não faz ou faz mal, e as razões.
- **Decisões técnicas** — porque é que escolhi ArcFace em vez de Facenet512, porque é que o liveness corre num servidor separado, etc.

A documentação serviu-me também como um "diário de bordo" em que ia apontando decisões e coisas que tinha aprendido, e que me foram úteis mais tarde para preparar a apresentação final e este próprio relatório.

---

## 11. Apresentação final

A apresentação final foi o momento em que mostrei aos orientadores e a alguns colegas da empresa tudo o que tinha feito. Foi marcada na penúltima semana do estágio e durou cerca de 45 minutos, dos quais 25 foram de apresentação em si e os restantes 20 de demonstrações ao vivo e perguntas.

A apresentação cobriu as seguintes secções:

- **Objetivo** — recordar o contexto e os objetivos iniciais.
- **Solução escolhida** — DeepFace + AWS, e as razões da escolha.
- **Arquitetura** — diagrama geral do projeto, com os três servidores e a separação entre DeepFace e AWS.
- **Demonstração ao vivo** — abri a aplicação e fiz em direto: registo, verificação 1:1, identificação 1:N, extração de documento, câmara ao vivo com identificação, liveness da AWS, comparação de modelos. Esta foi a parte mais stressante mas também a mais gratificante.
- **Resultados** — latências medidas, precisão observada, custos estimados.
- **Conclusões** — a tabela comparativa e a recomendação final.
- **Próximos passos** — o que faria a seguir se tivesse mais tempo (secção 14 deste relatório).

Foi uma das primeiras vezes que apresentei algo técnico num ambiente "real" de empresa, e apesar de estar um bocado nervoso no início, correu melhor do que esperava. Recebi feedback positivo dos orientadores, com sugestões para futuros desenvolvimentos.

---

## 12. Reuniões semanais com os orientadores

Desde o início do estágio, tinha reuniões semanais todas as segundas-feiras de manhã com os orientadores na empresa. Estas reuniões duravam tipicamente 30-45 minutos e serviam para:

- **Ponto de situação** — eu mostrava o que tinha feito na semana anterior.
- **Esclarecimento de dúvidas** — perguntava sobre decisões técnicas que tinha dúvidas.
- **Ajuste de prioridades** — às vezes os orientadores redirecionavam o foco para algo que eles achavam mais importante.
- **Feedback** — comentários sobre o que podia melhorar, tanto na parte técnica como na parte de comunicação.

As reuniões foram muito úteis, principalmente no início, porque me ajudaram a não perder o foco. Tive a tendência natural de "ir atrás" de funcionalidades extra que não estavam no plano, e os orientadores foram sempre muito bons a lembrar-me que o objetivo era ter uma POC comparativa bem feita, não um produto completo.

Uma das coisas que mais aprendi nestas reuniões foi a preparar o ponto de situação. Nas primeiras semanas ia um bocado "cru", e acabava por estar a pensar nas respostas durante a reunião. Com o tempo, passei a preparar sempre uma pequena lista antes, com o que tinha feito, o que estava a fazer, e que dúvidas tinha, o que tornou as reuniões muito mais eficientes.

---

## 13. Dificuldades encontradas

Ao longo do estágio encontrei várias dificuldades, algumas esperadas e outras completamente inesperadas. Faço aqui um apanhado das principais, porque acho que faz parte do processo mostrar também o que não correu bem.

**1. Instalação do DeepFace em Windows.** Já mencionei isto antes, mas volto a referir porque foi a primeira grande dor. As dependências do DeepFace (TensorFlow, protobuf, grpcio, numpy, e mais outras tantas) entram em conflito umas com as outras com muita facilidade, e em Windows há ainda o problema adicional de algumas delas não terem "wheels" pré-compilados para certas versões do Python. Acabei por fixar a versão do Python em 3.11 e fixar algumas bibliotecas no `requirements.txt` para garantir reprodutibilidade.

**2. Tamanho dos modelos e tempo de cold start.** Na primeira execução, o DeepFace descarrega automaticamente os pesos dos modelos, e alguns são grandes (VGG-Face tem cerca de 500 MB). Mesmo depois de descarregados, a primeira chamada demora vários segundos porque o TensorFlow precisa de compilar o grafo. Resolvi isto com o mecanismo de warmup no arranque do servidor.

**3. Anti-spoofing do DeepFace.** O anti-spoofing integrado no DeepFace requer `torch`, que não veio no meu `requirements.txt` inicial. Quando tentei ativar, deu erro. Decidi não forçar a instalação do torch (que é pesadíssimo) e em vez disso fiz o código detetar se o torch está disponível e desativar o anti-spoofing automaticamente se não estiver, devolvendo um aviso ao frontend. Isto está no `_get_anti_spoofing_runtime_status()` do `deepface_service.py`.

**4. Rotação das imagens.** Muitas fotos vinham rodadas (especialmente as tiradas com o telemóvel) e o DeepFace falhava a deteção. Tive de implementar um mecanismo de auto-rotação que tenta 4 orientações diferentes e escolhe a que dá melhor deteção. Isto envolveu bastante pesquisa sobre como o EXIF funciona e porque é que as bibliotecas nem sempre respeitam a orientação EXIF.

**5. Comparação documento vs câmara.** Como expliquei na secção 6.7, a comparação entre uma foto de documento (preto e branco, mal iluminada, comprimida) e uma selfie (a cores, iluminação ambiente) era muito pior do que esperado. Esta foi uma das partes que mais tempo me ocupou, e a solução (CLAHE + threshold mais permissivo) só apareceu depois de eu pesquisar bastante sobre normalização de imagem e ter testado várias alternativas (equalização simples, normalização de histograma, conversão para tons de cinzento, etc.).

**6. Integração do Face Liveness da AWS.** Esta foi difícil porque o componente `FaceLivenessDetector` é React-only e não tem uma forma fácil de ser embebido em HTML puro. Tive de aprender o suficiente de React e de Vite para fazer um mini-app que só tem uma página, servi-lo a partir de um Flask separado, e fazer a comunicação entre o frontend principal e o mini-app via query parameters. Também tive de perceber como funciona o CORS e o rate limiting.

**7. Cold start do servidor principal.** O FastAPI em si arranca rápido, mas a primeira chamada às rotas DeepFace demorava 5-10 segundos por causa do cold start do TensorFlow. Implementei o warmup em background, mas tive o cuidado de não bloquear o arranque do servidor — o warmup corre numa thread daemon separada, para que o servidor comece a atender pedidos imediatamente.

**8. Path traversal no registo.** Os orientadores chamaram-me a atenção para o risco de path traversal no `delete_person` (se alguém passasse um nome como `../../etc/passwd`). Tive de aprender sobre `os.path.realpath` e garantir que o caminho resolvido continua dentro do diretório esperado. Isto também me fez revisitar outros pontos do código onde havia manipulação de nomes de ficheiros.

**9. Limite de 5 MB das imagens AWS.** A AWS tem um limite de 5 MB por imagem nas chamadas ao Rekognition. Fotos de telemóveis modernos passam facilmente este limite. Criei um helper `_to_jpeg` que converte para JPEG e reduz progressivamente a qualidade (92, 80, 65, 50) até caber no limite.

**10. Gestão de sessões do documento de ID.** As sessões do documento guardam temporariamente a imagem em disco, e se o servidor for reiniciado ou se o utilizador fechar o browser sem terminar, as sessões ficam órfãs. Implementei um cleanup automático no arranque do servidor e um endpoint para limpar sessões com mais de 30 minutos.

**11. Velocidade da câmara ao vivo.** Na primeira versão, a câmara ao vivo analisava cada frame à medida que chegava, e o servidor ficava sobrecarregado. Tive de introduzir uma "fila de um só" — enquanto há um pedido a ser processado, os novos pedidos são ignorados, e o frame rate efetivo da análise acaba por ficar limitado pelo tempo de processamento. Do lado do cliente, há também um intervalo configurável (200/400/700 ms) entre análises.

---

## 14. Conclusões e próximos passos

Olhando para trás, sinto que o estágio correu muito melhor do que eu esperava. Comecei com uma ideia vaga do que era reconhecimento facial moderno e termino com uma aplicação completa, que cobre todos os casos de uso típicos, integrada com uma solução cloud e com uma biblioteca open-source, e com uma comparação fundamentada entre as duas.

Do ponto de vista técnico, os principais aprendizados foram:

- **Os modelos modernos são bons o suficiente para produção.** Tanto o ArcFace (local) como o Rekognition (cloud) têm precisões muito altas em condições controladas. O gargalo está quase sempre na qualidade das imagens de entrada, não no modelo.
- **A parte "aborrecida" é a que faz a diferença.** Coisas como auto-rotação, normalização de iluminação (CLAHE), avaliação de qualidade, gestão de cold starts — são essas coisas que separam uma POC que funciona em demonstração de uma aplicação que funciona com utilizadores reais.
- **Separação de responsabilidades paga-se por si própria.** Ter servidores separados para o liveness e para o webservice DeepFace deu-me muito mais flexibilidade e tornou o projeto mais fácil de raciocinar.

Se tivesse mais tempo, as coisas que faria a seguir seriam:

- **Substituir o `.env` por AWS Cognito Identity Pool** para as credenciais do liveness. Em produção, nunca se deve expor `AccessKey`/`SecretKey` diretos ao browser.
- **Persistir as métricas em base de dados** em vez de em memória, para não se perderem quando o servidor reinicia.
- **Adicionar testes unitários** para os serviços principais. Durante o estágio fui testando sempre manualmente, mas numa aplicação em produção seria importante ter uma suite de testes automatizada.
- **Melhorar o sistema de logging** para facilitar a análise em produção, com níveis configuráveis e formatação estruturada em JSON.
- **Medir a precisão em datasets públicos** (LFW, CFP, AgeDB) para ter uma comparação objetiva entre os dois motores, em vez de apenas testar com fotografias dos colegas.
- **Implementar cache dos embeddings** com indexação vetorial (FAISS, por exemplo) para permitir 1:N em bases de dados grandes sem perder performance.

Do ponto de vista pessoal, os aprendizados foram também muitos. Aprendi a organizar um projeto médio/grande em Python, a planear tarefas com issues e milestones, a comunicar em reuniões técnicas, a apresentar em público para um público técnico que sabe mais que eu, e a lidar com a frustração natural quando algo não funciona à primeira.

Termino este estágio com a sensação de que a POC ficou bem feita, os objetivos foram cumpridos e, principalmente, com a sensação de que aprendi muito mais do que aprenderia numa cadeira teórica. Agradeço mais uma vez à Asseco PST pela oportunidade.

---

## 15. Guia de execução do projeto

Esta secção explica como executar o projeto noutro computador, desde zero. É o guia que usaria eu próprio se tivesse de pegar no projeto daqui a seis meses.

### 15.1 Pré-requisitos

1. **Python 3.10 ou 3.11** (recomendo 3.11, que foi a versão que usei durante o estágio).
2. **Git** para clonar o repositório.
3. **Conta AWS ativa** com utilizador IAM que tenha permissões sobre o Rekognition (ver secção 7.1 para a lista de permissões).
4. **Ligação à internet** para a primeira execução (para descarregar os pesos dos modelos DeepFace).

O projeto foi desenvolvido em Windows 11, mas a mesma receita deve funcionar em Linux e macOS com pequenas adaptações (sobretudo na ativação do `venv`).

### 15.2 Clonar o repositório

```bash
git clone https://github.com/asseco-pst/poc-facial-recognition.git
cd poc-facial-recognition
```

### 15.3 Criar o ambiente virtual

```bash
python -m venv venv
```

Isto cria uma pasta `venv/` dentro do projeto, com um Python isolado. Em seguida, é preciso **ativar** o ambiente virtual, o que depende do sistema operativo:

- **Windows (PowerShell ou CMD):**
  ```
  .\venv\Scripts\activate
  ```
- **Linux / macOS:**
  ```
  source venv/bin/activate
  ```

Se a ativação funcionar, o prompt do terminal vai passar a ter `(venv)` à frente.

### 15.4 Instalar as dependências

```bash
pip install -r requirements.txt
```

Esta instalação pode demorar vários minutos (10-20 minutos não é raro), porque traz o TensorFlow, OpenCV, Pillow e outras bibliotecas pesadas. Se der algum erro, a solução habitual é atualizar o `pip` primeiro (`python -m pip install --upgrade pip`) e voltar a tentar.

Na primeira execução do DeepFace, a biblioteca também vai **descarregar automaticamente** os pesos dos modelos (ArcFace, YuNet, modelos de idade/género/emoção) para `~/.deepface/weights/`. Este download é feito na primeira chamada e não precisa de ser reexecutado nas vezes seguintes.

### 15.5 Configurar as credenciais AWS

As credenciais AWS não estão no repositório por razões de segurança. É preciso criar um ficheiro `.env` na raiz do projeto, copiando o template `.env.example`:

```bash
cp .env.example .env
```

E depois editar o `.env` e preencher com as credenciais reais:

```
AWS_ACCESS_KEY_ID=AKIA...                 # chave de acesso do utilizador IAM
AWS_SECRET_ACCESS_KEY=...                 # chave secreta
AWS_REGION=eu-west-1                      # região AWS (recomendo eu-west-1)
COLLECTION_ID=poc-asseco-faces            # nome da coleção no Rekognition
SIMILARITY_THRESHOLD=80.0                 # threshold de similaridade 1:N
ANTISPOOFING_THRESHOLD=50.0               # threshold de anti-spoofing local
```

**Importante:** nunca commitar este ficheiro. O `.gitignore` já o exclui.

### 15.6 Arrancar o servidor principal

Com o `venv` ativo e o `.env` configurado, basta correr:

```bash
python run.py
```

Isto arranca o servidor FastAPI principal na porta **8000**. No arranque, o servidor também inicia automaticamente, em background:

- O servidor Flask do liveness na porta **5001**.
- O warmup dos modelos DeepFace (ArcFace + YuNet) numa thread separada.

Depois do arranque, a aplicação fica acessível em:

- **Dashboard DeepFace:** http://localhost:8000
- **Dashboard AWS:** http://localhost:8000/aws
- **Documentação OpenAPI:** http://localhost:8000/docs
- **Liveness (servidor auxiliar):** http://localhost:5001

A primeira chamada à rota de verificação ou identificação pode demorar 5-10 segundos por causa do cold start do TensorFlow; as chamadas seguintes são muito mais rápidas.

### 15.7 Arrancar o webservice DeepFace dedicado (opcional)

Se for necessário expor apenas a API DeepFace, sem frontend e sem rotas AWS, é possível correr o webservice dedicado em paralelo:

```bash
python run_deepface.py
```

Isto arranca um segundo FastAPI na porta **8001**. A documentação do webservice está em http://localhost:8001/docs.

Pode correr em paralelo com o `python run.py` ou isoladamente.

### 15.8 Estrutura de execução típica

Para ter tudo a funcionar, a configuração típica é abrir **dois terminais**:

- **Terminal 1** (servidor principal + liveness + frontend):
  ```
  .\venv\Scripts\activate
  python run.py
  ```
- **Terminal 2** (webservice DeepFace isolado, opcional):
  ```
  .\venv\Scripts\activate
  python run_deepface.py
  ```

Depois é só abrir o browser em http://localhost:8000 e testar as funcionalidades.

### 15.9 Resolução de problemas comuns

- **"No module named 'deepface'"** — O `venv` não está ativo ou as dependências não foram instaladas. Reativar o `venv` e voltar a correr `pip install -r requirements.txt`.
- **"Anti-spoofing indisponível"** — O `torch` não está instalado, o que é intencional (é demasiado pesado para a POC). O anti-spoofing do DeepFace fica desativado, mas o anti-spoofing local (do lado AWS) continua a funcionar, e o Face Liveness da AWS também.
- **"Credenciais AWS não configuradas"** — O ficheiro `.env` não existe ou tem valores vazios. Criar a partir do `.env.example` e preencher.
- **"Port 8000 already in use"** — Há outro processo a usar a porta 8000. Matar esse processo ou alterar a porta em `run.py`.
- **"Collection not found"** — A coleção `poc-asseco-faces` ainda não foi criada na AWS. Basta fazer o primeiro registo pela página `/aws/register` — a coleção é criada automaticamente.

---

## 16. Tabela de horas

A tabela seguinte resume a distribuição das horas ao longo do estágio. As 210 horas foram distribuídas entre investigação, implementação, documentação e reuniões.

| #  | Tarefa                                                          | Horas |
|----|-----------------------------------------------------------------|-------|
| 1  | Levantamento de soluções de reconhecimento facial               | 18    |
| 2  | Análise de licenciamento e custos                               | 10    |
| 3  | Análise da arquitetura das soluções                             | 12    |
| 4  | Preparação da apresentação intermédia                           | 8     |
| 5  | DeepFace – Setup Python                                         | 4     |
| 6  | AWS – Configuração e preparação do ambiente                     | 5     |
| 7  | DeepFace – Pipeline de registo biométrico                       | 8     |
| 8  | DeepFace – Testes 1:1                                           | 6     |
| 9  | DeepFace – Testes 1:N                                           | 6     |
| 10 | AWS – Testes 1:1                                                | 5     |
| 11 | AWS – Testes 1:N                                                | 5     |
| 12 | AWS – Métricas de qualidade e performance                       | 6     |
| 13 | DeepFace – Atributos de qualidade da imagem                     | 8     |
| 14 | DeepFace – Avaliação dos melhores modelos                       | 10    |
| 15 | DeepFace – Extração da fotografia do documento de identificação | 18    |
| 16 | Liveness (AWS Face Liveness + anti-spoofing local)              | 14    |
| 17 | Frontend (HTML/CSS/JS, páginas DeepFace e AWS)                  | 15    |
| 18 | Comparação técnica e recomendação final                         | 8     |
| 19 | Documentação da POC                                             | 10    |
| 20 | Preparação da apresentação final                                | 8     |
| 21 | Relatório da Universidade                                       | 16    |
| 22 | Reuniões semanais com os orientadores                           | 10    |
|    | **Total**                                                       | **210** |

---

## 17. Referências

- Serengil, S. I. & Ozpinar, A. (2021). *LightFace: A Hybrid Deep Face Recognition Framework*. Documentação oficial do DeepFace: https://github.com/serengil/deepface
- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*. CVPR.
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering*. CVPR.
- Documentação oficial da AWS Rekognition: https://docs.aws.amazon.com/rekognition/
- Documentação do AWS Amplify UI (FaceLivenessDetector): https://ui.docs.amplify.aws/react/connected-components/liveness
- Documentação do boto3 (SDK Python AWS): https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/rekognition.html
- FastAPI – framework web: https://fastapi.tiangolo.com/
- OpenCV – biblioteca de visão por computador: https://docs.opencv.org/
- Face-api.js – biblioteca JavaScript de reconhecimento facial no browser: https://github.com/justadudewhohacks/face-api.js
- Pizer, S. M. et al. (1987). *Adaptive histogram equalization and its variations*. Computer Vision, Graphics, and Image Processing.