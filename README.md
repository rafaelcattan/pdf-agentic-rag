# PDF Agentic RAG Framework

Um framework agentico para análise de documentos PDF usando LangChain e OpenAI.

## 🤖 Agentes

- **search_pdf**: Busca informações no documento PDF
- **quality_check**: Verifica e melhora a qualidade das respostas

## 🚀 Instalação
```bash
@"
# PDF Agentic RAG Framework

Um framework agentico para análise de documentos PDF usando LangChain e OpenAI.

## 🤖 Agentes

- **search_pdf**: Busca informações no documento PDF
- **quality_check**: Verifica e melhora a qualidade das respostas

## 🚀 Instalação
```bash
# Criar ambiente virtual
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependências
pip install -r requirements.txt

# Configurar API key
# Criar arquivo .env com:
OPENAI_API_KEY=sua-chave-aqui
```

## 💻 Uso
```bash
python agentic_rag.py
```

## 📦 Dependências

- langchain
- langchain-openai
- faiss-cpu
- pypdf
- python-dotenv

## 📄 Estrutura
```
├── agentic_rag.py      # Script principal
├── requirements.txt    # Dependências
├── .env               # API keys (não commitado)
├── .gitignore         # Arquivos ignorados
└── README.md          # Este arquivo
```
