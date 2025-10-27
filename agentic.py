# For text splitting and model imports
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# For community integrations (PyPDFLoader, FAISS)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

# IMPORTANT: In LangChain 1.0, chains are now in langchain-classic
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# 🆕 NEW: For agentic behavior - agent can use tools
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

# =============================================================================
# SETUP: Load PDF and create retriever (same as before)
# =============================================================================
print("📄 Loading PDF...")
loader = PyPDFLoader("judea.pdf")
documents = loader.load()
print(f"✓ Loaded {len(documents)} pages.")

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"✓ Created {len(texts)} chunks\n")

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(texts, embeddings)
retriever = vector_store.as_retriever()

# =============================================================================
# 🆕 STEP 1: CREATE TOOLS (agents use these to take actions)
# =============================================================================
print("🔧 Creating agent tools...\n")

@tool
def search_pdf(query: str) -> str:
    """Search the PDF document for information. Use this when you need facts from the document."""
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs[:3]])
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"""Based on this context from the document, answer the question concisely.
    
Context:
{context}

Question: {query}

Answer:"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def quality_check(answer: str, question: str) -> str:
    """Check the quality of an answer and improve it if needed. Use this to verify answers."""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    prompt = f"""Review this answer and rate its quality (1-10). If score < 8, provide an improved version.

Question: {question}
Answer: {answer}

Respond in this format:
SCORE: [number]
IMPROVED: [improved answer if needed, or "No improvement needed"]
"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# 🆕 STEP 2: Give tools to the agent
tools = [search_pdf, quality_check]

# =============================================================================
# 🆕 STEP 3: CREATE THE AGENT (this is the key difference!)
# =============================================================================
print("🤖 Creating autonomous agent...\n")

# Define agent prompt - tells agent HOW to think
agent_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful research assistant with access to tools.

Your goal: Answer user questions accurately using the available tools.

You have access to:
1. search_pdf - Search the PDF document
2. quality_check - Verify and improve answers

Think step by step:
1. First, search the PDF for relevant information
2. Then, check the quality of your answer
3. Provide the final answer to the user

Always use tools to ensure accuracy."""),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # Agent's thinking process goes here
])

# Create the agent with tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# Create executor - this runs the agent's reasoning loop
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,  # Shows agent's thinking process
    max_iterations=5  # Prevents infinite loops
)

# =============================================================================
# 🆕 STEP 4: LET THE AGENT DECIDE (autonomous!)
# =============================================================================
print("="*70)
print("🚀 AGENT AUTONOMOUSLY WORKING...")
print("="*70 + "\n")

query = "Do I need to create a DAG to start any causal inference problem?"

# The agent will:
# 1. Decide to use search_pdf tool
# 2. Get the answer
# 3. Decide to use quality_check tool
# 4. Verify quality
# 5. Return final answer
# ALL WITHOUT YOU TELLING IT EACH STEP!

result = agent_executor.invoke({"input": query})

print("\n" + "="*70)
print("✅ FINAL RESULT")
print("="*70)
print(result["output"])