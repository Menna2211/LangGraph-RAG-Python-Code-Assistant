from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.output_parsers.string import StrOutputParser 
from langchain_core.runnables.passthrough import RunnablePassthrough  
from langchain_core.documents import Document
from datasets import load_dataset
import os
import shutil

# ----------------------------------------
# Configuration
# ----------------------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
PERSIST_DIR = "./chroma_langchain"

# ----------------------------------------
# Embedding model
# ----------------------------------------
def get_embedding_model():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    return HuggingFaceEmbeddings(model_name=model_name)

embedding_model = get_embedding_model()

# ----------------------------------------
# Load HumanEval dataset
# ----------------------------------------
def load_humaneval_documents():
    """Load HumanEval dataset as LangChain documents"""
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    documents = []
    for item in dataset:
        content = f"Task: {item['prompt']}\nSolution: {item['canonical_solution']}"
        metadata = {
            "task_id": item['task_id'],
            "entry_point": item['entry_point'],
            "source": "HumanEval"
        }
        documents.append(Document(page_content=content, metadata=metadata))
    
    print(f"✓ Loaded {len(documents)} examples as LangChain documents")
    return documents

# ----------------------------------------
# Setup RAG pipeline
# ----------------------------------------
def setup_rag_pipeline(persist_directory=PERSIST_DIR):
    """Setup the complete RAG pipeline with LangChain"""

    # Clean previous vectorstore if corrupted
    if os.path.exists(persist_directory):
        try:
            # Attempt to load existing store
            Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        except Exception as e:
            print(f"⚠️ Corrupt Chroma store detected: {e}. Rebuilding...")
            shutil.rmtree(persist_directory)

    docs = load_humaneval_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " "],
    )
    splits = splitter.split_documents(docs)
    print(f"✓ Split into {len(splits)} chunks")

    # Create or reload Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_directory,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Initialize LLM
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model="openai/gpt-oss-20b:free",
        temperature=0.2,
        max_tokens=1000,
    )

    # ----------------------------------------
    # Prompts
    # ----------------------------------------
    code_generation_prompt = ChatPromptTemplate.from_template("""
You are an expert Python programmer. Use the provided context of code examples to generate high-quality code.

Context Examples:
{context}

User Request: {question}

Generate clean, efficient, and well-documented Python code.
Include proper function definitions, type hints, and docstrings.

Code:
""")

    explanation_prompt = ChatPromptTemplate.from_template("""
You are an expert programming educator. Use the provided context to explain code concepts clearly.

Context Examples:
{context}

User Question: {question}

Provide a clear, structured explanation covering:
- What the code does
- How it works internally
- Key algorithms or design patterns used
- Practical use cases
- Performance or safety considerations

Explanation:
""")

    # ----------------------------------------
    # RAG Chains
    # ----------------------------------------
    code_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | code_generation_prompt
        | llm
        | StrOutputParser()
    )

    explain_rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | explanation_prompt
        | llm
        | StrOutputParser()
    )

    print("✓ RAG pipeline initialized successfully")
    return code_rag_chain, explain_rag_chain, retriever, vectorstore


# ----------------------------------------
# Initialize globally
# ----------------------------------------
code_rag_chain, explain_rag_chain, retriever, vectorstore = setup_rag_pipeline()