from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema import Document
from datasets import load_dataset
import os

OPENROUTER_API_KEY ="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"

# Initialize components
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def load_humaneval_documents():
    """Load HumanEval dataset as LangChain documents"""
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    
    documents = []
    for item in dataset:
        # Create document with combined content using LangChain Document class
        content = f"Task: {item['prompt']}\nSolution: {item['canonical_solution']}"
        metadata = {
            "task_id": item['task_id'],
            "entry_point": item['entry_point'],
            "source": "HumanEval"
        }
        documents.append(Document(
            page_content=content,
            metadata=metadata
        ))
    
    print(f"✓ Loaded {len(documents)} examples as LangChain documents")
    return documents

def setup_rag_pipeline(persist_directory="./chroma_langchain"):
    """Setup the complete RAG pipeline with LangChain"""
    
    # Load and split documents
    docs = load_humaneval_documents()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50, 
        separators=["\n\n", "\n", " "]
    )
    splits = splitter.split_documents(docs)
    print(f"✓ Split into {len(splits)} chunks")
    
    # Create Chroma vector store
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Initialize LLM
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        model="openai/gpt-oss-20b:free",
        temperature=0.2,
        max_tokens=1000
    )
    
    # Create RAG chains for different intents
    code_generation_prompt = ChatPromptTemplate.from_template("""
You are an expert Python programmer. Use the provided context of code examples to generate high-quality code.

Context Examples:
{context}

User Request: {question}

Generate clean, efficient, and well-documented Python code. Include proper function definitions, type hints, and docstrings.

Code:
""")
    
    explanation_prompt = ChatPromptTemplate.from_template("""
You are an expert programming educator. Use the provided context to explain concepts clearly.

Context Examples:
{context}

User Question: {question}

Provide a clear, comprehensive explanation covering:
- What it does and how it works
- Key algorithms or patterns used
- Practical use cases
- Important considerations

Explanation:
""")
    
    # Create RAG chains
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
    
    return code_rag_chain, explain_rag_chain, retriever, vectorstore

# Initialize globally
code_rag_chain, explain_rag_chain, retriever, vectorstore = setup_rag_pipeline()