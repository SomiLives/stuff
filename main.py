from flask import Flask, request, jsonify
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
import mysql.connector
from dotenv import load_dotenv
import os
import logging

load_dotenv()

app = Flask(__name__)
logger = logging.getLogger(__name__)

# Paths
DB_FAISS_PATH = 'vectorstore/'
os.makedirs(DB_FAISS_PATH, exist_ok=True)

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': os.getenv('DB_PASSWORD', 'somnotho'),
    'database': 'LargeECommerceDB',
}

PROMPT_TEMPLATE = """
You are a helpful assistant that generates SQL queries based on user questions and database schemas.

Context: {context}

User Question: {question}

Guidelines:
1. Generate a valid MySQL query for SQL server database to answer the user's question.
2. Use the provided context to construct queries that align with the database schema.
3. If the context is insufficient to construct the query, respond with: 'I need more information about the database schema to generate an accurate query.'
4. Avoid providing answers or explanations—respond only with the MySQL query.
6. Write the response on the same line, with no new lines or tabs. Replace new lines with spaces.
7. Do not use '*' for columns; list all required column names instead.
9. DO NOT INVENT TABLE NAMES.

SQL Query:
"""

def get_db_connection():
    """Establish a database connection."""
    try:
        logger.info("Attempting to connect to the database with config: %s", DB_CONFIG)
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            logger.info("Successfully connected to the database.")
            return connection
        else:
            logger.error("Failed to connect to the database.")
            return None
    except mysql.connector.Error as err:
        logger.error(f"Error connecting to MySQL: {err}")
        return None



def fetch_metadata():
    """Fetch table metadata and relationships from the database."""
    metadata = {}
    con = get_db_connection()
    try:
        with con as connection:
            with connection.cursor(dictionary=True) as cursor:
                # Fetch columns
                cursor.execute("""
                    SELECT TABLE_NAME, COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE();
                """)
                for row in cursor.fetchall():
                    table = row['TABLE_NAME']
                    column = row['COLUMN_NAME']
                    metadata.setdefault(table, {'columns': [], 'joins': []})['columns'].append(column)

                # Fetch relationships
                cursor.execute("""
                    SELECT TABLE_NAME, COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
                    FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
                    WHERE TABLE_SCHEMA = DATABASE() AND REFERENCED_TABLE_NAME IS NOT NULL;
                """)
                for row in cursor.fetchall():
                    table = row['TABLE_NAME']
                    join = (row['COLUMN_NAME'], row['REFERENCED_TABLE_NAME'], row['REFERENCED_COLUMN_NAME'])
                    metadata[table]['joins'].append(join)

    except mysql.connector.Error as err:
        logger.error(f"Error fetching metadata: {err}")
        print(f"Error fetching metadata: {err}")  # Add this for additional logging
    return metadata

def initialize_vector_store():
    """Initialize FAISS vector store with database metadata."""
    metadata = fetch_metadata()
    if not metadata:
        return "Failed to fetch metadata from the database."

    context = " ".join(
        [f"Table: {table}, Columns: {', '.join(details['columns'])}" for table, details in metadata.items()]
    )

    try:
        documents = [Document(page_content=context)]
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        vector_db = FAISS.from_documents(documents, embeddings)
        vector_db.save_local(DB_FAISS_PATH)
        return None
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        return str(e)

def setup_qa_chain():
    """Set up the QA chain."""
    llm = OllamaLLM(model = "llama3.2", temperature=0.3)
    try:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'})
        vector_store = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
        prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt},
        )
        return qa_chain, None
    except Exception as e:
        logger.error(f"Error setting up QA chain: {e}")
        return None, f"Error setting up QA chain: {e}"


@app.route('/initialize_ollama', methods=['POST'])
def initialize():
    """API endpoint to initialize the vector database."""
    error = initialize_vector_store()
    if error:
        return jsonify({"error": error}), 500
    return jsonify({"message": "Vector database initialized successfully."}), 200


@app.route('/answer_ollama', methods=['POST'])
def get_answer():
    """API endpoint to process user queries and return SQL answers."""
    query = request.json.get("query") or request.json.get("question")  # Accept both keys
    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    qa_chain, error = setup_qa_chain()
    if error:
        return jsonify({"error": error}), 500

    try:
        response = qa_chain.run({"query": query})

        # Check if response is a string (expected output) or a dictionary
        if isinstance(response, str):
            answer = response.replace('\n', ' ')  # Replace newlines with spaces to maintain formatting
        else:
            answer = response.get("result", "No answer found.")

        return jsonify({"answer": answer}), 200
    except Exception as e:
        logger.error(f"Error in QA processing: {e}")
        return jsonify({"error": "Internal server error occurred."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
