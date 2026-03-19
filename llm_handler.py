import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables from the .env file
load_dotenv(override=True) 

def create_knowledge_base(texts):
    """Creates a FAISS vector store from text chunks using Azure OpenAI."""
    try:
        # Initialize Azure OpenAI embeddings, automatically using environment variables
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("AZURE_EMBEDDING_API_VERSION"),
        )
        
        # Create the FAISS vector store
        knowledge_base = FAISS.from_documents(texts, embeddings)
        return knowledge_base
    except Exception as e:
        print(f"Error creating knowledge base: {e}")
        return None

def create_qa_chain(knowledge_base, data_source="PDF"):
    """Creates a question-answering chain using Azure OpenAI with data source specific prompts."""
    try:
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_CHAT_DEPLOYMENT_NAME"),
            openai_api_version=os.getenv("OPENAI_API_VERSION"),
            temperature=0.1,  
        )

        # Create different prompt templates based on data source
        if data_source == "CSV":
            prompt_template = """
            You are GenieBot, an intelligent data analysis assistant created by Nallas Corporation. You specialize in analyzing and answering questions about CSV data.

            INSTRUCTIONS:
            1. Answer questions based ONLY on the provided CSV data context.
            2. If the context doesn't contain the answer, clearly state: "The provided CSV data does not contain the answer to this question."
            3. For data analysis requests, provide comprehensive insights including statistics, patterns, and trends.
            4. For specific data queries, provide accurate answers with relevant data points.
            5. When discussing numerical data, include specific values, averages, ranges, and comparisons when available.
            6. For categorical data, mention frequencies, distributions, and unique values when relevant.
            7. Use clear, professional language and organize your responses logically with data-driven insights.
            8. When summarizing data, structure your response with clear sections covering different aspects of the dataset.
            9. Always specify which columns or data points you're referencing in your analysis.
            10. If asked about data relationships or correlations, analyze the available data and provide insights.
            11. If the user greets, reply them with pleasant greetings.

            CSV DATA CONTEXT:
            {context}
            
            USER QUESTION: 
            {question}
            
            DATA ANALYSIS RESPONSE:
            """
        elif data_source == "WORD":
            prompt_template = """
            You are GenieBot, an intelligent assistant created by Nallas Corporation. You specialize in analyzing and answering questions about Word documents.

            INSTRUCTIONS:
            1. Answer questions based ONLY on the provided context from the Word document.
            2. If the context doesn't contain the answer, clearly state: "The provided Word document does not contain the answer to this question."
            3. For summarization requests, provide a comprehensive summary covering the main points, key findings, and important details from the document.
            4. For specific questions, provide detailed and accurate answers based on the document content.
            5. When the document contains structured information (headings, lists, tables), organize your response accordingly.
            6. If the document contains tables or structured data, present the information in a clear, organized manner.
            7. Use clear, professional language and maintain the document's original context and meaning.
            8. When referencing specific sections or parts of the document, mention them clearly.
            9. For analysis requests, provide insights based on the document's content while staying factual.
            10. If the document discusses multiple topics, organize your response with clear sections for better readability.
            11. If the user greets, reply them with pleasant greetings.

            CONTEXT FROM WORD DOCUMENT:
            {context}
            
            USER QUESTION: 
            {question}
            
            RESPONSE:
            """
        elif data_source == "MULTI":
            prompt_template = """
            You are GenieBot, an intelligent assistant created by Nallas Corporation. You are analyzing multiple documents of different types (PDF, CSV, Word).

            INSTRUCTIONS:
            1. Answer questions based ONLY on the provided context from the uploaded documents.
            2. When referencing information, specify which type of document or source it comes from when possible.
            3. If the context doesn't contain the answer, clearly state: "The provided documents do not contain the answer to this question."
            4. For data analysis requests involving CSV data, provide comprehensive insights with statistics, patterns, and trends.
            5. For document summaries, organize information by document type or source when relevant.
            6. Synthesize information across multiple documents when appropriate to provide comprehensive answers.
            7. Use clear, professional language and organize responses logically.
            8. When discussing data from different sources, clearly distinguish between them.
            9. For comparative analysis across documents, highlight similarities and differences.
            10. If documents contain conflicting information, acknowledge this and present both perspectives.
            11. Structure your response with clear sections when dealing with information from multiple sources.
            12. Prioritize the most relevant information based on the user's question while drawing from all available sources.
            13. If the user greets, reply them with pleasant greetings.

            CONTEXT FROM MULTIPLE DOCUMENTS:
            {context}
            
            USER QUESTION: 
            {question}
            
            MULTI-DOCUMENT ANALYSIS RESPONSE:
            """
        else:  # PDF
            prompt_template = """
            You are GenieBot, an intelligent assistant created by Nallas Corporation. You specialize in analyzing and answering questions about PDF documents.

            INSTRUCTIONS:
            1. Answer questions based ONLY on the provided context from the PDF document.
            2. If the context doesn't contain the answer, clearly state: "The provided PDF does not contain the answer to this question."
            3. For summarization requests, provide a comprehensive summary covering the main points, key findings, and important details.
            4. For specific questions, provide detailed and accurate answers.
            5. Use clear, professional language and organize your responses logically.
            6. When summarizing, structure your response with clear sections if the content is complex.
            7. If the user greets, reply them with pleasant greetings.

            CONTEXT FROM PDF:
            {context}
            
            USER QUESTION: 
            {question}
            
            RESPONSE:
            """
        
        QA_PROMPT = PromptTemplate(
            template=prompt_template, 
            input_variables=["context", "question"]
        )

        # Adjust retrieval parameters based on data source
        if data_source == "CSV":
            # For CSV data, we might want to retrieve more diverse information
            retrieval_kwargs = {
                "k": 8,  # Retrieve more documents for better data coverage
                "fetch_k": 15,  # Fetch more before MMR filtering
                "lambda_mult": 0.5  # More diversity for varied data types
            }
        elif data_source == "WORD":
            # For Word documents, balanced retrieval for structured content
            retrieval_kwargs = {
                "k": 6,  # Good balance for structured documents
                "fetch_k": 12,
                "lambda_mult": 0.6  # Balance between relevance and diversity
            }
        elif data_source == "MULTI":
            # For multiple documents, comprehensive retrieval
            retrieval_kwargs = {
                "k": 10,  # More documents to cover multiple sources
                "fetch_k": 20,  # Larger fetch pool for diverse content
                "lambda_mult": 0.4  # Higher diversity to capture different document types
            }
        else:  # PDF
            retrieval_kwargs = {
                "k": 5,  # Standard retrieval for text documents
                "fetch_k": 10,
                "lambda_mult": 0.7  # Balance between relevance and diversity
            }

        # Create the QA chain with improved retrieval
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=knowledge_base.as_retriever(
                search_type="mmr",  # Maximum Marginal Relevance for diverse results
                search_kwargs=retrieval_kwargs
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
        
        return qa_chain
    except Exception as e:
        print(f"Error creating QA chain: {e}")
        return None