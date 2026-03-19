import streamlit as st
import tempfile
import os
from pdf_processor import process_pdf
from csv_processor import process_csv
from word_processor import process_word
from llm_handler import create_knowledge_base, create_qa_chain

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="GenieBot by Nallas Corporation",
    layout="wide"
)

# --- 2. CSS STYLING ---
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --- 3. HELPER FUNCTIONS ---
def get_file_type(filename):
    """Detect file type based on file extension."""
    filename_lower = filename.lower()
    if filename_lower.endswith('.pdf'):
        return "PDF"
    elif filename_lower.endswith('.csv'):
        return "CSV"
    elif filename_lower.endswith(('.docx', '.doc')):
        return "WORD"
    else:
        return None

def get_file_icon(file_type):
    """Get appropriate icon for file type."""
    if file_type == "PDF":
        return "📄"
    elif file_type == "CSV":
        return "📊"
    elif file_type == "WORD":
        return "📝"
    else:
        return "📁"

def display_uploaded_files():
    """Display all uploaded files with their status."""
    if st.session_state.processed_files:
        st.markdown('<div class="document-header">Uploaded Files</div>', unsafe_allow_html=True)
        
        for i, file_info in enumerate(st.session_state.processed_files):
            file_icon = get_file_icon(file_info['type'])
            col1, col2 = st.columns([4, 1])
            title=file_info['name']
            if len(title) > 15:
                title = title[:12] + '...'
            
            with col1:
                st.markdown(f'''
                    <div class="file-item" title="{file_info['name']}">
                        {file_icon} {title} ({file_info['type']})
                    </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                if st.button("❌", key=f"remove_{i}", help=f"Remove {file_info['name']}", type="secondary"):
                    remove_file(file_info['name'])
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

def remove_file(file_name):
    """Remove a specific file from processed files."""
    st.session_state.processed_files = [f for f in st.session_state.processed_files if f['name'] != file_name]
    
    # Rebuild all_documents and knowledge base
    st.session_state.all_documents = []
    for file_info in st.session_state.processed_files:
        st.session_state.all_documents.extend(file_info['documents'])
    
    if st.session_state.all_documents:
        knowledge_base = create_knowledge_base(st.session_state.all_documents)
        data_source = "MULTI" if len(st.session_state.processed_files) > 1 else st.session_state.processed_files[0]['type']
        st.session_state.chain = create_qa_chain(knowledge_base, data_source)
    else:
        st.session_state.chain = None
    
    st.session_state.history.clear()
    
    # Reset the file uploader by incrementing its key
    st.session_state.uploader_key += 1

def process_uploaded_files(uploaded_files):
    """Process multiple uploaded files."""
    if not uploaded_files:
        return
    
    new_files = []
    
    # Check which files are new
    existing_names = [f['name'] for f in st.session_state.processed_files]
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in existing_names:
            new_files.append(uploaded_file)
    
    if new_files:
        st.session_state.processing = True
        st.session_state.processing_files = [f.name for f in new_files]
        
        for uploaded_file in new_files:
            detected_type = get_file_type(uploaded_file.name)
            
            if detected_type:
                try:
                    # Process each file
                    file_extension_map = {
                        "PDF": ".pdf",
                        "CSV": ".csv", 
                        "WORD": ".docx"
                    }
                    file_extension = file_extension_map.get(detected_type, ".tmp")
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_file_path = tmp.name
                    
                    # Process based on file type
                    if detected_type == "PDF":
                        texts = process_pdf(tmp_file_path)
                    elif detected_type == "CSV":
                        texts = process_csv(tmp_file_path)
                    elif detected_type == "WORD":
                        texts = process_word(tmp_file_path)
                    
                    if texts:
                        # Add file info to processed files
                        file_info = {
                            'name': uploaded_file.name,
                            'type': detected_type,
                            'documents': texts
                        }
                        st.session_state.processed_files.append(file_info)
                        st.session_state.all_documents.extend(texts)
                        st.warning(f"{detected_type} '{uploaded_file.name}' is currently being processed!")
                    else:
                        st.error(f"Failed to process {uploaded_file.name}")
                    
                    os.remove(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            else:
                st.error(f"Unsupported file type for {uploaded_file.name}")
        
        # Create knowledge base from all documents
        if st.session_state.all_documents:
            knowledge_base = create_knowledge_base(st.session_state.all_documents)
            data_source = "MULTI" if len(st.session_state.processed_files) > 1 else st.session_state.processed_files[0]['type']
            st.session_state.chain = create_qa_chain(knowledge_base, data_source)
        
        st.session_state.processing = False
        st.session_state.processing_files = []
        st.rerun()

# --- 4. SESSION STATE INITIALIZATION ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'chain' not in st.session_state:
    st.session_state.chain = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'all_documents' not in st.session_state:
    st.session_state.all_documents = []
if 'processing_files' not in st.session_state:
    st.session_state.processing_files = []
if 'uploader_key' not in st.session_state:
    st.session_state.uploader_key = 0

# --- 5. SIDEBAR ---
with st.sidebar:
    st.markdown('<div class="sidebar-header">GENIEBOT BY NALLAS CORPORATION</div>', unsafe_allow_html=True)
    
    # Document Upload Section
    st.markdown('<div class="document-section">', unsafe_allow_html=True)
    st.markdown('<div class="document-header">Upload Documents</div>', unsafe_allow_html=True)

    # Show processing status if files are being processed
    if st.session_state.processing:
        processing_text = f"🔄 Processing {len(st.session_state.processing_files)} file(s)..."
        st.markdown(f'<div class="document-display processing">{processing_text}</div>', unsafe_allow_html=True)
        for file_name in st.session_state.processing_files:
            st.markdown(f'<div class="processing-file">• {file_name}</div>', unsafe_allow_html=True)
    
    # File uploader that accepts multiple files with dynamic key
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose PDF, CSV, or Word files",
        type=["pdf", "csv", "docx", "doc"],
        help="Drag and drop or browse to upload multiple documents",
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process uploaded files
    if uploaded_files:
        process_uploaded_files(uploaded_files)
    
    # Display uploaded files
    display_uploaded_files()
    
    # Clear all files button
    if st.session_state.processed_files and not st.session_state.processing:
        if st.button("🗑️ Clear All Files", type="secondary"):
            st.session_state.processed_files.clear()
            st.session_state.all_documents.clear()
            st.session_state.chain = None
            st.session_state.history.clear()
            st.session_state.uploader_key += 1  # Reset uploader
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Status indicator
    st.markdown('<div class="status-section">', unsafe_allow_html=True)
    if st.session_state.processing:
        st.markdown('<div class="status-processing">🟡 Status: Processing Files...</div>', unsafe_allow_html=True)
    elif st.session_state.processed_files:
        file_count = len(st.session_state.processed_files)
        file_types = list(set([f['type'] for f in st.session_state.processed_files]))
        types_text = ", ".join(file_types)
        st.markdown(f'<div class="status-ready">🟢 Status: Ready ({file_count} files loaded - {types_text})</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-waiting">⚪ Status: Waiting for Documents</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- 6. MAIN CHAT AREA ---
# Custom chat display instead of st.chat_message for better styling control
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.history):
        if message["role"] == "user":
            # User message - right side
            st.markdown(f'''
            <div class="chat-message-container user-message">
                <div class="user-bubble">
                    {message["content"]}
                </div>
            </div>
            ''', unsafe_allow_html=True)
        else:
            # Assistant message - left side
            st.markdown(f'''
            <div class="chat-message-container assistant-message">
                <div class="assistant-bubble">
                    <span class="assistant-label">GENIEBOT SAYS...</span>
                    {message["content"]}
                </div>
            </div>
            ''', unsafe_allow_html=True)

# --- 7. HANDLE NEW CHAT INPUT ---
if prompt := st.chat_input("ASK A QUESTION TO GENIEBOT", disabled=st.session_state.processing):
    if st.session_state.chain and not st.session_state.processing:
        # Add user message to history and display immediately
        st.session_state.history.append({"role": "user", "content": prompt})
        
        # Display user message
        st.markdown(f'''
        <div class="chat-message-container user-message">
            <div class="user-bubble">
                {prompt}
            </div>
        </div>
        ''', unsafe_allow_html=True)

        # Show thinking indicator
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown('''
        <div class="chat-message-container assistant-message">
            <div class="assistant-bubble thinking">
                <span class="assistant-label">GENIEBOT SAYS...</span>
                🤔 Thinking...
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        try:
            # Get response from chain
            response = st.session_state.chain.invoke({"query": prompt})
            answer = response["result"]
            
            # Clear thinking indicator
            thinking_placeholder.empty()
            
            # Add assistant response to history
            st.session_state.history.append({"role": "assistant", "content": answer})
            
            # Rerun to show the complete conversation
            st.rerun()
            
        except Exception as e:
            error_msg = "I apologize, but I encountered an error while processing your question. Please try again."
            thinking_placeholder.empty()
            st.session_state.history.append({"role": "assistant", "content": error_msg})
            st.rerun()
                
    elif st.session_state.processing:
        st.warning("Please wait while files are being processed.")
    else:
        st.warning("Please upload PDF, CSV, or Word documents first to start the chat.")

# Add clear chat button
if st.session_state.history and not st.session_state.processing:
    if st.button("🗑️", type="secondary",help="Clear Chat History"):
        st.session_state.history.clear()
        st.rerun()