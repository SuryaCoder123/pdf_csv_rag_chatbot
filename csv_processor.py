import pandas as pd
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import io

def process_csv(file_path):
    """
    Loads a CSV file, processes it, and converts it into text chunks for the knowledge base.
    Returns empty list if CSV processing fails.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if DataFrame is empty
        if df.empty:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("ERROR: The CSV file is empty or contains no data.")
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return []
        
        print("--- CSV PROCESSING DEBUG ---")
        print(f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns.")
        print(f"Columns: {list(df.columns)}")
        print(f"First few rows preview:")
        print(df.head().to_string())
        print("---------------------------")
        
        # Convert DataFrame to text documents
        documents = []
        
        # Method 1: Create a summary document with basic statistics and column info
        summary_text = create_csv_summary(df)
        summary_doc = Document(
            page_content=summary_text,
            metadata={"source": file_path, "type": "summary"}
        )
        documents.append(summary_doc)
        
        # Method 2: Convert each row to a text document
        for idx, row in df.iterrows():
            row_text = create_row_text(row, df.columns)
            row_doc = Document(
                page_content=row_text,
                metadata={"source": file_path, "type": "row", "row_number": idx + 1}
            )
            documents.append(row_doc)
        
        # Method 3: Create column-wise documents for better searchability
        for column in df.columns:
            column_text = create_column_text(df, column)
            column_doc = Document(
                page_content=column_text,
                metadata={"source": file_path, "type": "column", "column_name": column}
            )
            documents.append(column_doc)
        
        print(f"Created {len(documents)} documents from CSV data.")
        
        # Use text splitter for very long documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Smaller chunks for CSV data
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split documents if they're too long
        final_documents = []
        for doc in documents:
            if len(doc.page_content) > 1000:
                split_docs = text_splitter.split_documents([doc])
                final_documents.extend(split_docs)
            else:
                final_documents.append(doc)
        
        print(f"Final document count after splitting: {len(final_documents)}")
        
        return final_documents
        
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return []

def create_csv_summary(df):
    """Create a comprehensive summary of the CSV data."""
    summary_parts = []
    
    # Basic information
    summary_parts.append(f"CSV DATA SUMMARY")
    summary_parts.append(f"Total rows: {len(df)}")
    summary_parts.append(f"Total columns: {len(df.columns)}")
    summary_parts.append(f"Column names: {', '.join(df.columns)}")
    
    # Data types
    summary_parts.append("\nCOLUMN DATA TYPES:")
    for col, dtype in df.dtypes.items():
        summary_parts.append(f"- {col}: {dtype}")
    
    # Numerical columns statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary_parts.append("\nNUMERICAL COLUMNS STATISTICS:")
        for col in numeric_cols:
            stats = df[col].describe()
            summary_parts.append(f"\n{col}:")
            summary_parts.append(f"  - Mean: {stats['mean']:.2f}")
            summary_parts.append(f"  - Min: {stats['min']}")
            summary_parts.append(f"  - Max: {stats['max']}")
            summary_parts.append(f"  - Standard Deviation: {stats['std']:.2f}")
    
    # Categorical columns info
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        summary_parts.append("\nCATEGORICAL COLUMNS INFO:")
        for col in categorical_cols:
            unique_count = df[col].nunique()
            summary_parts.append(f"\n{col}:")
            summary_parts.append(f"  - Unique values: {unique_count}")
            if unique_count <= 10:
                unique_values = df[col].unique()
                summary_parts.append(f"  - Values: {', '.join(map(str, unique_values))}")
    
    # Missing values
    missing_data = df.isnull().sum()
    if missing_data.any():
        summary_parts.append("\nMISSING VALUES:")
        for col, missing_count in missing_data.items():
            if missing_count > 0:
                summary_parts.append(f"- {col}: {missing_count} missing values")
    
    return "\n".join(summary_parts)

def create_row_text(row, columns):
    """Convert a DataFrame row to readable text."""
    row_parts = []
    row_parts.append(f"Data Record:")
    
    for col in columns:
        value = row[col]
        if pd.isna(value):
            value = "Not Available"
        row_parts.append(f"{col}: {value}")
    
    return "\n".join(row_parts)

def create_column_text(df, column_name):
    """Create a text representation focusing on a specific column."""
    col_parts = []
    col_parts.append(f"COLUMN ANALYSIS: {column_name}")
    
    col_data = df[column_name]
    
    # Basic info
    col_parts.append(f"Data type: {col_data.dtype}")
    col_parts.append(f"Total values: {len(col_data)}")
    col_parts.append(f"Non-null values: {col_data.count()}")
    col_parts.append(f"Null values: {col_data.isnull().sum()}")
    
    # For numerical columns
    if pd.api.types.is_numeric_dtype(col_data):
        col_parts.append(f"Minimum value: {col_data.min()}")
        col_parts.append(f"Maximum value: {col_data.max()}")
        col_parts.append(f"Average value: {col_data.mean():.2f}")
        col_parts.append(f"Median value: {col_data.median()}")
    
    # For categorical columns
    elif pd.api.types.is_object_dtype(col_data):
        unique_values = col_data.unique()
        col_parts.append(f"Unique values count: {len(unique_values)}")
        
        # Show value counts for categorical data
        value_counts = col_data.value_counts().head(10)
        col_parts.append("Most frequent values:")
        for value, count in value_counts.items():
            col_parts.append(f"  - {value}: {count} occurrences")
    
    # Sample values
    sample_values = col_data.dropna().head(10).tolist()
    if sample_values:
        col_parts.append(f"Sample values: {', '.join(map(str, sample_values))}")
    
    return "\n".join(col_parts)