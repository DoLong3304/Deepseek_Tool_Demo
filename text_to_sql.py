import json
import re
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
import pandas as pd
import sqlparse

# Constants
MODEL_NAME = "deepseek-r1:8b"
TEMPLATE = """
You are a SQL generator. Your task is to generate valid and optimized SQL queries based on the provided database type, schema, and user questions.

1. Input Information:
- Database Type: {db_type}
- Schema: 
{schema}
- User Question: {query}

2. Output Requirements:
- Only return SQL queries wrapped in triple backticks with the `sql` language tag, like this:
```sql
SELECT * FROM table_name;
SELECT * FROM table2_name;
```
- Ensure the SQL queries strictly adhere to the provided schema. Do not use any table or column names not listed in the schema.
- If the user question is ambiguous, make reasonable assumptions based on the schema and generate the most likely query.
- Optimize the queries for performance where possible.
- Do not include any explanation, comments, or additional text outside the SQL block.

3. Notes:
- The user may ask multiple questions in a single input. Provide separate queries for each question.
- Use the schema to validate table and column names and ensure correctness.
"""

# Initialize model
model = OllamaLLM(model=MODEL_NAME)

# Check for session state integrity
def check_session_state():
    pass
    # if "connection_status" not in st.session_state or st.session_state.connection_status == "Not connected":
    #     st.markdown(
    #         """
    #         <style>
    #         .notification {
    #             position: fixed;
    #             top: 120px;
    #             right: 20px;
    #             background-color: #f8d7da;
    #             color: #721c24;
    #             padding: 10px 20px;
    #             border: 1px solid #f5c6cb;
    #             border-radius: 5px;
    #             box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    #             animation: slide-in 0.5s ease-out, fade-out 4s ease-in forwards;
    #         }
    #         @keyframes slide-in {
    #             from {
    #                 transform: translateX(100%);
    #             }
    #             to {
    #                 transform: translateX(0);
    #             }
    #         }
    #         @keyframes fade-out {
    #             to {
    #                 opacity: 0;
    #             }
    #         }
    #         </style>
    #         <div class="notification">
    #             ⚠️ Session state was reset. Please reconnect to the database.
    #         </div>
    #         """,
    #         unsafe_allow_html=True
    #     )

# Initialize session state
if "connection_status" not in st.session_state:
    st.session_state.connection_status = "Not connected"
if "schema" not in st.session_state:
    st.session_state.schema = None
if "db_type" not in st.session_state:
    st.session_state.db_type = "Unknown"

# Utility functions
def extract_schema(db_url):
    """Extracts the database schema and returns it as a text-oriented string with additional metadata."""
    check_session_state()
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            inspector = inspect(engine)
            schema_lines = []

            for table in inspector.get_table_names():
                schema_lines.append(f"Table: {table}")
                columns = inspector.get_columns(table)
                primary_keys = inspector.get_pk_constraint(table).get("constrained_columns", [])
                foreign_keys = inspector.get_foreign_keys(table)

                for col in columns:
                    col_info = f"  - {col['name']} ({col['type']})"
                    if col['name'] in primary_keys:
                        col_info += " [Primary Key]"
                    if not col['nullable']:
                        col_info += " [Not Null]"
                    if col.get('default') is not None:
                        col_info += f" [Default: {col['default']}]"
                    schema_lines.append(col_info)

                for fk in foreign_keys:
                    schema_lines.append(
                        f"  Foreign Key: {fk['constrained_columns'][0]} -> {fk['referred_table']}.{fk['referred_columns'][0]}"
                    )

                schema_lines.append("")  # Add a blank line between tables

            return "\n".join(schema_lines)
    except SQLAlchemyError as e:
        raise SQLAlchemyError(f"Failed to connect to the database: {str(e)}")

def get_database_type(db_url):
    """Extracts the database type from the URL."""
    match = re.match(r"(\w+):\/\/", db_url)
    return match.group(1) if match else "Unknown"

def to_sql_query(query, schema, db_type):
    """Generates SQL query using the model."""
    check_session_state()
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    chain = prompt | model
    response = chain.invoke({"query": query, "schema": schema, "db_type": db_type})
    extract_and_format_thinking(response)
    return clean_text(response)

def extract_and_format_thinking(text):
    """Extracts and formats the reasoning content."""
    think_content = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    if think_content:
        # Store the reasoning content in session state
        st.session_state.think_content = "\n".join(think_content)
        # Format the reasoning content
        formatted_content = st.session_state.think_content
        formatted_content = re.sub(r"\*\*(.*?)\*\*", r"**\1**", formatted_content)
        formatted_content = re.sub(r"'([a-zA-Z0-9_]+)'", r"`\1`", formatted_content)
        formatted_content = re.sub(r"__(.*?)__", r"_\1_", formatted_content)
        formatted_content = re.sub(r"~~(.*?)~~", r"~~\1~~", formatted_content)
        formatted_content = re.sub(r"==(.*?)==", r"<mark>\1</mark>", formatted_content)
        st.session_state.formatted_think_content = formatted_content

def clean_text(text):
    """Cleans and formats the generated SQL."""
    # Try to extract SQL from a ```sql block
    sql_match = re.search(r"```sql(.*?)```", text, flags=re.DOTALL)
    if sql_match:
        sql_code = sql_match.group(1).strip()
    else:
        # Fallback: Look for SQL-like patterns in the response
        sql_match = re.search(r"(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER).*", text, flags=re.DOTALL | re.IGNORECASE)
        sql_code = sql_match.group(0).strip() if sql_match else "-- SQL code could not be created."

    # Format the SQL code if valid
    if sql_code and not sql_code.startswith("-- SQL code could not be created"):
        return sqlparse.format(sql_code, reindent=True, reindent_aligned=True, keyword_case="upper")
    return sql_code

def connect_to_database():
    """Handles database connection and schema extraction."""
    check_session_state()
    try:
        st.session_state.schema = extract_schema(st.session_state.db_url)
        st.session_state.connection_status = "Connected successfully"
        st.session_state.db_type = get_database_type(st.session_state.db_url)
    except ModuleNotFoundError as e:
        st.session_state.connection_status = f"Driver missing: {str(e)}"
    except SQLAlchemyError as e:
        st.session_state.connection_status = f"Connection failed: {str(e)}"
    except Exception as e:
        st.session_state.connection_status = f"Unexpected error: {str(e)}"

def execute_sql(sql):
    """Executes the SQL query and displays results."""
    check_session_state()
    try:
        engine = create_engine(st.session_state.db_url)
        with engine.connect() as connection:
            # Split SQL into individual statements
            statements = sqlparse.split(sql)
            for statement in statements:
                statement = statement.strip()
                if statement:  # Skip empty statements
                    try:
                        result = connection.execute(text(statement))
                        if result.returns_rows:
                            # Handle SELECT or similar queries
                            data = result.fetchall()
                            columns = result.keys()
                            st.text("Query Results:")
                            st.dataframe(pd.DataFrame(data, columns=columns))
                        else:
                            # Handle INSERT, UPDATE, DELETE, etc.
                            st.success(f"Query executed successfully. Rows affected: {result.rowcount}")
                    except SQLAlchemyError as e:
                        st.error(f"Error executing statement: {statement}\n{str(e)}")
    except SQLAlchemyError as e:
        st.error(f"Error connecting to the database: {str(e)}")

# UI Components
def render_header():
    """Renders the app header."""
    st.markdown(
        """
        <div style="display: flex; align-items: center; margin-bottom: 20px;">
            <img src="https://cdn.prod.website-files.com/657639ebfb91510f45654149/67b4c293747043c9b6d86ec3_deepseek-color.png" alt="DeepSeek Logo" style="height: 50px; margin-right: 15px">
            <div>
                <h1 style="margin: 0; font-size: 2.5em;">TEXT-TO-SQL AI ASSISTANT</h1>
                <h3 style="margin: 0; color: gray;">Powered by deepseek-r1:8b</h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_connection_ui():
    """Renders the database connection UI."""
    st.text("")
    col1, col2, col3 = st.columns([4, 1, 1], vertical_alignment="bottom")
    with col1:
        st.text_input(
            "Enter your database URL:",
            placeholder="<type>://<username>:<password>@<host>:<port>/<database>",
            key="db_url",
            on_change=lambda: reset_connection_and_connect()  # Trigger reconnection on Enter
        )
    with col2:
        st.button("Connect", on_click=reset_connection_and_connect)  # Trigger reconnection on button click
    with col3:
        if st.session_state.connection_status.startswith("Connected"):
            st.text_input("Database Type", value=st.session_state.db_type, disabled=True)

    st.markdown(
        """
        <style>
        .status-badge {
            display: inline-block;
            padding: 5px 10px;
            border-radius: 5px;
            color: white;
        }
        .status-badge.connected { background-color: green; }
        .status-badge.error { background-color: red; }
        </style>
        """,
        unsafe_allow_html=True
    )

    status_class = "connected" if st.session_state.connection_status.startswith("Connected") else "error"
    st.markdown(f'<div class="status-badge {status_class}">{st.session_state.connection_status}</div>',
                unsafe_allow_html=True)

def render_query_ui():
    """Renders the query input and execution UI."""
    if st.session_state.connection_status == "Connected successfully":
        query = st.text_area(
            "Describe the data you want to retrieve from the database:",
            key="query",
            on_change=reset_and_generate_sql  # Trigger SQL generation on Enter
        )
        # Display the model's reasoning if available
        if "formatted_think_content" in st.session_state:
            with st.expander("Model's reasoning"):
                st.markdown(st.session_state.formatted_think_content, unsafe_allow_html=True)

        if "generated_sql" in st.session_state:
            st.text("Result(s):")
            statements = sqlparse.split(st.session_state.generated_sql)
            for statement in statements:
                st.code(statement, wrap_lines=True, language="sql")

        if "generated_sql" in st.session_state and st.button("Execute"):
            execute_sql(st.session_state.generated_sql)

def reset_state(keys):
    check_session_state()
    for key in keys:
        st.session_state[key] = None

def reset_connection_and_connect():
    reset_state(["connection_status", "schema", "db_type"])
    connect_to_database()

def reset_and_generate_sql():
    reset_state(["generated_sql"])
    if st.session_state.query:
        st.session_state.generated_sql = to_sql_query(
            st.session_state.query, st.session_state.schema, st.session_state.db_type
        )

# Main App
def main():
    check_session_state()
    render_header()
    render_connection_ui()
    render_query_ui()

if __name__ == "__main__":
    main()